"""Behavioral regression checks for the paper's core algorithmic contracts."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from config import AgentConfig, ObservationLayout
from encoder import Encoder3D_FeatureEnhanced
from hyenv import MovingEnv
from PER import PrioritizedExperienceReplay
from pdqn import MoEQNet, PDQN, soft_update, td_target


def small_config(**kwargs):
    cfg = AgentConfig(latent_dim=16, token_dim=24, encoder_hidden=32, gru_hidden=32,
                      capacity=32, batch_size=4, sequence_length=3)
    return replace(cfg, **kwargs)


class EnvironmentTests(unittest.TestCase):
    def test_gym_contract(self):
        env = MovingEnv(n_targets=2, n_obstacles=1)
        check_env(env, skip_render_check=True)
        wrapped = gym.make("AUVTracking-v0", n_targets=2)
        state, _ = wrapped.reset(seed=5)
        self.assertTrue(wrapped.observation_space.contains(state))
        wrapped.close()

    def test_seeded_trajectory_and_four_groups(self):
        a, b = MovingEnv(), MovingEnv()
        x, _ = a.reset(seed=42)
        y, _ = b.reset(seed=42)
        np.testing.assert_array_equal(x, y)
        self.assertEqual(a.layout.group_dims, (18, 8, 9, 4))
        initial_targets = a.targets.copy()
        for _ in range(5):
            action = (0, np.array([0.2, 0, 0, 0], dtype=np.float32))
            left, right = a.step(action), b.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])
            self.assertTrue(a.observation_space.contains(left[0]))
        self.assertFalse(np.array_equal(initial_targets, a.targets))

    def test_capture_each_target_once_and_complete(self):
        env = MovingEnv(n_targets=2, n_obstacles=0, target_speed=0, current_strength=0, noise_std=0)
        env.reset(seed=4)
        env.targets[:] = [[0, 0, 0], [0.7, 0.7, 0.7]]
        env.agent.reset([0, 0, 0])
        _, _, done, _, info = env.step((3, np.ones(4, dtype=np.float32)))
        self.assertFalse(done)
        self.assertEqual(info["captured_targets"], 1)
        self.assertEqual(info["reward_components"]["goal"], 10)
        _, _, _, _, info = env.step((3, np.ones(4, dtype=np.float32)))
        self.assertEqual(info["reward_components"]["goal"], 0)
        env.agent.reset(env.targets[1])
        _, _, done, _, info = env.step((3, np.ones(4, dtype=np.float32)))
        self.assertTrue(done)
        self.assertTrue(info["success"])

    def test_collision_boundary_and_timeout(self):
        for event in ("collision", "boundary", "timeout"):
            env = MovingEnv(n_targets=1, n_obstacles=1, max_step=1,
                            target_speed=0, obstacle_speed=0, noise_std=0, current_strength=0)
            env.reset(seed=1)
            env.agent.reset([0, 0, 0])
            env.targets[:] = [0.8, 0.8, 0.8]
            env.obstacles[:] = [0.6, -0.6, 0.6]
            if event == "collision":
                env.obstacles[:] = env.agent.position
            elif event == "boundary":
                env.agent.position[:] = [1.1, 0, 0]
            _, reward, terminated, truncated, info = env.step((3, np.ones(4, dtype=np.float32)))
            self.assertEqual(terminated, event != "timeout")
            self.assertEqual(truncated, event == "timeout")
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
            if event != "timeout":
                self.assertLess(info["reward_components"][event], 0)

    def test_braking_parameter_controls_strength(self):
        env = MovingEnv(n_obstacles=0, noise_std=0, current_strength=0)
        env.reset(seed=3)
        env.agent.reset([0, 0, 0])
        env.agent.velocity[:] = [1, 0, 0]
        env.step((3, np.array([0, 0, 0, -1], dtype=np.float32)))
        weak = env.agent.speed
        env.reset(seed=3)
        env.agent.reset([0, 0, 0])
        env.agent.velocity[:] = [1, 0, 0]
        env.step((3, np.ones(4, dtype=np.float32)))
        self.assertLess(env.agent.speed, weak)


class ReplayTests(unittest.TestCase):
    @staticmethod
    def row(value, end=False):
        # state(1), action, parameters(4), reward, terminated, episode_end, next_state(1)
        return np.array([value, 0, 0, 0, 0, 0, 0, end, end, value + 1], dtype=np.float32)

    def test_partial_buffer_finite_weights_and_updates(self):
        replay = PrioritizedExperienceReplay(31, seed=1)
        for i in range(7):
            replay.store(self.row(i))
        indices, rows, weights = replay.sample(4)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(rows[:, 0] < 7))
        first = replay.state_sequences([replay.tree.capacity - 1], 1, 3)
        np.testing.assert_array_equal(first[0, :, 0], [0, 0, 0])
        replay.batch_update(indices, torch.ones(4, 1))
        self.assertGreater(replay.tree.total_p, 0)

    def test_recurrence_respects_episode_and_ring_boundaries(self):
        replay = PrioritizedExperienceReplay(4)
        for i in range(6):
            replay.store(self.row(i, end=i == 3))
        offset = replay.tree.capacity - 1
        # Slots 0 and 1 contain rows 4, 5; row 3 terminates its episode.
        seq = replay.state_sequences([offset, offset + 1, offset + 2], 1, 3)
        np.testing.assert_array_equal(seq[:, :, 0], [[4, 4, 4], [4, 4, 5], [2, 2, 2]])


class NetworkTests(unittest.TestCase):
    def test_all_modalities_receive_gradient(self):
        encoder = Encoder3D_FeatureEnhanced(hidden_dim=32, output_dim=16, token_dim=24)
        out = encoder(torch.rand(2, 3, ObservationLayout().state_dim))
        self.assertEqual(out.shape, (2, 3, 16))
        out.square().mean().backward()
        for branch in encoder.branches:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters()))
        self.assertTrue(torch.isfinite(encoder(torch.rand(1, ObservationLayout().state_dim))).all())

    def test_sparse_execution_and_inactive_gradients(self):
        cfg = small_config()
        network = MoEQNet(cfg.latent_dim, config=cfg)
        with torch.no_grad():
            network.gating_network.query.weight.zero_()
            network.gating_network.bias.copy_(torch.tensor([10., 9., 0., -1., -2.]))
        counts = [0] * 5
        hooks = [e.register_forward_hook(lambda m, i, o, j=j: counts.__setitem__(j, counts[j] + 1))
                 for j, e in enumerate(network.experts)]
        q, gates = network(torch.rand(3, 3, cfg.latent_dim), torch.rand(3, 4))
        q.sum().backward()
        self.assertEqual(counts, [1, 1, 0, 0, 0])
        torch.testing.assert_close(gates.sum(dim=1), torch.ones(3))
        self.assertTrue(((gates > 0).sum(dim=1) == 2).all())
        for expert in list(network.experts)[2:]:
            self.assertTrue(all(p.grad is None for p in expert.parameters()))
        for hook in hooks:
            hook.remove()

    def test_terminal_mask_and_soft_update(self):
        target = td_target(torch.tensor([[2.], [2.]]), torch.tensor([[1.], [0.]]),
                           torch.tensor([[10., 20.], [10., 20.]]), 0.95)
        torch.testing.assert_close(target, torch.tensor([[2.], [21.]]))
        a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        before = a.weight.detach().clone()
        soft_update(a, b, 0.1)
        torch.testing.assert_close(a.weight, before * 0.9 + b.weight * 0.1)

    def test_history_changes_actor_output(self):
        agent = PDQN(config=small_config())
        actor = agent.xnet.eval()
        first = torch.rand(2, 3, agent.config.latent_dim)
        first[1, -1] = first[0, -1]
        result = actor(first)
        self.assertFalse(torch.allclose(result[0], result[1]))

    def test_learning_checkpoint_and_deterministic_evaluation(self):
        torch.manual_seed(7)
        env = MovingEnv(n_targets=1, n_obstacles=0, max_step=8)
        agent = PDQN(env.layout, small_config())
        state, _ = env.reset(seed=10)
        agent.reset_episode()
        for _ in range(8):
            action, parameters = agent.choose_action(state)
            nxt, reward, done, truncated, _ = env.step((int(action[0]), parameters[0]))
            agent.store_transition(state, action[0], parameters[0], reward, nxt, done, truncated)
            state = nxt
            if done or truncated:
                break
        before = {k: v.clone() for k, v in agent.encoder.state_dict().items()}
        losses = agent.learn()
        self.assertIsNotNone(losses)
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(any(not torch.equal(v, agent.encoder.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in agent.target_qnet.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.eval_qnet.parameters()))
        agent.reset_episode()
        a1, p1 = agent.choose_action(state, explore=False)
        agent.reset_episode()
        a2, p2 = agent.choose_action(state, explore=False)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(p1, p2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            agent.save(path, episode=3)
            loaded, episode = PDQN.load(path)
            self.assertEqual(episode, 3)
            self.assertEqual(len(loaded.memory), len(agent.memory))
            # Restoring RNG and replay must reproduce the next complete update.
            expected = agent.learn()
            loaded, _ = PDQN.load(path)
            actual = loaded.learn()
            np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-7)
            for key, value in agent.encoder.state_dict().items():
                torch.testing.assert_close(value, loaded.encoder.state_dict()[key])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
