"""Appendix alignment, genuine baseline updates and experiment-matrix tests."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from algorithms.baselines import BaselineAgent, BaselineConfig, Features, generalized_advantages, conjugate_gradient
from config import AgentConfig, ObservationLayout
from encoder import make_encoder
from hyenv import MovingEnv
from hyenv.observations import FEATURES
from pdqn import PDQN
from run_experiments import make_plan
from summarize import interval, aggregate, write_csv

ROOT = Path(__file__).resolve().parents[1]


def tiny(**kwargs):
    return AgentConfig(latent_dim=16, token_dim=24, encoder_hidden=32, gru_hidden=32,
                       capacity=32, batch_size=4, sequence_length=2, **kwargs)


class AppendixTests(unittest.TestCase):
    def test_9d_local_frame_velocity_time_and_capture(self):
        env = MovingEnv(n_targets=1, n_obstacles=0, observation_dim=9, max_step=100)
        env.reset(seed=6)
        env.agent.reset([0,0,0], yaw=np.pi/2, pitch=0)
        env.targets[:] = [[0.4,0,0]]
        env.agent.velocity[:] = [0.2,-0.4,0.6]
        env.current_step = 25
        # A world +x displacement is local -y when the AUV faces world +y.
        expected = [0.5, 0.5-0.4/(4*np.sqrt(3)), 0.5,
                    0.4/(2*np.sqrt(3)), 0, 0.55, 0.4, 0.65, 0.25]
        np.testing.assert_allclose(env.get_state(), expected, atol=1e-7)
        env.targets[:] = [[0.01,0,0]]
        self.assertEqual(env.get_state()[4], 1)

    def test_18d_attitude_and_vertical_boundary_slots(self):
        env = MovingEnv(n_targets=1, n_obstacles=0, observation_dim=18, max_step=100)
        env.reset(seed=6)
        env.agent.reset([0.2,-0.4,0.6], yaw=np.pi/2, pitch=np.pi/6)
        env.targets[:] = [[0,0,0]]
        env.current_step = 50
        expected = [0.6,0.3,0.8, 0.5,0.5,0.5,
                    np.sqrt(0.56)/(2*np.sqrt(3)),0,
                    0.5,0.5,0.5, 1,0.5, 0.75,(1+np.sqrt(3)/2)/2, 0.5, 0.2,0.8]
        np.testing.assert_allclose(env.get_state(), expected, atol=1e-7)

    def test_27d_current_obstacle_radius_and_missing_slot(self):
        env = MovingEnv(n_targets=1, n_obstacles=1, observation_dim=27)
        env.reset(seed=6)
        env.agent.reset([0,0,0])
        env.current[:] = [0.1,-0.05,0]
        env.obstacles[:] = [[0.4,0.2,-0.2]]
        expected_barrier = [1,0.25,0.5, 0.6,0.55,0.45,0.08, 0.5,0.5,0.5,0]
        np.testing.assert_allclose(env.get_state()[16:27], expected_barrier, atol=1e-7)

    def test_36d_yaw_rate_time_noise_and_confidence_slots(self):
        env = MovingEnv(n_targets=1, n_obstacles=0, observation_dim=36, max_step=100)
        env.reset(seed=6)
        env.yaw_rate = -0.5*env.max_turn
        env.current_step = 10
        env.noise[:] = env.noise_std*np.array([1,-1,0,2])
        state = env.get_state()
        np.testing.assert_allclose(state[18:20], [0.25,0.1], atol=1e-7)
        np.testing.assert_allclose(state[31:35], [0.75,0.25,0.5,0.5+np.arctan(2)/np.pi], atol=1e-7)
        self.assertAlmostEqual(float(state[35]), 0.5)

    def test_all_schemas_and_semantic_partitions(self):
        expected = {9: (5, 4, 0, 0), 18: (8, 8, 2, 0), 27: (8, 8, 11, 0), 36: (11, 9, 11, 5)}
        for dimension, groups in expected.items():
            env = MovingEnv(observation_dim=dimension)
            state, _ = env.reset(seed=6)
            self.assertEqual(state.shape, (dimension,))
            self.assertEqual(len(FEATURES[dimension]), dimension)
            self.assertEqual(env.layout.group_dims, groups)
            self.assertEqual(sorted(sum((list(x) for x in env.layout.group_indices), [])), list(range(dimension)))
            encoder = make_encoder(env.layout, tiny())
            output = encoder(torch.as_tensor(np.stack([state, state]))[:, None])
            self.assertEqual(output.shape, (2, 1, 16))
            output.square().mean().backward()
            for branch, indices in zip(encoder.branches, env.layout.group_indices):
                if indices:
                    self.assertTrue(any(p.grad is not None for p in branch.parameters()))
                else:
                    self.assertTrue(all(p.grad is None for p in branch.parameters()))

    def test_published_order_and_nearest_obstacles(self):
        env = MovingEnv(n_targets=2, n_obstacles=2, observation_dim=36, noise_std=0)
        env.reset(seed=6)
        env.agent.reset([0, 0, 0], yaw=0, pitch=0)
        env.targets[:] = [[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]]
        env.obstacles[:] = [[0.8, 0.8, 0.8], [0.1, 0.2, 0.3]]
        env.noise[:] = 0
        state = env.get_state()
        np.testing.assert_allclose(state[:3], [0.5]*3)
        np.testing.assert_allclose(state[3:6], [0.65, 0.75, 0.85])
        np.testing.assert_allclose(state[6:9], [0.575, 0.625, 0.675])
        np.testing.assert_allclose(state[14:18], [0.5, 1, 0.5, 1])
        np.testing.assert_allclose(state[23:26], [0.525, 0.55, 0.575])
        np.testing.assert_allclose(state[31:35], [0.5]*4)
        self.assertEqual(state[35], 1)
        env.captured[0] = True
        np.testing.assert_allclose(env.get_state()[3:6], [0.7, 0.8, 0.9])

    def test_energy_reference_equation_42(self):
        env = MovingEnv(max_step=1, n_obstacles=0, propulsion_weight=2, energy_reference=4)
        env.reset(seed=3)
        _, _, _, _, info = env.step((0, np.ones(4, dtype=np.float32)))
        self.assertAlmostEqual(info["energy_cost"], 2/4*env.delta_t)


class ExperimentTests(unittest.TestCase):
    def test_second_order_gru_matches_native_gru(self):
        model = Features(ObservationLayout(observation_dim=36), tiny()).eval()
        states = torch.rand(2,3,36)
        native = model(states)
        model.manual_gru = True
        manual = model(states)
        torch.testing.assert_close(native, manual)
        gradient = torch.autograd.grad(manual.square().sum(), model.gru.weight_ih_l0, create_graph=True)[0]
        second = torch.autograd.grad(gradient.square().sum(), model.gru.weight_ih_l0)[0]
        self.assertTrue(torch.isfinite(second).all())

    def test_aggregation_uses_seed_n_and_preserves_out_of_bin_paths(self):
        template = json.loads((ROOT / "experiments/paper.json").read_text())
        plan = make_plan(template, "comparison", seeds=(42,43,44), dimensions=(36,), algorithms=("dhea_mecd",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, run in enumerate(plan, start=1):
                run.update(training_status="complete", evaluation_status="complete")
                folder = root / run["run_id"]
                (folder / "evaluation").mkdir(parents=True)
                rows = [{"episode": i, "success": True, "collision": False, "reward": number,
                         "energy_cost": number, "trajectory_length": length,
                         "inference_mean_ms": 1, "inference_p95_ms": 2} for i,length in enumerate((50,600))]
                write_csv(folder / "evaluation/evaluation.csv", rows)
                write_csv(folder / "training.csv", [{"episode":0,"reward":number}])
            (root / "manifest.json").write_text(json.dumps({"protocol":"smoke", "episodes":1,
                                                          "eval_episodes":2, "runs":plan}))
            result = aggregate(root, length_bins=[200,300,350,400,500])
            energy = next(r for r in result[2] if r["metric"] == "energy_cost")
            self.assertEqual(energy["n_seeds"], 3)
            self.assertEqual(energy["n_eval_episodes"], 6)
            self.assertEqual(energy["sd"], 1)
            self.assertEqual(sum(r["count"] for r in result[4]), 6)
            self.assertEqual(sum(r["fraction"] for r in result[4]), 3)

    def test_complete_matrix_120_unique_runs(self):
        template = json.loads((ROOT / "experiments/paper.json").read_text())
        plan = make_plan(template)
        self.assertEqual(len(plan), 120)
        self.assertEqual(len(set(r["run_id"] for r in plan)), 120)
        self.assertEqual(template["environment"]["observation_dim"], 36)
        self.assertEqual(len(make_plan(template, "comparison")), 96)
        self.assertEqual(len(make_plan(template, "ablation")), 18)
        self.assertEqual(len(make_plan(template, "topk")), 15)

    def test_seed_ci_and_time_limit_gae(self):
        result = interval([1, 2, 3])
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["sd"], 1)
        self.assertAlmostEqual(result["ci_high"], 4.4841377, places=5)
        self.assertIsNone(interval([1])["ci_high"])
        # A truncation bootstraps the final value, but cannot import the next episode's advantage.
        advantage = generalized_advantages(np.array([1., 10.]), np.zeros(2), np.array([2., 3.]),
                                          np.array([0, 1]), np.ones(2), 0.9, 0.95)
        np.testing.assert_allclose(advantage, [2.8, 10.])
        matrix = torch.diag(torch.tensor([2., 3.]))
        solution = conjugate_gradient(lambda x: matrix@x, torch.tensor([4., 9.]), 10)
        torch.testing.assert_close(solution, torch.tensor([2., 3.]))

    def test_ablations_and_k_sweep_can_update(self):
        for variant in ["no_encoder_attention", "no_mecd", *range(1, 6)]:
            cfg = tiny(use_encoder_attention=variant != "no_encoder_attention",
                       use_moe=variant != "no_mecd", top_k=variant if isinstance(variant, int) else 2)
            env = MovingEnv(observation_dim=18, max_step=4)
            agent = PDQN(env.layout, cfg)
            state, _ = env.reset(seed=6)
            for _ in range(4):
                action, parameters = agent.choose_action(state)
                nxt, reward, terminal, end, _ = env.step((int(action[0]), parameters[0]))
                agent.store_transition(state, action[0], parameters[0], reward, nxt, terminal, end)
                state = nxt
            self.assertTrue(np.isfinite(agent.learn()).all(), variant)

    def test_all_six_baselines_update_and_checkpoint(self):
        for name in ("ppo", "ac", "trpo", "dqn", "ddpg", "sac"):
            for dimension in (9, 18, 27, 36):
                torch.manual_seed(42)
                env = MovingEnv(observation_dim=dimension, max_step=4)
                cfg = tiny()
                base = BaselineConfig(rollout_steps=4, epochs=1, cg_steps=2, line_search_steps=3)
                agent = BaselineAgent(name, env.layout, cfg, base)
                state, _ = env.reset(seed=6)
                main = "policy" if name in ("ppo", "ac", "trpo") else "q"
                before = {k: v.clone() for k, v in agent.networks[main].state_dict().items()}
                for _ in range(4):
                    action, parameters = agent.choose_action(state)
                    nxt, reward, terminal, end, _ = env.step((int(action[0]), parameters[0]))
                    agent.store_transition(state, action[0], parameters[0], reward, nxt, terminal, end)
                    state = nxt
                self.assertTrue(np.isfinite(agent.learn()).all(), (name, dimension))
                if name != "trpo" or agent.last_diagnostics["accepted"]:
                    self.assertTrue(any(not torch.equal(v, agent.networks[main].state_dict()[k]) for k, v in before.items()))
                if name == "trpo":
                    self.assertLessEqual(agent.last_diagnostics["kl"], base.max_kl)
                if dimension == 36:
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "model.pt"
                        agent.save(path, 1, {"test": True})
                        loaded, episode = BaselineAgent.load(path)
                        self.assertEqual(episode, 1)
                        self.assertEqual(loaded.metadata, {"test": True})
                        agent.reset_episode()
                        loaded.reset_episode()
                        a, p = agent.choose_action(state, explore=False)
                        b, q = loaded.choose_action(state, explore=False)
                        np.testing.assert_array_equal(a, b)
                        np.testing.assert_allclose(p, q)
                        self.assertEqual(loaded.learn_step_counter, 1)


if __name__ == "__main__":
    unittest.main()
