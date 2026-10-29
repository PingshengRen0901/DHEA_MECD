"""DHEA-MECD learner, paper Algorithm 1 and Eqs. (24)-(36)."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from config import AgentConfig, ObservationLayout
from encoder import make_encoder
from PER import PrioritizedExperienceReplay


class ExpertNetwork(nn.Module):
    """Independent GRU-MLP and learned stage-specific latent perception, Eq. 28."""
    def __init__(self, latent_dim, config, stage):
        super().__init__()
        self.gru = nn.GRU(latent_dim, config.gru_hidden, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(config.gru_hidden + config.param_dim, config.expert_hidden),
                                 nn.ReLU(), nn.Linear(config.expert_hidden, config.action_dim))
        # Stage supports are learned in latent coordinates, not raw-state indices.
        self.feature_weights = nn.Parameter(torch.full((latent_dim,), -2.0))
        with torch.no_grad():
            self.feature_weights[stage::config.num_experts] = 2.0
        self.perception = nn.Sequential(nn.Linear(latent_dim, config.expert_hidden), nn.ReLU(),
                                        nn.Linear(config.expert_hidden, config.action_dim))
        self.stage_bias = nn.Parameter(torch.zeros(config.action_dim))

    def forward(self, latent_sequence, parameters):
        outputs, _ = self.gru(latent_sequence)
        q = self.mlp(torch.cat([outputs[:, -1], parameters], dim=-1))
        stage_features = latent_sequence[:, -1] * self.feature_weights.sigmoid()
        return q + self.perception(stage_features) + self.stage_bias


class ImprovedMoEGatingNetwork(nn.Module):
    """Bilinear latent-to-expert-key matching and sparse softmax, Eqs. 26-27."""
    def __init__(self, input_dim, num_experts=5, hidden_dim=128, k=2):
        super().__init__()
        if not 1 <= k <= num_experts:
            raise ValueError("Invalid top-k")
        self.k = k
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.expert_keys = nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(num_experts))
        nn.init.normal_(self.expert_keys, std=hidden_dim ** -0.5)

    def forward(self, latent):
        scores = self.query(latent) @ self.expert_keys.T + self.bias
        values, indices = scores.topk(self.k, dim=-1)
        return torch.zeros_like(scores).scatter(1, indices, values.softmax(dim=-1))


class MoEQNet(nn.Module):
    stage_names = ("acceleration", "cruising", "braking", "height_adjustment", "fine_tuning")

    def __init__(self, state_dim, action_dim=4, num_experts=5, k=2, config=None):
        super().__init__()
        cfg = config or AgentConfig(latent_dim=state_dim, action_dim=action_dim,
                                    num_experts=num_experts, top_k=k)
        self.experts = nn.ModuleList([ExpertNetwork(state_dim, cfg, i) for i in range(cfg.num_experts)])
        self.gating_network = ImprovedMoEGatingNetwork(state_dim, cfg.num_experts, k=cfg.top_k)
        self.action_dim = cfg.action_dim

    def forward(self, latent_sequence, parameters):
        if latent_sequence.ndim == 2:
            latent_sequence = latent_sequence.unsqueeze(1)
        gates = self.gating_network(latent_sequence[:, -1])
        combined = latent_sequence.new_zeros((latent_sequence.shape[0], self.action_dim))
        # Sparse dispatch: each expert processes only the rows routing to it.
        for i, expert in enumerate(self.experts):
            rows = torch.nonzero(gates[:, i] > 0, as_tuple=True)[0]
            if rows.numel():
                q = expert(latent_sequence.index_select(0, rows), parameters.index_select(0, rows))
                combined = combined.index_add(0, rows, q * gates[rows, i, None])
        return combined, gates


class Xnet(nn.Module):
    def __init__(self, s_dim, param_dim=4, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(s_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 64),
                                 nn.ReLU(), nn.Linear(64, param_dim), nn.Tanh())

    def forward(self, latent_sequence):
        if latent_sequence.ndim == 2:
            latent_sequence = latent_sequence.unsqueeze(1)
        sequence, _ = self.gru(latent_sequence)
        return self.mlp(sequence[:, -1])


class SingleQNet(nn.Module):
    """Single GRU-MLP head used by PDQN and the w/o MECD ablation."""
    def __init__(self, config):
        super().__init__()
        self.gru = nn.GRU(config.latent_dim, config.gru_hidden, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(config.gru_hidden + config.param_dim, config.expert_hidden),
                                 nn.ReLU(), nn.Linear(config.expert_hidden, config.action_dim))

    def forward(self, latent_sequence, parameters):
        output, _ = self.gru(latent_sequence)
        q = self.mlp(torch.cat([output[:, -1], parameters], dim=-1))
        return q, q.new_ones((q.shape[0], 1))


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(parameter, tau)
        for target_buffer, buffer in zip(target.buffers(), source.buffers()):
            if target_buffer.is_floating_point():
                target_buffer.lerp_(buffer, tau)
            else:
                target_buffer.copy_(buffer)


def td_target(reward, terminated, next_q, gamma):
    return reward + gamma * (1 - terminated) * next_q.max(dim=1, keepdim=True).values


class PDQN:
    """Historical class name retained as the DHEA-MECD entry point."""
    def __init__(self, layout=None, config=None, seed=42):
        self.layout = layout or ObservationLayout()
        self.config = config or AgentConfig()
        cfg = self.config
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(seed)
        self.encoder = make_encoder(self.layout, cfg).to(self.device)
        self.xnet = Xnet(cfg.latent_dim, cfg.param_dim, cfg.gru_hidden).to(self.device)
        self.eval_qnet = (MoEQNet(cfg.latent_dim, config=cfg) if cfg.use_moe else SingleQNet(cfg)).to(self.device)
        self.target_encoder = deepcopy(self.encoder).eval().requires_grad_(False)
        self.target_xnet = deepcopy(self.xnet).eval().requires_grad_(False)
        self.target_qnet = deepcopy(self.eval_qnet).eval().requires_grad_(False)
        self.optimizer_q = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.eval_qnet.parameters()), lr=cfg.lr_q)
        self.optimizer_x = torch.optim.Adam(self.xnet.parameters(), lr=cfg.lr_actor)
        self.memory = PrioritizedExperienceReplay(cfg.capacity, alpha=0.6 if cfg.prioritized else 0.0, seed=seed)
        self.memory_counter = self.learn_step_counter = 0
        self.history = deque(maxlen=cfg.sequence_length)
        self.last_gate_weights = np.zeros(cfg.num_experts)

    def reset_episode(self):
        self.history.clear()

    def epsilon(self, episode):
        cfg = self.config
        return max(cfg.epsilon_start - episode * cfg.epsilon_decay, cfg.epsilon_min)

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def choose_action(self, state, episode=0, explore=True):
        self.history.append(np.asarray(state, dtype=np.float32).copy())
        states = [self.history[0]] * (self.config.sequence_length - len(self.history)) + list(self.history)
        # Dropout and BatchNorm must be deterministic during rollout/evaluation.
        modules = (self.encoder, self.xnet, self.eval_qnet)
        modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            latent = self.encoder(self._tensor(np.stack(states)[None]))
            parameters = self.xnet(latent)
            q, gates = self.eval_qnet(latent, parameters)
            self.last_gate_weights = gates[0].cpu().numpy()
            action = q.argmax(dim=1).cpu().numpy()
            parameters = parameters.cpu().numpy()
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)
        if explore and self.rng.random() < self.epsilon(episode):
            action = np.array([self.rng.integers(self.config.action_dim)])
            parameters = self.rng.uniform(-1, 1, (1, self.config.param_dim)).astype(np.float32)
        return action, parameters

    def store_transition(self, state, action, parameters, reward, next_state,
                         terminated=False, truncated=False):
        row = np.concatenate([np.asarray(state).reshape(-1), [int(np.asarray(action).item())],
                              np.asarray(parameters).reshape(-1), [reward, terminated, terminated or truncated],
                              np.asarray(next_state).reshape(-1)])
        expected = 2 * self.layout.state_dim + self.config.param_dim + 4
        if row.size != expected or not np.isfinite(row).all():
            raise ValueError("Malformed replay transition")
        self.memory.store(row)
        self.memory_counter += 1

    def learn(self):
        cfg = self.config
        if len(self.memory) < cfg.batch_size:
            return None
        indices, rows, weights = self.memory.sample(cfg.batch_size)
        n, p = self.layout.state_dim, cfg.param_dim
        states = self.memory.state_sequences(indices, n, cfg.sequence_length)
        next_states = np.concatenate([states[:, 1:], rows[:, -n:][:, None]], axis=1)
        actions = torch.as_tensor(rows[:, n], dtype=torch.long, device=self.device).unsqueeze(1)
        parameters = self._tensor(rows[:, n + 1:n + 1 + p])
        rewards = self._tensor(rows[:, n + 1 + p:n + 2 + p])
        terminal = self._tensor(rows[:, n + 2 + p:n + 3 + p])
        latent = self.encoder(self._tensor(states))
        q, _ = self.eval_qnet(latent, parameters)
        selected_q = q.gather(1, actions)
        with torch.no_grad():
            next_latent = self.target_encoder(self._tensor(next_states))
            next_parameters = self.target_xnet(next_latent)
            next_q, _ = self.target_qnet(next_latent, next_parameters)
            target = td_target(rewards, terminal, next_q, cfg.gamma)
        errors = target - selected_q
        loss_q = (self._tensor(weights) * errors.square()).mean()
        self.optimizer_q.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.eval_qnet.parameters()), cfg.grad_clip)
        self.optimizer_q.step()
        self.optimizer_q.zero_grad(set_to_none=True)

        # Eq. (35) uses the replayed discrete action. Q and encoder remain fixed.
        self.eval_qnet.requires_grad_(False)
        encoder_mode = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                actor_latent = self.encoder(self._tensor(states))
            actor_parameters = self.xnet(actor_latent)
            actor_q, _ = self.eval_qnet(actor_latent, actor_parameters)
            loss_actor = -actor_q.gather(1, actions).mean()
            self.optimizer_x.zero_grad(set_to_none=True)
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.xnet.parameters(), cfg.grad_clip)
            self.optimizer_x.step()
        finally:
            self.eval_qnet.requires_grad_(True)
            self.encoder.train(encoder_mode)
        self.memory.batch_update(indices, errors.detach().abs())
        self.learn_step_counter += 1
        update = cfg.target_update == "soft" or self.learn_step_counter % cfg.target_update_interval == 0
        if update:
            for target_module, module in ((self.target_encoder, self.encoder),
                                          (self.target_xnet, self.xnet), (self.target_qnet, self.eval_qnet)):
                soft_update(target_module, module, cfg.tau if cfg.target_update == "soft" else 1.0)
        return float(loss_q.detach()), float(loss_actor.detach())

    def save(self, path, episode, metadata=None):
        """Episode-boundary snapshot, including replay and all random generators."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {name: getattr(self, name).state_dict() for name in (
            "encoder", "target_encoder", "xnet", "target_xnet", "eval_qnet", "target_qnet",
            "optimizer_q", "optimizer_x")}
        state.update(kind="dhea_mecd", metadata=metadata or {}, config=asdict(self.config), layout=asdict(self.layout), episode=episode,
                     memory=self.memory, memory_counter=self.memory_counter,
                     learn_step_counter=self.learn_step_counter, rng=self.rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                     python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = AgentConfig(**{**state["config"], "device": device})
        agent = cls(ObservationLayout(**state["layout"]), cfg)
        for name in ("encoder", "target_encoder", "xnet", "target_xnet", "eval_qnet", "target_qnet",
                     "optimizer_q", "optimizer_x"):
            getattr(agent, name).load_state_dict(state[name])
        agent.memory = state["memory"]
        agent.metadata = state.get("metadata", {})
        agent.memory_counter, agent.learn_step_counter = state["memory_counter"], state["learn_step_counter"]
        agent.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        return agent, state["episode"]
