"""PPO, vanilla AC, TRPO, DQN, DDPG and SAC comparison implementations.

Baseline hybrid-action adaptations are explicit implementation choices: the
manuscript does not define them. See README for distributions and hyperparameters.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal, kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from config import AgentConfig, ObservationLayout
from encoder import make_encoder
from PER import PrioritizedExperienceReplay
from pdqn import soft_update


@dataclass
class BaselineConfig:
    rollout_steps: int = 256
    epochs: int = 10
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    max_kl: float = 0.01
    cg_steps: int = 10
    damping: float = 0.1
    line_search_steps: int = 10
    action_noise: float = 0.1
    sac_alpha: float = 0.2
    dqn_levels: int = 5

    def __post_init__(self):
        if min(self.rollout_steps, self.epochs, self.cg_steps, self.line_search_steps) < 1 or self.dqn_levels < 2:
            raise ValueError("Positive update lengths and at least two DQN levels required")
        if not 0 < self.clip_ratio < 1 or not 0 <= self.gae_lambda <= 1 or self.max_kl <= 0:
            raise ValueError("Invalid PPO/GAE/TRPO configuration")
        if min(self.damping, self.action_noise, self.sac_alpha, self.entropy_weight, self.value_weight) < 0:
            raise ValueError("Noise, loss weights and damping must be nonnegative")


class Features(nn.Module):
    def __init__(self, layout, config):
        super().__init__()
        self.encoder = make_encoder(layout, config)
        self.gru = nn.GRU(config.latent_dim, config.gru_hidden, batch_first=True)
        self.output_dim = config.gru_hidden
        self.manual_gru = False

    def forward(self, states):
        # Policy-ratio/KL calculations require deterministic features. eval()
        # freezes dropout/BN statistics but does not disable parameter gradients.
        self.encoder.eval()
        z = self.encoder(states)
        if self.manual_gru:
            # Equivalent GRU equations using differentiable primitives. cuDNN
            # RNN backward does not support the double backward needed by TRPO.
            hidden = z.new_zeros((z.shape[0], self.gru.hidden_size))
            for token in z.unbind(1):
                ir, iz, inn = F.linear(token, self.gru.weight_ih_l0, self.gru.bias_ih_l0).chunk(3, -1)
                hr, hz, hn = F.linear(hidden, self.gru.weight_hh_l0, self.gru.bias_hh_l0).chunk(3, -1)
                reset, update = (ir+hr).sigmoid(), (iz+hz).sigmoid()
                candidate = (inn + reset*hn).tanh()
                hidden = (1-update)*candidate + update*hidden
            return hidden
        output, _ = self.gru(z)
        return output[:, -1]


class HybridPolicy(nn.Module):
    def __init__(self, layout, config):
        super().__init__()
        self.features = Features(layout, config)
        self.head = nn.Sequential(nn.Linear(config.gru_hidden, 64), nn.ReLU())
        self.logits = nn.Linear(64, 4)
        self.mean = nn.Linear(64, 4)
        self.log_std = nn.Parameter(torch.full((4,), -0.5))

    def distributions(self, states):
        h = self.head(self.features(states))
        return Categorical(logits=self.logits(h)), Normal(self.mean(h), self.log_std.clamp(-5, 2).exp())

    @staticmethod
    def log_prob(categorical, normal, discrete, raw):
        return categorical.log_prob(discrete) + (normal.log_prob(raw) -
               torch.log(1 - raw.tanh().square() + 1e-6)).sum(-1)


class ValueNet(nn.Module):
    def __init__(self, layout, config):
        super().__init__()
        self.features = Features(layout, config)
        self.head = nn.Sequential(nn.Linear(config.gru_hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, states):
        return self.head(self.features(states)).squeeze(-1)


class ContinuousActor(nn.Module):
    def __init__(self, layout, config, stochastic=False):
        super().__init__()
        self.features = Features(layout, config)
        self.head = nn.Sequential(nn.Linear(config.gru_hidden, 64), nn.ReLU(), nn.Linear(64, 16 if stochastic else 8))
        self.stochastic = stochastic

    def forward(self, states, deterministic=False):
        output = self.head(self.features(states))
        if not self.stochastic:
            return output.tanh(), None
        mean, log_std = output.chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        action = raw.tanh()
        log_prob = (distribution.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob


class QNet(nn.Module):
    def __init__(self, layout, config, output_dim=1, continuous=True):
        super().__init__()
        self.features = Features(layout, config)
        self.head = nn.Sequential(nn.Linear(config.gru_hidden + (8 if continuous else 0), 64),
                                  nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, states, action=None):
        h = self.features(states)
        return self.head(h if action is None else torch.cat([h, action], -1))


def generalized_advantages(rewards, values, next_values, terminals, ends, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (1 - terminals[i]) * next_values[i] - values[i]
        carry = delta + gamma * lam * (1 - ends[i]) * carry
        advantages[i] = carry
    return advantages


def conjugate_gradient(operator, b, iterations):
    x, residual = torch.zeros_like(b), b.clone()
    direction = residual.clone()
    norm = residual.dot(residual)
    for _ in range(iterations):
        product = operator(direction)
        denominator = direction.dot(product)
        if denominator <= 0 or norm < 1e-12:
            break
        step = norm / denominator
        x += step * direction
        residual -= step * product
        next_norm = residual.dot(residual)
        direction = residual + next_norm / norm * direction
        norm = next_norm
    return x


class BaselineAgent:
    def __init__(self, algorithm, layout, config=None, baseline_config=None, seed=42):
        if algorithm not in ("ppo", "ac", "trpo", "dqn", "ddpg", "sac"):
            raise ValueError(f"Unknown baseline {algorithm}")
        self.algorithm, self.layout = algorithm, layout
        self.config, self.baseline_config = config or AgentConfig(), baseline_config or BaselineConfig()
        c, b = self.config, self.baseline_config
        self.device = torch.device(c.device)
        self.rng = np.random.default_rng(seed)
        self.history = deque(maxlen=c.sequence_length)
        self.memory = PrioritizedExperienceReplay(c.capacity, alpha=0, seed=seed)
        self.rollout = []
        self.memory_counter = self.learn_step_counter = 0
        self.networks, self.optimizers = {}, {}
        if algorithm in ("ppo", "ac", "trpo"):
            self.networks = {"policy": HybridPolicy(layout, c), "value": ValueNet(layout, c)}
            if algorithm == "trpo":
                self.networks["policy"].features.manual_gru = True
        elif algorithm == "dqn":
            self.networks = {"q": QNet(layout, c, output_dim=4*b.dqn_levels, continuous=False)}
            self.networks["target_q"] = deepcopy(self.networks["q"]).requires_grad_(False)
        else:
            self.networks = {"actor": ContinuousActor(layout, c, stochastic=algorithm == "sac"), "q": QNet(layout, c)}
            if algorithm == "sac":
                self.networks["q2"] = QNet(layout, c)
            for name in list(self.networks):
                self.networks["target_" + name] = deepcopy(self.networks[name]).requires_grad_(False)
        for name, network in self.networks.items():
            network.to(self.device).eval()
            if not name.startswith("target_"):
                self.optimizers[name] = torch.optim.Adam(network.parameters(), lr=c.lr_actor if name in ("policy", "actor") else c.lr_q)
        self.last_diagnostics = {}
        self.metadata = {}

    def reset_episode(self):
        self.history.clear()

    def epsilon(self, episode):
        if self.algorithm != "dqn":
            return 0.0
        return max(self.config.epsilon_start - episode*self.config.epsilon_decay, self.config.epsilon_min)

    def tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def choose_action(self, state, episode=0, explore=True):
        self.history.append(np.asarray(state, dtype=np.float32).copy())
        states = np.stack([self.history[0]] * (self.config.sequence_length - len(self.history)) + list(self.history))
        self.pending_states = states
        batch = self.tensor(states[None])
        if self.algorithm in ("ppo", "ac", "trpo"):
            categorical, normal = self.networks["policy"].distributions(batch)
            discrete = categorical.sample() if explore else categorical.logits.argmax(-1)
            raw = normal.sample() if explore else normal.mean
            parameters = raw.tanh()
            self.pending_raw = raw[0].cpu().numpy()
            self.pending_log_prob = float(HybridPolicy.log_prob(categorical, normal, discrete, raw)[0])
        elif self.algorithm == "dqn":
            index = int(self.networks["q"](batch).argmax(-1)[0])
            if explore and self.rng.random() < self.epsilon(episode):
                index = int(self.rng.integers(4*self.baseline_config.dqn_levels))
            self.pending_index = index
            action, level = divmod(index, self.baseline_config.dqn_levels)
            params = np.zeros((1, 4), dtype=np.float32)
            params[0, action] = np.linspace(-1, 1, self.baseline_config.dqn_levels)[level]
            return np.array([action]), params
        else:
            vector, _ = self.networks["actor"](batch, deterministic=not explore)
            if explore and self.algorithm == "ddpg":
                vector = (vector + self.tensor(self.rng.normal(0, self.baseline_config.action_noise, vector.shape))).clamp(-1, 1)
            self.pending_vector = vector[0].cpu().numpy()
            discrete, parameters = vector[:, :4].argmax(-1), vector[:, 4:]
        return discrete.cpu().numpy(), parameters.cpu().numpy()

    def store_transition(self, state, action, parameters, reward, next_state, terminated=False, truncated=False):
        self.memory_counter += 1
        if self.algorithm in ("ppo", "ac", "trpo"):
            next_history = np.concatenate([self.pending_states[1:], np.asarray(next_state)[None]], axis=0)
            self.rollout.append((self.pending_states.copy(), int(action), self.pending_raw.copy(), reward,
                                 next_history, bool(terminated), bool(terminated or truncated), self.pending_log_prob))
        else:
            executed = self.pending_index if self.algorithm == "dqn" else self.pending_vector
            action_vector = np.array([executed]) if self.algorithm == "dqn" else executed
            # Same causal replay row tail as DHEA-MECD.
            self.memory.store(np.concatenate([state, action_vector, [reward, terminated, terminated or truncated], next_state]))

    def _step(self, name, loss):
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.networks[name].parameters(), self.config.grad_clip)
        self.optimizers[name].step()

    def learn(self, force=False):
        if self.algorithm in ("ppo", "ac", "trpo"):
            if not self.rollout or (not force and len(self.rollout) < self.baseline_config.rollout_steps):
                return None
            result = self._learn_on_policy()
        else:
            if len(self.memory) < self.config.batch_size:
                return None
            result = self._learn_off_policy()
        self.learn_step_counter += 1
        return result

    def _learn_on_policy(self):
        c, b = self.config, self.baseline_config
        states, actions, raw, rewards, nxt, terminal, ends, old_log = zip(*self.rollout)
        states, nxt, raw = self.tensor(np.stack(states)), self.tensor(np.stack(nxt)), self.tensor(np.stack(raw))
        actions = torch.tensor(actions, device=self.device)
        with torch.no_grad():
            values = self.networks["value"](states).cpu().numpy()
            next_values = self.networks["value"](nxt).cpu().numpy()
        advantage = generalized_advantages(np.array(rewards), values, next_values, np.array(terminal), np.array(ends), c.gamma, b.gae_lambda)
        returns = self.tensor(advantage + values)
        advantage = self.tensor((advantage - advantage.mean()) / max(float(advantage.std()), 1e-8))
        old_log = self.tensor(old_log)
        if self.algorithm == "trpo":
            policy_loss = self._trpo(states, actions, raw, old_log, advantage)
        else:
            for _ in range(b.epochs if self.algorithm == "ppo" else 1):
                categorical, normal = self.networks["policy"].distributions(states)
                log_prob = HybridPolicy.log_prob(categorical, normal, actions, raw)
                entropy = categorical.entropy() + normal.entropy().sum(-1)
                if self.algorithm == "ppo":
                    ratio = (log_prob - old_log).exp()
                    objective = torch.minimum(ratio*advantage, ratio.clamp(1-b.clip_ratio, 1+b.clip_ratio)*advantage)
                else:
                    objective = log_prob * advantage
                policy_loss = -objective.mean() - b.entropy_weight*entropy.mean()
                self._step("policy", policy_loss)
            policy_loss = float(policy_loss.detach())
        for _ in range(b.epochs if self.algorithm in ("ppo", "trpo") else 1):
            value_loss = b.value_weight * F.mse_loss(self.networks["value"](states), returns)
            self._step("value", value_loss)
        self.rollout.clear()
        return float(value_loss.detach()), policy_loss

    def _trpo(self, states, actions, raw, old_log, advantage):
        b = self.baseline_config
        policy = self.networks["policy"]
        parameters = list(policy.parameters())
        def flat_grad(output, create_graph=False):
            gradients = torch.autograd.grad(output, parameters, create_graph=create_graph, allow_unused=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                              for p, g in zip(parameters, gradients)])
        with torch.no_grad():
            old_cat, old_normal = policy.distributions(states)
        def objective_and_kl():
            cat, normal = policy.distributions(states)
            objective = ((HybridPolicy.log_prob(cat, normal, actions, raw) - old_log).exp()*advantage).mean()
            kl = (kl_divergence(old_cat, cat) + kl_divergence(old_normal, normal).sum(-1)).mean()
            return objective, kl
        objective, _ = objective_and_kl()
        gradient = flat_grad(objective).detach()
        def fisher(vector):
            _, kl = objective_and_kl()
            first = flat_grad(kl, create_graph=True)
            return flat_grad((first*vector).sum()).detach() + b.damping*vector
        direction = conjugate_gradient(fisher, gradient, b.cg_steps)
        curvature = direction.dot(fisher(direction))
        original = parameters_to_vector(parameters).detach().clone()
        accepted, actual_kl = False, 0.0
        if torch.isfinite(curvature) and curvature > 0:
            full_step = direction * torch.sqrt(2*b.max_kl / curvature)
            for j in range(b.line_search_steps):
                with torch.no_grad():
                    vector_to_parameters(original + (0.5**j)*full_step, parameters)
                    candidate, kl = objective_and_kl()
                    if torch.isfinite(candidate) and torch.isfinite(kl) and candidate > objective and kl <= b.max_kl:
                        accepted, actual_kl = True, float(kl)
                        break
        if not accepted:
            with torch.no_grad():
                vector_to_parameters(original, parameters)
        self.last_diagnostics = {"accepted": accepted, "kl": actual_kl}
        return -float(objective.detach())

    def _learn_off_policy(self):
        c, b = self.config, self.baseline_config
        indices, rows, _ = self.memory.sample(c.batch_size)
        n = self.layout.state_dim
        states = self.memory.state_sequences(indices, n, c.sequence_length)
        nxt = np.concatenate([states[:, 1:], rows[:, -n:][:, None]], axis=1)
        states, nxt = self.tensor(states), self.tensor(nxt)
        width = 1 if self.algorithm == "dqn" else 8
        executed = self.tensor(rows[:, n:n+width])
        reward = self.tensor(rows[:, n+width:n+width+1])
        terminal = self.tensor(rows[:, n+width+1:n+width+2])
        if self.algorithm == "dqn":
            with torch.no_grad():
                target = reward + c.gamma*(1-terminal)*self.networks["target_q"](nxt).max(-1, keepdim=True).values
            prediction = self.networks["q"](states).gather(1, executed.long())
            loss = F.mse_loss(prediction, target)
            self._step("q", loss)
            # Table I T_update=200 for the discrete DQN baseline.
            if (self.learn_step_counter + 1) % c.target_update_interval == 0:
                soft_update(self.networks["target_q"], self.networks["q"], 1.0)
            return float(loss.detach()), 0.0
        with torch.no_grad():
            next_action, log_prob = self.networks["target_actor" if self.algorithm == "ddpg" else "actor"](nxt)
            next_q = self.networks["target_q"](nxt, next_action)
            if self.algorithm == "sac":
                next_q = torch.minimum(next_q, self.networks["target_q2"](nxt, next_action)) - b.sac_alpha*log_prob
            target = reward + c.gamma*(1-terminal)*next_q
        q_names = ["q", "q2"] if self.algorithm == "sac" else ["q"]
        loss_q = 0.0
        for name in q_names:
            loss = F.mse_loss(self.networks[name](states, executed), target)
            self._step(name, loss)
            loss_q += float(loss.detach())
            self.networks[name].requires_grad_(False)
        generated, log_prob = self.networks["actor"](states)
        value = self.networks["q"](states, generated)
        if self.algorithm == "sac":
            value = torch.minimum(value, self.networks["q2"](states, generated))
            loss_actor = (b.sac_alpha*log_prob - value).mean()
        else:
            loss_actor = -value.mean()
        self._step("actor", loss_actor)
        for name in q_names:
            self.networks[name].requires_grad_(True)
        for name in self.networks:
            if name.startswith("target_"):
                soft_update(self.networks[name], self.networks[name[7:]], c.tau)
        return loss_q, float(loss_actor.detach())

    def save(self, path, episode, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(kind="baseline", algorithm=self.algorithm, config=asdict(self.config),
                     baseline_config=asdict(self.baseline_config), layout=asdict(self.layout),
                     networks={k: v.state_dict() for k, v in self.networks.items()},
                     optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
                     memory=self.memory, rollout=self.rollout, memory_counter=self.memory_counter,
                     learn_step_counter=self.learn_step_counter, episode=episode, metadata=metadata or {},
                     rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = AgentConfig(**{**state["config"], "device": device})
        agent = cls(state["algorithm"], ObservationLayout(**state["layout"]), cfg, BaselineConfig(**state["baseline_config"]))
        for name, values in state["networks"].items():
            agent.networks[name].load_state_dict(values)
        for name, values in state["optimizers"].items():
            agent.optimizers[name].load_state_dict(values)
        for name in ("memory", "rollout", "memory_counter", "learn_step_counter", "metadata"):
            setattr(agent, name, state[name])
        agent.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        return agent, state["episode"]
