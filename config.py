"""Paper Table I defaults and explicit implementation choices."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationLayout:
    n_targets: int = 3
    n_obstacles: int = 2
    observation_dim: int | None = None

    def __post_init__(self):
        if self.n_targets < 1 or self.n_obstacles < 0:
            raise ValueError("Require at least one target and nonnegative obstacle count")
        if self.observation_dim not in (None, 9, 18, 27, 36):
            raise ValueError("observation_dim must be 9, 18, 27, 36, or None (full state)")

    @property
    def group_indices(self):
        """Published appendix order, gathered into space/motion/barrier/noise."""
        if self.observation_dim is None:
            sizes = (3 + 5 * self.n_targets, 8, 3 + 3 * self.n_obstacles, 4)
            ends = [0]
            for size in sizes:
                ends.append(ends[-1] + size)
            return tuple(tuple(range(ends[i], ends[i + 1])) for i in range(4))
        if self.observation_dim == 9:
            return (tuple(range(5)), tuple(range(5, 9)), (), ())
        if self.observation_dim == 18:
            return (tuple(range(8)), tuple(range(8, 16)), (16, 17), ())
        if self.observation_dim == 27:
            return (tuple(range(8)), tuple(range(8, 16)), tuple(range(16, 27)), ())
        return (tuple(range(11)), tuple(range(11, 20)), tuple(range(20, 31)), tuple(range(31, 36)))

    @property
    def group_dims(self):
        # p_a, all p_j, all distances, capture flags; v_a, sin/cos attitude, time;
        # local current, obstacle positions; four noise sources (Section V-A).
        return tuple(len(indices) for indices in self.group_indices)

    @property
    def state_dim(self):
        return sum(self.group_dims)


@dataclass
class AgentConfig:
    latent_dim: int = 128
    token_dim: int = 96  # divisible by the six attention heads in Eq. (20)
    attention_heads: int = 6
    encoder_hidden: int = 128
    gru_hidden: int = 256
    expert_hidden: int = 64
    num_experts: int = 5
    top_k: int = 2
    param_dim: int = 4
    action_dim: int = 4
    capacity: int = 200_000
    batch_size: int = 64
    lr_q: float = 0.001
    lr_actor: float = 0.001
    gamma: float = 0.95
    tau: float = 0.005  # Eq. (36); not specified numerically in the paper
    grad_clip: float = 1.0
    sequence_length: int = 8  # truncated recurrent context, implementation choice
    prioritized: bool = True
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.0002
    epsilon_min: float = 0.1
    device: str = "cpu"
    use_encoder_attention: bool = True
    use_moe: bool = True
    target_update: str = "soft"
    target_update_interval: int = 200

    def __post_init__(self):
        if self.num_experts != 5 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("The five-stage expert bank requires 1 <= top_k <= 5")
        if self.param_dim != 4 or self.action_dim != 4:
            raise ValueError("The paper uses four discrete actions and four parameters")
        if self.batch_size < 1 or self.capacity < self.batch_size or self.sequence_length < 1:
            raise ValueError("Capacity must cover a batch; sequence_length must be positive")
        if min(self.latent_dim, self.token_dim, self.attention_heads, self.encoder_hidden,
               self.gru_hidden, self.expert_hidden) < 1 or self.encoder_hidden < 2:
            raise ValueError("Network dimensions must be positive; encoder_hidden must be >= 2")
        if self.token_dim % self.attention_heads:
            raise ValueError("token_dim must be divisible by attention_heads")
        if min(self.lr_q, self.lr_actor, self.grad_clip) <= 0:
            raise ValueError("Learning rates and grad_clip must be positive")
        if not 0 < self.tau <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("Invalid tau or gamma")
        if self.target_update not in ("soft", "hard") or self.target_update_interval < 1:
            raise ValueError("Invalid target update schedule")
