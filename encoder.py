"""Four semantic encoders and six-head fusion, paper Eqs. (12)-(23)."""
import torch
from torch import nn
from torch.nn import functional as F

from config import ObservationLayout


class ResidualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.activation = activation
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + self.activation(layer(x)))
        return self.output(x)


class SpatialEncoder3D(ResidualEncoder):
    def __init__(self, input_dim, hidden_dim=128, output_dim=96):
        super().__init__(input_dim, hidden_dim, output_dim, F.gelu)


class EnvironmentEncoder3D(ResidualEncoder):
    def __init__(self, input_dim, hidden_dim=128, output_dim=96):
        super().__init__(input_dim, hidden_dim, output_dim, F.relu)


class DynamicEncoder3D(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=96):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])
        groups = next(g for g in (8, 4, 2, 1) if hidden_dim % g == 0 and hidden_dim // g > 1)
        self.norms = nn.ModuleList([nn.GroupNorm(groups, hidden_dim) for _ in self.layers])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = norm(F.silu(layer(x)))
        return self.output(x)


class NoiseEncoder3D(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=96):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in self.layers])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(layer(x))
            # A single observation has no batch variance. Use stored statistics.
            if self.training and x.shape[0] == 1:
                x = F.batch_norm(x, norm.running_mean, norm.running_var,
                                 norm.weight, norm.bias, training=False, eps=norm.eps)
            else:
                x = norm(x)
        return self.output(x)


class Encoder3D_FeatureEnhanced(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=128, output_dim=128,
                 layout=None, token_dim=96, num_heads=6):
        super().__init__()
        self.layout = layout or ObservationLayout()
        if input_dim is not None and input_dim != self.layout.state_dim:
            raise ValueError("input_dim must match the explicit four-group observation layout")
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        dims = self.layout.group_dims
        self.branches = nn.ModuleList([
            SpatialEncoder3D(max(1, dims[0]), hidden_dim, token_dim),
            DynamicEncoder3D(max(1, dims[1]), hidden_dim, token_dim),
            EnvironmentEncoder3D(max(1, dims[2]), hidden_dim, token_dim),
            NoiseEncoder3D(max(1, dims[3]), hidden_dim, token_dim),
        ])
        self.projections = nn.ModuleList([nn.Linear(token_dim, token_dim) for _ in dims])
        self.attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.enhancer = nn.Sequential(
            nn.Linear(token_dim, 2 * token_dim), nn.GELU(),
            nn.Linear(2 * token_dim, 2 * token_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2 * token_dim, token_dim))
        self.enhancement_norm = nn.LayerNorm(token_dim)
        self.fusion = nn.Sequential(
            nn.Linear(4 * token_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(256),
            nn.Linear(256, output_dim))

    def forward(self, x):
        if x.shape[-1] != self.layout.state_dim:
            raise ValueError(f"Expected {self.layout.state_dim} observation features, got {x.shape[-1]}")
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        tokens = []
        absent = []
        for branch, proj, indices in zip(self.branches, self.projections, self.layout.group_indices):
            absent.append(not indices)
            if indices:
                tokens.append(proj(branch(x[:, list(indices)])))
            else:
                tokens.append(x.new_zeros((x.shape[0], self.attention.embed_dim)))
        tokens = torch.stack(tokens, dim=1)
        mask = torch.tensor(absent, device=x.device).expand(x.shape[0], -1)
        # Explicit attention path also supports TRPO's second-order KL products.
        fused, _ = self.attention(tokens, tokens, tokens, key_padding_mask=mask, need_weights=True)
        enhanced = self.enhancement_norm(fused + self.enhancer(fused))
        return self.fusion(enhanced.flatten(1)).reshape(*shape, -1)


class FlatEncoder(nn.Module):
    """Explicit w/o encoder-attention ablation: unstructured MLP projection."""
    def __init__(self, layout, output_dim=128, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(layout.state_dim, 256), nn.ReLU(),
                                    nn.Linear(256, output_dim))

    def forward(self, x):
        return self.layers(x)


def make_encoder(layout, config):
    kind = Encoder3D_FeatureEnhanced if config.use_encoder_attention else FlatEncoder
    return kind(layout=layout, hidden_dim=config.encoder_hidden, output_dim=config.latent_dim,
                token_dim=config.token_dim, num_heads=config.attention_heads)
