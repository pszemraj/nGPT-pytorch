# nMamba2.py
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

# Special imports
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


# Helper functions
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(tensor, dim=-1, eps=1e-6):
    return F.normalize(tensor, p=2, dim=dim, eps=eps)


# Scale module
class Scale(nn.Module):
    def __init__(self, dim, init=1.0, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale


# Residual module
class Residual(nn.Module):
    def __init__(self, fn, dim, init, scale):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, default(scale, dim**-0.5))

    def forward(self, x, **kwargs):
        residual = x
        out = self.fn(x, **kwargs)
        out = l2norm(out)
        out = l2norm(residual.lerp(out, self.branch_scale()))
        return out


# Normalized Linear layer
class NormLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))
        self.scale = dim_in**-0.5

    def forward(self, x):
        weight = l2norm(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale


# Simplified Normalized Mamba2 Layer with SSM Computation
class NormalizedMamba2Layer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_inner = self.expand * self.dim
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path

        # Input projection
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = NormLinear(self.dim, d_in_proj)

        # Convolutional layer
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
        )

        # Learnable initial states
        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state)
            )

        self.act = nn.SiLU()

        # Time-step bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # State matrix A
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))

        # Normalization layers
        self.x_norm = nn.LayerNorm(self.dim)
        self.xBC_norm = nn.LayerNorm(conv_dim)
        self.norm = RMSNormGated(self.d_inner, eps=1e-5)

        # Output projection
        self.out_proj = NormLinear(self.d_inner, self.dim)

    def forward(self, u, seq_idx=None):
        batch, seqlen, _ = u.shape

        # Input normalization
        u = self.x_norm(u)

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads)

        # Split components
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        # Normalize xBC
        xBC = self.xBC_norm(xBC)

        # Time-step dt
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        initial_states = (
            repeat(self.init_states, "... -> b ...", b=batch)
            if self.learnable_init_states
            else None
        )

        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        if self.use_mem_eff_path:
            # Fully fused path using mamba_split_conv1d_scan_combined
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=l2norm(self.out_proj.weight, dim=-1),
                outproj_bias=None,  # Assuming no bias in NormLinear
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            # Non-fused path
            # Convolution
            xBC = xBC.transpose(1, 2)  # (B, conv_dim, L)
            if causal_conv1d_fn is not None and self.activation in ["silu", "swish"]:
                xBC = causal_conv1d_fn(
                    x=xBC,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                xBC = xBC.transpose(1, 2)  # (B, L, conv_dim)
            else:
                xBC = self.conv1d(xBC)
                xBC = xBC.transpose(1, 2)
                xBC = self.act(xBC)

            # Split xBC into x, B, C
            x, B, C = torch.split(
                xBC,
                [
                    self.d_inner,
                    self.ngroups * self.d_state,
                    self.ngroups * self.d_state,
                ],
                dim=-1,
            )

            # Reshape for SSM computation
            x = x.view(batch, seqlen, self.nheads, self.headdim)
            B = B.view(batch, seqlen, self.ngroups, self.d_state)
            C = C.view(batch, seqlen, self.ngroups, self.d_state)

            # SSM computation using mamba_chunk_scan_combined
            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )

            y = y.view(batch, seqlen, self.d_inner)

            # Gating and normalization
            out = self.norm(y, z)

            # Output projection
            out = self.out_proj(out)

        return out


# Normalized Mamba2 Model
class nMamba2(nn.Module):
    def __init__(self, num_tokens, dim, depth, **kwargs):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, dim))
        self.ignore_index = ce_ignore_index
        self.ignore_index = ce_ignore_index

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            layer = NormalizedMamba2Layer(dim, **kwargs)
            self.layers.append(Residual(layer, dim, init=1.0 / depth, scale=dim**-0.5))

        self.to_logits = NormLinear(dim, num_tokens)
        self.logit_scale = Scale(num_tokens, init=1.0, scale=dim**-0.5)
        self.ignore_index = ce_ignore_index

    def forward(self, ids, return_loss=False):
        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]
            
        tokens = self.token_emb(ids) + self.pos_emb[:, :ids.size(1), :]

        for layer in self.layers:
            tokens = layer(tokens)

        logits = self.to_logits(tokens) * self.logit_scale()

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, "b n c -> b c n"),
            labels,
            ignore_index=self.ignore_index
        )
        return loss
