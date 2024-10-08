from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

# for use with parametrize

class L2Norm(Module):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return l2norm(t, dim = self.dim)

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim = -1
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias = False)

        parametrize.register_parametrization(
            self.linear,
            'weight',
            L2Norm(dim = norm_dim)
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

# attention and feedforward

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        norm_qk = False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_q = NormLinear(dim, dim_inner, norm_dim = 0)
        self.to_k = NormLinear(dim, dim_inner, norm_dim = 0)
        self.to_v = NormLinear(dim, dim_inner, norm_dim = 0)

        self.norm_qk = norm_qk
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear(dim_inner, dim)

    def forward(
        self,
        x
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        if self.norm_qk:
            q, k = map(l2norm, (q, k))

        # scale is 1., as scaling factor is moved to s_qk (dk ^ 0.25) - eq. 16

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True,
            scale = 1.
        )

        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        expand_factor = 4
    ):
        super().__init__()
        dim_inner = int(dim * expand_factor * 2 / 3)
        self.to_hidden = NormLinear(dim, dim_inner, norm_dim = 0)
        self.to_gate = NormLinear(dim, dim_inner, norm_dim = 0)
        self.to_out = NormLinear(dim_inner, dim)

    def forward(self, x):
        x, gate = self.to_hidden(x), self.to_gate(x)
        x = F.silu(gate) * x
        return self.to_out(x)

# classes

class nGPT(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_norm_qk = False,  # they say the query/key normalization is optional
        ff_expand_factor = 4.,
        ce_ignore_index = -1,
        residual_lerp_scale_init = None
    ):
        super().__init__()

        residual_lerp_scale_init = default(residual_lerp_scale_init, 1. / depth)

        self.token_embed = NormLinear(dim, num_tokens)

        self.layers = ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, norm_qk = attn_norm_qk),
                FeedForward(dim, expand_factor = ff_expand_factor),
            ]))

            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init),
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init),
            ]))

        self.token_unembed = NormLinear(dim, num_tokens, norm_dim = 0)
        self.ignore_index = ce_ignore_index

    def forward(
        self,
        ids,
        return_loss = False
    ):

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        tokens = self.token_embed.weight[ids]

        for (attn, ff), (attn_alpha, ff_alpha) in zip(self.layers, self.residual_lerp_scales):

            attn_out = attn(tokens)
            tokens = l2norm(tokens.lerp(attn_out, attn_alpha))

            ff_out = ff(tokens)
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha))

        logits = self.token_unembed(tokens)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
