# nGPT_pytorch/mamba.py

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP
from torch import nn


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=2,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Mamba2LM(nn.Module):
    """
    Mamba2LM class - 'standard' Mamba2 model for language modeling
    """

    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        ce_ignore_index=-1,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.ignore_index = ce_ignore_index

        norm_cls = nn.LayerNorm
        mixer_cls = lambda layer_idx: lambda dim: Mamba2(
            d_model=dim, layer_idx=layer_idx, **kwargs
        )
        mlp_cls = lambda dim: GatedMLP(dim)

        self.layers = nn.ModuleList([])
        for i in range(depth):
            block = Block(
                dim,
                mixer_cls=mixer_cls(i),
                mlp_cls=mlp_cls,
                norm_cls=norm_cls,
            )
            self.layers.append(block)

        self.norm_f = norm_cls(dim)
        self.lm_head = nn.Linear(dim, num_tokens, bias=False)
        # Optionally, tie embeddings
        self.lm_head.weight = self.token_emb.weight

        # Apply custom weight initialization
        self.apply(
            lambda module: _init_weights(
                module,
                n_layer=self.depth,
                initializer_range=initializer_range,
                rescale_prenorm_residual=True,
                n_residuals_per_layer=2,  # Updated to 2 since we have MLP layers
            )
        )

    def forward(self, ids, return_loss=False):
        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        tokens = self.token_emb(ids)

        # Generate sequence indices with dtype torch.int32
        batch_size, seq_len = ids.size()
        device = ids.device
        seq_idx = (
            torch.arange(seq_len, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        residual = None
        for layer in self.layers:
            tokens, residual = layer(tokens, residual=residual, seq_idx=seq_idx)

        # Final layer norm
        if residual is not None:
            tokens = tokens + residual
        tokens = self.norm_f(tokens)

        logits = self.lm_head(tokens)

        if not return_loss:
            return logits

        # Cross-entropy loss expects input as (batch_size, num_classes, seq_length)
        loss = F.cross_entropy(
            rearrange(logits, "b n c -> b c n"), labels, ignore_index=self.ignore_index
        )
        return loss
