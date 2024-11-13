# nGPT_pytorch/mamba.py

import torch
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba2
from torch import nn


class Mamba2LM(nn.Module):
    """
    Mamba2LM class - 'standard' Mamba2 model for language modeling
    """

    def __init__(self, num_tokens, dim, depth, ce_ignore_index=-1, **kwargs):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, dim))
        self.ignore_index = ce_ignore_index

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            layer = Mamba2(d_model=dim, **kwargs)
            self.layers.append(layer)

        self.to_logits = nn.Linear(dim, num_tokens)
        # Optionally, tie embeddings
        self.to_logits.weight = self.token_emb.weight

    def forward(self, ids, return_loss=False):
        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        tokens = self.token_emb(ids) + self.pos_emb[:, : ids.size(1), :]

        for layer in self.layers:
            tokens = layer(tokens)

        logits = self.to_logits(tokens)

        if not return_loss:
            return logits

        # Cross-entropy loss expects input as (batch_size, num_classes, seq_length)
        loss = F.cross_entropy(
            rearrange(logits, "b n c -> b c n"), labels, ignore_index=self.ignore_index
        )
        return loss
