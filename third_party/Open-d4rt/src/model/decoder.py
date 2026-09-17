"""Independent-query cross-attention decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Decoder block with cross-attention only (no query self-attention)."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        ff_dim = int(math.ceil(hidden_dim * mlp_ratio))
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(memory_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = query_tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class IndependentQueryDecoder(nn.Module):
    """Stack of cross-attention blocks."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        x = query_tokens
        for block in self.blocks:
            x = block(x, memory_tokens)
        return self.out_norm(x)

