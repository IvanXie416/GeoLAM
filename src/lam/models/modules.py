from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FeedForward(dim, ffn_ratio)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.q_norm(query), self.kv_norm(context), self.kv_norm(context), need_weights=False)
        query = query + attn_out
        return query + self.ffn(query)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FeedForward(dim, ffn_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        return x + self.ffn(x)


class AdaLNZeroBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[0] != x.shape[0] or condition.shape[-1] != x.shape[-1]:
            raise ValueError("AdaLN condition must be [B, C] and match the state tokens")
        modulation = self.modulation(condition).chunk(6, dim=-1)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation

        attn_input = self._modulate(self.attn_norm(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out

        ffn_input = self._modulate(self.ffn_norm(x), shift_ffn, scale_ffn)
        return x + gate_ffn.unsqueeze(1) * self.ffn(ffn_input)
