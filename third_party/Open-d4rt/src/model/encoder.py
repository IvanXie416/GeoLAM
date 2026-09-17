"""Video encoder backbone for D4RT."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sinusoidal_position_embedding


def _normalize_attention_pattern(pattern: str | None) -> str:
    raw = (pattern or "global").strip().lower()
    aliases = {
        "interleaved_local_framewise_and_global": "interleaved_local_global",
        "interleaved_local_framewise_global": "interleaved_local_global",
        "interleaved_local_and_global": "interleaved_local_global",
        "global": "global",
        "full_global": "global",
    }
    return aliases.get(raw, raw)


class SelfAttentionBlock(nn.Module):
    """Pre-norm transformer block with self-attention + MLP."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1) -> None:
        super().__init__()
        ff_dim = int(math.ceil(hidden_dim * mlp_ratio))
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_attn(tokens)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class VideoPatchTransformerEncoder(nn.Module):
    """Patchify video then encode tokens with interleaved local/global attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size_t_h_w: tuple[int, int, int],
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        max_tokens: int = 4096,
        attention_pattern: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.attention_pattern = _normalize_attention_pattern(attention_pattern)
        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size_t_h_w,
            stride=patch_size_t_h_w,
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.block_modes = self._build_block_modes(num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _build_block_modes(self, num_layers: int) -> list[Literal["local", "global"]]:
        if self.attention_pattern == "interleaved_local_global":
            return ["local" if (i % 2 == 0) else "global" for i in range(num_layers)]
        return ["global"] * num_layers

    def _token_cap(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T', H', W']
        b, c, tp, hp, wp = x.shape
        token_count = tp * hp * wp
        if token_count <= self.max_tokens:
            return x
        scale = math.sqrt(self.max_tokens / float(token_count))
        out_h = max(1, int(round(hp * scale)))
        out_w = max(1, int(round(wp * scale)))
        return F.adaptive_avg_pool3d(x, output_size=(tp, out_h, out_w))

    def forward(self, video_b_t_c_h_w: torch.Tensor, extra_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if video_b_t_c_h_w.ndim != 5:
            raise ValueError(f"Expected video tensor with ndim=5, got {video_b_t_c_h_w.shape}")
        x = video_b_t_c_h_w.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.patch_embed(x)
        x = self._token_cap(x)

        b, c, tp, hp, wp = x.shape
        video_tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        token_count = video_tokens.shape[1]
        pos = sinusoidal_position_embedding(token_count, self.hidden_dim, video_tokens.device)
        video_tokens = video_tokens + pos.unsqueeze(0)
        if extra_tokens is not None:
            if extra_tokens.ndim != 3:
                raise ValueError(f"Expected extra_tokens [B, N_extra, C], got {extra_tokens.shape}")
            if extra_tokens.shape[0] != b or extra_tokens.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"extra_tokens must match batch and hidden dim: expected [B={b}, *, C={self.hidden_dim}], got {extra_tokens.shape}"
                )

        spatial_tokens = hp * wp
        for mode, block in zip(self.block_modes, self.blocks):
            if mode == "local":
                local = video_tokens.reshape(b, tp, spatial_tokens, c).reshape(b * tp, spatial_tokens, c)
                local = block(local)
                video_tokens = local.reshape(b, tp, spatial_tokens, c).reshape(b, tp * spatial_tokens, c)
                continue

            if extra_tokens is None:
                video_tokens = block(video_tokens)
                continue

            merged = torch.cat([video_tokens, extra_tokens], dim=1)
            merged = block(merged)
            video_tokens = merged[:, :token_count]
            extra_tokens = merged[:, token_count:]

        if extra_tokens is None:
            encoded = video_tokens
        else:
            encoded = torch.cat([video_tokens, extra_tokens], dim=1)
        return self.final_norm(encoded)
