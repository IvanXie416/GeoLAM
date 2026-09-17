"""Query embeddings for D4RT decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Fourier feature mapping for normalized UV coordinates."""

    def __init__(self, input_dim: int = 2, num_bands: int = 8, include_input: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_bands = num_bands
        self.include_input = include_input
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_bands, dtype=torch.float32) * torch.pi,
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + self.input_dim * self.num_bands * 2

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        # uv: [B, M, 2], normalized to [0, 1]
        uv = uv.to(dtype=torch.float32)
        x = uv.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1)
        sin = torch.sin(x)
        cos = torch.cos(x)
        out = [sin, cos]
        if self.include_input:
            out.insert(0, uv.unsqueeze(-1))
        merged = torch.cat(out, dim=-1)
        return merged.flatten(start_dim=2)


class QueryEmbedder(nn.Module):
    """Builds per-query token from UV/time and local RGB patch."""

    def __init__(
        self,
        hidden_dim: int,
        clip_frames: int,
        local_patch_enabled: bool,
        local_patch_size: int = 9,
        uv_num_bands: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_frames = clip_frames
        self.local_patch_enabled = local_patch_enabled
        self.local_patch_size = local_patch_size

        self.uv_encoder = FourierFeatures(input_dim=2, num_bands=uv_num_bands, include_input=True)
        self.uv_proj = nn.Linear(self.uv_encoder.output_dim, hidden_dim)
        self.t_src_embed = nn.Embedding(clip_frames, hidden_dim)
        self.t_tgt_embed = nn.Embedding(clip_frames, hidden_dim)
        self.t_cam_embed = nn.Embedding(clip_frames, hidden_dim)

        if local_patch_enabled:
            patch_dim = 3 * local_patch_size * local_patch_size
            self.patch_proj = nn.Sequential(
                nn.Linear(patch_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.patch_proj = None

        self.out_norm = nn.LayerNorm(hidden_dim)

    def _clamp_t(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0, max=self.max_frames - 1)

    def _extract_local_patches(
        self,
        video: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        t_src: torch.Tensor,
    ) -> torch.Tensor:
        # video: [B, T, C, H, W]
        bsz, _, channels, height, width = video.shape
        _, num_queries = u.shape

        batch_idx = torch.arange(bsz, device=video.device).view(-1, 1).expand(-1, num_queries)
        src_frames = video[batch_idx, t_src]  # [B, M, C, H, W]
        src_frames = src_frames.reshape(-1, channels, height, width)

        p = self.local_patch_size
        offsets = torch.linspace(-(p - 1) / 2.0, (p - 1) / 2.0, p, device=video.device, dtype=torch.float32)
        dx = offsets * (2.0 / max(1, width - 1))
        dy = offsets * (2.0 / max(1, height - 1))
        grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [P, P, 2]

        centers_x = (u * 2.0 - 1.0).reshape(-1)
        centers_y = (v * 2.0 - 1.0).reshape(-1)
        centers = torch.stack([centers_x, centers_y], dim=-1)

        grid = base_grid.unsqueeze(0).repeat(centers.shape[0], 1, 1, 1)
        grid[..., 0] = grid[..., 0] + centers[:, 0].view(-1, 1, 1)
        grid[..., 1] = grid[..., 1] + centers[:, 1].view(-1, 1, 1)

        patches = F.grid_sample(
            src_frames,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        patches = patches.reshape(bsz, num_queries, channels * p * p)
        return patches

    def forward(
        self,
        video: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
    ) -> torch.Tensor:
        uv = torch.stack([u, v], dim=-1)
        uv_token = self.uv_proj(self.uv_encoder(uv))

        t_src = self._clamp_t(t_src)
        t_tgt = self._clamp_t(t_tgt)
        t_cam = self._clamp_t(t_cam)
        time_token = self.t_src_embed(t_src) + self.t_tgt_embed(t_tgt) + self.t_cam_embed(t_cam)

        token = uv_token + time_token
        if self.patch_proj is not None:
            patches = self._extract_local_patches(video, u, v, t_src)
            token = token + self.patch_proj(patches)

        return self.out_norm(token)

