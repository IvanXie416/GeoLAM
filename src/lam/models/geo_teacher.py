from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GeoMotionTarget:
    tokens: torch.Tensor


class GeoMotionTokenizer(nn.Module):
    # Target channels are motion only. Confidence and visibility are used to
    # weight reliable queries, but are not learnable constants in the target.
    feature_dim = 7

    def __init__(
        self,
        hidden_dim: int = 1280,
        c_geo: int = 7,
        k_geo: int = 16,
        residual_motion_weight: float = 4.0,
        xyz_motion_scale: float = 0.1,
        uv_motion_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.c_geo = int(c_geo)
        self.k_geo = int(k_geo)
        self.residual_motion_weight = float(residual_motion_weight)
        self.xyz_motion_scale = float(xyz_motion_scale)
        self.uv_motion_scale = float(uv_motion_scale)
        self.eps = 1.0e-6
        if self.k_geo <= 0:
            raise ValueError("k_geo must be positive")
        if self.c_geo < self.feature_dim:
            raise ValueError(f"c_geo must be at least {self.feature_dim}")
        if not math.isfinite(self.xyz_motion_scale) or self.xyz_motion_scale <= 0.0:
            raise ValueError("xyz_motion_scale must be finite and positive")
        if not math.isfinite(self.uv_motion_scale) or self.uv_motion_scale <= 0.0:
            raise ValueError("uv_motion_scale must be finite and positive")

    @staticmethod
    def _require_heads(g: dict[str, torch.Tensor], prefix: str) -> None:
        required = ("xyz_3d", "uv_2d", "visibility", "normal", "confidence")
        missing = [key for key in required if key not in g]
        if missing:
            raise KeyError(f"{prefix} missing D4RT head(s): {missing}")

    @staticmethod
    def _weights(g_t: dict[str, torch.Tensor], g_tk: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            torch.sigmoid(g_t["confidence"])
            * torch.sigmoid(g_tk["confidence"])
            * torch.sigmoid(g_t["visibility"])
            * torch.sigmoid(g_tk["visibility"])
        )

    def _scene_scale(self, xyz: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        norms = xyz.norm(dim=-1)
        weights = weights.clamp_min(self.eps)
        scale = (norms * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return scale.clamp_min(self.eps).unsqueeze(-1)

    def _pool_weights(self, g_t: dict[str, torch.Tensor], g_tk: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = self._weights(g_t, g_tk).clamp_min(self.eps)
        if self.residual_motion_weight <= 0.0:
            return weights
        delta_uv = g_tk["uv_2d"] - g_t["uv_2d"]
        background_delta = delta_uv.median(dim=1, keepdim=True).values
        residual_norm = (delta_uv - background_delta).norm(dim=-1).detach().clamp_max(1.0)
        motion_floor = 0.05
        motion_focus = (motion_floor + residual_norm).pow(self.residual_motion_weight)
        return weights * motion_focus

    def _geometry_features(self, g_t: dict[str, torch.Tensor], g_tk: dict[str, torch.Tensor]) -> torch.Tensor:
        self._require_heads(g_t, "current")
        self._require_heads(g_tk, "future")
        weights = self._weights(g_t, g_tk)
        scale = self._scene_scale(g_t["xyz_3d"], weights)
        delta_xyz = (
            (g_tk["xyz_3d"] - g_t["xyz_3d"]) / scale / self.xyz_motion_scale
        ).clamp(-1.0, 1.0)
        delta_uv = g_tk["uv_2d"] - g_t["uv_2d"]
        background_delta_uv = delta_uv.median(dim=1, keepdim=True).values
        residual_uv_raw = delta_uv - background_delta_uv
        residual_uv = (residual_uv_raw / self.uv_motion_scale).clamp(-1.0, 1.0)
        normal_t = F.normalize(g_t["normal"], dim=-1, eps=self.eps)
        normal_tk = F.normalize(g_tk["normal"], dim=-1, eps=self.eps)
        normal_delta = normal_tk - normal_t
        normal_delta_norm = normal_delta.norm(dim=-1, keepdim=True).clamp_max(2.0) / 2.0
        residual_uv_norm = (residual_uv_raw.norm(dim=-1, keepdim=True) / self.uv_motion_scale).clamp(0.0, 1.0)
        features = torch.cat(
            [
                delta_xyz,
                residual_uv,
                normal_delta_norm,
                residual_uv_norm,
            ],
            dim=-1,
        )
        if self.c_geo == self.feature_dim:
            return features
        return F.pad(features, (0, self.c_geo - self.feature_dim))

    def _grid_slot_pool(self, motion: torch.Tensor, weights: torch.Tensor) -> torch.Tensor | None:
        query_side = int(math.isqrt(motion.shape[1]))
        slot_side = int(math.isqrt(self.k_geo))
        if query_side * query_side != motion.shape[1]:
            return None
        if slot_side * slot_side != self.k_geo:
            return None
        if query_side % slot_side != 0:
            return None

        batch = motion.shape[0]
        cell = query_side // slot_side
        motion_grid = motion.reshape(batch, query_side, query_side, self.c_geo)
        weight_grid = weights.reshape(batch, query_side, query_side, 1).clamp_min(1.0e-6)
        slots: list[torch.Tensor] = []
        for row in range(slot_side):
            for col in range(slot_side):
                y0, y1 = row * cell, (row + 1) * cell
                x0, x1 = col * cell, (col + 1) * cell
                region_motion = motion_grid[:, y0:y1, x0:x1]
                region_weights = weight_grid[:, y0:y1, x0:x1]
                pooled = (region_motion * region_weights).sum(dim=(1, 2)) / region_weights.sum(dim=(1, 2)).clamp_min(1.0e-6)
                slots.append(pooled)
        return torch.stack(slots, dim=1)

    def _topk_slot_pool(self, motion: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        topk = min(self.k_geo, motion.shape[1])
        indices = torch.topk(weights, k=topk, dim=1, largest=True, sorted=True).indices
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, motion.shape[-1])
        slots = torch.gather(motion, dim=1, index=gather_idx)
        if topk == self.k_geo:
            return slots
        pad = slots[:, -1:].expand(-1, self.k_geo - topk, -1)
        return torch.cat([slots, pad], dim=1)

    def forward(
        self,
        h_t: torch.Tensor,
        g_t: dict[str, torch.Tensor],
        h_tk: torch.Tensor,
        g_tk: dict[str, torch.Tensor],
        detach: bool = True,
    ) -> GeoMotionTarget:
        if h_t.shape[:2] != h_tk.shape[:2]:
            raise ValueError(f"Current/future D4RT hidden shapes must match in batch/query dims, got {h_t.shape} and {h_tk.shape}")
        motion = self._geometry_features(g_t, g_tk)
        if motion.shape[:2] != h_t.shape[:2]:
            raise ValueError(f"D4RT head batch/query dims must match hidden dims, got {motion.shape[:2]} and {h_t.shape[:2]}")
        weights = self._pool_weights(g_t, g_tk).clamp_min(1.0e-6)
        tokens = self._grid_slot_pool(motion, weights)
        if tokens is None:
            tokens = self._topk_slot_pool(motion, weights)
        target = GeoMotionTarget(tokens=tokens)
        if detach:
            return GeoMotionTarget(tokens=target.tokens.detach())
        return target


GeoMotionPool = GeoMotionTokenizer
