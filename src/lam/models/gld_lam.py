from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CrossAttentionBlock, SelfAttentionBlock


@dataclass(frozen=True)
class GLDState:
    tokens: torch.Tensor


@dataclass(frozen=True)
class DeterministicLatentAction:
    tokens: torch.Tensor
    low_dim: torch.Tensor


@dataclass(frozen=True)
class LatentAction(DeterministicLatentAction):
    vector: torch.Tensor


@dataclass(frozen=True)
class GLDLAMOutput:
    state_t: GLDState
    state_tk: GLDState
    latent_action: LatentAction
    pred_tokens_tk: torch.Tensor
    f0_hat: torch.Tensor
    f1_hat: torch.Tensor
    z_geo_tokens_pred: torch.Tensor


def init_grid_pos_embed(num_slots: int, dim: int) -> torch.Tensor:
    slot_side = int(num_slots**0.5)
    if slot_side * slot_side != num_slots:
        return torch.zeros(1, num_slots, dim)

    coords = torch.linspace(-1.0, 1.0, slot_side)
    y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
    xy = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
    num_freqs = max(1, dim // 4)
    freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * torch.pi
    x_angles = xy[:, 0:1] * freqs.unsqueeze(0)
    y_angles = xy[:, 1:2] * freqs.unsqueeze(0)
    pos = torch.cat([xy, x_angles.sin(), x_angles.cos(), y_angles.sin(), y_angles.cos()], dim=-1)
    if pos.shape[-1] < dim:
        pos = F.pad(pos, (0, dim - pos.shape[-1]))
    return pos[:, :dim].unsqueeze(0)


class GLDStateEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        c_state: int = 1024,
        heads: int = 16,
        depth: int = 2,
        num_tokens: int = 256,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.f0_norm = nn.LayerNorm(in_dim)
        self.f1_norm = nn.LayerNorm(in_dim)
        self.f0_proj = nn.Linear(in_dim, c_state)
        self.f1_proj = nn.Linear(in_dim, c_state)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, c_state))
        self.fuse = nn.Sequential(
            nn.LayerNorm(c_state * 2),
            nn.Linear(c_state * 2, c_state * 2),
            nn.GELU(),
            nn.Linear(c_state * 2, c_state),
        )
        self.blocks = nn.ModuleList([SelfAttentionBlock(c_state, heads, ffn_ratio) for _ in range(depth)])

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> GLDState:
        if f0.shape != f1.shape:
            raise ValueError(f"F0/F1 shapes must match, got {tuple(f0.shape)} and {tuple(f1.shape)}")
        if f0.ndim != 3:
            raise ValueError(f"Expected F0/F1 [B, N, C], got {tuple(f0.shape)}")
        if f0.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {f0.shape[1]}")

        x0 = self.f0_proj(self.f0_norm(f0)) + self.pos_embed
        x1 = self.f1_proj(self.f1_norm(f1)) + self.pos_embed
        tokens = self.fuse(torch.cat([x0, x1], dim=-1))
        for block in self.blocks:
            tokens = block(tokens)
        return GLDState(tokens=tokens)


class DeterministicLatentActionBottleneck(nn.Module):
    def __init__(
        self,
        token_dim: int = 1024,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)
        self.project_in = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, latent_dim))
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.project_out = nn.Linear(latent_dim, token_dim)

    def forward(self, tokens: torch.Tensor) -> DeterministicLatentAction:
        if tokens.ndim != 3:
            raise ValueError(f"Expected latent action tokens [B, K, C], got {tuple(tokens.shape)}")
        low_dim = self.latent_norm(self.project_in(tokens))
        latent_tokens = self.project_out(low_dim)
        return DeterministicLatentAction(
            tokens=latent_tokens,
            low_dim=low_dim,
        )


class GLDIDM(nn.Module):
    def __init__(
        self,
        c_state: int = 1024,
        c_z: int = 1024,
        z_dim: int = 512,
        k_action: int = 16,
        latent_dim: int = 32,
        heads: int = 16,
        depth: int = 4,
        bottleneck_depth: int = 2,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.k_action = int(k_action)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(c_state),
            nn.Linear(c_state, c_z),
            nn.GELU(),
            nn.Linear(c_z, c_z),
        )
        self.blocks = nn.ModuleList([SelfAttentionBlock(c_z, heads, ffn_ratio) for _ in range(depth)])
        self.latent_queries = nn.Parameter(torch.randn(1, k_action, c_z) * 0.02)
        self.latent_grid_pos_embed = nn.Parameter(init_grid_pos_embed(k_action, c_z), requires_grad=False)
        self.latent_blocks = nn.ModuleList([CrossAttentionBlock(c_z, heads, ffn_ratio) for _ in range(bottleneck_depth)])
        self.bottleneck = DeterministicLatentActionBottleneck(token_dim=c_z, latent_dim=latent_dim)
        self.z_dim = int(z_dim)

    def forward(self, tokens_t: torch.Tensor, tokens_tk: torch.Tensor) -> LatentAction:
        if tokens_t.shape != tokens_tk.shape:
            raise ValueError("Current and future state token shapes must match")
        motion_tokens = self.pair_mlp(tokens_tk - tokens_t)
        for block in self.blocks:
            motion_tokens = block(motion_tokens)

        z = self.latent_queries + self.latent_grid_pos_embed.to(device=tokens_t.device, dtype=tokens_t.dtype)
        z = z.expand(tokens_t.shape[0], -1, -1)
        for block in self.latent_blocks:
            z = block(z, motion_tokens)
        latent = self.bottleneck(z)
        # Keep the legacy vector interface without a second trainable action
        # head. The FDM and geometry projector consume latent tokens directly.
        pooled = latent.tokens.mean(dim=1)
        if pooled.shape[-1] >= self.z_dim:
            vector = pooled[..., : self.z_dim]
        else:
            vector = F.pad(pooled, (0, self.z_dim - pooled.shape[-1]))
        return LatentAction(
            tokens=latent.tokens,
            low_dim=latent.low_dim,
            vector=vector,
        )


class GLDFDM(nn.Module):
    def __init__(
        self,
        c_state: int = 1024,
        heads: int = 16,
        depth: int = 4,
        self_attn_every: int = 2,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if self_attn_every < 0:
            raise ValueError("self_attn_every must be non-negative")
        self.self_attn_every = int(self_attn_every)
        self.register_buffer("spatial_self_attention_version", torch.tensor(1, dtype=torch.int64))
        self.blocks = nn.ModuleList([CrossAttentionBlock(c_state, heads, ffn_ratio) for _ in range(depth)])
        num_spatial_blocks = depth // self.self_attn_every if self.self_attn_every else 0
        self.spatial_blocks = nn.ModuleList(
            [SelfAttentionBlock(c_state, heads, ffn_ratio) for _ in range(num_spatial_blocks)]
        )
        self.out_norm = nn.LayerNorm(c_state)

    def forward(self, tokens_t: torch.Tensor, z_tokens: torch.Tensor) -> torch.Tensor:
        if tokens_t.ndim != 3 or z_tokens.ndim != 3:
            raise ValueError("FDM expects state and latent action tokens with shape [B, N, C]")
        if tokens_t.shape[0] != z_tokens.shape[0] or tokens_t.shape[-1] != z_tokens.shape[-1]:
            raise ValueError("FDM state and latent action batch/channel dimensions must match")
        tokens = tokens_t
        spatial_idx = 0
        for block_idx, block in enumerate(self.blocks, start=1):
            tokens = block(tokens, z_tokens)
            if self.self_attn_every and block_idx % self.self_attn_every == 0:
                tokens = self.spatial_blocks[spatial_idx](tokens)
                spatial_idx += 1
        return self.out_norm(tokens)


class GLDFeatureHead(nn.Module):
    def __init__(self, c_state: int = 1024, out_dim: int = 1536) -> None:
        super().__init__()
        self.register_buffer("residual_version", torch.tensor(1, dtype=torch.int64))
        self.f0_head = self._make_delta_head(c_state, out_dim)
        self.f1_head = self._make_delta_head(c_state, out_dim)

    @staticmethod
    def _make_delta_head(c_state: int, out_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.LayerNorm(c_state),
            nn.Linear(c_state, c_state),
            nn.GELU(),
            nn.Linear(c_state, out_dim),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(
        self,
        tokens: torch.Tensor,
        f0_current: torch.Tensor,
        f1_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if f0_current.shape != f1_current.shape:
            raise ValueError("Current F0/F1 feature shapes must match")
        if tokens.shape[:2] != f0_current.shape[:2]:
            raise ValueError("Prediction tokens and current features must share batch/token dimensions")
        return f0_current + self.f0_head(tokens), f1_current + self.f1_head(tokens)


class GeoGridProjector(nn.Module):
    def __init__(
        self,
        token_dim: int = 1024,
        c_geo: int = 7,
        k_geo: int = 16,
        heads: int = 16,
        depth: int = 2,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.k_geo = int(k_geo)
        self.geo_queries = nn.Parameter(torch.randn(1, k_geo, token_dim) * 0.02)
        self.geo_grid_pos_embed = nn.Parameter(init_grid_pos_embed(k_geo, token_dim), requires_grad=False)
        self.blocks = nn.ModuleList([CrossAttentionBlock(token_dim, heads, ffn_ratio) for _ in range(depth)])
        self.out = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, c_geo),
        )

    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:
        if z_tokens.ndim != 3:
            raise ValueError(f"Expected latent action tokens [B, K, C], got {tuple(z_tokens.shape)}")
        geo_tokens = self.geo_queries + self.geo_grid_pos_embed.to(device=z_tokens.device, dtype=z_tokens.dtype)
        geo_tokens = geo_tokens.expand(z_tokens.shape[0], -1, -1)
        for block in self.blocks:
            geo_tokens = block(geo_tokens, z_tokens)
        return self.out(geo_tokens)


GeoActionProjector = GeoGridProjector


class GLDLAM(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        c_state: int = 1024,
        c_z: int = 1024,
        z_dim: int = 512,
        c_geo: int = 7,
        k_action: int = 16,
        k_geo: int | None = None,
        latent_dim: int = 32,
        heads: int = 16,
        action_bottleneck_depth: int = 4,
        idm_depth: int = 8,
        fdm_depth: int = 8,
        fdm_self_attn_every: int = 2,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        k_geo = k_action if k_geo is None else int(k_geo)
        self.state_encoder = GLDStateEncoder(
            in_dim=in_dim,
            c_state=c_state,
            heads=heads,
            depth=action_bottleneck_depth,
            ffn_ratio=ffn_ratio,
        )
        self.bottleneck = self.state_encoder
        self.idm = GLDIDM(
            c_state=c_state,
            c_z=c_z,
            z_dim=z_dim,
            k_action=k_action,
            latent_dim=latent_dim,
            heads=heads,
            depth=idm_depth,
            bottleneck_depth=action_bottleneck_depth,
            ffn_ratio=ffn_ratio,
        )
        self.fdm = GLDFDM(
            c_state=c_state,
            heads=heads,
            depth=fdm_depth,
            self_attn_every=fdm_self_attn_every,
            ffn_ratio=ffn_ratio,
        )
        self.feature_head = GLDFeatureHead(c_state=c_state, out_dim=in_dim)
        self.geo_projector = GeoGridProjector(
            token_dim=c_z,
            c_geo=c_geo,
            k_geo=k_geo,
            heads=heads,
            depth=2,
            ffn_ratio=ffn_ratio,
        )
        self.register_buffer(
            "architecture_depths",
            torch.tensor(
                [action_bottleneck_depth, idm_depth, fdm_depth, fdm_self_attn_every],
                dtype=torch.int64,
            ),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "feature_head.residual_version" not in state_dict:
            raise RuntimeError(
                "Checkpoint uses the legacy absolute GLD feature head and is incompatible with the "
                "current-frame residual feature head. Retrain with the current architecture."
            )
        if "fdm.spatial_self_attention_version" not in state_dict:
            raise RuntimeError(
                "Checkpoint predates FDM spatial self-attention and is incompatible with the current "
                "alternating cross/self-attention FDM. Retrain with the current architecture."
            )
        checkpoint_depths = state_dict.get("architecture_depths")
        if checkpoint_depths is None:
            raise RuntimeError(
                "Checkpoint does not record the IDM/FDM architecture depths and cannot be safely "
                "loaded into the current model. Retrain with the current architecture."
            )
        expected_depths = self.architecture_depths.to(device=checkpoint_depths.device)
        if not torch.equal(checkpoint_depths, expected_depths):
            raise RuntimeError(
                "Checkpoint IDM/FDM depths are incompatible: "
                f"checkpoint={checkpoint_depths.tolist()} current={expected_depths.tolist()}."
            )
        legacy_bottleneck_weight = state_dict.get("idm.bottleneck.project_in.1.weight")
        if (
            legacy_bottleneck_weight is not None
            and legacy_bottleneck_weight.shape[0] == self.idm.bottleneck.latent_dim * 2
        ):
            raise RuntimeError(
                "Checkpoint uses the legacy variational latent bottleneck and is incompatible "
                "with the deterministic normalized bottleneck. Retrain with the current architecture."
            )
        legacy_fdm_keys = [
            key for key in state_dict
            if key.startswith("fdm.blocks.") and ".modulation." in key
        ]
        if legacy_fdm_keys:
            raise RuntimeError(
                "Checkpoint uses the legacy mean-pooled AdaLN FDM and is incompatible with "
                "the cross-attention FDM. Retrain with the current architecture, or explicitly "
                "remove every fdm.* key when using the checkpoint only as a warm start."
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, f0_t: torch.Tensor, f1_t: torch.Tensor, f0_tk: torch.Tensor, f1_tk: torch.Tensor) -> GLDLAMOutput:
        state_t = self.state_encoder(f0_t, f1_t)
        state_tk = self.state_encoder(f0_tk, f1_tk)
        latent = self.idm(state_t.tokens, state_tk.tokens)
        pred_tokens = self.fdm(state_t.tokens, latent.tokens)
        f0_hat, f1_hat = self.feature_head(pred_tokens, f0_t, f1_t)
        z_geo_tokens_pred = self.geo_projector(latent.tokens)
        return GLDLAMOutput(
            state_t=state_t,
            state_tk=state_tk,
            latent_action=latent,
            pred_tokens_tk=pred_tokens,
            f0_hat=f0_hat,
            f1_hat=f1_hat,
            z_geo_tokens_pred=z_geo_tokens_pred,
        )
