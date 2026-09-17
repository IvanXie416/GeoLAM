from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .gld_lam import GLDLAM, GLDLAMOutput, init_grid_pos_embed
from .modules import SelfAttentionBlock


class RGBPatchDecoder(nn.Module):
    """Small fallback decoder retained for standalone shape tests.

    Production pixel training replaces it with :class:`GLDMAERGBDecoder`
    through ``set_rgb_decoder``.
    """
    def __init__(self, token_dim=1024, image_size=224, patch_size=14, heads=16, depth=4, ffn_ratio=4.0):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size, self.patch_size = image_size, patch_size
        self.grid_size = image_size // patch_size
        n = self.grid_size ** 2
        self.pos_embed = nn.Parameter(init_grid_pos_embed(n, token_dim), requires_grad=False)
        self.blocks = nn.ModuleList([SelfAttentionBlock(token_dim, heads, ffn_ratio) for _ in range(depth)])
        self.out = nn.Linear(token_dim, 3 * patch_size * patch_size)

    def forward(self, tokens):
        n = self.grid_size ** 2
        if tokens.ndim != 3 or tokens.shape[1] != n:
            raise ValueError(f"RGB decoder expects {n} tokens")
        x = tokens + self.pos_embed.to(tokens)
        for block in self.blocks:
            x = block(x)
        p = self.out(x).sigmoid().reshape(tokens.shape[0], self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
        return p.permute(0, 3, 1, 4, 2, 5).reshape(tokens.shape[0], 3, self.image_size, self.image_size)


@dataclass(frozen=True)
class LAM3DPixelOutput(GLDLAMOutput):
    rgb_hat: torch.Tensor


class GLDMAERGBDecoder(nn.Module):
    """Differentiably apply the frozen GLD MAE decoder to predicted features."""

    def __init__(
        self,
        decoder: nn.Module,
        propagator: nn.Module | None = None,
        image_size: int = 224,
        encoder_mean: torch.Tensor | None = None,
        encoder_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        # Keep the frozen DA3 owner out of the LAM state_dict/DDP parameter
        # broadcast; only its differentiable forward is needed here.
        object.__setattr__(self, "propagator", propagator)
        self.image_size = int(image_size)
        self.register_buffer("encoder_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) if encoder_mean is None else encoder_mean.reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("encoder_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) if encoder_std is None else encoder_std.reshape(1, 3, 1, 1), persistent=False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.decoder.eval()

    def train(self, mode: bool = True):
        # The decoder is a frozen teacher; never enable dropout when the LAM
        # parent switches to training mode.
        super().train(mode)
        self.decoder.eval()
        return self

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        if f0.shape != f1.shape or f0.ndim != 3:
            raise ValueError("Expected matching GLD feature tensors [B, N, C]")
        if self.propagator is None:
            raise RuntimeError("GLD RGB decoder requires the DA3 feature propagator")
        side = int(f1.shape[1] ** 0.5)
        if side * side != f1.shape[1]:
            raise ValueError(f"Expected square GLD token grid, got {f1.shape[1]} tokens")
        feature_map = f1.transpose(1, 2).reshape(f1.shape[0], f1.shape[2], side, side)
        cls = torch.zeros(f1.shape[0], f1.shape[2], device=f1.device, dtype=f1.dtype)
        propagated = self.propagator.propagate_features(
            feature_map, from_level=1, total_view=1, cls_token=cls
        )
        if len(propagated) < 3:
            raise RuntimeError(f"Expected DA3 propagated levels 1-3, got {len(propagated)}")
        # Keep F0/F1 predictions and use DA3's native propagation for F2/F3.
        feats = torch.cat([f0, f1, propagated[1][0][:, 0], propagated[2][0][:, 0]], dim=-1)
        output = self.decoder(feats, input_size=(self.image_size, self.image_size), drop_cls_token=False)
        logits = output.logits if hasattr(output, "logits") else output
        rgb = self.decoder.unpatchify(logits, (self.image_size, self.image_size))
        return rgb * self.encoder_std.to(rgb) + self.encoder_mean.to(rgb)


class LAM3DPixel(GLDLAM):
    """GLD IDM/FDM latent action model constrained by future RGB reconstruction."""

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
        action_bottleneck_depth: int = 2,
        idm_depth: int = 4,
        fdm_depth: int = 4,
        fdm_self_attn_every: int = 2,
        ffn_ratio: float = 4.0,
        image_size: int = 224,
        rgb_decoder: nn.Module | None = None,
        rgb_propagator: nn.Module | None = None,
        rgb_patch_size: int = 14,
        rgb_decoder_depth: int = 4,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            c_state=c_state,
            c_z=c_z,
            z_dim=z_dim,
            c_geo=c_geo,
            k_action=k_action,
            k_geo=k_geo,
            latent_dim=latent_dim,
            heads=heads,
            action_bottleneck_depth=action_bottleneck_depth,
            idm_depth=idm_depth,
            fdm_depth=fdm_depth,
            fdm_self_attn_every=fdm_self_attn_every,
            ffn_ratio=ffn_ratio,
        )
        self.image_size = int(image_size)
        self.rgb_decoder = (GLDMAERGBDecoder(rgb_decoder, propagator=rgb_propagator, image_size=image_size)
                            if rgb_decoder is not None else RGBPatchDecoder(c_state, image_size, rgb_patch_size, heads, rgb_decoder_depth, ffn_ratio))

    def set_rgb_decoder(self, decoder: nn.Module, propagator: nn.Module | None = None) -> None:
        adapter = GLDMAERGBDecoder(decoder, propagator=propagator, image_size=self.image_size)
        # ``set_rgb_decoder`` may be called after the LAM itself was moved.
        # Keep the adapter buffers (normalization statistics) on that device.
        try:
            device = next(self.parameters()).device
            adapter = adapter.to(device)
        except StopIteration:
            pass
        self.rgb_decoder = adapter

    def forward(
        self,
        f0_t: torch.Tensor,
        f1_t: torch.Tensor,
        f0_tk: torch.Tensor,
        f1_tk: torch.Tensor,
    ) -> LAM3DPixelOutput:
        output = super().forward(f0_t, f1_t, f0_tk, f1_tk)
        return LAM3DPixelOutput(
            state_t=output.state_t,
            state_tk=output.state_tk,
            latent_action=output.latent_action,
            pred_tokens_tk=output.pred_tokens_tk,
            f0_hat=output.f0_hat,
            f1_hat=output.f1_hat,
            z_geo_tokens_pred=output.z_geo_tokens_pred,
            rgb_hat=(self.rgb_decoder(output.f0_hat, output.f1_hat)
                     if isinstance(self.rgb_decoder, GLDMAERGBDecoder)
                     else self.rgb_decoder(output.pred_tokens_tk)),
        )
