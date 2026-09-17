from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _strip_cls_and_validate(feat: torch.Tensor, expected_tokens: int, name: str) -> torch.Tensor:
    if feat.ndim != 3:
        raise ValueError(f"{name} must be [B, N, C], got {tuple(feat.shape)}")
    if feat.shape[1] == expected_tokens + 1:
        feat = feat[:, 1:]
    if feat.shape[1] != expected_tokens:
        raise ValueError(f"{name} expected {expected_tokens} patch tokens, got {feat.shape[1]}")
    return feat


def extract_gld_f0_f1_tokens(
    features: dict[int, torch.Tensor],
    batch_size: int,
    expected_tokens: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if 0 not in features or 1 not in features:
        raise KeyError("GLD feature dict must contain level 0 and level 1")
    f0 = _strip_cls_and_validate(features[0], expected_tokens, "F0")
    f1 = _strip_cls_and_validate(features[1], expected_tokens, "F1")
    if f0.shape[0] != batch_size or f1.shape[0] != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got F0={f0.shape[0]} F1={f1.shape[0]}")
    return f0, f1


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "ema", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


class GLDBackboneWrapper(nn.Module):
    """Loads GLD Stage1 DA3 and exposes F0/F1 patch tokens for LAM."""

    def __init__(
        self,
        gld_root: str | Path,
        checkpoint: str | Path | None = None,
        encoder_pretrained_path: str | Path | None = None,
        mae_weight: str | Path | None = None,
        decoder_config_path: str | Path | None = None,
        image_size: int = 224,
        expected_tokens: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.gld_root = Path(gld_root)
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.encoder_pretrained_path = self._resolve_encoder_pretrained_path(encoder_pretrained_path)
        self.mae_weight = None if mae_weight is None else Path(mae_weight)
        self.decoder_config_path = None if decoder_config_path is None else Path(decoder_config_path)
        self.image_size = int(image_size)
        self.expected_tokens = int(expected_tokens)
        self.model = self._load_model()
        if self.checkpoint is not None and self.checkpoint.exists():
            self._load_checkpoint(self.checkpoint)
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _resolve_encoder_pretrained_path(self, encoder_pretrained_path: str | Path | None) -> str:
        if encoder_pretrained_path is not None:
            return str(encoder_pretrained_path)
        local_da3 = self.gld_root / "pretrained_models" / "da3"
        if local_da3.exists():
            return str(local_da3)
        return "depth-anything/DA3-Base"

    def _load_model(self) -> nn.Module:
        src_root = self.gld_root / "src"
        with _prepend_sys_path(src_root), _prepend_sys_path(self.gld_root):
            from stage1.rae_da3 import RAE_DA3

            return RAE_DA3(
                encoder_pretrained_path=self.encoder_pretrained_path,
                encoder_input_size=self.image_size,
                encoder_type="DA3EncoderDirect",
                reshape_to_2d=False,
                mae_weight=None if self.mae_weight is None else str(self.mae_weight),
                decoder_config_path=None if self.decoder_config_path is None else str(self.decoder_config_path),
            )

    def _load_checkpoint(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu")
        state_dict = _unwrap_state_dict(payload)
        if state_dict:
            result = self.model.load_state_dict(state_dict, strict=False)
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            if missing or unexpected:
                warnings.warn(
                    f"GLD checkpoint loaded with missing_keys={missing} unexpected_keys={unexpected}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        mean = self.model.encoder_mean.to(device=images.device, dtype=images.dtype)
        std = self.model.encoder_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}")
        features = self.model.encode(self._normalize_images(images), mode="all")
        return extract_gld_f0_f1_tokens(features, batch_size=images.shape[0], expected_tokens=self.expected_tokens)

    @torch.no_grad()
    def all_features(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}")
        return self.model.encode(self._normalize_images(images), mode="all")
