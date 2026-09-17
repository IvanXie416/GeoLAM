"""D4RT model skeleton aligned with paper query interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import IndependentQueryDecoder
from .encoder import VideoPatchTransformerEncoder
from .heads import D4RTHeads
from .query_embedding import QueryEmbedder
from .utils import resolve_int

_LOGGER = logging.getLogger(__name__)
_WRAPPER_PREFIXES = ("", "module.", "model.", "backbone.", "trunk.", "visual.", "network.", "net.")


def _resolve_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unpack_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload  # already a raw state dict
    return {}


def _encoder_variant_defaults(variant: str) -> tuple[int, int, int, float]:
    key = variant.strip().lower().replace("_", "-")
    table = {
        "vit-b": (768, 12, 12, 4.0),
        "vit-l": (1024, 16, 24, 4.0),
        # VideoMAE2 ViT-g uses hidden=1408 and MLP hidden=6144.
        "vit-g": (1408, 16, 40, 6144.0 / 1408.0),
    }
    return table.get(key, (512, 8, 8, 4.0))


def _candidate_pretrained_keys(target_key: str) -> list[str]:
    base = [target_key]
    if target_key.startswith("patch_embed."):
        base.append(target_key.replace("patch_embed.", "patch_embed.proj.", 1))
    if target_key.startswith("final_norm."):
        base.append(target_key.replace("final_norm.", "norm.", 1))

    prefixes = ("", "module.", "model.", "encoder.", "backbone.", "trunk.", "visual.")
    out: list[str] = []
    for item in base:
        for prefix in prefixes:
            out.append(prefix + item)
    return out


def _wrapped_keys(base_key: str) -> list[str]:
    return [prefix + base_key for prefix in _WRAPPER_PREFIXES]


def _find_tensor(
    src_state: dict[str, torch.Tensor],
    candidate_keys: tuple[str, ...] | list[str],
    expected_shape: torch.Size | None = None,
) -> torch.Tensor | None:
    for key in candidate_keys:
        for wrapped in _wrapped_keys(key):
            value = src_state.get(wrapped, None)
            if not torch.is_tensor(value):
                continue
            if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
                continue
            return value
    return None


def _resize_patch_embed_weight(src: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor | None:
    if src.ndim != 5 or len(dst_shape) != 5:
        return None
    if src.shape[0] != dst_shape[0] or src.shape[1] != dst_shape[1]:
        return None
    if tuple(src.shape) == tuple(dst_shape):
        return src

    so, si, st, sh, sw = src.shape
    dt, dh, dw = int(dst_shape[2]), int(dst_shape[3]), int(dst_shape[4])
    kernel = src.reshape(so * si, 1, st, sh, sw).to(dtype=torch.float32)
    resized = F.interpolate(
        kernel,
        size=(dt, dh, dw),
        mode="trilinear",
        align_corners=False,
    )
    return resized.reshape(so, si, dt, dh, dw).to(dtype=src.dtype)


def _structured_pretrained_tensor(dst_key: str, dst_value: torch.Tensor, src_state: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if dst_key == "patch_embed.weight":
        for src_key in ("encoder.patch_embed.proj.weight", "patch_embed.proj.weight", "patch_embed.weight"):
            src = _find_tensor(src_state, [src_key])
            if src is None:
                continue
            out = _resize_patch_embed_weight(src, dst_value.shape)
            if out is not None and tuple(out.shape) == tuple(dst_value.shape):
                return out
        return None

    if dst_key == "patch_embed.bias":
        return _find_tensor(src_state, ["encoder.patch_embed.proj.bias", "patch_embed.proj.bias", "patch_embed.bias"], dst_value.shape)

    if dst_key.startswith("final_norm."):
        suffix = dst_key.split(".", 1)[1]
        return _find_tensor(src_state, [f"encoder.norm.{suffix}", f"norm.{suffix}", dst_key], dst_value.shape)

    if not dst_key.startswith("blocks."):
        return None

    parts = dst_key.split(".")
    if len(parts) < 4:
        return None
    try:
        block_id = int(parts[1])
    except ValueError:
        return None
    suffix = ".".join(parts[2:])
    base = f"encoder.blocks.{block_id}"

    direct_map = {
        "norm_attn.weight": f"{base}.norm1.weight",
        "norm_attn.bias": f"{base}.norm1.bias",
        "attn.in_proj_weight": f"{base}.attn.qkv.weight",
        "attn.out_proj.weight": f"{base}.attn.proj.weight",
        "attn.out_proj.bias": f"{base}.attn.proj.bias",
        "norm_ff.weight": f"{base}.norm2.weight",
        "norm_ff.bias": f"{base}.norm2.bias",
        "ff.0.weight": f"{base}.mlp.fc1.weight",
        "ff.0.bias": f"{base}.mlp.fc1.bias",
        "ff.3.weight": f"{base}.mlp.fc2.weight",
        "ff.3.bias": f"{base}.mlp.fc2.bias",
    }
    mapped_key = direct_map.get(suffix)
    if mapped_key is not None:
        src = _find_tensor(src_state, [mapped_key], dst_value.shape)
        if src is not None:
            return src

    if suffix == "attn.in_proj_bias":
        q_bias = _find_tensor(src_state, [f"{base}.attn.q_bias"])
        v_bias = _find_tensor(src_state, [f"{base}.attn.v_bias"])
        if q_bias is not None and v_bias is not None and tuple(q_bias.shape) == tuple(v_bias.shape):
            k_bias = torch.zeros_like(q_bias)
            in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            if tuple(in_proj_bias.shape) == tuple(dst_value.shape):
                return in_proj_bias

    return None


class D4RTModel(nn.Module):
    """Unified model: encoder + independent query decoder + task heads."""

    def __init__(self, model_cfg: Any) -> None:
        super().__init__()
        input_cfg = model_cfg["input"]
        encoder_cfg = model_cfg["encoder"]
        decoder_cfg = model_cfg["decoder"]
        query_cfg = model_cfg["query_embedding"]

        enc_variant = str(encoder_cfg.get("variant", "vit-b"))
        default_hidden, default_heads, default_layers, default_mlp = _encoder_variant_defaults(enc_variant)
        enc_hidden = resolve_int(encoder_cfg.get("hidden_dim"), default_hidden)
        enc_heads = resolve_int(encoder_cfg.get("num_heads"), default_heads)
        enc_layers = resolve_int(encoder_cfg.get("num_layers"), default_layers)
        enc_mlp = _resolve_float(encoder_cfg.get("mlp_ratio"), default_mlp)

        dec_hidden = resolve_int(decoder_cfg.get("hidden_dim"), enc_hidden)
        dec_heads = resolve_int(decoder_cfg.get("num_heads"), max(1, min(16, dec_hidden // 64)))
        dec_layers = resolve_int(decoder_cfg.get("num_layers"), 8)
        dec_mlp = _resolve_float(decoder_cfg.get("mlp_ratio"), 4.0)

        patch_t_h_w = encoder_cfg.get("patch_size_t_h_w", [2, 16, 16])
        self.encoder = VideoPatchTransformerEncoder(
            in_channels=int(input_cfg.get("channels", 3)),
            hidden_dim=enc_hidden,
            patch_size_t_h_w=(int(patch_t_h_w[0]), int(patch_t_h_w[1]), int(patch_t_h_w[2])),
            num_layers=enc_layers,
            num_heads=enc_heads,
            mlp_ratio=enc_mlp,
            max_tokens=resolve_int(encoder_cfg.get("max_tokens"), 4096),
            attention_pattern=str(encoder_cfg.get("attention_pattern", "global")),
        )
        self.use_aspect_ratio_token = bool(input_cfg.get("use_aspect_ratio_token", False))
        self.aspect_ratio_proj = (
            nn.Sequential(
                nn.Linear(1, enc_hidden),
                nn.GELU(),
                nn.Linear(enc_hidden, enc_hidden),
            )
            if self.use_aspect_ratio_token
            else None
        )

        self.memory_proj = nn.Identity() if dec_hidden == enc_hidden else nn.Linear(enc_hidden, dec_hidden)
        local_patch = query_cfg.get("local_rgb_patch", {})
        self.query_embedder = QueryEmbedder(
            hidden_dim=dec_hidden,
            clip_frames=int(input_cfg.get("clip_frames", 48)),
            local_patch_enabled=bool(local_patch.get("enabled", True)),
            local_patch_size=int(local_patch.get("patch_size", 9)),
            uv_num_bands=resolve_int(query_cfg.get("uv_num_bands"), 8),
        )
        self.decoder = IndependentQueryDecoder(
            hidden_dim=dec_hidden,
            num_layers=dec_layers,
            num_heads=dec_heads,
            mlp_ratio=dec_mlp,
        )
        self.heads = D4RTHeads(hidden_dim=dec_hidden)
        self._load_pretrained_encoder_weights(encoder_cfg.get("pretrained", None))

    def _project_aspect_ratio_token(
        self,
        video: torch.Tensor,
        aspect_ratio: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.aspect_ratio_proj is None:
            return None
        if aspect_ratio is None:
            aspect_ratio = torch.ones((video.shape[0], 1), dtype=video.dtype, device=video.device)
        else:
            if not torch.is_tensor(aspect_ratio):
                aspect_ratio = torch.as_tensor(aspect_ratio, dtype=video.dtype, device=video.device)
            aspect_ratio = aspect_ratio.to(device=video.device, dtype=video.dtype)
            if aspect_ratio.ndim == 1:
                aspect_ratio = aspect_ratio.unsqueeze(-1)
            if aspect_ratio.ndim != 2 or aspect_ratio.shape[0] != video.shape[0]:
                raise ValueError(f"Expected aspect_ratio [B,1] or [B], got {tuple(aspect_ratio.shape)}")
        return self.aspect_ratio_proj(aspect_ratio).unsqueeze(1)

    def encode_video(
        self,
        video: torch.Tensor,
        aspect_ratio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {video.shape}")
        extra_tokens = self._project_aspect_ratio_token(video=video, aspect_ratio=aspect_ratio)
        return self.memory_proj(self.encoder(video, extra_tokens=extra_tokens))

    def decode_queries(
        self,
        video: torch.Tensor,
        query: dict[str, torch.Tensor],
        memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {video.shape}")

        u = query["u"].to(dtype=video.dtype)
        v = query["v"].to(dtype=video.dtype)
        t_src = query["t_src"].long()
        t_tgt = query["t_tgt"].long()
        t_cam = query["t_cam"].long()

        query_tokens = self.query_embedder(
            video=video,
            u=u,
            v=v,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
        )
        decoded = self.decoder(query_tokens, memory)
        return self.heads(decoded)

    def _load_pretrained_encoder_weights(self, pretrained_cfg: Any) -> None:
        if not isinstance(pretrained_cfg, dict):
            return
        if "enabled" in pretrained_cfg and not bool(pretrained_cfg.get("enabled", True)):
            return
        pretrained_type = str(pretrained_cfg.get("type", "")).strip().lower()
        strict = bool(pretrained_cfg.get("strict", False))
        must_succeed = bool(pretrained_cfg.get("must_succeed", False)) or strict or ("videomae" in pretrained_type)

        def _raise_or_warn(message: str) -> None:
            if must_succeed:
                raise RuntimeError(message)
            _LOGGER.warning(message)

        path_raw = pretrained_cfg.get("path", None)
        if path_raw in (None, ""):
            _raise_or_warn("Encoder pretrained is enabled but `pretrained.path` is empty.")
            return
        path = Path(str(path_raw))
        if not path.exists():
            _raise_or_warn(f"Encoder pretrained path does not exist: {path}")
            return

        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive
            _raise_or_warn(f"Failed to load pretrained checkpoint {path}: {exc}")
            return

        src_state = _unpack_state_dict(payload)
        if not src_state:
            _raise_or_warn(f"No usable state_dict found in pretrained checkpoint: {path}")
            return

        dst_state = self.encoder.state_dict()
        matched: dict[str, torch.Tensor] = {}
        for dst_key, dst_value in dst_state.items():
            structured = _structured_pretrained_tensor(dst_key, dst_value, src_state)
            if structured is not None:
                matched[dst_key] = structured
                continue
            for src_key in _candidate_pretrained_keys(dst_key):
                src_value = src_state.get(src_key, None)
                if src_value is None:
                    continue
                if not torch.is_tensor(src_value):
                    continue
                if tuple(src_value.shape) != tuple(dst_value.shape):
                    continue
                matched[dst_key] = src_value
                break

        if not matched:
            _raise_or_warn(f"No encoder tensors matched pretrained checkpoint: {path}")
            return

        core_ok = (
            "patch_embed.weight" in matched
            and "final_norm.weight" in matched
            and any(key.startswith("blocks.") for key in matched)
        )
        if not core_ok:
            _raise_or_warn(
                "Pretrained checkpoint matched only partial/non-core encoder tensors; "
                "refusing to continue because VideoMAE init is required."
            )
            return

        self.encoder.load_state_dict(matched, strict=False)
        loaded_elems = int(sum(v.numel() for v in matched.values()))
        total_elems = int(sum(v.numel() for v in dst_state.values()))
        _LOGGER.info(
            "Loaded encoder pretrained tensors: %d/%d params (%d/%d elems) from %s",
            len(matched),
            len(dst_state),
            loaded_elems,
            total_elems,
            path,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        video = batch["video"]
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {video.shape}")
        memory = self.encode_video(video=video, aspect_ratio=batch.get("aspect_ratio"))
        return self.decode_queries(video=video, query=batch["query"], memory=memory)
