from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    gld_root: str = "third_party/GLD"
    gld_encoder_pretrained_path: str = "third_party/GLD/pretrained_models/da3"
    d4rt_root: str = "third_party/Open-d4rt"
    gld_checkpoint: str | None = None
    gld_mae_weight: str | None = None
    gld_decoder_config: str | None = None
    d4rt_config: str = "checkpoints/d4rt/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
    d4rt_checkpoint: str = "checkpoints/d4rt/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
    output_dir: str = "outputs/d4rt_pixel"


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 224
    d4rt_image_size: int = 256
    d4rt_context_frames: int = 6
    strides: list[int] = field(default_factory=lambda: [5, 10, 15, 20])
    num_sample_frames: int = 9


@dataclass(frozen=True)
class ModelConfig:
    type: str = "gld"
    f_dim: int = 1536
    c_state: int = 1024
    c_z: int = 1024
    z_dim: int = 512
    c_geo: int = 7
    k_action: int = 16
    k_geo: int = 16
    latent_dim: int = 32
    heads: int = 16
    action_bottleneck_depth: int = 4
    idm_depth: int = 8
    fdm_depth: int = 8
    fdm_self_attn_every: int = 2
    ffn_ratio: float = 4.0
    rgb_patch_size: int = 14
    rgb_decoder_depth: int = 4


@dataclass(frozen=True)
class D4RTConfig:
    query_grid_size: int = 32
    residual_motion_weight: float = 2.0
    xyz_motion_scale: float = 0.1
    uv_motion_scale: float = 0.05


@dataclass(frozen=True)
class LossConfig:
    phase0_warmup_steps: int = 10000
    phase1_geo_ramp_steps: int = 20000
    lambda_gld: float = 1.0
    lambda_pixel: float = 1.0
    lambda_geo_action: float = 0.03


@dataclass(frozen=True)
class LAMConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    d4rt: D4RTConfig = field(default_factory=D4RTConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{name}` must be a mapping")
    return value


def load_lam_config(path: str | Path) -> LAMConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    return LAMConfig(
        paths=PathsConfig(**_section(raw, "paths")),
        data=DataConfig(**_section(raw, "data")),
        model=ModelConfig(**_section(raw, "model")),
        d4rt=D4RTConfig(**_section(raw, "d4rt")),
        loss=LossConfig(**_section(raw, "loss")),
    )
