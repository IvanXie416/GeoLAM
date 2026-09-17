"""Dataset and dataloader builders for the 9Mix OpenD4RT training recipe."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from src.core.registry import Registry

from .blendermvs_raw_dataset import BlendermvsRawConfig, BlendermvsRawDataset
from .co3d_raw_dataset import Co3dRawConfig, Co3dRawDataset
from .dynamic_replica_raw_dataset import DynamicReplicaRawConfig, DynamicReplicaRawDataset
from .kubric_full_robust_dataset import KubricFullRobustConfig, KubricFullRobustDataset
from .kubric_full_robust_preprocess_dataset import (
    KubricFullRobustPreprocessConfig,
    KubricFullRobustPreprocessDataset,
)
from .mixture_dataset import MixtureDataset, MixtureDatasetConfig
from .mvs_synth_raw_dataset import MvsSynthRawConfig, MvsSynthRawDataset
from .pointodyssey_raw_dataset import PointOdysseyRawConfig, PointOdysseyRawDataset
from .raw_augment import (
    STATIC_LOCAL_GLOBAL_TGT_DELTA_CHOICES,
    STATIC_LOCAL_GLOBAL_TGT_DELTA_PROBS,
    augment_cfg_from_train_config,
)
from .scannet_raw_dataset import ScannetRawConfig, ScannetRawDataset
from .seeding import configure_dataset_seeding, seed_dataloader_worker
from .tartanair_raw_dataset import TartanairRawConfig, TartanairRawDataset
from .virtual_kitti2_raw_dataset import VirtualKitti2RawConfig, VirtualKitti2RawDataset

DATASET_REGISTRY = Registry("dataset")


def _image_size(cfg: Any) -> tuple[int, int]:
    image_size_cfg = cfg.get_path("data.image_size", [256, 256])
    if not isinstance(image_size_cfg, (list, tuple)) or len(image_size_cfg) != 2:
        raise ValueError(f"data.image_size must be [H, W], got: {image_size_cfg}")
    return int(image_size_cfg[0]), int(image_size_cfg[1])


def _queries_per_clip(cfg: Any) -> int:
    return int(cfg.get_path("train_sampling.queries_per_clip", 4096))


def _hard_query_ratio(cfg: Any) -> float:
    return float(cfg.get_path("train_sampling.hard_query_ratio", 0.0))


def _prob_t_tgt_equals_t_cam(cfg: Any) -> float:
    return float(cfg.get_path("train_sampling.timestep_sampling.prob_t_tgt_equals_t_cam", 0.0))


def _query_timestep_delta_kwargs(cfg: Any) -> dict[str, tuple[int | None, ...] | tuple[float, ...] | None]:
    path = "train_sampling.timestep_sampling"
    mode_raw = cfg.get_path(f"{path}.t_src_tgt_delta_mode", None)
    choices_raw = cfg.get_path(f"{path}.t_src_tgt_delta_choices", None)
    probs_raw = cfg.get_path(f"{path}.t_src_tgt_delta_probs", None)

    if choices_raw is None:
        mode = str(mode_raw).strip().lower().replace("-", "_") if mode_raw is not None else ""
        if mode in {"", "none", "off", "disabled", "uniform", "independent_uniform"}:
            choices: tuple[int | None, ...] | None = None
            probs: tuple[float, ...] | None = None
        elif mode in {"static_local_global", "local_global", "local_global_static"}:
            choices = STATIC_LOCAL_GLOBAL_TGT_DELTA_CHOICES
            probs = STATIC_LOCAL_GLOBAL_TGT_DELTA_PROBS
        else:
            raise ValueError(
                f"Unsupported {path}.t_src_tgt_delta_mode={mode_raw!r}; "
                "supported: off, independent_uniform, static_local_global"
            )
    else:
        if not isinstance(choices_raw, (list, tuple)):
            raise ValueError(f"{path}.t_src_tgt_delta_choices must be a list, got {choices_raw!r}")
        parsed_choices: list[int | None] = []
        for item in choices_raw:
            if item is None:
                parsed_choices.append(None)
                continue
            if isinstance(item, str):
                token = item.strip().lower().replace("-", "_")
                if token in {"none", "null", "full", "full_range", "global", "all"}:
                    parsed_choices.append(None)
                    continue
            parsed_choices.append(int(item))
        if not parsed_choices:
            raise ValueError(f"{path}.t_src_tgt_delta_choices must be non-empty")
        choices = tuple(parsed_choices)
        if probs_raw is None:
            probs = tuple([1.0 / float(len(choices))] * len(choices))
        else:
            if not isinstance(probs_raw, (list, tuple)):
                raise ValueError(f"{path}.t_src_tgt_delta_probs must be a list, got {probs_raw!r}")
            probs = tuple(float(v) for v in probs_raw)
            if len(probs) != len(choices):
                raise ValueError(f"{path}.t_src_tgt_delta_probs length does not match choices length")

    return {"t_src_tgt_delta_choices": choices, "t_src_tgt_delta_probs": probs}


def _clip_frames(cfg: Any) -> int:
    return int(cfg.get_path("data.clip_frames", 48))


def _bad_sample_registry_path(cfg: Any) -> Path:
    return Path(str(cfg.get_path("data.bad_sample_registry.path", "data/meta/bad_sample.json")))


def _bad_sample_max_retries(cfg: Any) -> int:
    return int(cfg.get_path("data.bad_sample_registry.max_retries", 64))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_dataset_type(dataset_type: str) -> str:
    key = str(dataset_type).strip().lower()
    aliases = {
        "blendermvs": "blendedmvs_raw",
        "blendermvs_raw": "blendedmvs_raw",
        "blendedmvs": "blendedmvs_raw",
        "blended_mvs": "blendedmvs_raw",
        "co3d": "co3d_raw",
        "co3dv2": "co3d_raw",
        "co3d_v2": "co3d_raw",
        "dynamic_replica": "dynamic_replica_raw",
        "dynamic-replica": "dynamic_replica_raw",
        "kubric_full": "kubric_full_robust",
        "kubric-full": "kubric_full_robust",
        "kubric_full_preprocess": "kubric_full_robust_preprocess",
        "kubric_full_processed": "kubric_full_robust_preprocess",
        "mvs_synth": "mvs_synth_raw",
        "mvs-synth": "mvs_synth_raw",
        "pointodyssey": "pointodyssey_raw",
        "point_odyssey": "pointodyssey_raw",
        "scannet": "scannet_raw",
        "tartanair": "tartanair_raw",
        "virtual_kitti2": "virtual_kitti2_raw",
        "virtual-kitti-2": "virtual_kitti2_raw",
        "virtualkitti2": "virtual_kitti2_raw",
        "vkitti2": "virtual_kitti2_raw",
        "vitual-kitti-2": "virtual_kitti2_raw",
        "vitual_kitti_2": "virtual_kitti2_raw",
        "mixture": "mixture_raw",
    }
    return aliases.get(key, key)


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, str):
        out: list[int] = []
        for part in value.split(","):
            try:
                out.append(int(part.strip()))
            except ValueError:
                continue
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _blendermvs_roots(cfg: Any) -> list[Path]:
    roots_value = cfg.get_path("data.blendermvs.roots", cfg.get_path("data.blendedmvs.roots", None))
    roots = [Path(item) for item in _to_str_list(roots_value)]
    if roots:
        return roots
    root_value = cfg.get_path(
        "data.blendermvs.root",
        cfg.get_path("data.blendedmvs.root", "data/blendermvs/base-low-res/BlendedMVS"),
    )
    return [Path(item) for item in _to_str_list(root_value)] or [Path(str(root_value))]


@DATASET_REGISTRY.register("blendermvs_raw")
@DATASET_REGISTRY.register("blendedmvs_raw")
def _build_blendedmvs_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    roots = _blendermvs_roots(cfg)
    augment_cfg = augment_cfg_from_train_config(cfg)
    split_map_raw = cfg.get_path("data.blendermvs.split_map", cfg.get_path("data.blendedmvs.split_map", None))
    split_map = split_map_raw if isinstance(split_map_raw, dict) else None
    return BlendermvsRawDataset(
        BlendermvsRawConfig(
            root=roots[0],
            roots=tuple(roots),
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            split_map=split_map,
            max_scenes=_optional_int(cfg.get_path("data.blendermvs.max_scenes", cfg.get_path("data.blendedmvs.max_scenes", None))),
            split_modulo=int(cfg.get_path("data.blendermvs.split_modulo", cfg.get_path("data.blendedmvs.split_modulo", 20))),
            use_masked_images=bool(cfg.get_path("data.blendermvs.use_masked_images", cfg.get_path("data.blendedmvs.use_masked_images", False))),
            max_depth_m=float(cfg.get_path("data.blendermvs.max_depth_m", cfg.get_path("data.blendedmvs.max_depth_m", 1e5))),
            depth_clip_percentile=float(
                cfg.get_path("data.blendermvs.depth_clip_percentile", cfg.get_path("data.blendedmvs.depth_clip_percentile", 0.0))
            ),
            min_depth_valid_ratio=float(
                cfg.get_path("data.blendermvs.min_depth_valid_ratio", cfg.get_path("data.blendedmvs.min_depth_valid_ratio", 0.0))
            ),
            min_valid_frames_ratio=float(
                cfg.get_path("data.blendermvs.min_valid_frames_ratio", cfg.get_path("data.blendedmvs.min_valid_frames_ratio", 0.0))
            ),
            require_complete_frames=bool(
                cfg.get_path("data.blendermvs.require_complete_frames", cfg.get_path("data.blendedmvs.require_complete_frames", False))
            ),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
        )
    )


@DATASET_REGISTRY.register("co3d_raw")
def _build_co3d_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.co3d.root", "data/co3d/v2"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    categories = _to_str_list(cfg.get_path("data.co3d.categories", []))
    split_map_raw = cfg.get_path("data.co3d.split_map", None)
    split_map = split_map_raw if isinstance(split_map_raw, dict) else None
    return Co3dRawDataset(
        Co3dRawConfig(
            root=root,
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            split_map=split_map,
            max_scenes=_optional_int(cfg.get_path("data.co3d.max_scenes", None)),
            categories=categories or None,
            min_viewpoint_quality=float(cfg.get_path("data.co3d.min_viewpoint_quality", 0.5)),
            use_depth_masks=bool(cfg.get_path("data.co3d.use_depth_masks", False)),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
        )
    )


@DATASET_REGISTRY.register("kubric_full_robust")
def _build_kubric_full_robust(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.kubric_full.root", "data/kubric/movi-f_full/512x512"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    split_map_raw = cfg.get_path("data.kubric_full.tfds_split_map", {"train": "train", "val": "validation", "test": "validation"})
    split_map = split_map_raw if isinstance(split_map_raw, dict) else {"train": "train", "val": "validation", "test": "validation"}
    return KubricFullRobustDataset(
        KubricFullRobustConfig(
            root=root,
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            max_scenes=_optional_int(cfg.get_path("data.kubric_full.max_scenes", None)),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
            tfds_split_map=split_map,
            shuffle_buffer_size=int(cfg.get_path("data.kubric_full.shuffle_buffer_size", 256)),
            eval_cache_max_items=int(cfg.get_path("data.kubric_full.eval_cache_max_items", 4)),
            benchmark_tracking_enabled=bool(cfg.get_path("data.kubric_full.benchmark_tracking.enabled", False)),
            benchmark_max_queries=int(cfg.get_path("data.kubric_full.benchmark_tracking.max_queries", 4096)),
        )
    )


@DATASET_REGISTRY.register("kubric_full_robust_preprocess")
def _build_kubric_full_robust_preprocess(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.kubric_full.processed_root", cfg.get_path("data.kubric_full.root", "data/kubric_full/kubric_full_process_v1")))
    augment_cfg = augment_cfg_from_train_config(cfg)
    split_map_raw = cfg.get_path("data.kubric_full.processed_split_map", {"train": "train", "val": "validation", "test": "validation"})
    split_map = split_map_raw if isinstance(split_map_raw, dict) else {"train": "train", "val": "validation", "test": "validation"}
    mmap_mode_raw = cfg.get_path("data.kubric_full.mmap_mode", "r")
    mmap_mode = None if str(mmap_mode_raw).lower() in {"", "none", "false"} else str(mmap_mode_raw)
    return KubricFullRobustPreprocessDataset(
        KubricFullRobustPreprocessConfig(
            root=root,
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            training=(split == "train"),
            max_scenes=_optional_int(cfg.get_path("data.kubric_full.max_scenes", None)),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
            split_map=split_map,
            mmap_mode=mmap_mode,
            eval_cache_max_items=int(cfg.get_path("data.kubric_full.eval_cache_max_items", 4)),
            benchmark_tracking_enabled=bool(cfg.get_path("data.kubric_full.benchmark_tracking.enabled", False)),
            benchmark_max_queries=int(cfg.get_path("data.kubric_full.benchmark_tracking.max_queries", 4096)),
        )
    )


@DATASET_REGISTRY.register("pointodyssey_raw")
def _build_pointodyssey_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.pointodyssey.root", "data/pointodyssey/v2"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    split_map_raw = cfg.get_path("data.pointodyssey.split_map", None)
    split_map = split_map_raw if isinstance(split_map_raw, dict) else None
    return PointOdysseyRawDataset(
        PointOdysseyRawConfig(
            root=root,
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            split_map=split_map,
            max_scenes=_optional_int(cfg.get_path("data.pointodyssey.max_scenes", None)),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
            max_cached_scenes=int(cfg.get_path("data.pointodyssey.max_cached_scenes", 2)),
            val_clips_per_scene=int(cfg.get_path("data.pointodyssey.val_clips_per_scene", 1)),
        )
    )


@DATASET_REGISTRY.register("virtual_kitti2_raw")
def _build_virtual_kitti2_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.virtual_kitti2.root", "data/vitual-kitti-2/v2"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    split_scenes_raw = cfg.get_path("data.virtual_kitti2.split_scenes", None)
    split_scenes = split_scenes_raw if isinstance(split_scenes_raw, dict) else None
    variants = _to_str_list(cfg.get_path("data.virtual_kitti2.variants", []))
    camera_ids = _to_int_list(cfg.get_path("data.virtual_kitti2.camera_ids", [0]))
    return VirtualKitti2RawDataset(
        VirtualKitti2RawConfig(
            root=root,
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            split_scenes=split_scenes,
            variants=variants or None,
            camera_ids=camera_ids or [0],
            max_scenes=_optional_int(cfg.get_path("data.virtual_kitti2.max_scenes", None)),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
        )
    )


def _dynamic_replica_split(split: str, cfg: Any) -> str:
    split_map = cfg.get_path("data.dynamic_replica.split_map", {"train": "train", "val": "valid", "test": "test"})
    if isinstance(split_map, dict):
        mapped = split_map.get(split)
        if mapped:
            return str(mapped)
    return {"train": "train", "val": "valid", "test": "test"}.get(split, split)


@DATASET_REGISTRY.register("dynamic_replica_raw")
def _build_dynamic_replica_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.dynamic_replica.root", "data/dynamic-replica/v2"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    return DynamicReplicaRawDataset(
        DynamicReplicaRawConfig(
            root=root,
            split=_dynamic_replica_split(split, cfg),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            max_scenes=_optional_int(cfg.get_path("data.dynamic_replica.max_scenes", None)),
            camera_convention=str(cfg.get_path("data.dynamic_replica.camera_convention", "dynamic_replica_v2")),
            depth_decode_mode=str(cfg.get_path("data.dynamic_replica.depth_decode_mode", "auto")),
            depth_divisor=float(cfg.get_path("data.dynamic_replica.depth_divisor", 10000.0)),
            reprojection_self_check_enabled=bool(cfg.get_path("data.dynamic_replica.reprojection_self_check.enabled", True)),
            reprojection_self_check_mode=str(cfg.get_path("data.dynamic_replica.reprojection_self_check.mode", "warn")),
            reprojection_self_check_median_threshold_px=float(cfg.get_path("data.dynamic_replica.reprojection_self_check.median_threshold_px", 5.0)),
            reprojection_self_check_max_scenes=int(cfg.get_path("data.dynamic_replica.reprojection_self_check.max_scenes", 1)),
            reprojection_self_check_max_frames=int(cfg.get_path("data.dynamic_replica.reprojection_self_check.max_frames", 4)),
            reprojection_self_check_max_points=int(cfg.get_path("data.dynamic_replica.reprojection_self_check.max_points_per_frame", 4096)),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
            benchmark_tracking_enabled=bool(cfg.get_path("data.dynamic_replica.benchmark_tracking.enabled", False)),
            benchmark_max_queries=int(cfg.get_path("data.dynamic_replica.benchmark_tracking.max_queries", 0)),
        )
    )


@DATASET_REGISTRY.register("mvs_synth_raw")
def _build_mvs_synth_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.mvs_synth.root", "data/mvs-synth/v1"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    split_map_raw = cfg.get_path("data.mvs_synth.split_map", None)
    split_map = split_map_raw if isinstance(split_map_raw, dict) else None
    return MvsSynthRawDataset(
        MvsSynthRawConfig(
            root=root,
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            split_map=split_map,
            sequence_dir=str(cfg.get_path("data.mvs_synth.sequence_dir", "GTAV_540")),
            max_scenes=_optional_int(cfg.get_path("data.mvs_synth.max_scenes", None)),
            split_modulo=int(cfg.get_path("data.mvs_synth.split_modulo", 20)),
            depth_scale=float(cfg.get_path("data.mvs_synth.depth_scale", 1.0)),
            max_depth_m=float(cfg.get_path("data.mvs_synth.max_depth_m", 1e5)),
            depth_clip_percentile=float(cfg.get_path("data.mvs_synth.depth_clip_percentile", 0.0)),
            min_depth_valid_ratio=float(cfg.get_path("data.mvs_synth.min_depth_valid_ratio", 0.0)),
            min_valid_frames_ratio=float(cfg.get_path("data.mvs_synth.min_valid_frames_ratio", 0.0)),
            require_complete_frames=bool(cfg.get_path("data.mvs_synth.require_complete_frames", False)),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
        )
    )


def _scannet_split_file(split: str, cfg: Any) -> Path:
    split_files = cfg.get_path("data.scannet.split_files", {})
    if isinstance(split_files, dict):
        picked = split_files.get(split)
        if picked:
            return Path(str(picked))
    defaults = {
        "train": "data/scannet/plus-v2/splits/nvs_sem_train.txt",
        "val": "data/scannet/plus-v2/splits/nvs_sem_val.txt",
        "test": "data/scannet/plus-v2/splits/nvs_test.txt",
    }
    return Path(defaults.get(split, defaults["val"]))


@DATASET_REGISTRY.register("scannet_raw")
def _build_scannet_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.scannet.root", "data/scannet/plus-v2/data"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    return ScannetRawDataset(
        ScannetRawConfig(
            root=root,
            split_file=_scannet_split_file(split, cfg),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            max_scenes=_optional_int(cfg.get_path("data.scannet.max_scenes", None)),
            training=(split == "train"),
            source=str(cfg.get_path("data.scannet.source", "auto")),
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
        )
    )


@DATASET_REGISTRY.register("tartanair_raw")
def _build_tartanair_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    root = Path(cfg.get_path("data.tartanair.root", "data/tartanair/v2"))
    augment_cfg = augment_cfg_from_train_config(cfg)
    split_map_raw = cfg.get_path("data.tartanair.split_map", None)
    split_map = split_map_raw if isinstance(split_map_raw, dict) else None
    difficulties = _to_str_list(cfg.get_path("data.tartanair.difficulties", ["Data_easy", "Data_hard"]))
    intrinsics_raw = cfg.get_path("data.tartanair.intrinsics", None)
    intrinsics = [float(v) for v in intrinsics_raw] if isinstance(intrinsics_raw, (list, tuple)) else None
    return TartanairRawDataset(
        TartanairRawConfig(
            root=root,
            split=str(split),
            clip_frames=_clip_frames(cfg),
            image_size=_image_size(cfg),
            queries_per_clip=_queries_per_clip(cfg),
            hard_query_ratio=_hard_query_ratio(cfg),
            prob_t_tgt_equals_t_cam=_prob_t_tgt_equals_t_cam(cfg),
            **_query_timestep_delta_kwargs(cfg),
            training=(split == "train"),
            split_map=split_map,
            camera_name=str(cfg.get_path("data.tartanair.camera_name", "lcam_front")),
            difficulties=difficulties or ["Data_easy", "Data_hard"],
            max_scenes=_optional_int(cfg.get_path("data.tartanair.max_scenes", None)),
            split_modulo=int(cfg.get_path("data.tartanair.split_modulo", 20)),
            max_depth_m=float(cfg.get_path("data.tartanair.max_depth_m", 1000.0)),
            intrinsics=intrinsics,
            augment=augment_cfg,
            bad_sample_registry_path=_bad_sample_registry_path(cfg),
            max_sample_retries=_bad_sample_max_retries(cfg),
        )
    )


def _normalize_mixture_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "blendermvs_raw": "blendermvs",
        "blendedmvs": "blendermvs",
        "blendedmvs_raw": "blendermvs",
        "co3d_raw": "co3d",
        "co3dv2": "co3d",
        "dynamic_replica_raw": "dynamic_replica",
        "kubric_full_robust": "kubric_full",
        "kubric_full_robust_preprocess": "kubric_full",
        "mvs_synth_raw": "mvs_synth",
        "pointodyssey_raw": "pointodyssey",
        "point_odyssey": "pointodyssey",
        "scannet_raw": "scannet",
        "tartanair_raw": "tartanair",
        "virtual_kitti2_raw": "virtual_kitti2",
        "virtualkitti2": "virtual_kitti2",
        "vkitti2": "virtual_kitti2",
        "vitual_kitti_2": "virtual_kitti2",
    }
    return aliases.get(key, key)


def _resolve_mixture_sources(split: str, cfg: Any) -> list[tuple[str, str]]:
    names = _to_str_list(cfg.get_path(f"data.{split}_dataset_mixture", None))
    if not names:
        names = _to_str_list(cfg.get_path("data.dataset_mixture", []))

    kubric_full_backend = str(cfg.get_path("data.kubric_full.backend", "tfds")).strip().lower()
    kubric_full_builder = (
        "kubric_full_robust_preprocess"
        if kubric_full_backend in {"preprocess", "processed", "npy", "preprocessed"}
        else "kubric_full_robust"
    )
    source_to_builder = {
        "blendermvs": "blendedmvs_raw",
        "co3d": "co3d_raw",
        "dynamic_replica": "dynamic_replica_raw",
        "kubric_full": kubric_full_builder,
        "mvs_synth": "mvs_synth_raw",
        "pointodyssey": "pointodyssey_raw",
        "scannet": "scannet_raw",
        "tartanair": "tartanair_raw",
        "virtual_kitti2": "virtual_kitti2_raw",
    }

    resolved: list[tuple[str, str]] = []
    unknown_sources: list[str] = []
    for name in names:
        normalized = _normalize_mixture_name(name)
        builder_name = source_to_builder.get(normalized)
        if builder_name is None:
            unknown_sources.append(str(name))
        else:
            resolved.append((name, builder_name))
    if unknown_sources:
        supported = ", ".join(sorted(source_to_builder))
        unknown = ", ".join(unknown_sources)
        raise ValueError(f"Unsupported mixture source(s): {unknown}. Supported sources: {supported}")
    return resolved


def _resolve_mixture_weights(cfg: Any, selected_sources: list[str]) -> list[float] | None:
    raw = cfg.get_path("data.mixture_sampling_weights", None)
    if raw is None or not selected_sources:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) < len(selected_sources):
            return None
        return [float(raw[idx]) for idx in range(len(selected_sources))]
    if isinstance(raw, dict):
        out: list[float] = []
        for name in selected_sources:
            normalized = _normalize_mixture_name(name)
            candidate = raw.get(name, raw.get(normalized))
            if candidate is None:
                return None
            out.append(float(candidate))
        return out
    return None


@DATASET_REGISTRY.register("mixture_raw")
def _build_mixture_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    del manifest_paths
    resolved_sources = _resolve_mixture_sources(split, cfg)
    if not resolved_sources:
        raise ValueError("mixture_raw requires at least one source in data.dataset_mixture")

    datasets = []
    source_names: list[str] = []
    for source_name, dataset_type in resolved_sources:
        builder = DATASET_REGISTRY.get(dataset_type)
        try:
            datasets.append(builder(split=split, cfg=cfg, manifest_paths=None))
            source_names.append(source_name)
        except Exception as exc:
            warnings.warn(f"Skip mixture source '{source_name}' ({dataset_type}): {exc}", stacklevel=2)

    if not datasets:
        raise ValueError("No valid datasets could be built for mixture_raw")

    return MixtureDataset(
        MixtureDatasetConfig(
            datasets=datasets,
            weights=_resolve_mixture_weights(cfg, source_names),
        )
    )


def _normalize_manifest_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def build_dataset(split: str, cfg: Any, manifest_arg: str | None = None):
    manifest_paths = _normalize_manifest_arg(manifest_arg)
    dataset_type = cfg.get_path(f"data.{split}_dataset_type")
    if dataset_type is None:
        if manifest_paths:
            raise ValueError("Manifest-backed training was removed from this minimal OpenD4RT training release.")
        if _resolve_mixture_sources(split, cfg):
            dataset_type = "mixture_raw"
        else:
            raise ValueError("No dataset type or 9Mix source list configured.")

    dataset_type = _normalize_dataset_type(str(dataset_type))
    builder = DATASET_REGISTRY.get(dataset_type)
    dataset = builder(split=split, cfg=cfg, manifest_paths=manifest_paths)
    if isinstance(dataset, list):
        dataset = ConcatDataset(dataset)
    configure_dataset_seeding(dataset, base_seed=int(cfg.get_path("experiment.seed", 42)))
    return dataset


def _worker_count(split: str, cfg: Any) -> int:
    return int(cfg.get_path(f"runtime.{split}_num_workers", cfg.get_path("runtime.num_workers", 4)))


def _loader_bool(split: str, cfg: Any, key: str, default: bool) -> bool:
    return bool(cfg.get_path(f"runtime.{split}_{key}", cfg.get_path(f"runtime.{key}", default)))


def _loader_int(split: str, cfg: Any, key: str, default: int | None) -> int | None:
    raw = cfg.get_path(f"runtime.{split}_{key}", cfg.get_path(f"runtime.{key}", default))
    if raw is None:
        return None
    return int(raw)


def build_dataloader(
    split: str,
    cfg: Any,
    manifest_arg: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = build_dataset(split=split, cfg=cfg, manifest_arg=manifest_arg)
    batch_size = int(cfg.get_path(f"runtime.{split}_batch_size", cfg.get_path("runtime.batch_size", 1)))
    drop_last = bool(cfg.get_path(f"runtime.{split}_drop_last", split == "train"))
    num_workers = _worker_count(split, cfg)

    sampler = None
    shuffle = split == "train"
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get_path("runtime.pin_memory", True)),
        "drop_last": drop_last if sampler is None else False,
        "worker_init_fn": seed_dataloader_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = _loader_bool(split, cfg, "persistent_workers", False)
        prefetch_factor = _loader_int(split, cfg, "prefetch_factor", 2)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        timeout = _loader_int(split, cfg, "timeout", 0)
        if timeout is not None:
            loader_kwargs["timeout"] = max(0, int(timeout))

    loader = DataLoader(**loader_kwargs)
    loader.dist_sampler = sampler  # type: ignore[attr-defined]
    return loader
