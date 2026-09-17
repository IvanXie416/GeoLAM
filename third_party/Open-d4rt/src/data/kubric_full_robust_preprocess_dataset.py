"""Kubric MOVi-F full-annotation robust dataset adapter (preprocessed npy scenes)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bad_sample_registry import BadSampleRegistry, RetryableSampleError, is_retryable_data_error
from .kubric_full_robust_dataset import (
    KubricFullRobustConfig,
    KubricFullRobustDataset,
    _dedup_str_list,
)
from .raw_augment import RawAugmentConfig


_REQUIRED_FILES = (
    "rgb.npy",
    "depth_uint16.npy",
    "segmentation.npy",
    "normal_uint16.npy",
    "object_coordinates_uint16.npy",
    "camera_positions.npy",
    "camera_quaternions.npy",
    "instances_bboxes_3d.npy",
    "instances_positions.npy",
    "instances_quaternions.npy",
    "meta.json",
)


@dataclass
class KubricFullRobustPreprocessConfig(KubricFullRobustConfig):
    split_map: dict[str, str] | None = None
    mmap_mode: str | None = "r"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RetryableSampleError(
            f"Failed to read Kubric processed metadata: {type(exc).__name__}: {exc}",
            failed_paths=[str(path)],
        ) from exc
    if not isinstance(payload, dict):
        raise RetryableSampleError(f"Invalid Kubric processed metadata payload: {path}", failed_paths=[str(path)])
    return payload


def _as_np_scalar(value: Any, dtype: Any) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


class KubricFullRobustPreprocessDataset(KubricFullRobustDataset):
    """Loads preprocessed Kubric MOVi-F scenes and reuses robust supervision construction."""

    def __init__(self, config: KubricFullRobustPreprocessConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="kubric_full_robust_preprocess", default_seed=20260418)
        self.augment = config.augment or RawAugmentConfig()
        if not config.training:
            self.augment = RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        self._warned_skip_keys: set[str] = set()

        self.root = Path(config.root)
        self.tfds_dir = self.root
        self.mmap_mode = config.mmap_mode
        split_map = dict(config.split_map or {"train": "train", "val": "validation", "valid": "validation", "test": "validation"})
        self.processed_split = split_map.get(str(config.split).lower(), str(config.split))
        self.scene_dirs = self._discover_scene_dirs()
        if config.max_scenes is not None:
            self.scene_dirs = self.scene_dirs[: int(config.max_scenes)]
        self.num_examples = len(self.scene_dirs)
        if self.num_examples <= 0:
            raise ValueError(f"Empty Kubric processed split '{self.processed_split}' under: {self.root}")

        self._validate_scene_dir(self.scene_dirs[0])

    def _reset_train_iter(self) -> None:
        return

    def _reset_eval_iter(self) -> None:
        return

    def _discover_scene_dirs(self) -> list[Path]:
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            scenes = manifest.get("scenes", [])
            if isinstance(scenes, list):
                out: list[Path] = []
                for item in scenes:
                    if not isinstance(item, dict) or str(item.get("split", "")) != self.processed_split:
                        continue
                    rel = item.get("relative_dir")
                    if rel:
                        out.append(self.root / str(rel))
                if out:
                    return out

        split_dir = self.root / self.processed_split
        if not split_dir.exists():
            raise FileNotFoundError(f"Kubric processed split dir not found: {split_dir}")
        return sorted([p for p in split_dir.iterdir() if p.is_dir()])

    def _validate_scene_dir(self, scene_dir: Path) -> None:
        missing = [name for name in _REQUIRED_FILES if not (scene_dir / name).is_file()]
        if missing:
            raise RetryableSampleError(
                f"Kubric processed scene missing files: scene={scene_dir}; missing={missing}",
                failed_paths=[str(scene_dir / name) for name in missing],
            )

    def _load_npy(self, scene_dir: Path, name: str) -> np.ndarray:
        path = scene_dir / name
        try:
            return np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)
        except Exception as exc:
            raise RetryableSampleError(
                f"Failed to load Kubric processed npy: {type(exc).__name__}: {exc}",
                failed_paths=[str(path)],
            ) from exc

    def _sample_by_index(self, index: int) -> tuple[dict[str, Any], Path]:
        scene_dir = self.scene_dirs[int(index) % self.num_examples]
        self._validate_scene_dir(scene_dir)
        meta = _load_json(scene_dir / "meta.json")

        sample = {
            "video": self._load_npy(scene_dir, "rgb.npy"),
            "segmentations": self._load_npy(scene_dir, "segmentation.npy"),
            "depth": self._load_npy(scene_dir, "depth_uint16.npy"),
            "normal": self._load_npy(scene_dir, "normal_uint16.npy"),
            "object_coordinates": self._load_npy(scene_dir, "object_coordinates_uint16.npy"),
            "camera": {
                "positions": self._load_npy(scene_dir, "camera_positions.npy"),
                "quaternions": self._load_npy(scene_dir, "camera_quaternions.npy"),
                "field_of_view": _as_np_scalar(meta.get("camera_field_of_view", np.nan), np.float32),
            },
            "instances": {
                "bboxes_3d": self._load_npy(scene_dir, "instances_bboxes_3d.npy"),
                "positions": self._load_npy(scene_dir, "instances_positions.npy"),
                "quaternions": self._load_npy(scene_dir, "instances_quaternions.npy"),
            },
            "metadata": {
                "depth_range": np.asarray(meta.get("depth_range", [np.nan, np.nan]), dtype=np.float32),
                "num_frames": _as_np_scalar(meta.get("num_frames", 0), np.int32),
                "num_instances": _as_np_scalar(meta.get("num_instances", 0), np.int32),
                "video_name": str(meta.get("video_name", meta.get("scene_id", scene_dir.name))),
                "height": _as_np_scalar(meta.get("height", 0), np.int32),
                "width": _as_np_scalar(meta.get("width", 0), np.int32),
            },
        }
        return sample, scene_dir

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))
        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene_dir = self.scene_dirs[query_index % self.num_examples]
            sample_key = ""
            sample_paths = [str(scene_dir)]
            try:
                sample, scene_dir = self._sample_by_index(query_index)
                sample_paths = [str(scene_dir)]
                self._validate_runtime_sample(sample)
                idxs = self._select_clip(sample=sample, index=query_index)
                clip_start = int(idxs[0]) if idxs else 0
                video_name = str(sample["metadata"]["video_name"])
                sample_key = f"kubric_full_robust_preprocess::{video_name}::frames={','.join(str(int(v)) for v in idxs)}"
                if self.bad_registry.is_bad_sample(sample_key) or self.bad_registry.has_any_bad_path(sample_paths):
                    continue
                out = self._build_sample(sample=sample, idxs=idxs, clip_start=clip_start)
                out["meta"]["dataset"] = "kubric_full_robust_preprocess"
                out["meta"]["source_mode"] = "kubric_movi_full_preprocessed_objectcoord"
                out["meta"]["sample_key"] = sample_key
                return out
            except Exception as exc:
                failed_paths = self._failed_paths_from_exception(exc, default=sample_paths)
                if not is_retryable_data_error(exc):
                    exc = RetryableSampleError(
                        f"Converted non-retryable processed sample failure: {type(exc).__name__}: {exc}",
                        failed_paths=failed_paths,
                    )
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="kubric_full_robust_preprocess",
                    sample_key=sample_key or f"kubric_full_robust_preprocess::index={query_index}",
                    sample_paths=_dedup_str_list(sample_paths + failed_paths),
                    failed_paths=failed_paths,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._warn_skip(
                    sample_key=sample_key or f"kubric_full_robust_preprocess::index={query_index}",
                    reason=f"{type(exc).__name__}: {exc}",
                    failed_paths=failed_paths,
                )
                continue

        raise RuntimeError(
            "KubricFullRobustPreprocessDataset failed to produce a valid sample after "
            f"{self.max_sample_retries} retries. last_error={last_error}"
        )
