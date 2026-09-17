"""ScanNet raw dataset adapter.

Preferred mode is `iphone_rgbd`:
- RGB from `iphone/rgb.mkv`
- Depth from `iphone/depth.bin`
- Pose/intrinsics from `iphone/pose_intrinsic_imu.json`

Fallback mode is `dslr_colmap`:
- RGB + camera + sparse tracks from DSLR COLMAP text files.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bad_sample_registry import (
    BadSampleRegistry,
    RetryableSampleError,
    failed_paths_from_exception,
    is_retryable_data_error,
)
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    sample_frame_indices_with_stride,
    sample_hard_query_flags,
    sample_t_tgt_t_cam,
)
from .seeding import SeededDatasetMixin
from .depth_query_builder import build_queries_from_depth

try:
    import lz4.block as lz4_block  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    lz4_block = None


def _read_image(path: Path, width: int, height: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_video_frames(path: Path, frame_ids: list[int], width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RetryableSampleError(f"Failed to open video: {path}", failed_paths=[str(path)])
    frames: list[np.ndarray] = []
    try:
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fid))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RetryableSampleError(f"Failed to read frame {fid} from video {path}", failed_paths=[str(path)])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != width or rgb.shape[0] != height:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb.astype(np.uint8))
    finally:
        cap.release()
    return np.stack(frames, axis=0)


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float32)
    q /= n
    qw, qx, qy, qz = q.tolist()
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _world_to_cam(x_world: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    x_h = np.concatenate([x_world.astype(np.float32), np.array([1.0], dtype=np.float32)], axis=0)
    x_cam = t_cw @ x_h
    return x_cam[:3]


def _parse_colmap_camera(cameras_path: Path) -> tuple[np.ndarray, int, int] | None:
    if not cameras_path.exists():
        return None
    for line in cameras_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        fx = float(params[0]) if len(params) > 0 else 1.0
        fy = float(params[1]) if len(params) > 1 else fx
        cx = float(params[2]) if len(params) > 2 else width * 0.5
        cy = float(params[3]) if len(params) > 3 else height * 0.5
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return k, width, height
    return None


def _parse_colmap_images(images_path: Path, image_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray, dict[int, np.ndarray], np.ndarray]]:
    if not images_path.exists():
        return []
    lines = [line.strip() for line in images_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    out: list[tuple[str, np.ndarray, np.ndarray, dict[int, np.ndarray], np.ndarray]] = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        header = lines[i].split()
        if len(header) < 10:
            continue
        name = header[9]
        image_path = image_dir / name
        if not image_path.exists():
            continue

        qw, qx, qy, qz = [float(v) for v in header[1:5]]
        tx, ty, tz = [float(v) for v in header[5:8]]
        rot = _quat_to_rot(qw, qx, qy, qz)
        t_cw = np.eye(4, dtype=np.float32)
        t_cw[:3, :3] = rot
        t_cw[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        t_wc = np.linalg.inv(t_cw).astype(np.float32)

        p_tokens = lines[i + 1].split()
        if len(p_tokens) % 3 != 0:
            continue
        arr = np.asarray(p_tokens, dtype=np.float64).reshape(-1, 3)
        uv_by_pid: dict[int, np.ndarray] = {}
        for x, y, pid_raw in arr:
            pid = int(pid_raw)
            if pid < 0:
                continue
            if pid not in uv_by_pid:
                uv_by_pid[pid] = np.array([float(x), float(y)], dtype=np.float32)
        pid_ids = np.array(list(uv_by_pid.keys()), dtype=np.int64)
        out.append((name, t_cw, t_wc, uv_by_pid, pid_ids))
    return out


def _parse_colmap_points(points_path: Path, needed_ids: set[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not points_path.exists() or not needed_ids:
        return out
    for line in points_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        pid = int(parts[0])
        if pid not in needed_ids:
            continue
        out[pid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
    return out


def _parse_iphone_pose_intrinsic(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[tuple[int, np.ndarray, np.ndarray]] = []
    for key, value in raw.items():
        if not key.startswith("frame_"):
            continue
        try:
            fid = int(key.split("_")[-1])
        except ValueError:
            continue
        pose = np.asarray(value.get("aligned_pose", value.get("pose", [])), dtype=np.float32)
        intrinsic = np.asarray(value.get("intrinsic", []), dtype=np.float32)
        if pose.shape != (4, 4) or intrinsic.shape != (3, 3):
            continue
        items.append((fid, pose, intrinsic))
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    frame_ids = np.array([x[0] for x in items], dtype=np.int64)
    t_wc = np.stack([x[1] for x in items], axis=0).astype(np.float32)
    k = np.stack([x[2] for x in items], axis=0).astype(np.float32)
    return frame_ids, t_wc, k


def _build_depth_chunk_index(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return []
    index: list[tuple[int, int]] = []
    with path.open("rb") as handle:
        while True:
            head = handle.read(4)
            if not head:
                break
            if len(head) < 4:
                break
            length = struct.unpack("<I", head)[0]
            offset = int(handle.tell())
            index.append((offset, int(length)))
            handle.seek(length, 1)
    return index


def _decode_depth_chunk(payload: bytes, out_h: int = 192, out_w: int = 256) -> np.ndarray:
    """Decode one ScanNet++ iPhone depth chunk to depth in meters.

    In practice, scenes can mix at least two encodings:
    - lz4.block compressed uint16 depth in millimeters
    - zlib raw-deflate compressed float32 depth in meters
    """

    out_u16_bytes = out_h * out_w * 2
    out_f32_bytes = out_h * out_w * 4

    if lz4_block is not None:
        try:
            raw = lz4_block.decompress(payload, uncompressed_size=out_u16_bytes)
            if len(raw) == out_u16_bytes:
                depth_mm = np.frombuffer(raw, dtype="<u2").reshape(out_h, out_w)
                return depth_mm.astype(np.float32) / 1000.0
        except Exception:
            pass

    for wbits in (-15, zlib.MAX_WBITS):
        try:
            raw = zlib.decompress(payload, wbits)
        except Exception:
            continue
        if len(raw) == out_f32_bytes:
            return np.frombuffer(raw, dtype="<f4").reshape(out_h, out_w).astype(np.float32, copy=False)
        if len(raw) == out_u16_bytes:
            depth_mm = np.frombuffer(raw, dtype="<u2").reshape(out_h, out_w)
            return depth_mm.astype(np.float32) / 1000.0

    raise RuntimeError(
        f"depth chunk decompression failed: payload={len(payload)} bytes, expected decoded={out_u16_bytes} or {out_f32_bytes}"
    )


def _resize_depth(depth_m: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(depth_m.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32, copy=False)


@dataclass
class ScannetRawConfig:
    root: Path
    split_file: Path
    clip_frames: int
    image_size: tuple[int, int]  # (H, W)
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    training: bool
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    max_scenes: int | None = None
    source: str = "auto"  # auto | iphone_rgbd | dslr_colmap
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _DSLRFrameObs:
    name: str
    t_cw: np.ndarray
    t_wc: np.ndarray
    uv_by_pid: dict[int, np.ndarray]
    pid_ids: np.ndarray


@dataclass
class _DSLRScene:
    scene_id: str
    image_dir: Path
    frames: list[_DSLRFrameObs]
    points_xyz: dict[int, np.ndarray]
    intrinsics: np.ndarray
    src_w: int
    src_h: int


@dataclass
class _IphoneScene:
    scene_id: str
    rgb_path: Path
    depth_path: Path
    frame_ids: np.ndarray
    t_wc_seq: np.ndarray
    k_seq: np.ndarray
    depth_index: list[tuple[int, int]]
    src_w: int
    src_h: int


def _load_dslr_scene(scene_root: Path) -> _DSLRScene | None:
    dslr_root = scene_root / "dslr"
    image_dir = dslr_root / "resized_images"
    if not image_dir.exists():
        image_dir = dslr_root / "resized_undistorted_images"
    if not image_dir.exists():
        return None

    colmap_root = dslr_root / "colmap"
    camera_info = _parse_colmap_camera(colmap_root / "cameras.txt")
    if camera_info is None:
        return None
    intrinsics, src_w, src_h = camera_info

    image_entries = _parse_colmap_images(colmap_root / "images.txt", image_dir=image_dir)
    if not image_entries:
        return None

    frames: list[_DSLRFrameObs] = []
    needed_ids: set[int] = set()
    for name, t_cw, t_wc, uv_by_pid, pid_ids in image_entries:
        if pid_ids.size == 0:
            continue
        needed_ids.update(int(pid) for pid in pid_ids.tolist())
        frames.append(_DSLRFrameObs(name=name, t_cw=t_cw, t_wc=t_wc, uv_by_pid=uv_by_pid, pid_ids=pid_ids))
    if not frames:
        return None

    points_xyz = _parse_colmap_points(colmap_root / "points3D.txt", needed_ids=needed_ids)
    if not points_xyz:
        return None

    for frame in frames:
        valid_ids = [pid for pid in frame.pid_ids.tolist() if pid in points_xyz]
        frame.pid_ids = np.array(valid_ids, dtype=np.int64)
        frame.uv_by_pid = {pid: frame.uv_by_pid[pid] for pid in valid_ids}
    frames = [frame for frame in frames if frame.pid_ids.size > 0]
    if not frames:
        return None

    return _DSLRScene(
        scene_id=scene_root.name,
        image_dir=image_dir,
        frames=frames,
        points_xyz=points_xyz,
        intrinsics=intrinsics,
        src_w=src_w,
        src_h=src_h,
    )


def _load_iphone_scene(scene_root: Path) -> _IphoneScene | None:
    iphone_root = scene_root / "iphone"
    rgb_path = iphone_root / "rgb.mkv"
    depth_path = iphone_root / "depth.bin"
    pose_path = iphone_root / "pose_intrinsic_imu.json"
    if not (rgb_path.exists() and depth_path.exists() and pose_path.exists()):
        return None

    pose_pack = _parse_iphone_pose_intrinsic(pose_path)
    if pose_pack is None:
        return None
    frame_ids, t_wc_seq, k_seq = pose_pack

    cap = cv2.VideoCapture(str(rgb_path))
    if not cap.isOpened():
        return None
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_w <= 0 or src_h <= 0:
        src_w, src_h = 1920, 1440

    depth_index = _build_depth_chunk_index(depth_path)
    if not depth_index:
        return None

    n = min(video_frames, len(depth_index), frame_ids.shape[0], t_wc_seq.shape[0], k_seq.shape[0])
    if n <= 0:
        return None
    frame_ids = frame_ids[:n]
    t_wc_seq = t_wc_seq[:n]
    k_seq = k_seq[:n]
    depth_index = depth_index[:n]

    return _IphoneScene(
        scene_id=scene_root.name,
        rgb_path=rgb_path,
        depth_path=depth_path,
        frame_ids=frame_ids,
        t_wc_seq=t_wc_seq,
        k_seq=k_seq,
        depth_index=depth_index,
        src_w=src_w,
        src_h=src_h,
    )


class ScannetRawDataset(SeededDatasetMixin, Dataset):
    """Loads ScanNet scenes with dense iPhone RGBD supervision when available."""

    def __init__(self, config: ScannetRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="scannet_raw", default_seed=20260320)
        self.augment = config.augment or RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        if not config.training:
            self.augment = RawAugmentConfig()

        if not config.root.exists():
            raise FileNotFoundError(f"ScanNet root not found: {config.root}")
        if not config.split_file.exists():
            raise FileNotFoundError(f"ScanNet split file not found: {config.split_file}")

        self.scene_ids: list[str] = []
        for sid in [line.strip() for line in config.split_file.read_text(encoding="utf-8").splitlines() if line.strip()]:
            if (config.root / sid).exists():
                self.scene_ids.append(sid)
            if config.max_scenes is not None and len(self.scene_ids) >= int(config.max_scenes):
                break
        if not self.scene_ids:
            raise ValueError(f"No scenes from split {config.split_file}")

        self.scene_roots = {sid: config.root / sid for sid in self.scene_ids}
        self.scene_cache: dict[str, _IphoneScene | _DSLRScene] = {}

    def __len__(self) -> int:
        base = len(self.scene_ids) * 20 if self.cfg.training else len(self.scene_ids)
        return max(base, len(self.scene_ids))

    def _load_scene_cached(self, sid: str) -> _IphoneScene | _DSLRScene:
        cached = self.scene_cache.get(sid)
        if cached is not None:
            return cached

        root = self.scene_roots[sid]
        source = str(self.cfg.source).strip().lower()

        loaded: _IphoneScene | _DSLRScene | None = None
        if source in {"auto", "iphone_rgbd", "iphone"}:
            loaded = _load_iphone_scene(root)
            if isinstance(loaded, _IphoneScene) and loaded.frame_ids.shape[0] < self.cfg.clip_frames:
                loaded = None

        if loaded is None and source in {"auto", "dslr_colmap", "dslr"}:
            loaded = _load_dslr_scene(root)
            if isinstance(loaded, _DSLRScene) and len(loaded.frames) < self.cfg.clip_frames:
                loaded = None

        if loaded is None:
            raise ValueError(f"Failed to load scene {sid} with source={self.cfg.source}")

        self.scene_cache[sid] = loaded
        return loaded

    def _scene(self, index: int) -> _IphoneScene | _DSLRScene:
        if self.cfg.training:
            for _ in range(max(4, len(self.scene_ids))):
                sid = self.scene_ids[int(self.rng.integers(0, len(self.scene_ids)))]
                try:
                    return self._load_scene_cached(sid)
                except Exception:
                    continue
            raise ValueError("Unable to sample a valid ScanNet scene")
        sid = self.scene_ids[index % len(self.scene_ids)]
        return self._load_scene_cached(sid)

    def _start(self, scene: _IphoneScene | _DSLRScene, index: int) -> int:
        scene_len = int(scene.frame_ids.shape[0]) if isinstance(scene, _IphoneScene) else len(scene.frames)
        max_start = scene_len - self.cfg.clip_frames
        if max_start <= 0:
            return 0
        if self.cfg.training:
            return int(self.rng.integers(0, max_start + 1))
        return int((index * self.cfg.clip_frames) % (max_start + 1))

    def _frame_indices(self, scene_len: int, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng, scene_len=scene_len, clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment, training=bool(self.cfg.training), index=index,
        )

    def _getitem_iphone(self, scene: _IphoneScene, start: int, idxs: list[int] | None = None) -> dict[str, Any]:
        if idxs is None:
            idxs = self._frame_indices(scene.frame_ids.shape[0], start)
        frame_ids = [int(scene.frame_ids[i]) for i in idxs]

        rgb = _read_video_frames(scene.rgb_path, frame_ids=frame_ids, width=self.w, height=self.h)
        depth_m_list: list[np.ndarray] = []
        with scene.depth_path.open("rb") as handle:
            for i in idxs:
                offset, length = scene.depth_index[i]
                handle.seek(offset)
                payload = handle.read(length)
                try:
                    d_raw_m = _decode_depth_chunk(payload)  # [192, 256] float32 meters
                except Exception as exc:
                    raise RetryableSampleError(
                        f"Failed to decode depth chunk: scene={scene.scene_id} frame_index={i} path={scene.depth_path}: {exc}",
                        failed_paths=[str(scene.depth_path)],
                    ) from exc
                d_resized = _resize_depth(d_raw_m, width=self.w, height=self.h)
                depth_m_list.append(d_resized)

        depth = np.stack(depth_m_list, axis=0).astype(np.float32, copy=False)
        depth_valid = np.isfinite(depth) & (depth > 0.0)

        k_src = scene.k_seq[idxs].copy()
        t_wc_seq = scene.t_wc_seq[idxs].copy()
        sx = self.w / float(scene.src_w)
        sy = self.h / float(scene.src_h)
        for i in range(k_src.shape[0]):
            k_src[i, 0, 0] *= sx
            k_src[i, 0, 2] *= sx
            k_src[i, 1, 1] *= sy
            k_src[i, 1, 2] *= sy

        video = np.transpose(rgb.astype(np.float32) / 255.0, (0, 3, 1, 2))
        aspect_ratio = np.array([scene.src_w / max(1.0, scene.src_h)], dtype=np.float32)
        _crop_info = {}
        if self.cfg.training:
            video = apply_photometric_augment(video, self.rng, self.augment)
            (video, depth, depth_valid, k_src, aspect_ratio) = apply_spatial_crop_images_only(
                video_t_chw=video,
                depth_t_hw=depth,
                depth_valid_t_hw=depth_valid,
                k_t_33=k_src,
                camera_valid_t=np.ones((len(idxs),), dtype=np.bool_),
                rng=self.rng,
                cfg=self.augment,
                native_aspect_ratio=aspect_ratio,
                out_info=_crop_info,
            )

        query, target, mask, query_stats = build_queries_from_depth(
            rng=self.rng,
            depth=depth,
            depth_valid=depth_valid,
            k_seq=k_src,
            t_wc_seq=t_wc_seq,
            camera_valid=np.ones((len(idxs),), dtype=np.bool_),
            queries_per_clip=int(self.cfg.queries_per_clip),
            hard_query_ratio=float(self.cfg.hard_query_ratio),
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
        )

        return {
            "video": torch.from_numpy(video).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio.astype(np.float32)),
            "depth_m": torch.from_numpy(depth).float(),
            "depth_valid": torch.from_numpy(depth_valid).bool(),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()},
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_src).float(),
                "T_wc": torch.from_numpy(t_wc_seq).float(),
                "camera_valid": torch.ones((len(idxs),), dtype=torch.bool),
            },
            "augment_info": {k: torch.from_numpy(v) for k, v in build_augment_info(_crop_info, image_hw=(self.h, self.w)).items()},
            "meta": {
                "dataset": "scannet_raw",
                "scene_id": scene.scene_id,
                "clip_start": start,
                "source_mode": "iphone_rgbd",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }

    def _getitem_dslr(self, scene: _DSLRScene, start: int, idxs: list[int] | None = None) -> dict[str, Any]:
        if idxs is None:
            idxs = self._frame_indices(len(scene.frames), start)
        clip_frames = [scene.frames[i] for i in idxs]
        t = len(clip_frames)
        m = self.cfg.queries_per_clip

        frames: list[np.ndarray] = []
        k_seq: list[np.ndarray] = []
        t_wc_seq: list[np.ndarray] = []
        for obs in clip_frames:
            frames.append(_read_image(scene.image_dir / obs.name, self.w, self.h))
            sx = self.w / float(scene.src_w)
            sy = self.h / float(scene.src_h)
            k = scene.intrinsics.copy()
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy
            k_seq.append(k)
            t_wc_seq.append(obs.t_wc)

        edge_masks: list[np.ndarray] = []
        if self.cfg.hard_query_ratio > 0.0:
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))
                thr = float(np.quantile(mag, 0.8)) if mag.size > 0 else np.inf
                edge_masks.append(mag >= thr)

        video = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video = np.transpose(video, (0, 3, 1, 2))
        depth = np.full((t, self.h, self.w), np.nan, dtype=np.float32)
        depth_valid = np.zeros((t, self.h, self.w), dtype=np.bool_)

        aspect_ratio = np.array([scene.src_w / max(1.0, scene.src_h)], dtype=np.float32)
        _crop_info: dict = {}
        k_arr = np.stack(k_seq, axis=0).astype(np.float32)
        t_wc_arr = np.stack(t_wc_seq, axis=0).astype(np.float32)
        cam_valid = np.ones((t,), dtype=np.bool_)

        if self.cfg.training:
            video = apply_photometric_augment(video, self.rng, self.augment)
            (video, depth, depth_valid, k_arr, aspect_ratio) = apply_spatial_crop_images_only(
                video_t_chw=video,
                depth_t_hw=depth,
                depth_valid_t_hw=depth_valid,
                k_t_33=k_arr,
                camera_valid_t=cam_valid,
                rng=self.rng,
                cfg=self.augment,
                native_aspect_ratio=aspect_ratio,
                out_info=_crop_info,
            )
            # Recompute edge masks on the cropped video for hard-query sampling.
            edge_masks = []
            if self.cfg.hard_query_ratio > 0.0:
                for fi in range(t):
                    img_chw = video[fi]  # (3, h, w) float32 [0,1]
                    gray = np.transpose(img_chw, (1, 2, 0))  # (h, w, 3)
                    gray = cv2.cvtColor((gray * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                    mag = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))
                    thr = float(np.quantile(mag, 0.8)) if mag.size > 0 else np.inf
                    edge_masks.append(mag >= thr)

        # Determine the UV transform from original image space to (possibly
        # cropped) output pixel space.  Without a crop the mapping is simply
        # src -> resized.  With a crop the mapping becomes:
        #   u_out = (u_resized - x0) * (w / crop_w)
        # where u_resized = uv_orig * (self.w / scene.src_w).
        crop_x0 = float(_crop_info.get("crop_xy", (0, 0))[0])
        crop_y0 = float(_crop_info.get("crop_xy", (0, 0))[1])
        crop_w_px = float(_crop_info.get("crop_hw", (self.h, self.w))[1])
        crop_h_px = float(_crop_info.get("crop_hw", (self.h, self.w))[0])
        # Composite scale: original-image-pixel -> output-pixel
        sx_composite = (self.w / max(float(scene.src_w), 1.0)) * (float(self.w) / max(crop_w_px, 1.0))
        sy_composite = (self.h / max(float(scene.src_h), 1.0)) * (float(self.h) / max(crop_h_px, 1.0))
        ox_composite = -crop_x0 * (float(self.w) / max(crop_w_px, 1.0))
        oy_composite = -crop_y0 * (float(self.h) / max(crop_h_px, 1.0))

        q_u = np.zeros((m,), dtype=np.float32)
        q_v = np.zeros((m,), dtype=np.float32)
        q_t_src = self.rng.integers(0, t, size=(m,), dtype=np.int64)
        q_t_tgt, q_t_cam, _ = sample_t_tgt_t_cam(
            rng=self.rng,
            queries_per_clip=m,
            clip_frames=t,
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            q_t_src=q_t_src,
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
        )

        y_uv = np.zeros((m, 2), dtype=np.float32)
        y_xyz = np.zeros((m, 3), dtype=np.float32)
        y_disp = np.zeros((m, 3), dtype=np.float32)
        y_normal = np.zeros((m, 3), dtype=np.float32)
        y_vis = np.zeros((m,), dtype=np.float32)

        m_uv = np.zeros((m,), dtype=np.bool_)
        m_xyz = np.zeros((m,), dtype=np.bool_)
        m_disp = np.zeros((m,), dtype=np.bool_)
        m_vis = np.zeros((m,), dtype=np.bool_)
        m_normal = np.zeros((m,), dtype=np.bool_)
        is_hard_query = np.zeros((m,), dtype=np.bool_)

        hard_target = int(sample_hard_query_flags(self.rng, m, float(self.cfg.hard_query_ratio)).sum())
        use_hard = np.zeros((m,), dtype=np.bool_)
        hard_candidates_cache: dict[int, np.ndarray] = {}

        if hard_target > 0:
            order = self.rng.permutation(m)
            picked = 0
            for raw_idx in order.tolist():
                i = int(raw_idx)
                if picked >= hard_target:
                    break
                fs = int(q_t_src[i])
                src_obs = clip_frames[fs]
                if src_obs.pid_ids.size == 0:
                    continue
                candidates = src_obs.pid_ids.astype(np.int64)
                hard_cands: list[int] = []
                for pid_v in candidates.tolist():
                    uv_src = src_obs.uv_by_pid.get(int(pid_v))
                    if uv_src is None:
                        continue
                    if not np.isfinite(uv_src).all():
                        continue
                    u_px = int(np.clip(np.rint(float(uv_src[0]) * sx_composite + ox_composite), 0, max(self.w - 1, 0)))
                    v_px = int(np.clip(np.rint(float(uv_src[1]) * sy_composite + oy_composite), 0, max(self.h - 1, 0)))
                    if fs < len(edge_masks) and edge_masks[fs][v_px, u_px]:
                        hard_cands.append(int(pid_v))
                if hard_cands:
                    use_hard[i] = True
                    hard_candidates_cache[i] = np.asarray(hard_cands, dtype=np.int64)
                    picked += 1

        # Normalize output-pixel coordinates to [0, 1].
        w_norm = max(1.0, float(self.w - 1))
        h_norm = max(1.0, float(self.h - 1))
        for i in range(m):
            fs = int(q_t_src[i])
            ft = int(q_t_tgt[i])
            fc = int(q_t_cam[i])
            src_obs = clip_frames[fs]
            tgt_obs = clip_frames[ft]
            cam_obs = clip_frames[fc]

            if src_obs.pid_ids.size == 0:
                continue
            candidates = src_obs.pid_ids.astype(np.int64)
            picked_hard = bool(use_hard[i])
            if picked_hard:
                cached = hard_candidates_cache.get(i)
                if cached is not None and cached.size > 0:
                    candidates = cached
                else:
                    picked_hard = False
            pid = int(candidates[int(self.rng.integers(0, candidates.size))])
            is_hard_query[i] = picked_hard
            src_uv = src_obs.uv_by_pid.get(pid)
            if src_uv is None:
                continue
            if np.isfinite(src_uv).all():
                u_out = float(src_uv[0]) * sx_composite + ox_composite
                v_out = float(src_uv[1]) * sy_composite + oy_composite
                if u_out < 0.0 or u_out > self.w - 1 or v_out < 0.0 or v_out > self.h - 1:
                    continue
                q_u[i] = np.clip(u_out / w_norm, 0.0, 1.0)
                q_v[i] = np.clip(v_out / h_norm, 0.0, 1.0)
            else:
                q_u[i] = 0.5
                q_v[i] = 0.5
                continue
            m_vis[i] = True

            tgt_uv = tgt_obs.uv_by_pid.get(pid)
            if tgt_uv is None:
                continue
            if np.isfinite(tgt_uv).all():
                u_tgt_out = float(tgt_uv[0]) * sx_composite + ox_composite
                v_tgt_out = float(tgt_uv[1]) * sy_composite + oy_composite
                if u_tgt_out < 0.0 or u_tgt_out > self.w - 1 or v_tgt_out < 0.0 or v_tgt_out > self.h - 1:
                    continue
                y_uv[i, 0] = np.clip(u_tgt_out / w_norm, 0.0, 1.0)
                y_uv[i, 1] = np.clip(v_tgt_out / h_norm, 0.0, 1.0)
            else:
                y_uv[i, 0] = 0.5
                y_uv[i, 1] = 0.5
                continue
            y_vis[i] = 1.0
            m_uv[i] = True

            xyz_world = scene.points_xyz.get(pid)
            if xyz_world is None:
                continue
            xyz_cam = _world_to_cam(xyz_world, cam_obs.t_cw)
            if not np.isfinite(xyz_cam).all() or float(xyz_cam[2]) <= 0.0:
                continue
            y_xyz[i] = xyz_cam
            y_disp[i] = 0.0
            m_xyz[i] = True
            m_disp[i] = True

        query = {
            "u": q_u,
            "v": q_v,
            "t_src": q_t_src,
            "t_tgt": q_t_tgt,
            "t_cam": q_t_cam,
        }
        query_stats = {
            "is_hard_query": is_hard_query,
        }
        target = {
            "xyz_3d": y_xyz,
            "uv_2d": y_uv,
            "visibility": y_vis,
            "displacement": y_disp,
            "normal": y_normal,
        }
        mask = {
            "xyz_3d": m_xyz,
            "uv_2d": m_uv,
            "visibility": m_vis,
            "displacement": m_disp,
            "normal": m_normal,
        }

        return {
            "video": torch.from_numpy(video).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio.astype(np.float32)),
            "depth_m": torch.from_numpy(depth).float(),
            "depth_valid": torch.from_numpy(depth_valid).bool(),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()},
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_arr).float(),
                "T_wc": torch.from_numpy(t_wc_arr).float(),
                "camera_valid": torch.from_numpy(cam_valid).bool(),
            },
            "augment_info": {k: torch.from_numpy(v) for k, v in build_augment_info(_crop_info, image_hw=(self.h, self.w)).items()},
            "meta": {
                "dataset": "scannet_raw",
                "scene_id": scene.scene_id,
                "clip_start": start,
                "source_mode": "dslr_colmap",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }

    def _sample_key_iphone(self, scene: _IphoneScene, idxs: list[int]) -> str:
        frame_ids = ",".join(str(int(scene.frame_ids[i])) for i in idxs)
        return f"scannet_raw::iphone_rgbd::{scene.scene_id}::frames={frame_ids}"

    def _sample_paths_iphone(self, scene: _IphoneScene) -> list[str]:
        pose_path = scene.rgb_path.parent / "pose_intrinsic_imu.json"
        return [str(scene.rgb_path), str(scene.depth_path), str(pose_path)]

    def _sample_key_dslr(self, scene: _DSLRScene, idxs: list[int]) -> str:
        names = ",".join(scene.frames[i].name for i in idxs)
        return f"scannet_raw::dslr_colmap::{scene.scene_id}::frames={names}"

    def _sample_paths_dslr(self, scene: _DSLRScene, idxs: list[int]) -> list[str]:
        out: list[str] = []
        for i in idxs:
            out.append(str(scene.image_dir / scene.frames[i].name))
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            start = self._start(scene, query_index)

            if isinstance(scene, _IphoneScene):
                idxs = self._frame_indices(scene.frame_ids.shape[0], start)
                sample_key = self._sample_key_iphone(scene, idxs)
                sample_paths = self._sample_paths_iphone(scene)
                if self.bad_registry.is_bad_sample(sample_key):
                    continue
                if self.bad_registry.has_any_bad_path(sample_paths):
                    continue
                try:
                    sample = self._getitem_iphone(scene, start, idxs=idxs)
                except Exception as exc:
                    if not is_retryable_data_error(exc):
                        raise
                    last_error = exc
                    self.bad_registry.mark_bad(
                        dataset="scannet_raw",
                        sample_key=sample_key,
                        sample_paths=sample_paths,
                        failed_paths=failed_paths_from_exception(exc),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue
                sample["meta"]["sample_key"] = sample_key
                return sample

            idxs = self._frame_indices(len(scene.frames), start)
            sample_key = self._sample_key_dslr(scene, idxs)
            sample_paths = self._sample_paths_dslr(scene, idxs)
            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue
            try:
                sample = self._getitem_dslr(scene, start, idxs=idxs)
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="scannet_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"ScannetRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
