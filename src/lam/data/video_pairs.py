from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import Dataset

cv2.setNumThreads(int(os.environ.get("LAM_CV2_NUM_THREADS", "0")))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VIDEO_INDEX_CACHE_VERSION = 2


def _iter_video_files(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def _has_video_files(root: Path, extensions: set[str]) -> bool:
    if not root.exists():
        return False
    for dirpath, _, filenames in os.walk(root):
        if any(Path(filename).suffix.lower() in extensions for filename in filenames):
            return True
    return False


def _default_video_index_cache_path(root: Path) -> Path:
    cache_root = Path(os.environ.get("LAM_VIDEO_INDEX_CACHE_DIR", root / ".lam_cache"))
    root_key = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"video_index_v{VIDEO_INDEX_CACHE_VERSION}_{root.name}_{root_key}.json"


def _load_video_index_cache(cache_path: Path, root: Path, extensions: set[str]) -> list[tuple[Path, int]] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("version") != VIDEO_INDEX_CACHE_VERSION:
        return None
    if payload.get("root") != str(root.resolve()):
        return None
    if sorted(payload.get("extensions", [])) != sorted(extensions):
        return None

    videos: list[tuple[Path, int]] = []
    for item in payload.get("videos", []):
        try:
            path = root / item["path"]
            frame_count = int(item["frame_count"])
        except (KeyError, TypeError, ValueError):
            return None
        if frame_count > 1:
            videos.append((path, frame_count))
    return videos


def _write_video_index_cache(cache_path: Path, root: Path, extensions: set[str], videos: list[tuple[Path, int]]) -> None:
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    payload = {
        "version": VIDEO_INDEX_CACHE_VERSION,
        "root": str(root.resolve()),
        "extensions": sorted(extensions),
        "videos": [
            {"path": str(path.relative_to(root)), "frame_count": int(frame_count)}
            for path, frame_count in videos
        ],
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def root_has_video_files(root: str | Path, extensions: set[str] | None = None) -> bool:
    root = Path(root)
    suffixes = extensions or VIDEO_EXTENSIONS
    return _has_video_files(root, suffixes)


def larybench_uniform_indices(total_frames: int, num_frames: int = 9) -> list[int]:
    if total_frames <= 0:
        return []
    interval = total_frames / num_frames
    return [min(total_frames - 1, int(interval * idx)) for idx in range(num_frames)]


def larybench_motion_guided_indices(diff_scores: Sequence[float], num_frames: int = 9) -> list[int]:
    scores = np.asarray(diff_scores, dtype=np.float64)
    if scores.size == 0:
        return []

    scores = np.power(np.maximum(scores, 0.0), 0.5)
    total = float(np.sum(scores))
    if not np.isfinite(total) or total <= 0.0:
        return []

    cumulative = np.cumsum(scores / total)
    indices: list[int] = []
    for idx in range(num_frames):
        target = 1 / (num_frames * 2) + idx / num_frames
        nearest = int(np.abs(cumulative - target).argmin() + 1)
        indices.append(nearest)
    return indices


def larybench_sample_indices(total_frames: int, diff_scores: Sequence[float] | None, num_frames: int = 9) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if total_frames <= 0:
        return []

    if diff_scores is None or total_frames <= num_frames * 2:
        current_indices = larybench_uniform_indices(total_frames, num_frames)
    else:
        current_indices = larybench_motion_guided_indices(diff_scores, num_frames)
        if not current_indices:
            current_indices = larybench_uniform_indices(total_frames, num_frames)

    processed = sorted({max(0, min(total_frames - 1, int(idx))) for idx in current_indices})
    if not processed or processed == [0]:
        processed = larybench_uniform_indices(total_frames, num_frames)
    if not processed:
        return []

    while len(processed) < num_frames:
        processed.append(processed[-1])
    return processed[:num_frames]


def larybench_sample_pairs(total_frames: int, diff_scores: Sequence[float] | None, num_frames: int = 9) -> list[tuple[int, int]]:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if total_frames <= 1:
        return []

    sample_indices = larybench_sample_indices(total_frames, diff_scores, num_frames)
    pairs = [(int(src), int(tgt)) for src, tgt in zip(sample_indices, sample_indices[1:]) if int(tgt) > int(src)]
    if not pairs:
        pairs = [(idx, idx + 1) for idx in range(total_frames - 1)]

    target_count = num_frames - 1
    valid_pairs = list(pairs)
    while len(pairs) < target_count:
        pairs.append(valid_pairs[len(pairs) % len(valid_pairs)])
    return pairs[:target_count]


def uniform_context_indices(src_index: int, tgt_index: int, num_frames: int) -> list[int]:
    """Return fixed-length, endpoint-preserving indices between two frames.

    Short pairs may contain repeated indices. This keeps the tensor shape
    stable while ensuring the first and last context frames are the exact
    source and target used by the GLD branch.
    """
    src_index = int(src_index)
    tgt_index = int(tgt_index)
    num_frames = int(num_frames)
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if tgt_index < src_index:
        raise ValueError("tgt_index must be greater than or equal to src_index")
    span = tgt_index - src_index
    denominator = num_frames - 1
    return [src_index + (span * position + denominator // 2) // denominator for position in range(num_frames)]


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _video_diff_scores(path: Path, num_frames_limit: int) -> list[float] | None:
    total_frames = _video_frame_count(path)
    if 0 < total_frames <= num_frames_limit:
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    diffs: list[float] = []
    prev_gray: np.ndarray | None = None
    read_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            read_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (224, 224))
            if prev_gray is not None:
                score = structural_similarity(prev_gray, gray_small)
                diffs.append(float(1.0 - score))
            prev_gray = gray_small
    finally:
        cap.release()

    if read_frames <= num_frames_limit:
        return None
    return diffs


def _load_video_frame(path: Path, frame_idx: int, image_size: int) -> torch.Tensor:
    frames = _load_video_frames(path, [frame_idx], image_size)
    return frames[0]


def _format_video_read_error(path: Path, frame_idx: int, frame_count: int) -> str:
    return (
        f"Could not read frame {int(frame_idx)} from video: {path} "
        f"(reported_frame_count={frame_count}). "
        "OpenCV/FFmpeg may have logged the decoder error to stderr above."
    )


def _load_video_frames(path: Path, frame_indices: Sequence[int], image_size: int) -> list[torch.Tensor]:
    if not frame_indices:
        return []

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(
                f"Could not open video with cv2.VideoCapture: {path}. "
                "OpenCV/FFmpeg may have failed to initialize the codec; see stderr for VIDEOIO/FFMPEG details."
            )

        frames: list[torch.Tensor] = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                raise ValueError(_format_video_read_error(path, int(frame_idx), frame_count))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            array = np.asarray(frame, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
        return frames
    finally:
        cap.release()


def _is_video_frame_read_error(exc: Exception) -> bool:
    return "Could not read frame" in str(exc) or "Could not open video with cv2.VideoCapture" in str(exc)


class VideoFolderFrameDataset(Dataset):
    """Samples frame pairs from `root/video_id/frame.png` folders."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        strides: list[int] | tuple[int, ...] = (5, 10, 15, 20),
        split: str = "train",
        eval_stride: int | None = None,
        d4rt_context_frames: int = 0,
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.strides = [int(item) for item in strides]
        self.split = split
        self.eval_stride = int(eval_stride) if eval_stride is not None else self.strides[0]
        self.d4rt_context_frames = int(d4rt_context_frames)
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not self.strides:
            raise ValueError("At least one stride is required")
        if self.d4rt_context_frames == 1 or self.d4rt_context_frames < 0:
            raise ValueError("d4rt_context_frames must be zero (disabled) or at least 2")
        self._items = self._build_index()
        if not self._items:
            raise ValueError(f"No valid frame pairs found under {self.root}")

    def _video_frames(self) -> list[list[Path]]:
        videos: list[list[Path]] = []
        for video_dir in sorted(item for item in self.root.iterdir() if item.is_dir()):
            frames = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
            if frames:
                videos.append(frames)
        return videos

    def _build_index(self) -> list[tuple[list[Path], int]]:
        index: list[tuple[list[Path], int]] = []
        max_stride = max(self.strides if self.split == "train" else [self.eval_stride])
        for frames in self._video_frames():
            if len(frames) <= max_stride:
                continue
            for base_idx in range(0, len(frames) - max_stride):
                index.append((frames, base_idx))
        return index

    def __len__(self) -> int:
        return len(self._items)

    def _choose_stride(self, frames: list[Path], base_idx: int) -> int:
        if self.split != "train":
            return self.eval_stride
        valid = [stride for stride in self.strides if base_idx + stride < len(frames)]
        return random.choice(valid)

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        frames, base_idx = self._items[idx]
        stride = self._choose_stride(frames, base_idx)
        path_t = frames[base_idx]
        path_tk = frames[base_idx + stride]
        sample: dict[str, torch.Tensor | int | str] = {
            "stride": stride,
            "path_t": str(path_t),
            "path_tk": str(path_tk),
        }
        if self.d4rt_context_frames:
            context_indices = uniform_context_indices(base_idx, base_idx + stride, self.d4rt_context_frames)
            context = torch.stack([self._load_image(frames[frame_idx]) for frame_idx in context_indices], dim=0)
            sample.update(
                {
                    "image_t": context[0],
                    "image_tk": context[-1],
                    "d4rt_video": context,
                    "d4rt_t_src": 0,
                    "d4rt_t_tgt": self.d4rt_context_frames - 1,
                    "d4rt_t_cam": 0,
                }
            )
        else:
            sample.update(
                {
                    "image_t": self._load_image(path_t),
                    "image_tk": self._load_image(path_tk),
                }
            )
        return sample


class VideoFilePairDataset(Dataset):
    """Samples frame pairs from flat video files."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        num_sample_frames: int = 9,
        strides: list[int] | tuple[int, ...] = (5, 10, 15, 20),
        split: str = "train",
        video_sampling: str = "motion-guided",
        pairs_per_video: int | None = None,
        extensions: set[str] | None = None,
        index_cache_path: str | Path | None = None,
        use_index_cache: bool = True,
        max_resample_attempts: int = 16,
        d4rt_context_frames: int = 0,
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.num_sample_frames = int(num_sample_frames)
        self.strides = [int(item) for item in strides]
        self.split = split
        self.video_sampling = video_sampling
        self._pairs_per_video = int(pairs_per_video) if pairs_per_video is not None else self.num_sample_frames - 1
        self.extensions = extensions or VIDEO_EXTENSIONS
        self.use_index_cache = bool(use_index_cache)
        self.max_resample_attempts = max(0, int(max_resample_attempts))
        self.d4rt_context_frames = int(d4rt_context_frames)
        self.index_cache_path = Path(index_cache_path) if index_cache_path is not None else _default_video_index_cache_path(self.root)
        if self.num_sample_frames < 2:
            raise ValueError("num_sample_frames must be at least 2")
        if self.d4rt_context_frames == 1 or self.d4rt_context_frames < 0:
            raise ValueError("d4rt_context_frames must be zero (disabled) or at least 2")
        if not self.strides:
            raise ValueError("At least one stride is required")
        if any(stride <= 0 for stride in self.strides):
            raise ValueError("strides must be positive")
        if self.video_sampling not in {"motion-guided", "sliding-window"}:
            raise ValueError("video_sampling must be motion-guided or sliding-window")
        if self._pairs_per_video <= 0:
            raise ValueError("pairs_per_video must be positive")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self._videos = self._build_video_index()
        self._pair_cache: dict[int, list[tuple[int, int]]] = {}
        self._sample_cache: dict[int, list[int]] = {}
        self._bad_video_indices: set[int] = set()
        if not self._videos:
            raise ValueError(f"No readable video files found under {self.root}")

    def _build_video_index(self) -> list[tuple[Path, int]]:
        if self.use_index_cache:
            cached = _load_video_index_cache(self.index_cache_path, self.root, self.extensions)
            if cached is not None:
                return cached

        videos: list[tuple[Path, int]] = []
        for path in _iter_video_files(self.root, self.extensions):
            frame_count = _video_frame_count(path)
            if frame_count > 1:
                videos.append((path, frame_count))

        if self.use_index_cache:
            _write_video_index_cache(self.index_cache_path, self.root, self.extensions, videos)
        return videos

    @property
    def pairs_per_video(self) -> int:
        return self._pairs_per_video

    def __len__(self) -> int:
        return len(self._videos) * self.pairs_per_video

    def _sample_indices(self, video_idx: int) -> list[int]:
        path, total_frames = self._videos[video_idx]
        diffs = None
        if total_frames > self.num_sample_frames * 2:
            diffs = _video_diff_scores(path, self.num_sample_frames * 2)
        return larybench_sample_indices(total_frames, diffs, self.num_sample_frames)

    def _sample_pairs(self, video_idx: int) -> list[tuple[int, int]]:
        if video_idx in self._pair_cache:
            return self._pair_cache[video_idx]

        path, total_frames = self._videos[video_idx]
        diffs = None
        if total_frames > self.num_sample_frames * 2:
            diffs = _video_diff_scores(path, self.num_sample_frames * 2)
        pairs = larybench_sample_pairs(total_frames, diffs, self.num_sample_frames)
        if len(pairs) < self.pairs_per_video:
            valid_pairs = list(pairs)
            while len(pairs) < self.pairs_per_video:
                pairs.append(valid_pairs[len(pairs) % len(valid_pairs)])
        pairs = pairs[: self.pairs_per_video]
        sample_indices = [pairs[0][0], *[tgt for _, tgt in pairs]]
        self._pair_cache[video_idx] = pairs
        self._sample_cache[video_idx] = sample_indices
        return pairs

    def _sample_sliding_pair(self, total_frames: int) -> tuple[int, int]:
        valid_strides = [stride for stride in self.strides if total_frames > stride]
        if valid_strides:
            stride = random.choice(valid_strides)
            src_index = random.randint(0, total_frames - stride - 1)
            return src_index, src_index + stride

        src_index = random.randint(0, total_frames - 2)
        return src_index, src_index + 1

    def _make_sample(self, video_idx: int, pair_idx: int) -> dict[str, torch.Tensor | int | str | list[int]]:
        path, total_frames = self._videos[video_idx]
        if self.video_sampling == "sliding-window":
            src_index, tgt_index = self._sample_sliding_pair(total_frames)
            sample_pairs = [(src_index, tgt_index)]
            sample_indices = [src_index, tgt_index]
        else:
            sample_pairs = self._sample_pairs(video_idx)
            sample_indices = self._sample_cache[video_idx]
            src_index, tgt_index = sample_pairs[pair_idx]
        stride = tgt_index - src_index
        if self.d4rt_context_frames:
            context_indices = uniform_context_indices(src_index, tgt_index, self.d4rt_context_frames)
            context_frames = _load_video_frames(path, context_indices, self.image_size)
            image_t, image_tk = context_frames[0], context_frames[-1]
        else:
            context_indices = None
            context_frames = None
            image_t, image_tk = _load_video_frames(path, [src_index, tgt_index], self.image_size)
        sample: dict[str, torch.Tensor | int | str | list[int]] = {
            "image_t": image_t,
            "image_tk": image_tk,
            "stride": stride,
            "src_index": src_index,
            "tgt_index": tgt_index,
            "sample_indices": sample_indices,
            "sample_pairs": sample_pairs,
            "video_path": str(path),
            "path_t": f"{path}:{src_index}",
            "path_tk": f"{path}:{tgt_index}",
        }
        if context_frames is not None and context_indices is not None:
            sample.update(
                {
                    "d4rt_video": torch.stack(context_frames, dim=0),
                    "d4rt_t_src": 0,
                    "d4rt_t_tgt": self.d4rt_context_frames - 1,
                    "d4rt_t_cam": 0,
                }
            )
        return sample

    def _sample_replacement_index(self) -> int | None:
        available_video_indices = [
            video_idx for video_idx in range(len(self._videos))
            if video_idx not in self._bad_video_indices
        ]
        if not available_video_indices:
            return None
        video_idx = random.choice(available_video_indices)
        pair_idx = random.randrange(self.pairs_per_video)
        return video_idx * self.pairs_per_video + pair_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str | list[int]]:
        video_idx = idx // self.pairs_per_video
        pair_idx = idx % self.pairs_per_video
        last_error: ValueError | None = None
        for attempt in range(self.max_resample_attempts + 1):
            if video_idx in self._bad_video_indices:
                next_idx = self._sample_replacement_index()
                if next_idx is None:
                    break
                video_idx = next_idx // self.pairs_per_video
                pair_idx = next_idx % self.pairs_per_video
            try:
                return self._make_sample(video_idx, pair_idx)
            except ValueError as exc:
                if not _is_video_frame_read_error(exc):
                    raise
                self._bad_video_indices.add(video_idx)
                last_error = exc
                if attempt >= self.max_resample_attempts:
                    break
                next_idx = self._sample_replacement_index()
                if next_idx is None:
                    break
                next_video_idx = next_idx // self.pairs_per_video
                next_pair_idx = next_idx % self.pairs_per_video
                warnings.warn(
                    f"Blacklisting unreadable video for this dataset worker idx={idx} "
                    f"video_idx={video_idx} pair_idx={pair_idx}; "
                    f"bad_video_count={len(self._bad_video_indices)}; "
                    f"resampling idx={next_idx} video_idx={next_video_idx} pair_idx={next_pair_idx}. {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                video_idx = next_video_idx
                pair_idx = next_pair_idx

        if last_error is not None:
            raise last_error
        raise ValueError(f"No readable videos remain under {self.root}")


def build_pair_dataset(
    root: str | Path,
    image_size: int = 224,
    split: str = "train",
    strides: list[int] | tuple[int, ...] = (5, 10, 15, 20),
    eval_stride: int | None = None,
    num_sample_frames: int = 9,
    video_sampling: str = "motion-guided",
    pairs_per_video: int | None = None,
    d4rt_context_frames: int = 0,
) -> Dataset:
    if root_has_video_files(root):
        return VideoFilePairDataset(
            root,
            image_size=image_size,
            num_sample_frames=num_sample_frames,
            strides=strides,
            split=split,
            video_sampling=video_sampling,
            pairs_per_video=pairs_per_video,
            d4rt_context_frames=d4rt_context_frames,
        )
    return VideoFolderFrameDataset(
        root,
        image_size=image_size,
        strides=strides,
        split=split,
        eval_stride=eval_stride,
        d4rt_context_frames=d4rt_context_frames,
    )
