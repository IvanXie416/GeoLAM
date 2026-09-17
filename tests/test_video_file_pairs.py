from pathlib import Path

import cv2
import numpy as np
import torch

from lam.data.video_pairs import (
    VideoFilePairDataset,
    build_pair_dataset,
    larybench_sample_indices,
    larybench_sample_pairs,
    uniform_context_indices,
)


def _write_video(path: Path, frames: int = 16, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (size, size))
    for idx in range(frames):
        frame = np.full((size, size, 3), idx * 10 % 255, dtype=np.uint8)
        frame[:, :, 1] = 255 - frame[:, :, 0]
        writer.write(frame)
    writer.release()


def test_larybench_sample_indices_uses_uniform_for_short_videos():
    indices = larybench_sample_indices(total_frames=12, diff_scores=None, num_frames=9)

    assert indices == [0, 1, 2, 4, 5, 6, 8, 9, 10]


def test_larybench_sample_indices_uses_motion_guided_diffs_for_long_videos():
    diff_scores = [0.01] * 31
    diff_scores[10] = 10.0
    diff_scores[20] = 10.0

    indices = larybench_sample_indices(total_frames=32, diff_scores=diff_scores, num_frames=9)

    assert len(indices) == 9
    assert indices == sorted(indices)
    assert len(set(indices)) > 1
    assert all(0 <= idx < 32 for idx in indices)


def test_video_file_pair_dataset_reads_mp4_pair(tmp_path):
    _write_video(tmp_path / "0_2.mp4", frames=12)

    dataset = VideoFilePairDataset(tmp_path, image_size=8, num_sample_frames=9)
    sample = dataset[0]

    assert len(dataset) == 8
    assert sample["image_t"].shape == (3, 8, 8)
    assert sample["image_tk"].shape == (3, 8, 8)
    assert sample["sample_indices"] == [0, 1, 2, 4, 5, 6, 8, 9, 10]
    assert sample["tgt_index"] - sample["src_index"] == sample["stride"]
    assert sample["stride"] == 1
    assert sample["video_path"].endswith("0_2.mp4")


def test_video_file_pair_dataset_returns_d4rt_context_clip(tmp_path, monkeypatch):
    _write_video(tmp_path / "0_2.mp4", frames=20)

    dataset = VideoFilePairDataset(
        tmp_path,
        image_size=8,
        video_sampling="sliding-window",
        strides=[10],
        pairs_per_video=1,
        d4rt_context_frames=6,
    )
    monkeypatch.setattr(dataset, "_sample_sliding_pair", lambda total_frames: (2, 12))
    sample = dataset[0]

    assert sample["d4rt_video"].shape == (6, 3, 8, 8)
    assert sample["d4rt_t_src"] == 0
    assert sample["d4rt_t_tgt"] == 5
    assert sample["d4rt_t_cam"] == 0
    assert torch.equal(sample["image_t"], sample["d4rt_video"][0])
    assert torch.equal(sample["image_tk"], sample["d4rt_video"][-1])


def test_larybench_sample_pairs_repeats_valid_adjacent_pairs_for_tiny_videos():
    pairs = larybench_sample_pairs(total_frames=2, diff_scores=None, num_frames=9)

    assert pairs == [(0, 1)] * 8
    assert all(tgt > src for src, tgt in pairs)


def test_uniform_context_indices_preserves_endpoints_and_length():
    assert uniform_context_indices(3, 23, 6) == [3, 7, 11, 15, 19, 23]
    assert uniform_context_indices(4, 5, 6) == [4, 4, 4, 5, 5, 5]


def test_video_file_pair_dataset_repeats_valid_pairs_without_zero_stride(tmp_path):
    _write_video(tmp_path / "tiny.mp4", frames=2)

    dataset = VideoFilePairDataset(tmp_path, image_size=8, num_sample_frames=9)
    strides = [dataset[idx]["stride"] for idx in range(len(dataset))]
    pairs = [dataset[idx]["sample_pairs"][idx % dataset.pairs_per_video] for idx in range(len(dataset))]

    assert len(dataset) == 8
    assert strides == [1] * 8
    assert pairs == [(0, 1)] * 8


def test_video_file_pair_dataset_motion_guided_does_not_write_sample_pair_cache(tmp_path, monkeypatch):
    _write_video(tmp_path / "long.mp4", frames=32)
    calls = {"count": 0}

    def fake_diff_scores(path, num_frames_limit):
        calls["count"] += 1
        return [0.01] * 31

    monkeypatch.setattr("lam.data.video_pairs._video_diff_scores", fake_diff_scores)
    dataset = VideoFilePairDataset(tmp_path, image_size=8, num_sample_frames=9)
    sample = dataset[0]

    assert calls["count"] == 1
    assert sample["sample_pairs"]
    assert not list(tmp_path.rglob("*sample_pairs*.sqlite"))


def test_video_file_pair_dataset_sliding_window_skips_diff_scores(tmp_path, monkeypatch):
    _write_video(tmp_path / "long.mp4", frames=20)

    def fail_diff_scores(path, num_frames_limit):
        raise AssertionError("sliding-window sampling should not compute video diff scores")

    monkeypatch.setattr("lam.data.video_pairs._video_diff_scores", fail_diff_scores)
    dataset = VideoFilePairDataset(
        tmp_path,
        image_size=8,
        video_sampling="sliding-window",
        strides=[3, 6, 9, 12],
        pairs_per_video=8,
    )

    assert len(dataset) == 8
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert sample["stride"] in {3, 6, 9, 12}
        assert sample["tgt_index"] - sample["src_index"] == sample["stride"]
        assert 0 <= sample["src_index"] < sample["tgt_index"] < 20
        assert sample["sample_indices"] == [sample["src_index"], sample["tgt_index"]]
        assert sample["sample_pairs"] == [(sample["src_index"], sample["tgt_index"])]


def test_video_file_pair_dataset_sliding_window_falls_back_to_adjacent_for_short_videos(tmp_path):
    _write_video(tmp_path / "short.mp4", frames=2)

    dataset = VideoFilePairDataset(
        tmp_path,
        image_size=8,
        video_sampling="sliding-window",
        strides=[3, 6, 9, 12],
        pairs_per_video=8,
    )
    pairs = [(dataset[idx]["src_index"], dataset[idx]["tgt_index"]) for idx in range(len(dataset))]

    assert len(dataset) == 8
    assert pairs == [(0, 1)] * 8


def test_video_file_pair_dataset_resamples_unreadable_video_samples(tmp_path, monkeypatch):
    bad_path = tmp_path / "bad.mp4"
    good_path = tmp_path / "good.mp4"
    _write_video(bad_path, frames=12)
    _write_video(good_path, frames=12)
    calls_by_path = {bad_path: 0, good_path: 0}

    def fake_load_video_frames(path, frame_indices, image_size):
        calls_by_path[path] += 1
        if path == bad_path:
            raise ValueError(
                f"Could not read frame {frame_indices[-1]} from video: {path} "
                "(reported_frame_count=12). OpenCV/FFmpeg may have logged the decoder error to stderr above."
            )
        return [torch.zeros(3, image_size, image_size) for _ in frame_indices]

    monkeypatch.setattr("lam.data.video_pairs._load_video_frames", fake_load_video_frames)
    dataset = VideoFilePairDataset(
        tmp_path,
        image_size=8,
        video_sampling="sliding-window",
        pairs_per_video=1,
        max_resample_attempts=8,
    )

    sample = dataset[0]
    sample_again = dataset[0]

    assert sample["video_path"] == str(good_path)
    assert sample_again["video_path"] == str(good_path)
    assert sample["image_t"].shape == (3, 8, 8)
    assert calls_by_path[bad_path] == 1
    assert dataset._bad_video_indices == {0}


def test_video_file_pair_dataset_raises_after_resample_attempts_exhausted(tmp_path, monkeypatch):
    _write_video(tmp_path / "bad.mp4", frames=12)

    def fake_load_video_frames(path, frame_indices, image_size):
        raise ValueError(
            f"Could not read frame {frame_indices[-1]} from video: {path} "
            "(reported_frame_count=12). OpenCV/FFmpeg may have logged the decoder error to stderr above."
        )

    monkeypatch.setattr("lam.data.video_pairs._load_video_frames", fake_load_video_frames)
    dataset = VideoFilePairDataset(
        tmp_path,
        image_size=8,
        video_sampling="sliding-window",
        pairs_per_video=1,
        max_resample_attempts=2,
    )

    try:
        dataset[0]
    except ValueError as exc:
        assert "Could not read frame" in str(exc)
    else:
        raise AssertionError("expected unreadable sample to raise after retries are exhausted")


def test_build_pair_dataset_auto_detects_flat_mp4_root(tmp_path):
    _write_video(tmp_path / "0_2.mp4", frames=16)
    _write_video(tmp_path / "1_0.mp4", frames=16)

    dataset = build_pair_dataset(tmp_path, image_size=8, split="train", num_sample_frames=9)

    assert isinstance(dataset, VideoFilePairDataset)
    assert len(dataset) == 16
