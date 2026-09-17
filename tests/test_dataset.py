from pathlib import Path

from PIL import Image

import torch

from lam.data.video_pairs import VideoFolderFrameDataset


def _write_frame(path: Path, shade: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(shade, shade, shade)).save(path)


def test_video_folder_dataset_samples_training_stride_from_choices(tmp_path):
    video_dir = tmp_path / "video_a"
    for idx in range(20):
        _write_frame(video_dir / f"{idx:05d}.png", idx)

    dataset = VideoFolderFrameDataset(tmp_path, image_size=8, strides=[3, 6, 9, 12], split="train")
    sample = dataset[0]

    assert sample["stride"] in {3, 6, 9, 12}
    assert sample["image_t"].shape == (3, 8, 8)
    assert sample["image_tk"].shape == (3, 8, 8)


def test_video_folder_dataset_uses_fixed_validation_stride(tmp_path):
    video_dir = tmp_path / "video_a"
    for idx in range(20):
        _write_frame(video_dir / f"{idx:05d}.png", idx)

    dataset = VideoFolderFrameDataset(tmp_path, image_size=8, strides=[3, 6, 9, 12], split="val", eval_stride=9)
    sample = dataset[0]

    assert sample["stride"] == 9


def test_video_folder_dataset_returns_d4rt_context_clip(tmp_path):
    video_dir = tmp_path / "video_a"
    for idx in range(20):
        _write_frame(video_dir / f"{idx:05d}.png", idx)

    dataset = VideoFolderFrameDataset(
        tmp_path,
        image_size=8,
        strides=[10],
        split="val",
        eval_stride=10,
        d4rt_context_frames=6,
    )
    sample = dataset[0]

    assert sample["d4rt_video"].shape == (6, 3, 8, 8)
    assert sample["d4rt_t_src"] == 0
    assert sample["d4rt_t_tgt"] == 5
    assert sample["d4rt_t_cam"] == 0
    assert torch.equal(sample["image_t"], sample["d4rt_video"][0])
    assert torch.equal(sample["image_tk"], sample["d4rt_video"][-1])
