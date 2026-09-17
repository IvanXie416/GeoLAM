from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import Dataset

from lam.train import (
    build_training_dataset,
    compute_steps_per_epoch,
    dataloader_performance_kwargs,
    format_duration,
    format_progress,
    parse_args,
    resolve_log_file,
    resolve_save_steps,
    split_dataset_for_validation,
)


class _VideoLikeDataset(Dataset):
    pairs_per_video = 3

    def __init__(self, videos: int = 5) -> None:
        self._videos = [(Path(f"video_{idx}.mp4"), 10) for idx in range(videos)]

    def __len__(self) -> int:
        return len(self._videos) * self.pairs_per_video

    def __getitem__(self, idx: int) -> int:
        return idx


def test_parse_args_accepts_save_steps_and_log_file():
    args = parse_args(
        [
            "data/root",
            "--save-steps",
            "25",
            "--log-steps",
            "20",
            "--log-file",
            "logs/run.log",
            "--val-steps",
            "50",
            "--val-ratio",
            "0.2",
            "--val-batches",
            "4",
        ]
    )

    assert args.data_root == Path("data/root")
    assert args.save_steps == 25
    assert args.log_steps == 20
    assert args.log_file == Path("logs/run.log")
    assert args.val_steps == 50
    assert args.val_ratio == 0.2
    assert args.val_batches == 4


def test_parse_args_accepts_sliding_window_video_sampling_options():
    args = parse_args(
        [
            "data/root",
            "--video-sampling",
            "sliding-window",
            "--strides",
            "5",
            "10",
            "15",
            "20",
            "--pairs-per-video",
            "8",
        ]
    )

    assert args.video_sampling == "sliding-window"
    assert args.strides == [5, 10, 15, 20]
    assert args.pairs_per_video == 8


def test_build_training_dataset_applies_positional_data_root_sample_rates(monkeypatch):
    def fake_build_pair_dataset(data_root, **kwargs):
        videos = 4 if Path(data_root).name == "small" else 8
        return _VideoLikeDataset(videos=videos)

    monkeypatch.setattr("lam.train.build_pair_dataset", fake_build_pair_dataset)
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            image_size=8,
            strides=[3, 6],
            num_sample_frames=9,
        )
    )
    args = parse_args(
        [
            "data/small",
            "data/agibot",
            "--data-root-sample-rates",
            "1.0",
            "0.5",
        ]
    )

    dataset = build_training_dataset(args, cfg)

    assert len(dataset.datasets[0]) == 12
    assert len(dataset.datasets[1]) == 12


def test_build_training_dataset_only_loads_context_when_d4rt_is_enabled(monkeypatch):
    context_lengths = []

    def fake_build_pair_dataset(data_root, **kwargs):
        context_lengths.append(kwargs["d4rt_context_frames"])
        return _VideoLikeDataset(videos=1)

    monkeypatch.setattr("lam.train.build_pair_dataset", fake_build_pair_dataset)
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            image_size=8,
            strides=[5],
            num_sample_frames=9,
            d4rt_context_frames=6,
        )
    )

    build_training_dataset(parse_args(["data/root"]), cfg)
    build_training_dataset(parse_args(["data/root", "--no-d4rt"]), cfg)

    assert context_lengths == [6, 0]


def test_resolve_save_steps_prefers_new_flag_and_keeps_legacy_alias():
    args = parse_args(["data/root", "--save-every", "10", "--save-steps", "5"])
    assert resolve_save_steps(args) == 5

    legacy_args = parse_args(["data/root", "--save-every", "10"])
    assert resolve_save_steps(legacy_args) == 10


def test_compute_steps_per_epoch_matches_drop_last_loader_length():
    assert compute_steps_per_epoch(dataset_size=26288, batch_size=1, drop_last=True) == 26288
    assert compute_steps_per_epoch(dataset_size=26288, batch_size=4, drop_last=True) == 6572
    assert compute_steps_per_epoch(dataset_size=26288, batch_size=4, drop_last=False) == 6572
    assert compute_steps_per_epoch(dataset_size=26289, batch_size=4, drop_last=False) == 6573


def test_dataloader_performance_kwargs_enable_pinned_persistent_prefetch():
    kwargs = dataloader_performance_kwargs(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    assert kwargs == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }


def test_dataloader_performance_kwargs_skip_worker_only_options_without_workers():
    kwargs = dataloader_performance_kwargs(
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_resolve_log_file_defaults_to_output_dir_train_log():
    assert resolve_log_file(Path("outputs/run"), None) == Path("outputs/run/train.log")
    assert resolve_log_file(Path("outputs/run"), Path("custom.log")) == Path("custom.log")


def test_format_duration_is_compact_and_readable():
    assert format_duration(5.2) == "5s"
    assert format_duration(65.0) == "1m05s"
    assert format_duration(3661.0) == "1h01m01s"
    assert format_duration(float("inf")) == "unknown"


def test_format_progress_includes_bar_elapsed_and_eta():
    progress = format_progress(completed_steps=5, total_steps=10, start_time=0.0, now=10.0)

    assert "progress=[############------------]" in progress
    assert "5/10" in progress
    assert "50.00%" in progress
    assert "elapsed=10s" in progress
    assert "eta=10s" in progress


def test_split_dataset_for_validation_uses_disjoint_video_groups():
    dataset = _VideoLikeDataset(videos=5)

    train_subset, val_subset, info = split_dataset_for_validation(dataset, val_ratio=0.4, seed=123)

    assert info.full_size == 15
    assert info.group_count == 5
    assert info.train_group_count == 3
    assert info.val_group_count == 2
    assert len(train_subset) == 9
    assert len(val_subset) == 6
    train_videos = {idx // dataset.pairs_per_video for idx in train_subset.indices}
    val_videos = {idx // dataset.pairs_per_video for idx in val_subset.indices}
    assert train_videos.isdisjoint(val_videos)


def test_split_dataset_for_validation_preserves_groups_after_downsampling(monkeypatch):
    source_dataset = _VideoLikeDataset(videos=6)

    def fake_build_pair_dataset(data_root, **kwargs):
        return source_dataset

    monkeypatch.setattr("lam.train.build_pair_dataset", fake_build_pair_dataset)
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            image_size=8,
            strides=[3, 6],
            num_sample_frames=9,
        )
    )
    args = parse_args(["data/agibot", "--data-root-sample-rates", "0.5"])
    downsampled = build_training_dataset(args, cfg)

    train_subset, val_subset, info = split_dataset_for_validation(downsampled, val_ratio=0.4, seed=123)

    assert len(downsampled) == 9
    assert info.group_count == 3
    train_videos = {downsampled.indices[idx] // source_dataset.pairs_per_video for idx in train_subset.indices}
    val_videos = {downsampled.indices[idx] // source_dataset.pairs_per_video for idx in val_subset.indices}
    assert train_videos.isdisjoint(val_videos)


def test_split_dataset_for_validation_can_disable_validation():
    dataset = _VideoLikeDataset(videos=5)

    train_subset, val_subset, info = split_dataset_for_validation(dataset, val_ratio=0.0, seed=123)

    assert train_subset is dataset
    assert val_subset is None
    assert info.train_size == len(dataset)
    assert info.val_size == 0
