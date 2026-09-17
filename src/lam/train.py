from __future__ import annotations

import argparse
import datetime
import math
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from .config import load_lam_config
from .data.video_pairs import build_pair_dataset
from .integrations.d4rt import D4RTWrapper
from .integrations.gld import GLDBackboneWrapper
from .losses import (
    LAMLossInputs,
    LossSchedule,
    LossWeights,
    PixelLAMLossInputs,
    compute_lam_losses,
    compute_pixel_lam_losses,
)
from .models.geo_teacher import GeoMotionTokenizer
from .models.gld_lam import GLDLAM
from .models.lam_3d_pixel import LAM3DPixel


@dataclass(frozen=True)
class DatasetSplitInfo:
    full_size: int
    train_size: int
    val_size: int
    group_count: int
    train_group_count: int
    val_group_count: int


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    device: torch.device


class TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        written = self.primary.write(data)
        self.secondary.write(data)
        return written

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


@contextmanager
def tee_output(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_file.open("a", encoding="utf-8", buffering=1) as handle:
        sys.stdout = TeeStream(original_stdout, handle)
        sys.stderr = TeeStream(original_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def compute_steps_per_epoch(dataset_size: int, batch_size: int, drop_last: bool = True) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")
    if drop_last:
        return dataset_size // batch_size
    return math.ceil(dataset_size / batch_size)


def dataloader_performance_kwargs(
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers <= 0:
        return kwargs

    kwargs["persistent_workers"] = bool(persistent_workers)
    if prefetch_factor is not None and prefetch_factor > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def is_distributed_env() -> bool:
    return env_int("WORLD_SIZE", 1) > 1


def setup_distributed(requested_device: str) -> DistributedContext:
    distributed = is_distributed_env()
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    device = torch.device(requested_device)

    if distributed:
        use_cuda = device.type == "cuda"
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        backend = "nccl" if use_cuda else "gloo"
        if not dist.is_initialized():
            init_kwargs = {"backend": backend}
            if use_cuda:
                init_kwargs["device_id"] = device
            timeout_seconds = env_int("LAM_DISTRIBUTED_TIMEOUT_SECONDS", 3600)
            if timeout_seconds > 0:
                init_kwargs["timeout"] = datetime.timedelta(seconds=timeout_seconds)
            dist.init_process_group(**init_kwargs)

    return DistributedContext(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main=rank == 0,
        device=device,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        if ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def autocast_context(device: torch.device, amp: str):
    if amp == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def checkpoint_payload(model: torch.nn.Module, step: int) -> dict:
    return {
        "model": unwrap_model(model).state_dict(),
        "step": int(step),
        "architecture": {
            "fdm_conditioning": "cross_attention",
            "idm_input": "state_delta",
            "latent_type": "deterministic_normalized",
        },
    }


def reduce_loss_parts(parts: dict[str, torch.Tensor], ctx: DistributedContext) -> dict[str, torch.Tensor]:
    if not ctx.distributed or not dist.is_initialized():
        return parts
    reduced: dict[str, torch.Tensor] = {}
    for key, value in parts.items():
        detached = value.detach().clone()
        dist.all_reduce(detached, op=dist.ReduceOp.SUM)
        detached /= ctx.world_size
        reduced[key] = detached
    return reduced


def reduce_scalar(value: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if not ctx.distributed or not dist.is_initialized():
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= ctx.world_size
    return reduced


def latent_stat_parts(latent_action) -> dict[str, torch.Tensor]:
    low_dim = latent_action.low_dim.float()
    return {
        "z_low_dim_abs": low_dim.abs().mean(),
        "z_low_dim_std": low_dim.std(unbiased=False),
    }


def reduce_validation_result(
    metrics: dict[str, float], batch_count: int, device: torch.device, ctx: DistributedContext
) -> tuple[dict[str, float], int]:
    if not ctx.distributed or not dist.is_initialized():
        return metrics, batch_count
    keys = tuple(metrics)
    values = [
        0.0 if batch_count <= 0 or not math.isfinite(value) else value * batch_count
        for value in metrics.values()
    ]
    stats = torch.tensor([*values, float(batch_count)], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_count = int(stats[-1].item())
    if total_count <= 0:
        return {key: float("nan") for key in keys}, 0
    return {key: float((stats[index] / stats[-1]).item()) for index, key in enumerate(keys)}, total_count


def should_log(ctx: DistributedContext) -> bool:
    return ctx.is_main


def resolve_save_steps(args: argparse.Namespace) -> int:
    if args.save_steps is not None:
        return int(args.save_steps)
    if args.save_every is not None:
        return int(args.save_every)
    return 1000


def resolve_log_file(output_dir: Path, log_file: Path | None) -> Path:
    return output_dir / "train.log" if log_file is None else log_file


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_progress(
    completed_steps: int,
    total_steps: int,
    start_time: float,
    now: float,
    width: int = 24,
) -> str:
    elapsed = max(0.0, now - start_time)
    if total_steps <= 0:
        return f"progress=[{'-' * max(0, width)}] {completed_steps}/{total_steps} 0.00% elapsed={format_duration(elapsed)} eta=unknown"

    bounded_completed = min(max(0, completed_steps), total_steps)
    ratio = bounded_completed / total_steps
    bar_width = max(1, width)
    filled = min(bar_width, int(bar_width * ratio))
    bar = "#" * filled + "-" * (bar_width - filled)
    if bounded_completed > 0:
        seconds_per_step = elapsed / bounded_completed
        eta = seconds_per_step * (total_steps - bounded_completed)
    else:
        eta = float("inf")
    return (
        f"progress=[{bar}] {bounded_completed}/{total_steps} {ratio * 100:.2f}% "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


def _dataset_index_groups(dataset: Dataset) -> list[list[int]]:
    if isinstance(dataset, Subset):
        child_groups = _dataset_index_groups(dataset.dataset)
        subset_positions_by_child_idx: dict[int, list[int]] = {}
        for subset_pos, child_idx in enumerate(dataset.indices):
            subset_positions_by_child_idx.setdefault(int(child_idx), []).append(subset_pos)

        groups: list[list[int]] = []
        for child_group in child_groups:
            group: list[int] = []
            for child_idx in child_group:
                group.extend(subset_positions_by_child_idx.get(int(child_idx), []))
            if group:
                groups.append(group)
        return groups

    if isinstance(dataset, ConcatDataset):
        groups: list[list[int]] = []
        offset = 0
        for child in dataset.datasets:
            child_groups = _dataset_index_groups(child)
            groups.extend([[offset + idx for idx in group] for group in child_groups])
            offset += len(child)
        return groups

    if hasattr(dataset, "_videos") and hasattr(dataset, "pairs_per_video"):
        pairs_per_video = int(getattr(dataset, "pairs_per_video"))
        video_count = len(getattr(dataset, "_videos"))
        groups: list[list[int]] = []
        for video_idx in range(video_count):
            start = video_idx * pairs_per_video
            groups.append([idx for idx in range(start, min(start + pairs_per_video, len(dataset)))])
        return [group for group in groups if group]

    if hasattr(dataset, "_items"):
        grouped: dict[str, list[int]] = {}
        for idx, item in enumerate(getattr(dataset, "_items")):
            frames = item[0]
            key = str(frames[0].parent) if frames else str(idx)
            grouped.setdefault(key, []).append(idx)
        return list(grouped.values())

    return [[idx] for idx in range(len(dataset))]


def resolve_data_root_sample_rates(data_roots: Sequence[Path], sample_rates: Sequence[float] | None) -> list[float]:
    if sample_rates is None:
        return [1.0 for _ in data_roots]
    if len(sample_rates) != len(data_roots):
        raise ValueError(
            "--data-root-sample-rates must have exactly one value per data root "
            f"({len(sample_rates)} rates for {len(data_roots)} roots)"
        )

    rates = [float(rate) for rate in sample_rates]
    for rate in rates:
        if not math.isfinite(rate) or rate <= 0.0 or rate > 1.0:
            raise ValueError("--data-root-sample-rates values must be in the range (0, 1]")
    return rates


def downsample_dataset_by_groups(dataset: Dataset, sample_rate: float, seed: int = 42) -> Dataset:
    if sample_rate >= 1.0:
        return dataset

    groups = _dataset_index_groups(dataset)
    if not groups:
        return dataset

    keep_group_count = max(1, math.floor(len(groups) * sample_rate))
    if keep_group_count >= len(groups):
        return dataset

    group_order = list(range(len(groups)))
    rng = random.Random(seed)
    rng.shuffle(group_order)
    selected_group_ids = set(group_order[:keep_group_count])

    indices: list[int] = []
    for group_idx, group in enumerate(groups):
        if group_idx in selected_group_ids:
            indices.extend(group)
    return Subset(dataset, indices)


def build_training_dataset(args: argparse.Namespace, cfg) -> Dataset:
    strides = args.strides if args.strides is not None else cfg.data.strides
    datasets = []
    for data_root, sample_rate in zip(args.data_roots, args.data_root_sample_rates):
        dataset = build_pair_dataset(
            data_root,
            image_size=cfg.data.image_size,
            strides=strides,
            split="train",
            num_sample_frames=cfg.data.num_sample_frames,
            video_sampling=args.video_sampling,
            pairs_per_video=args.pairs_per_video,
            d4rt_context_frames=(
                getattr(cfg.data, "d4rt_context_frames", 0)
                if not getattr(args, "no_d4rt", False)
                else 0
            ),
        )
        datasets.append(downsample_dataset_by_groups(dataset, sample_rate, seed=args.data_root_sample_seed))
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_lam_model(cfg) -> torch.nn.Module:
    model_type = getattr(cfg.model, "type", "gld")
    model_kwargs = dict(
        in_dim=cfg.model.f_dim,
        c_state=cfg.model.c_state,
        c_z=cfg.model.c_z,
        z_dim=cfg.model.z_dim,
        c_geo=cfg.model.c_geo,
        k_action=cfg.model.k_action,
        k_geo=cfg.model.k_geo,
        latent_dim=cfg.model.latent_dim,
        heads=cfg.model.heads,
        action_bottleneck_depth=cfg.model.action_bottleneck_depth,
        idm_depth=cfg.model.idm_depth,
        fdm_depth=cfg.model.fdm_depth,
        fdm_self_attn_every=cfg.model.fdm_self_attn_every,
        ffn_ratio=cfg.model.ffn_ratio,
    )
    if model_type == "gld":
        return GLDLAM(**model_kwargs)
    if model_type == "lam_3D_pixel":
        return LAM3DPixel(
            **model_kwargs,
            image_size=cfg.data.image_size,
            rgb_patch_size=cfg.model.rgb_patch_size,
            rgb_decoder_depth=cfg.model.rgb_decoder_depth,
        )
    raise ValueError(f"Unsupported model.type: {model_type!r}")

def split_dataset_for_validation(
    dataset: Dataset,
    val_ratio: float,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None, DatasetSplitInfo]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    groups = _dataset_index_groups(dataset)
    if val_ratio <= 0.0 or len(groups) < 2:
        info = DatasetSplitInfo(
            full_size=len(dataset),
            train_size=len(dataset),
            val_size=0,
            group_count=len(groups),
            train_group_count=len(groups),
            val_group_count=0,
        )
        return dataset, None, info

    val_group_count = min(max(1, math.ceil(len(groups) * val_ratio)), len(groups) - 1)
    group_order = list(range(len(groups)))
    rng = random.Random(seed)
    rng.shuffle(group_order)
    val_group_ids = set(group_order[:val_group_count])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for group_idx, group in enumerate(groups):
        if group_idx in val_group_ids:
            val_indices.extend(group)
        else:
            train_indices.extend(group)

    train_dataset: Dataset = Subset(dataset, train_indices)
    val_dataset: Dataset = Subset(dataset, val_indices)
    info = DatasetSplitInfo(
        full_size=len(dataset),
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        group_count=len(groups),
        train_group_count=len(groups) - val_group_count,
        val_group_count=val_group_count,
    )
    return train_dataset, val_dataset, info


def run_validation(
    model: torch.nn.Module,
    gld: GLDBackboneWrapper,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 16,
    model_type: str = "gld",
    lambda_gld: float = 1.0,
    amp: str = "none",
) -> tuple[dict[str, float], int]:
    was_training = model.training
    model.eval()
    totals: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            image_t = batch["image_t"].to(device)
            image_tk = batch["image_tk"].to(device)
            with autocast_context(device, amp):
                f0_t, f1_t = gld(image_t)
                f0_tk, f1_tk = gld(image_tk)
                out = model(f0_t, f1_t, f0_tk, f1_tk)
                if model_type == "lam_3D_pixel":
                    losses = compute_pixel_lam_losses(
                        PixelLAMLossInputs(
                            rgb_hat=out.rgb_hat,
                            rgb_target=image_tk,
                            z_tokens=out.latent_action.tokens,
                            z_vector=out.latent_action.vector,
                            f0_hat=out.f0_hat,
                            f1_hat=out.f1_hat,
                            z_geo_tokens_pred=out.z_geo_tokens_pred,
                        ),
                    )
                    totals.setdefault("pixel_mse", []).append(float(losses.parts["pixel_mse"].detach().item()))
                else:
                    losses = compute_lam_losses(
                        LAMLossInputs(
                            f0_hat=out.f0_hat,
                            f1_hat=out.f1_hat,
                            f0_target=f0_tk,
                            f1_target=f1_tk,
                            z_tokens=out.latent_action.tokens,
                            z_vector=out.latent_action.vector,
                            z_geo_tokens_pred=out.z_geo_tokens_pred,
                            m_geo_tokens=None,
                        ),
                        LossWeights(
                            lambda_gld=lambda_gld,
                            lambda_geo_action=0.0,
                        ),
                    )
                    for key in ("gld_f0", "gld_f1", "gld_f01"):
                        totals.setdefault(key, []).append(float(losses.parts[key].detach().item()))
    if was_training:
        model.train()
    if not totals:
        keys = ("pixel_mse",) if model_type == "lam_3D_pixel" else ("gld_f0", "gld_f1", "gld_f01")
        return {key: float("nan") for key in keys}, 0
    return {key: sum(values) / len(values) for key, values in totals.items()}, len(next(iter(totals.values())))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GLD-main LAM with D4RT geometry teacher")
    parser.add_argument("data_roots", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/lam_3d_pixel.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["none", "bf16"], default="none", help="Enable bfloat16 autocast")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor when num_workers > 0; <=0 disables explicit prefetching")
    parser.add_argument("--video-sampling", choices=["motion-guided", "sliding-window"], default="motion-guided")
    parser.add_argument("--strides", nargs="+", type=int, default=None, help="Temporal strides for folder data and sliding-window video sampling")
    parser.add_argument("--pairs-per-video", type=int, default=None, help="Number of video pairs exposed per flat video; defaults to num_sample_frames - 1")
    parser.add_argument("--data-root-sample-rates", nargs="+", type=float, default=None, help="Per-data-root sample rates in positional order; each value must be in (0, 1]")
    parser.add_argument("--data-root-sample-seed", type=int, default=42, help="Seed for deterministic per-root downsampling")
    parser.add_argument("--log-steps", type=int, default=10, help="Print loss, progress, elapsed time, and ETA every N optimizer steps; <=0 disables periodic progress logs")
    parser.add_argument("--save-steps", type=int, default=None, help="Save an intermediate checkpoint every N optimizer steps; <=0 disables intermediate saves")
    parser.add_argument("--save-every", type=int, default=None, help="Deprecated alias for --save-steps")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None, help="Write a copy of stdout/stderr to this file; defaults to output_dir/train.log")
    parser.add_argument("--val-steps", type=int, default=0, help="Run validation every N optimizer steps; <=0 disables validation")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Held-out validation fraction, split by video/sequence group")
    parser.add_argument("--val-batches", type=int, default=16, help="Maximum validation batches per validation run; <=0 uses all validation batches")
    parser.add_argument("--val-seed", type=int, default=42, help="Seed for deterministic train/validation split")
    parser.add_argument("--no-d4rt", action="store_true", help="Disable the D4RT geometry teacher loss")
    args = parser.parse_args(argv)
    try:
        args.data_root_sample_rates = resolve_data_root_sample_rates(args.data_roots, args.data_root_sample_rates)
    except ValueError as exc:
        parser.error(str(exc))
    args.data_root = args.data_roots[0]
    return args


def main() -> int:
    args = parse_args()
    cfg = load_lam_config(args.config)
    ctx = setup_distributed(args.device)
    if ctx.device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.amp == "bf16" and ctx.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp bf16 requires CUDA bfloat16 support")
    device = ctx.device
    output_dir = args.output_dir if args.output_dir is not None else Path(cfg.paths.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(ctx)
    log_file = resolve_log_file(output_dir, args.log_file)
    save_steps = resolve_save_steps(args)

    output_context = tee_output(log_file) if ctx.is_main else nullcontext()
    try:
        with output_context:
            if ctx.is_main:
                print(f"logging_to={log_file}", flush=True)
                print(
                    f"distributed={ctx.distributed} rank={ctx.rank} local_rank={ctx.local_rank} world_size={ctx.world_size} device={ctx.device}",
                    flush=True,
                )

            if ctx.distributed:
                if ctx.is_main:
                    print("building_dataset_index_cache=rank0", flush=True)
                    build_training_dataset(args, cfg)
                    print("building_dataset_index_cache=done", flush=True)
                distributed_barrier(ctx)

            full_dataset = build_training_dataset(args, cfg)
            if args.val_steps > 0:
                dataset, val_dataset, split_info = split_dataset_for_validation(full_dataset, args.val_ratio, args.val_seed)
            else:
                dataset, val_dataset, split_info = split_dataset_for_validation(full_dataset, 0.0, args.val_seed)

            train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if ctx.distributed else None
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if ctx.distributed and val_dataset is not None else None
            effective_train_size = len(dataset)
            if ctx.distributed:
                effective_train_size = len(train_sampler) if train_sampler is not None else math.ceil(len(dataset) / ctx.world_size)
            loader_kwargs = dataloader_performance_kwargs(
                args.num_workers,
                args.pin_memory,
                args.persistent_workers,
                args.prefetch_factor,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                drop_last=True,
                **loader_kwargs,
            )
            val_loader = None
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=val_sampler,
                    drop_last=False,
                    **loader_kwargs,
                )
            steps_per_epoch = len(loader)
            computed_steps = compute_steps_per_epoch(effective_train_size, args.batch_size, drop_last=True)
            if steps_per_epoch != computed_steps:
                raise RuntimeError(f"DataLoader length mismatch: len(loader)={steps_per_epoch} computed={computed_steps}")
            if steps_per_epoch <= 0:
                raise ValueError(f"Batch size {args.batch_size} is larger than per-rank dataset size with drop_last=True")
            if ctx.is_main:
                global_batch_size = args.batch_size * ctx.world_size
                print(
                    f"data_roots={len(args.data_roots)} full_dataset_size={split_info.full_size} train_size={split_info.train_size} val_size={split_info.val_size} "
                    f"group_count={split_info.group_count} train_groups={split_info.train_group_count} val_groups={split_info.val_group_count} "
                    f"batch_size_per_rank={args.batch_size} global_batch_size={global_batch_size} steps_per_epoch={steps_per_epoch} max_steps={args.steps} "
                    f"video_sampling={args.video_sampling} strides={args.strides if args.strides is not None else cfg.data.strides} pairs_per_video={args.pairs_per_video} "
                    f"data_root_sample_rates={args.data_root_sample_rates} data_root_sample_seed={args.data_root_sample_seed} "
                    f"num_workers={args.num_workers} pin_memory={args.pin_memory} persistent_workers={args.persistent_workers} prefetch_factor={args.prefetch_factor} "
                    f"log_steps={args.log_steps} save_steps={save_steps} val_steps={args.val_steps} val_batches={args.val_batches} amp={args.amp}",
                    flush=True,
                )

            gld = GLDBackboneWrapper(
                cfg.paths.gld_root,
                checkpoint=cfg.paths.gld_checkpoint,
                encoder_pretrained_path=cfg.paths.gld_encoder_pretrained_path,
                image_size=cfg.data.image_size,
                mae_weight=cfg.paths.gld_mae_weight if cfg.model.type == "lam_3D_pixel" else None,
                decoder_config_path=cfg.paths.gld_decoder_config if cfg.model.type == "lam_3D_pixel" else None,
            ).to(device)
            model = build_lam_model(cfg)
            if cfg.model.type == "lam_3D_pixel":
                if gld.model.mae_decoder is None:
                    raise RuntimeError("lam_3D_pixel requires a GLD MAE decoder; set paths.gld_mae_weight and paths.gld_decoder_config")
                unwrap_model(model).set_rgb_decoder(gld.model.mae_decoder, propagator=gld.model)
            model = model.to(device)
            if ctx.distributed:
                ddp_kwargs = {"device_ids": [ctx.local_rank], "output_device": ctx.local_rank} if device.type == "cuda" else {}
                model = DistributedDataParallel(model, **ddp_kwargs)
            d4rt = None
            geo_tokenizer = None
            if not args.no_d4rt:
                d4rt = D4RTWrapper(
                    cfg.paths.d4rt_root,
                    model_config=cfg.paths.d4rt_config,
                    checkpoint=cfg.paths.d4rt_checkpoint,
                    query_grid_size=cfg.d4rt.query_grid_size,
                    image_size=cfg.data.d4rt_image_size,
                    device=device,
                )
                geo_tokenizer = GeoMotionTokenizer(
                    hidden_dim=1280,
                    c_geo=cfg.model.c_geo,
                    k_geo=cfg.model.k_geo,
                    residual_motion_weight=cfg.d4rt.residual_motion_weight,
                    xyz_motion_scale=getattr(cfg.d4rt, "xyz_motion_scale", 0.1),
                    uv_motion_scale=getattr(cfg.d4rt, "uv_motion_scale", 0.05),
                ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            schedule = LossSchedule(
                phase0_warmup_steps=cfg.loss.phase0_warmup_steps,
                phase1_geo_ramp_steps=cfg.loss.phase1_geo_ramp_steps,
                lambda_gld=cfg.loss.lambda_gld,
                lambda_geo_action=cfg.loss.lambda_geo_action,
            )
            step = 0
            epoch = 0
            model.train()
            train_start_time = time.monotonic()
            while step < args.steps:
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                for batch in loader:
                    image_t = batch["image_t"].to(device, non_blocking=True)
                    image_tk = batch["image_tk"].to(device, non_blocking=True)
                    with autocast_context(device, args.amp):
                        with torch.no_grad():
                            f0_t, f1_t = gld(image_t)
                            f0_tk, f1_tk = gld(image_tk)

                        out = model(f0_t, f1_t, f0_tk, f1_tk)
                        loss_weights = schedule.weights(step)
                        geo_target = None
                        # D4RT is frozen and expensive. There is no geometry
                        # gradient during warmup, so defer the teacher forward
                        # until its scheduled weight becomes non-zero.
                        if d4rt is not None and geo_tokenizer is not None and loss_weights.lambda_geo_action > 0.0:
                            d4rt_video = batch.get("d4rt_video")
                            if torch.is_tensor(d4rt_video):
                                teacher = d4rt.forward_clip(
                                    d4rt_video,
                                    t_src=batch.get("d4rt_t_src", 0),
                                    t_tgt=batch.get("d4rt_t_tgt", 1),
                                    t_cam=batch.get("d4rt_t_cam", 0),
                                )
                            else:
                                # Compatibility path for custom/legacy datasets
                                # that only provide the GLD image pair.
                                teacher = d4rt.forward_pair(image_t, image_tk)
                            geo_target = geo_tokenizer(teacher.hidden_t, teacher.heads_t, teacher.hidden_tk, teacher.heads_tk, detach=True)

                        if cfg.model.type == "lam_3D_pixel":
                            losses = compute_pixel_lam_losses(
                                PixelLAMLossInputs(
                                    rgb_hat=out.rgb_hat,
                                    rgb_target=image_tk,
                                    z_tokens=out.latent_action.tokens,
                                    z_vector=out.latent_action.vector,
                                    f0_hat=out.f0_hat,
                                    f1_hat=out.f1_hat,
                                    z_geo_tokens_pred=out.z_geo_tokens_pred,
                                    m_geo_tokens=geo_target.tokens if geo_target is not None else None,
                                ),
                                lambda_pixel=cfg.loss.lambda_pixel,
                                lambda_geo_action=loss_weights.lambda_geo_action,
                            )
                        else:
                            losses = compute_lam_losses(
                                LAMLossInputs(
                                    f0_hat=out.f0_hat,
                                    f1_hat=out.f1_hat,
                                    f0_target=f0_tk,
                                    f1_target=f1_tk,
                                    z_tokens=out.latent_action.tokens,
                                    z_vector=out.latent_action.vector,
                                    z_geo_tokens_pred=out.z_geo_tokens_pred,
                                    m_geo_tokens=geo_target.tokens if geo_target is not None else None,
                                ),
                                loss_weights,
                            )
                    optimizer.zero_grad(set_to_none=True)
                    losses.total.backward()
                    optimizer.step()

                    step += 1
                    if ctx.is_main and args.log_steps > 0 and (step == 1 or step % args.log_steps == 0 or step >= args.steps):
                        reduced_parts = reduce_loss_parts(losses.parts, ctx)
                        reduced_latent_parts = reduce_loss_parts(latent_stat_parts(out.latent_action), ctx)
                        reduced_total = reduce_scalar(losses.total, ctx)
                        parts = " ".join(
                            f"{key}={value.item():.4f}"
                            for key, value in {**reduced_parts, **reduced_latent_parts}.items()
                            if key != "geo_token"
                        )
                        progress = format_progress(step, args.steps, train_start_time, time.monotonic())
                        print(f"step={step} total={reduced_total.item():.4f} {parts} {progress}", flush=True)
                    elif not ctx.is_main and args.log_steps > 0 and (step == 1 or step % args.log_steps == 0 or step >= args.steps):
                        reduce_loss_parts(losses.parts, ctx)
                        reduce_loss_parts(latent_stat_parts(out.latent_action), ctx)
                        reduce_scalar(losses.total, ctx)
                    if ctx.is_main and save_steps > 0 and step % save_steps == 0:
                        ckpt_path = output_dir / f"lam_step_{step:07d}.pt"
                        torch.save(checkpoint_payload(model, step), ckpt_path)
                        print(f"saved_checkpoint={ckpt_path}", flush=True)
                    if val_loader is not None and args.val_steps > 0 and step % args.val_steps == 0:
                        val_metrics, val_batches = run_validation(
                            unwrap_model(model), gld, val_loader, device, args.val_batches, cfg.model.type,
                            cfg.loss.lambda_gld,
                            args.amp,
                        )
                        val_metrics, val_batches = reduce_validation_result(val_metrics, val_batches, device, ctx)
                        if ctx.is_main:
                            if cfg.model.type == "lam_3D_pixel":
                                metrics_text = f"pixel_mse={val_metrics['pixel_mse']:.6f}"
                            else:
                                metrics_text = " ".join(
                                    f"{key}={val_metrics[key]:.6f}"
                                    for key in ("gld_f0", "gld_f1", "gld_f01")
                                )
                            print(f"val step={step} {metrics_text} batches={val_batches}", flush=True)
                    if step >= args.steps:
                        break
                epoch += 1
            if ctx.is_main:
                torch.save(checkpoint_payload(model, step), output_dir / "lam_last.pt")
                print(f"saved_checkpoint={output_dir / 'lam_last.pt'}", flush=True)
            distributed_barrier(ctx)
        return 0
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
