#!/usr/bin/env python3
"""Training entrypoint for MyD4RT."""


from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging as stdlib_logging
import os
import platform
import shlex
import socket
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core import apply_overrides, build_logger, load_checkpoint, load_yaml_config, seed_everything
from src.data import build_dataloader
from src.data.builder import _normalize_dataset_type, _resolve_mixture_sources
from src.engine import Trainer
from src.losses import build_loss
from src.model import build_model


def _init_distributed() -> tuple[int, int, int]:
    """Initialize DDP from torchrun env vars. Returns (rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _split_overrides(overrides: list[str] | None) -> tuple[list[str], list[str]]:
    model_overrides: list[str] = []
    train_overrides: list[str] = []
    for item in overrides or []:
        if item.startswith("model."):
            tail = item[len("model.") :]
            if tail.startswith("model."):
                model_overrides.append(tail)
            else:
                model_overrides.append(f"model.{tail}")
        elif item.startswith("train."):
            train_overrides.append(item[len("train.") :])
        else:
            train_overrides.append(item)
    return model_overrides, train_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyD4RT")
    parser.add_argument(
        "--model-config",
        default="configs/model_effective.yaml",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train_effective.yaml",
    )
    parser.add_argument("--train-manifest", default=None, help="Comma-separated manifest paths for train split.")
    parser.add_argument("--val-manifest", default=None, help="Comma-separated manifest paths for val split.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint for resume.")
    parser.add_argument("--init-model", default=None, help="Path to checkpoint used only for model-weight initialization.")
    parser.add_argument(
        "--init-timestep-embed-resize",
        choices=("linear", "nearest", "repeat_last"),
        default="linear",
        help=(
            "How to resize learned query timestep embeddings when --init-model has fewer clip frames "
            "than the current model. Shape-matched checkpoints are loaded unchanged."
        ),
    )
    parser.add_argument("--tb_log", action="store_true", help="Enable TensorBoard scalar logging.")
    parser.add_argument("--override", action="append", default=[], help="Override config by key=value. Use prefix model./train.")
    return parser.parse_args()


def _split_csv_paths(raw: str | None) -> list[Path]:
    if not raw:
        return []
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def _git_commit(cwd: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _collect_repro_file_entries(args: argparse.Namespace, train_cfg: Any) -> list[dict[str, object]]:
    entries: list[Path] = []
    entries.extend(_split_csv_paths(args.train_manifest))
    entries.extend(_split_csv_paths(args.val_manifest))

    split_files = train_cfg.get_path("data.scannet.split_files", {})
    if isinstance(split_files, dict):
        for value in split_files.values():
            entries.append(Path(str(value)))

    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for path in entries:
        p = Path(path)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        info: dict[str, object] = {
            "path": key,
            "exists": p.exists(),
            "is_file": p.is_file(),
        }
        if p.exists() and p.is_file():
            info["size_bytes"] = int(p.stat().st_size)
            info["sha256"] = _sha256_file(p)
        out.append(info)
    return out


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


def _resize_timestep_embedding(
    value: torch.Tensor,
    target_value: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Resize [T, C] learned timestep embeddings along T."""
    if mode not in {"linear", "nearest", "repeat_last"}:
        raise ValueError(f"Unsupported timestep embedding resize mode: {mode!r}")
    src_len = int(value.shape[0])
    dst_len = int(target_value.shape[0])
    if src_len <= 0 or dst_len <= 0:
        return target_value.detach().clone()

    source = value.detach().to(device=target_value.device)
    if mode == "repeat_last" or src_len == 1:
        resized = target_value.detach().clone()
        copy_len = min(src_len, dst_len)
        resized[:copy_len] = source[:copy_len].to(dtype=target_value.dtype)
        if dst_len > copy_len:
            resized[copy_len:] = source[copy_len - 1 : copy_len].to(dtype=target_value.dtype).expand(
                dst_len - copy_len,
                -1,
            )
        return resized

    interp_mode = "nearest" if mode == "nearest" else "linear"
    src = source.to(dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    kwargs: dict[str, object] = {"size": dst_len, "mode": interp_mode}
    if interp_mode == "linear":
        kwargs["align_corners"] = True
    resized = F.interpolate(src, **kwargs).squeeze(0).transpose(0, 1)
    return resized.to(dtype=target_value.dtype)


def _prepare_init_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    timestep_resize_mode: str = "linear",
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Filter init weights and resize learned timestep embeddings when clip length grows."""
    target_state = model.state_dict()
    prepared: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    expanded: list[str] = []
    timestep_keys = {
        "query_embedder.t_src_embed.weight",
        "query_embedder.t_tgt_embed.weight",
        "query_embedder.t_cam_embed.weight",
    }

    for key, value in state_dict.items():
        target_key = key[7:] if key.startswith("module.") else key
        target_value = target_state.get(target_key)
        if target_value is None:
            continue
        if tuple(value.shape) == tuple(target_value.shape):
            prepared[target_key] = value
            continue
        if (
            target_key in timestep_keys
            and value.ndim == 2
            and target_value.ndim == 2
            and int(value.shape[1]) == int(target_value.shape[1])
            and int(value.shape[0]) < int(target_value.shape[0])
        ):
            resized = _resize_timestep_embedding(
                value=value,
                target_value=target_value,
                mode=timestep_resize_mode,
            )
            prepared[target_key] = resized
            expanded.append(
                f"{target_key}:{tuple(value.shape)}->{tuple(target_value.shape)}:{timestep_resize_mode}"
            )
            continue
        skipped.append(f"{target_key}:{tuple(value.shape)}!={tuple(target_value.shape)}")

    return prepared, skipped, expanded


def _align_model_input_with_train_data(model_cfg: Any, train_cfg: Any) -> str | None:
    train_clip_frames = train_cfg.get_path("data.clip_frames", None)
    if train_clip_frames is None:
        return None

    try:
        train_clip_frames_i = int(train_clip_frames)
    except (TypeError, ValueError):
        return None

    model_clip_frames = model_cfg.get_path("model.input.clip_frames", None)
    if model_clip_frames is None:
        model_cfg.set_path("model.input.clip_frames", train_clip_frames_i)
        return (
            "model.input.clip_frames is missing; "
            f"auto-setting it to data.clip_frames={train_clip_frames_i}."
        )

    try:
        model_clip_frames_i = int(model_clip_frames)
    except (TypeError, ValueError):
        model_cfg.set_path("model.input.clip_frames", train_clip_frames_i)
        return (
            f"model.input.clip_frames={model_clip_frames!r} is invalid; "
            f"auto-setting it to data.clip_frames={train_clip_frames_i}."
        )

    if model_clip_frames_i == train_clip_frames_i:
        return None

    model_cfg.set_path("model.input.clip_frames", train_clip_frames_i)
    return (
        "model.input.clip_frames does not match data.clip_frames; "
        f"auto-aligning model clip length from {model_clip_frames_i} to {train_clip_frames_i}."
    )


def _apply_default_per_dataset_validation(train_cfg: Any) -> None:
    raw_val_dataset_type = train_cfg.get_path("data.val_dataset_type", None)
    val_dataset_type = _normalize_dataset_type(str(raw_val_dataset_type or ""))
    if val_dataset_type != "mixture_raw":
        if raw_val_dataset_type is not None:
            return
        try:
            if not _resolve_mixture_sources(split="val", cfg=train_cfg):
                return
        except ValueError:
            return

    cfg = train_cfg.get_path("logging.per_dataset_validation", None)
    if not isinstance(cfg, dict):
        train_cfg.set_path("logging.per_dataset_validation", {})

    if train_cfg.get_path("logging.per_dataset_validation.enabled", None) is None:
        train_cfg.set_path("logging.per_dataset_validation.enabled", True)

    if (
        bool(train_cfg.get_path("logging.per_dataset_validation.enabled", False))
        and train_cfg.get_path("logging.per_dataset_validation.max_samples_global", None) is None
    ):
        train_cfg.set_path("logging.per_dataset_validation.max_samples_global", 512)


def _build_reference_val_loader(
    train_cfg: Any,
    logger: Any,
    rank: int,
    world_size: int,
    manifest_arg: str | None,
) -> Any | None:
    max_samples_global = int(train_cfg["logging"].get("validate_max_samples_global", 0))
    if max_samples_global <= 0:
        return None
    if world_size > 1:
        return None
    if rank != 0:
        return None

    loader = build_dataloader(split="val", cfg=train_cfg, manifest_arg=manifest_arg, rank=0, world_size=1)
    logger.info(
        "Built fixed-sample validation loader: samples=%d dataset=%s batch_size=%s",
        max_samples_global,
        train_cfg.get_path("data.val_dataset_type", None),
        getattr(loader, "batch_size", "unknown"),
    )
    return loader


def _build_per_dataset_val_loaders(train_cfg: Any, logger: Any, rank: int, world_size: int) -> dict[str, Any]:
    cfg = train_cfg.get_path("logging.per_dataset_validation", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {}

    val_dataset_type = _normalize_dataset_type(str(train_cfg.get_path("data.val_dataset_type", "")))
    if val_dataset_type != "mixture_raw":
        if rank == 0:
            logger.info(
                "Skip per-dataset validation loaders because data.val_dataset_type=%s is not mixture_raw",
                val_dataset_type or "<unset>",
            )
        return {}

    fixed_sample_mode = int(cfg.get("max_samples_global", 0)) > 0 and world_size <= 1
    if fixed_sample_mode and rank != 0:
        return {}

    resolved_sources = _resolve_mixture_sources(split="val", cfg=train_cfg)
    loaders: dict[str, Any] = {}
    for source_name, dataset_type in resolved_sources:
        single_cfg = train_cfg.clone()
        single_cfg.set_path("data.val_dataset_type", dataset_type)
        try:
            loader = build_dataloader(
                split="val",
                cfg=single_cfg,
                manifest_arg=None,
                rank=(0 if fixed_sample_mode else rank),
                world_size=(1 if fixed_sample_mode else world_size),
            )
        except Exception as exc:
            warnings.warn(
                f"Skip per-dataset validation source '{source_name}' ({dataset_type}): {exc}",
                stacklevel=2,
            )
            continue
        loaders[str(source_name)] = loader
        if rank == 0:
            logger.info(
                "Built per-dataset validation loader: source=%s dataset_type=%s scenes=%d batch_size=%s",
                source_name,
                dataset_type,
                len(getattr(loader, "dataset", [])),
                getattr(loader, "batch_size", "unknown"),
            )
    return loaders


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = _init_distributed()
    distributed = world_size > 1

    model_cfg = load_yaml_config(args.model_config)
    train_cfg = load_yaml_config(args.train_config)

    model_overrides, train_overrides = _split_overrides(args.override)
    model_cfg = apply_overrides(model_cfg, model_overrides)
    train_cfg = apply_overrides(train_cfg, train_overrides)
    clip_alignment_warning = _align_model_input_with_train_data(model_cfg, train_cfg)
    _apply_default_per_dataset_validation(train_cfg)

    output_dir = Path(train_cfg.get_path("experiment.output_dir", "tmp/experiments/d4rt_train"))

    # Rank-0-only: directory creation, config saving, metadata, logger
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = build_logger("train", output_dir)

        model_cfg_path = output_dir / "config" / "model_effective.yaml"
        train_cfg_path = output_dir / "config" / "train_effective.yaml"
        _write_yaml(model_cfg_path, model_cfg.to_dict())
        _write_yaml(train_cfg_path, train_cfg.to_dict())
        if clip_alignment_warning is not None:
            logger.warning("%s", clip_alignment_warning)
            warnings.warn(clip_alignment_warning, stacklevel=2)

        repro_files = _collect_repro_file_entries(args, train_cfg)
        run_metadata_path = output_dir / "run_metadata.json"
        run_metadata = {
            "status": "initialized",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "git_commit": _git_commit(Path.cwd()),
            "command": " ".join(shlex.quote(part) for part in sys.argv),
            "distributed": {
                "enabled": distributed,
                "world_size": world_size,
                "backend": "nccl" if distributed else None,
            },
            "paths": {
                "cwd": str(Path.cwd()),
                "model_config_input": str(args.model_config),
                "train_config_input": str(args.train_config),
                "model_config_effective": str(model_cfg_path),
                "train_config_effective": str(train_cfg_path),
                "output_dir": str(output_dir),
                "resume": str(args.resume) if args.resume else None,
                "init_model": str(args.init_model) if args.init_model else None,
                "tensorboard_dir": str(output_dir / "tensorboard") if args.tb_log else None,
            },
            "data": {
                "train_dataset_type": train_cfg.get_path("data.train_dataset_type", None),
                "val_dataset_type": train_cfg.get_path("data.val_dataset_type", None),
                "dataset_mixture": train_cfg.get_path("data.dataset_mixture", None),
                "train_dataset_mixture": train_cfg.get_path("data.train_dataset_mixture", None),
                "val_dataset_mixture": train_cfg.get_path("data.val_dataset_mixture", None),
                "mixture_sampling_weights": train_cfg.get_path("data.mixture_sampling_weights", None),
                "clip_frames": train_cfg.get_path("data.clip_frames", None),
                "model_input_clip_frames": model_cfg.get_path("model.input.clip_frames", None),
                "image_size": train_cfg.get_path("data.image_size", None),
                "queries_per_clip": train_cfg.get_path("train_sampling.queries_per_clip", None),
                "clip_frames_auto_aligned": bool(clip_alignment_warning is not None),
            },
            "initialization": {
                "init_model": str(args.init_model) if args.init_model else None,
                "timestep_embed_resize": str(args.init_timestep_embed_resize),
            },
            "repro_files": repro_files,
            "logging": {
                "tensorboard_enabled": bool(args.tb_log),
                "validate_max_batches_global": train_cfg.get_path("logging.validate_max_batches_global", None),
                "validate_max_samples_global": train_cfg.get_path("logging.validate_max_samples_global", None),
                "per_dataset_validation": train_cfg.get_path("logging.per_dataset_validation", None),
            },
        }
        run_metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved effective configs and run metadata to %s", output_dir)
    else:
        logger = stdlib_logging.getLogger("train")
        logger.setLevel(stdlib_logging.WARNING)
        if not logger.handlers:
            logger.addHandler(stdlib_logging.StreamHandler())
        run_metadata_path = output_dir / "run_metadata.json"

    seed = int(train_cfg.get_path("experiment.seed", model_cfg.get_path("experiment.seed", 42)))
    seed_everything(seed + rank, deterministic=False)
    logger.info("Using seed=%d (base=%d, rank=%d)", seed + rank, seed, rank)
    logger.info("Output dir: %s", output_dir)
    if distributed:
        logger.info("DDP: rank=%d local_rank=%d world_size=%d", rank, local_rank, world_size)

    train_loader = build_dataloader(split="train", cfg=train_cfg, manifest_arg=args.train_manifest,
                                     rank=rank, world_size=world_size)
    val_loader = build_dataloader(split="val", cfg=train_cfg, manifest_arg=args.val_manifest,
                                   rank=rank, world_size=world_size)
    reference_val_loader = _build_reference_val_loader(
        train_cfg=train_cfg,
        logger=logger,
        rank=rank,
        world_size=world_size,
        manifest_arg=args.val_manifest,
    )
    per_dataset_val_loaders = _build_per_dataset_val_loaders(
        train_cfg=train_cfg,
        logger=logger,
        rank=rank,
        world_size=world_size,
    )
    model = build_model(model_cfg["model"])
    loss_fn = build_loss(model_name=model_cfg["model"].get("name", "d4rt"), train_cfg=train_cfg)

    if args.init_model:
        payload = load_checkpoint(args.init_model, map_location="cpu")
        state_dict = _unwrap_state_dict(payload)
        if not state_dict:
            raise RuntimeError(f"No model weights found in checkpoint: {args.init_model}")
        init_state_dict, skipped_shape, expanded_shape = _prepare_init_state_dict(
            model,
            state_dict,
            timestep_resize_mode=str(args.init_timestep_embed_resize),
        )
        missing, unexpected = model.load_state_dict(init_state_dict, strict=False)
        logger.info(
            "Initialized model weights from %s (loaded=%d missing=%d unexpected=%d expanded=%d skipped_shape=%d)",
            args.init_model,
            len(init_state_dict),
            len(missing),
            len(unexpected),
            len(expanded_shape),
            len(skipped_shape),
        )
        if expanded_shape:
            logger.info("Expanded init tensors: %s", ", ".join(expanded_shape))
        if skipped_shape:
            logger.warning("Skipped init tensors with incompatible shapes: %s", "; ".join(skipped_shape[:20]))

    use_data_parallel = bool(train_cfg.get_path("runtime.data_parallel", False))
    visible_gpus = int(torch.cuda.device_count())

    if distributed:
        device = torch.device("cuda", local_rank)
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        logger.info("DDP enabled: rank=%d/%d local_rank=%d", rank, world_size, local_rank)
    elif use_data_parallel and visible_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(visible_gpus)))
        logger.info("DataParallel enabled with %d GPUs: %s", visible_gpus, list(range(visible_gpus)))
    elif use_data_parallel and visible_gpus <= 1:
        logger.warning("runtime.data_parallel=true but only %d GPU visible; fallback to single device.", visible_gpus)

    if rank == 0:
        run_metadata_update = {}
        if run_metadata_path.exists():
            run_metadata_update = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        run_metadata_update.setdefault("runtime", {})
        run_metadata_update["runtime"]["data_parallel_enabled"] = bool(use_data_parallel and visible_gpus > 1 and not distributed)
        run_metadata_update["runtime"]["ddp_enabled"] = distributed
        run_metadata_update["runtime"]["visible_gpu_count"] = visible_gpus
        run_metadata_update["runtime"]["world_size"] = world_size
        run_metadata_path.write_text(json.dumps(run_metadata_update, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tb_writer = None
    if args.tb_log and rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise RuntimeError(
                "Failed to enable --tb_log because TensorBoard dependencies are missing. "
                "Install `tensorboard` in the active environment."
            ) from exc
        tb_dir = output_dir / "tensorboard"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info("TensorBoard logging enabled: %s", tb_dir)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        reference_val_loader=reference_val_loader,
        per_dataset_val_loaders=per_dataset_val_loaders,
        train_cfg=train_cfg,
        output_dir=output_dir,
        logger=logger,
        run_metadata_path=run_metadata_path,
        tb_writer=tb_writer,
        rank=rank,
        world_size=world_size,
    )
    trainer.resume(args.resume)
    trainer.train()
    logger.info("Training finished.")

    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(record(main)())
