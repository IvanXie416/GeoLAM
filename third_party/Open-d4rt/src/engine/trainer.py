"""Trainer implementation for D4RT."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from src.core.checkpoint import load_checkpoint, save_checkpoint
from src.core.logging import MetricLogger
from src.data.seeding import set_dataset_epoch_recursive
from src.vis import QueryPredictionVisualizer


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    return value


def _amp_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _debug_scalar(value: Any) -> str:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return "tensor(empty)"
        item = value.reshape(-1)[0].detach().cpu().item()
        return str(item)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return _debug_scalar(value[0])
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _batch_size(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, dict):
        video = value.get("video")
        if torch.is_tensor(video) and video.ndim > 0:
            return int(video.shape[0])
        for child in value.values():
            size = _batch_size(child)
            if size > 0:
                return size
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _slice_batch(value: Any, limit: int) -> Any:
    take = max(0, int(limit))
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        return value[:take]
    if isinstance(value, dict):
        return {k: _slice_batch(v, take) for k, v in value.items()}
    if isinstance(value, list):
        return [_slice_batch(v, take) for v in value[:take]]
    if isinstance(value, tuple):
        return tuple(_slice_batch(v, take) for v in value[:take])
    return value


def _format_stage_times(stage_times: dict[str, float]) -> str:
    ordered = []
    for key, value in stage_times.items():
        ordered.append(f"{key}={value:.3f}s")
    return " ".join(ordered)


def _metric_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value).strip().lower()).strip("_")
    return token or "unknown"


def _best_mode_for_metric(metric_name: str, raw_mode: str) -> str:
    mode = str(raw_mode).strip().lower()
    if mode in {"min", "max"}:
        return mode

    key = str(metric_name).strip().lower()
    if any(token in key for token in ("loss", "l1", "absrel", "ate", "rpe")):
        return "min"
    if any(token in key for token in ("aj", "apd", "oa", "pck", "auc", "acc", "accuracy", "f1")):
        return "max"
    return "min"


def _bool_cfg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, peak_lr: float, final_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.step_id = 0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _lr_at_step(self, step_id: int) -> float:
        if step_id <= self.warmup_steps:
            ratio = step_id / float(self.warmup_steps)
            return self.peak_lr * ratio
        progress = (step_id - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr + (self.peak_lr - self.final_lr) * cosine

    def step(self) -> None:
        self.step_id += 1
        lr = self._lr_at_step(self.step_id)
        self._set_lr(lr)

    def state_dict(self) -> dict[str, Any]:
        return {"step_id": self.step_id}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_id = int(state.get("step_id", 0))
        self._set_lr(self._lr_at_step(self.step_id))


def _infinite(loader: Iterable[Any]) -> Iterable[Any]:
    sampler = getattr(loader, "dist_sampler", None)
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            set_dataset_epoch_recursive(dataset, epoch)
        for batch in loader:
            yield batch
        epoch += 1


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_loader: Any,
        val_loader: Any | None,
        reference_val_loader: Any | None,
        per_dataset_val_loaders: dict[str, Any] | None,
        train_cfg: Any,
        output_dir: Path,
        logger: Any,
        run_metadata_path: Path | None = None,
        tb_writer: Any | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.reference_val_loader = reference_val_loader
        self.per_dataset_val_loaders = per_dataset_val_loaders or {}
        self.train_cfg = train_cfg
        self.output_dir = output_dir
        self.logger = logger
        self.run_metadata_path = Path(run_metadata_path) if run_metadata_path is not None else None
        self.tb_writer = tb_writer
        self.rank = rank
        self.world_size = world_size
        self.distributed = world_size > 1

        if self.rank == 0:
            self.metric_logger = MetricLogger(output_dir / "metrics.jsonl")
            self.visualizer = QueryPredictionVisualizer(
                train_cfg=train_cfg,
                output_dir=output_dir,
                tb_writer=tb_writer,
                logger=logger,
            )
        else:
            self.metric_logger = None
            self.visualizer = None

        if self.distributed:
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        optim_cfg = train_cfg["optimizer"]
        lr_cfg = optim_cfg["learning_rate"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(lr_cfg.get("peak_lr", 1e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.03)),
        )

        total_steps = int(train_cfg["schedule"]["total_steps"])
        local_override = train_cfg["schedule"].get("local_repro_override", {})
        if bool(local_override.get("enabled", False)):
            total_steps = int(local_override.get("total_steps", total_steps))
        self.total_steps = total_steps

        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=int(lr_cfg.get("warmup_steps", 2500)),
            total_steps=self.total_steps,
            peak_lr=float(lr_cfg.get("peak_lr", 1e-4)),
            final_lr=float(lr_cfg.get("final_lr", 1e-6)),
        )
        self.grad_clip = float(optim_cfg.get("gradient_clip_l2_norm", 0.0))
        self.use_amp = bool(train_cfg.get_path("runtime.mixed_precision", True))
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        self.amp_skip_abort_after = int(train_cfg.get_path("runtime.amp_skip_abort_after", 25))
        self.amp_consecutive_skips = 0

        self.log_every = int(train_cfg["logging"].get("log_every_steps", 50))
        self.validate_every = int(train_cfg["logging"].get("validate_every_steps", 2000))
        self.validate_max_batches_global = int(
            train_cfg["logging"].get(
                "validate_max_batches_global",
                train_cfg["logging"].get("validate_max_batches", 16),
            )
        )
        self.validate_max_samples_global = int(train_cfg["logging"].get("validate_max_samples_global", 0))
        per_dataset_val_cfg = train_cfg.get_path("logging.per_dataset_validation", {})
        if not isinstance(per_dataset_val_cfg, dict):
            per_dataset_val_cfg = {}
        self.per_dataset_validate_max_batches_global = int(
            per_dataset_val_cfg.get("max_batches_global", self.validate_max_batches_global)
        )
        self.per_dataset_validate_max_samples_global = int(
            per_dataset_val_cfg.get("max_samples_global", 0)
        )
        checkpoint_cfg = train_cfg["checkpoint"]
        self.save_every = int(checkpoint_cfg.get("save_every_steps", 5000))
        self.step_save_every = int(checkpoint_cfg.get("step_save_every_steps", 5000))
        auto_eval_cfg = checkpoint_cfg.get("auto_eval_worldtrack_step", {})
        if not isinstance(auto_eval_cfg, dict):
            auto_eval_cfg = {}
        self.auto_eval_worldtrack_enabled = _bool_cfg(auto_eval_cfg.get("enabled", True), default=True)
        self.auto_eval_worldtrack_num_frames = int(auto_eval_cfg.get("num_frames", 64))
        self.auto_eval_worldtrack_script = str(
            auto_eval_cfg.get(
                "script",
                "scripts/eval_worldtrack/run_batch_eval_worldtrack_step_ckpts.sh",
            )
        )
        self.auto_eval_worldtrack_log_dir_name = str(
            auto_eval_cfg.get("log_dir_name", "eval_worldtrack_step_auto_logs")
        )
        self.auto_eval_worldtrack_extra_env = auto_eval_cfg.get("env", {})
        if not isinstance(self.auto_eval_worldtrack_extra_env, dict):
            self.auto_eval_worldtrack_extra_env = {}
        self._worldtrack_eval_processes: list[tuple[int, subprocess.Popen[Any], Path]] = []
        self.global_step = 0
        self.best_metric_name = str(train_cfg.get_path("checkpoint.keep_best_by", "val_loss_total"))
        self.best_metric_mode = _best_mode_for_metric(
            metric_name=self.best_metric_name,
            raw_mode=str(train_cfg.get_path("checkpoint.keep_best_mode", "auto")),
        )
        self.best_metric_value = float("inf") if self.best_metric_mode == "min" else float("-inf")
        self.best_val = float("inf")
        self.fail_on_non_finite = bool(train_cfg.get_path("runtime.fail_on_non_finite", True))
        vis_cfg = train_cfg.get_path("logging.visualization", {})
        if not isinstance(vis_cfg, dict):
            vis_cfg = {}
        self.train_vis_enabled = bool(vis_cfg.get("enabled", self.tb_writer is not None))
        self.train_vis_every = int(vis_cfg.get("train_every_steps", 0))
        self.validation_round = 0
        self.slow_stage_warn_seconds = float(train_cfg.get_path("runtime.debug_slow_stage_seconds", 10.0))
        self.stage_time_log_every = int(train_cfg.get_path("runtime.debug_stage_time_log_every_steps", self.log_every))

        self._query_total = 0
        self._query_tgt_eq_cam_total = 0
        self._query_hard_total = 0
        self._query_hard_known_total = 0
        self._mask_totals: dict[str, dict[str, float]] = {
            "xyz_3d": {"valid": 0.0, "total": 0.0},
            "uv_2d": {"valid": 0.0, "total": 0.0},
            "visibility": {"valid": 0.0, "total": 0.0},
            "displacement": {"valid": 0.0, "total": 0.0},
            "normal": {"valid": 0.0, "total": 0.0},
        }

    def _batch_meta_summary(self, batch: Any) -> str:
        if not isinstance(batch, dict):
            return "meta=unavailable"
        meta = batch.get("meta")
        if not isinstance(meta, dict):
            return "meta=unavailable"

        fields: list[str] = []
        for key in ("dataset", "scene_id", "source_mode", "sample_key"):
            if key not in meta:
                continue
            try:
                fields.append(f"{key}={_debug_scalar(meta[key])}")
            except Exception:
                fields.append(f"{key}=<unavailable>")
        return " ".join(fields) if fields else "meta=empty"

    def _log_stage_times(
        self,
        step: int,
        stage_times: dict[str, float],
        *,
        batch: Any | None = None,
        batch_id: int | None = None,
        force: bool = False,
        prefix: str = "timing",
    ) -> None:
        if not stage_times:
            return
        should_log = force
        if self.stage_time_log_every > 0 and (step % self.stage_time_log_every == 0):
            should_log = True
        if any(value >= self.slow_stage_warn_seconds for value in stage_times.values()):
            should_log = True
        if not should_log:
            return
        base = f"[rank {self.rank}] step={step}"
        if batch_id is not None:
            base += f" batch_id={batch_id}"
        message = f"{base} {prefix} {_format_stage_times(stage_times)}"
        if batch is not None:
            message += f" {self._batch_meta_summary(batch)}"
        print(message, flush=True)

    def _model_module(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _checkpoint_payload(self) -> dict[str, Any]:
        base_model = self._model_module()
        return {
            "model": base_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_val": self.best_val,
            "best_metric_name": self.best_metric_name,
            "best_metric_mode": self.best_metric_mode,
            "best_metric_value": self.best_metric_value,
            "train_config": self.train_cfg.to_dict() if hasattr(self.train_cfg, "to_dict") else dict(self.train_cfg),
        }

    def save_last_checkpoint(self) -> None:
        if self.rank != 0:
            return
        save_checkpoint(self.output_dir / "checkpoints" / "last.ckpt", self._checkpoint_payload())

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _poll_worldtrack_eval_processes(self) -> None:
        if self.rank != 0 or not self._worldtrack_eval_processes:
            return
        running: list[tuple[int, subprocess.Popen[Any], Path]] = []
        for step, proc, log_path in self._worldtrack_eval_processes:
            returncode = proc.poll()
            if returncode is None:
                running.append((step, proc, log_path))
                continue
            if returncode == 0:
                self.logger.info("WorldTrack auto-eval finished: step=%d log=%s", step, log_path)
            else:
                self.logger.error(
                    "WorldTrack auto-eval failed: step=%d exit_code=%d log=%s",
                    step,
                    returncode,
                    log_path,
                )
        self._worldtrack_eval_processes = running

    def _display_worldtrack_eval_command(self, env: dict[str, str], script_arg: str) -> str:
        keys = ("MIN_STEP", "MAX_STEP", "NUM_FRAMES", "CHECKPOINT_DIR")
        parts = [f"{key}={shlex.quote(env[key])}" for key in keys if key in env]
        parts.append(f"bash {shlex.quote(script_arg)}")
        return " ".join(parts)

    def _launch_worldtrack_step_eval(self, step_ckpt: Path) -> None:
        if self.rank != 0 or not self.auto_eval_worldtrack_enabled:
            return
        self._poll_worldtrack_eval_processes()

        repo_root = self._repo_root()
        script_path = Path(self.auto_eval_worldtrack_script)
        if not script_path.is_absolute():
            script_path = repo_root / script_path
        if not script_path.exists():
            self.logger.error("WorldTrack auto-eval script not found: %s", script_path)
            return

        try:
            script_arg = str(script_path.relative_to(repo_root))
        except ValueError:
            script_arg = str(script_path)

        step = int(self.global_step)
        checkpoint_dir = step_ckpt.parent
        log_dir = self.output_dir / self.auto_eval_worldtrack_log_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"step_{step:07d}.log"
        report_path = log_dir / f"step_{step:07d}_report.json"

        env = os.environ.copy()
        env.update(
            {
                "MIN_STEP": str(step),
                "MAX_STEP": str(step + 1),
                "NUM_FRAMES": str(self.auto_eval_worldtrack_num_frames),
                "CHECKPOINT_DIR": str(checkpoint_dir),
                "TENSORBOARD_LOGDIR": str(self.output_dir / "tensorboard"),
                "REPORT_PATH": str(report_path),
            }
        )
        for key, value in self.auto_eval_worldtrack_extra_env.items():
            env[str(key)] = str(value)

        display_command = self._display_worldtrack_eval_command(env, script_arg)
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"[auto_eval] step={step}\n")
                log_handle.write(f"[auto_eval] command: {display_command}\n")
                log_handle.flush()
                proc = subprocess.Popen(
                    ["bash", script_arg],
                    cwd=str(repo_root),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except Exception as exc:
            self.logger.error("Failed to launch WorldTrack auto-eval for step=%d: %s", step, exc)
            return

        self._worldtrack_eval_processes.append((step, proc, log_path))
        self.logger.info(
            "Launched WorldTrack auto-eval: step=%d pid=%d log=%s",
            step,
            proc.pid,
            log_path,
        )

    def maybe_save_step_checkpoint(self) -> None:
        if self.global_step % self.save_every != 0:
            return
        if self.distributed:
            dist.barrier()
        if self.rank == 0:
            self.save_last_checkpoint()
            if self.step_save_every > 0 and (self.global_step % self.step_save_every == 0):
                step_ckpt = self.output_dir / "checkpoints" / f"step_{self.global_step:07d}.ckpt"
                save_checkpoint(step_ckpt, self._checkpoint_payload())
                self._launch_worldtrack_step_eval(step_ckpt)
        if self.distributed:
            dist.barrier()

    def maybe_save_best(self, metrics: dict[str, float]) -> None:
        if self.rank != 0:
            return
        candidate = _safe_float(metrics.get(self.best_metric_name))
        if candidate is None:
            return
        is_better = candidate < self.best_metric_value if self.best_metric_mode == "min" else candidate > self.best_metric_value
        if not is_better:
            return
        self.best_metric_value = candidate
        val_loss = _safe_float(metrics.get("val_loss_total"))
        if val_loss is not None:
            self.best_val = val_loss
        save_checkpoint(self.output_dir / "checkpoints" / "best.ckpt", self._checkpoint_payload())
        self.logger.info(
            "New best checkpoint: %s=%.6f mode=%s at step=%d",
            self.best_metric_name,
            candidate,
            self.best_metric_mode,
            self.global_step,
        )

    def resume(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        payload = load_checkpoint(ckpt_path, map_location=self.device)
        self._model_module().load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload.get("scheduler", {}))
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload.get("global_step", 0))
        self.best_val = float(payload.get("best_val", self.best_val))
        self.best_metric_name = str(payload.get("best_metric_name", self.best_metric_name))
        self.best_metric_mode = _best_mode_for_metric(
            metric_name=self.best_metric_name,
            raw_mode=str(payload.get("best_metric_mode", self.best_metric_mode)),
        )
        if "best_metric_value" in payload:
            self.best_metric_value = float(payload.get("best_metric_value", self.best_metric_value))
        elif self.best_metric_name == "val_loss_total":
            self.best_metric_value = float(payload.get("best_val", self.best_metric_value))
        self.logger.info("Resumed from %s at step %d", ckpt_path, self.global_step)
        self._update_run_metadata(
            {
                "training_state": {
                    "resumed_from": str(ckpt_path),
                    "resume_step": self.global_step,
                }
            }
        )

    def _merge_dict_inplace(self, target: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_dict_inplace(target[key], value)
            else:
                target[key] = value

    def _update_run_metadata(self, update: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        if self.run_metadata_path is None:
            return
        payload: dict[str, Any] = {}
        if self.run_metadata_path.exists():
            try:
                payload = json.loads(self.run_metadata_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        self._merge_dict_inplace(payload, update)
        self.run_metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _collect_batch_supervision_stats(self, batch: dict[str, Any]) -> dict[str, float]:
        stats: dict[str, float] = {}

        query = batch.get("query", {})
        t_tgt = query.get("t_tgt") if isinstance(query, dict) else None
        t_cam = query.get("t_cam") if isinstance(query, dict) else None
        if torch.is_tensor(t_tgt) and torch.is_tensor(t_cam):
            eq = (t_tgt == t_cam)
            stats["query_prob_t_tgt_eq_t_cam"] = float(eq.float().mean().item())
            self._query_total += int(eq.numel())
            self._query_tgt_eq_cam_total += int(eq.sum().item())

        query_stats = batch.get("query_stats", {})
        is_hard = query_stats.get("is_hard_query") if isinstance(query_stats, dict) else None
        if torch.is_tensor(is_hard):
            hard_bool = is_hard.bool()
            stats["query_hard_ratio"] = float(hard_bool.float().mean().item())
            self._query_hard_known_total += int(hard_bool.numel())
            self._query_hard_total += int(hard_bool.sum().item())

        mask = batch.get("mask", {})
        if isinstance(mask, dict):
            for key in self._mask_totals:
                m = mask.get(key)
                if not torch.is_tensor(m):
                    continue
                m_float = m.float()
                stats[f"mask_coverage_{key}"] = float(m_float.mean().item())
                self._mask_totals[key]["valid"] += float(m_float.sum().item())
                self._mask_totals[key]["total"] += float(m_float.numel())

        running = self._running_supervision_stats()
        for key, value in running.items():
            if isinstance(value, float):
                stats[f"{key}_running"] = value
        return stats

    def _running_supervision_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._query_total > 0:
            out["query_prob_t_tgt_eq_t_cam"] = float(self._query_tgt_eq_cam_total) / float(self._query_total)
        if self._query_hard_known_total > 0:
            out["query_hard_ratio"] = float(self._query_hard_total) / float(self._query_hard_known_total)
        for key, accum in self._mask_totals.items():
            total = float(accum["total"])
            if total > 0.0:
                out[f"mask_coverage_{key}"] = float(accum["valid"]) / total
        return out

    def _detect_non_finite(self, loss: torch.Tensor, metrics: dict[str, Any]) -> str | None:
        if not torch.isfinite(loss).all():
            return "loss_total"
        for key, value in metrics.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return key
        return None

    def _sync_non_finite_status(self, bad_key: str | None) -> tuple[bool, int | None, str | None]:
        local_bad = bad_key is not None
        if not self.distributed:
            return local_bad, (self.rank if local_bad else None), bad_key

        bad_flag = torch.tensor([1 if local_bad else 0], device=self.device, dtype=torch.int32)
        dist.all_reduce(bad_flag, op=dist.ReduceOp.SUM)
        if int(bad_flag.item()) == 0:
            return False, None, None

        bad_rank = torch.tensor(
            [self.rank if local_bad else self.world_size],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(bad_rank, op=dist.ReduceOp.MIN)

        gathered_keys: list[str | None] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_keys, bad_key)
        first_bad_key = next((item for item in gathered_keys if item is not None), None)
        return True, int(bad_rank.item()), first_bad_key

    def _scalarize_metric_dict(self, metrics: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                out[key] = float(value.item())
            else:
                out[key] = float(value)
        return out

    def _distributed_mean_scalar_dict(self, payload: dict[str, float]) -> dict[str, float]:
        if not self.distributed or not payload:
            return payload
        keys = sorted(payload.keys())
        values = torch.tensor([float(payload[key]) for key in keys], device=self.device, dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(self.world_size)
        return {key: float(val.item()) for key, val in zip(keys, values)}

    def _tb_add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.tb_writer is None:
            return
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        self.tb_writer.add_scalar(tag, scalar, step)

    def _tb_log_train_payload(self, payload: dict[str, Any]) -> None:
        if self.tb_writer is None:
            return
        step = int(payload.get("step", self.global_step))
        for key, value in payload.items():
            if key == "step":
                continue
            self._tb_add_scalar(f"train/{key}", value, step)

    def _validate(
        self,
        loader: Any | None = None,
        max_batches_global: int | None = None,
        max_samples_global: int | None = None,
        sample_weighted: bool = False,
        eval_model: torch.nn.Module | None = None,
        visualize: bool = False,
    ) -> tuple[float, int, dict[str, float]]:
        if loader is None:
            loader = self.val_loader
        if loader is None:
            return 0.0, 0, {}

        if max_batches_global is None:
            max_batches_global = self.validate_max_batches_global
        max_batches_global = max(0, int(max_batches_global))
        if max_samples_global is None:
            max_samples_global = 0
        max_samples_global = max(0, int(max_samples_global))
        if max_batches_global <= 0 and max_samples_global <= 0:
            return 0.0, 0, {}

        if eval_model is None:
            eval_model = self.model
        eval_model.eval()
        total = 0.0
        count = 0
        metric_sums: dict[str, float] = {}
        vis_done = False
        samples_seen = 0
        try:
            with torch.no_grad():
                for batch_id, batch in enumerate(loader):
                    val_stage_times: dict[str, float] = {}
                    global_batch_id = batch_id
                    if self.distributed:
                        global_batch_id = batch_id * self.world_size + self.rank
                    if max_batches_global > 0 and global_batch_id >= max_batches_global:
                        break
                    batch_size = _batch_size(batch)
                    if batch_size <= 0:
                        continue
                    if max_samples_global > 0:
                        remaining = max_samples_global - samples_seen
                        if remaining <= 0:
                            break
                        if batch_size > remaining:
                            batch = _slice_batch(batch, remaining)
                            batch_size = remaining
                    t_stage = time.perf_counter()
                    batch = _to_device(batch, self.device)
                    val_stage_times["val_to_device"] = time.perf_counter() - t_stage
                    t_stage = time.perf_counter()
                    outputs = eval_model(batch)
                    val_stage_times["val_forward"] = time.perf_counter() - t_stage
                    t_stage = time.perf_counter()
                    loss, metrics = self.loss_fn(outputs, batch)
                    val_stage_times["val_loss"] = time.perf_counter() - t_stage
                    if sample_weighted:
                        total += float(loss.item()) * float(batch_size)
                        count += int(batch_size)
                    else:
                        total += float(loss.item())
                        count += 1
                    for key, value in self._scalarize_metric_dict(metrics).items():
                        if sample_weighted:
                            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * float(batch_size)
                        else:
                            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                    samples_seen += int(batch_size)
                    if visualize and (not vis_done):
                        t_stage = time.perf_counter()
                        self.visualizer.log_batch(batch=batch, outputs=outputs, split="val", step=self.global_step)
                        val_stage_times["val_visualization"] = time.perf_counter() - t_stage
                        vis_done = True
                    self._log_stage_times(
                        self.global_step,
                        val_stage_times,
                        batch=batch,
                        batch_id=batch_id,
                        prefix="val_timing",
                    )
        finally:
            eval_model.train()
        return total, count, metric_sums

    def _global_batches_for_sample_limit(self, loader: Any | None, max_samples_global: int) -> int:
        if max_samples_global <= 0:
            return 0
        batch_size = int(getattr(loader, "batch_size", 0) or 0)
        if batch_size <= 0:
            batch_size = int(self.train_cfg.get_path("runtime.val_batch_size", self.train_cfg.get_path("runtime.batch_size", 1)))
        batch_size = max(1, batch_size)
        return max(1, math.ceil(int(max_samples_global) / batch_size))

    def _reduce_validation_stats(
        self,
        val_loss_sum: float,
        val_count: int,
        val_metric_sums: dict[str, float],
    ) -> tuple[float, int, dict[str, float]]:
        if self.distributed:
            val_stats = torch.tensor([val_loss_sum, float(val_count)], device=self.device, dtype=torch.float64)
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            val_loss_sum = float(val_stats[0].item())
            val_count = int(round(float(val_stats[1].item())))
            gathered_metric_keys: list[list[str]] = [[] for _ in range(self.world_size)]
            dist.all_gather_object(gathered_metric_keys, sorted(val_metric_sums.keys()))
            metric_keys = sorted({key for keys in gathered_metric_keys for key in keys})
            if metric_keys:
                metric_values = torch.tensor(
                    [float(val_metric_sums.get(key, 0.0)) for key in metric_keys],
                    device=self.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
                val_metric_sums = {
                    key: float(value.item())
                    for key, value in zip(metric_keys, metric_values)
                }
        return val_loss_sum, val_count, val_metric_sums

    def _finalize_validation_metrics(
        self,
        val_loss_sum: float,
        val_count: int,
        val_metric_sums: dict[str, float],
    ) -> dict[str, float]:
        val_loss = val_loss_sum / max(1, val_count)
        val_metrics = {"val_loss_total": float(val_loss)}
        for key, value in val_metric_sums.items():
            if key == "loss_total":
                continue
            val_metrics[f"val_{key}"] = float(value) / max(1, val_count)
        return val_metrics

    def _validate_per_dataset(self) -> dict[str, dict[str, float]]:
        if not self.per_dataset_val_loaders:
            return {}
        max_batches_global = max(0, int(self.per_dataset_validate_max_batches_global))
        max_samples_global = max(0, int(self.per_dataset_validate_max_samples_global))
        if max_batches_global <= 0 and max_samples_global <= 0:
            return {}

        results: dict[str, dict[str, float]] = {}
        if max_samples_global > 0 and self.rank != 0:
            if not self.distributed:
                return results
        for source_name, loader in self.per_dataset_val_loaders.items():
            if max_samples_global > 0 and self.distributed:
                effective_max_batches_global = self._global_batches_for_sample_limit(loader, max_samples_global)
                effective_max_samples_global = 0
            else:
                effective_max_batches_global = 0 if max_samples_global > 0 else max_batches_global
                effective_max_samples_global = max_samples_global
            val_loss_sum, val_count, val_metric_sums = self._validate(
                loader=loader,
                max_batches_global=effective_max_batches_global,
                max_samples_global=effective_max_samples_global,
                sample_weighted=(max_samples_global > 0),
                eval_model=(self._model_module() if max_samples_global > 0 else None),
                visualize=False,
            )
            if max_samples_global <= 0 or self.distributed:
                val_loss_sum, val_count, val_metric_sums = self._reduce_validation_stats(
                    val_loss_sum=val_loss_sum,
                    val_count=val_count,
                    val_metric_sums=val_metric_sums,
                )
            results[str(source_name)] = self._finalize_validation_metrics(
                val_loss_sum=val_loss_sum,
                val_count=val_count,
                val_metric_sums=val_metric_sums,
            )
        return results

    def _flatten_per_dataset_metrics(self, metrics_by_source: dict[str, dict[str, float]]) -> dict[str, float]:
        flat: dict[str, float] = {}
        for source_name, metrics in metrics_by_source.items():
            source_token = _metric_token(source_name)
            for key, value in metrics.items():
                metric_name = key[4:] if key.startswith("val_") else key
                flat[f"val_dataset_{source_token}_{metric_name}"] = float(value)
        return flat

    def _tb_log_per_dataset_metrics(self, metrics_by_source: dict[str, dict[str, float]]) -> None:
        if self.rank != 0:
            return
        for source_name, metrics in metrics_by_source.items():
            source_token = _metric_token(source_name)
            for key, value in metrics.items():
                metric_name = key[4:] if key.startswith("val_") else key
                self._tb_add_scalar(
                    f"val_by_dataset/{source_token}/{metric_name.replace('_', '/')}",
                    value,
                    self.global_step,
                )

    def train(self) -> None:
        self.model.train()
        train_iter = _infinite(self.train_loader)
        start = time.time()
        self._update_run_metadata(
            {
                "status": "running",
                "training_state": {
                    "start_step": self.global_step,
                    "target_total_steps": self.total_steps,
                },
            }
        )

        while self.global_step < self.total_steps:
            self.global_step += 1
            step = self.global_step
            stage_times: dict[str, float] = {}

            t_stage = time.perf_counter()
            cpu_batch = next(train_iter)
            stage_times["next_train_iter"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()
            batch = _to_device(cpu_batch, self.device)
            stage_times["to_device"] = time.perf_counter() - t_stage

            self.optimizer.zero_grad(set_to_none=True)
            t_stage = time.perf_counter()
            with autocast(enabled=self.use_amp and self.device.type == "cuda"):
                outputs = self.model(batch)
                stage_times["forward"] = time.perf_counter() - t_stage
                t_stage = time.perf_counter()
                loss, metrics = self.loss_fn(outputs, batch)
                stage_times["loss_fn"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()
            stat_metrics = self._collect_batch_supervision_stats(batch)
            stage_times["collect_stats"] = time.perf_counter() - t_stage
            bad_key = self._detect_non_finite(loss, metrics)
            bad_found, bad_rank, synced_bad_key = self._sync_non_finite_status(bad_key)
            if bad_found:
                self._update_run_metadata(
                    {
                        "status": "failed_non_finite",
                        "training_summary": {
                            "failed_step": self.global_step,
                            "failed_metric": synced_bad_key,
                            "failed_rank": bad_rank,
                            "running_supervision_stats": self._running_supervision_stats(),
                        },
                    }
                )
                if self.fail_on_non_finite:
                    raise FloatingPointError(
                        f"Non-finite value detected at step {self.global_step}: "
                        f"rank={bad_rank} key={synced_bad_key}"
                    )
                if self.rank == 0:
                    self.logger.error(
                        "Non-finite metric detected at step=%d rank=%s key=%s",
                        self.global_step,
                        bad_rank,
                        synced_bad_key,
                    )
                continue

            t_stage = time.perf_counter()
            self.scaler.scale(loss).backward()
            stage_times["backward"] = time.perf_counter() - t_stage
            if self.grad_clip > 0:
                t_stage = time.perf_counter()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                stage_times["grad_clip"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()
            scaler_scale_before = float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scaler_scale_after = float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0
            if self.scaler.is_enabled() and scaler_scale_after < scaler_scale_before:
                self.amp_consecutive_skips += 1
                message = (
                    f"AMP skipped optimizer step at train step {self.global_step}: "
                    f"scale {scaler_scale_before:g} -> {scaler_scale_after:g}, "
                    f"consecutive_skips={self.amp_consecutive_skips}"
                )
                if self.rank == 0:
                    self.logger.warning(message)
                if self.amp_consecutive_skips >= self.amp_skip_abort_after or scaler_scale_after <= 0.0:
                    raise FloatingPointError(message)
            else:
                self.amp_consecutive_skips = 0
            self.scheduler.step()
            stage_times["optimizer_step"] = time.perf_counter() - t_stage
            self._log_stage_times(step, stage_times, batch=batch)

            if self.global_step % self.log_every == 0:
                reduced_metrics = self._distributed_mean_scalar_dict(self._scalarize_metric_dict(metrics))
                if self.rank == 0:
                    elapsed = max(1e-6, time.time() - start)
                    steps_per_sec = self.global_step / elapsed
                    payload = {
                        "step": self.global_step,
                        "lr": self.scheduler.get_lr(),
                        "steps_per_sec": steps_per_sec,
                    }
                    for key, value in reduced_metrics.items():
                        payload[key] = float(value)
                    for key, value in stat_metrics.items():
                        payload[key] = float(value)
                    if self.metric_logger is not None:
                        self.metric_logger.log(payload)
                    self._tb_log_train_payload(payload)
                    self.logger.info(
                        "step=%d loss=%.6f lr=%.3e step/s=%.2f hard=%.3f p(t_tgt=t_cam)=%.3f",
                        self.global_step,
                        payload["loss_total"],
                        payload["lr"],
                        payload["steps_per_sec"],
                        payload.get("query_hard_ratio", float("nan")),
                        payload.get("query_prob_t_tgt_eq_t_cam", float("nan")),
                    )

            train_vis_due = (
                self.train_vis_enabled
                and self.train_vis_every > 0
                and (self.global_step % self.train_vis_every == 0)
            )
            
            if train_vis_due and self.distributed:
                dist.barrier()
            if (
                train_vis_due
                and self.rank == 0
                and self.visualizer is not None
                and self.visualizer.should_log_train(self.global_step)
            ):
                self.visualizer.log_batch(batch=batch, outputs=outputs, split="train", step=self.global_step)
            if train_vis_due and self.distributed:
                dist.barrier()
                
            if self.global_step % self.validate_every == 0 and self.val_loader is not None:
                val_vis = (
                    self.rank == 0
                    and self.visualizer is not None
                    and self.visualizer.should_log_val(self.global_step)
                )
                if self.validate_max_samples_global > 0:
                    val_loss_sum = 0.0
                    val_count = 0
                    val_metric_sums: dict[str, float] = {}
                    if self.distributed:
                        val_loss_sum, val_count, val_metric_sums = self._validate(
                            loader=self.val_loader,
                            max_batches_global=self._global_batches_for_sample_limit(
                                self.val_loader,
                                self.validate_max_samples_global,
                            ),
                            max_samples_global=0,
                            sample_weighted=True,
                            eval_model=self._model_module(),
                            visualize=val_vis,
                        )
                        val_loss_sum, val_count, val_metric_sums = self._reduce_validation_stats(
                            val_loss_sum=val_loss_sum,
                            val_count=val_count,
                            val_metric_sums=val_metric_sums,
                        )
                    elif self.rank == 0:
                        val_loss_sum, val_count, val_metric_sums = self._validate(
                            loader=self.reference_val_loader,
                            max_batches_global=0,
                            max_samples_global=self.validate_max_samples_global,
                            sample_weighted=True,
                            eval_model=self._model_module(),
                            visualize=val_vis,
                        )
                else:
                    val_loss_sum, val_count, val_metric_sums = self._validate(visualize=val_vis)
                    val_loss_sum, val_count, val_metric_sums = self._reduce_validation_stats(
                        val_loss_sum=val_loss_sum,
                        val_count=val_count,
                        val_metric_sums=val_metric_sums,
                    )
                val_metrics = self._finalize_validation_metrics(
                    val_loss_sum=val_loss_sum,
                    val_count=val_count,
                    val_metric_sums=val_metric_sums,
                )
                val_loss = float(val_metrics["val_loss_total"])
                best_metrics = dict(val_metrics)
                per_dataset_metrics_by_source = self._validate_per_dataset()
                per_dataset_metrics = self._flatten_per_dataset_metrics(per_dataset_metrics_by_source)
                best_metrics.update(per_dataset_metrics)
                self.validation_round += 1
                if self.rank == 0:
                    if self.metric_logger is not None:
                        self.metric_logger.log({"step": self.global_step, **val_metrics, **per_dataset_metrics})
                    for key, value in val_metrics.items():
                        self._tb_add_scalar(f"val/{key.removeprefix('val_')}", value, self.global_step)
                    self._tb_log_per_dataset_metrics(per_dataset_metrics_by_source)
                    self.logger.info(
                        "validation step=%d val_loss=%.6f %s=%d",
                        self.global_step,
                        val_loss,
                        ("samples" if self.validate_max_samples_global > 0 else "batches"),
                        val_count,
                    )
                    if per_dataset_metrics_by_source:
                        per_dataset_summary = " ".join(
                            f"{source}={metrics.get('val_loss_total', float('nan')):.6f}"
                            for source, metrics in per_dataset_metrics_by_source.items()
                        )
                        self.logger.info(
                            "validation_by_dataset step=%d %s",
                            self.global_step,
                            per_dataset_summary,
                        )
                if self.distributed:
                    dist.barrier()
                if self.rank == 0:
                    self.maybe_save_best(best_metrics)
                if self.distributed:
                    dist.barrier()

            self.maybe_save_step_checkpoint()
            if self.rank == 0 and self.log_every > 0 and (self.global_step % self.log_every == 0):
                self._poll_worldtrack_eval_processes()

        if self.distributed:
            dist.barrier()
        self.save_last_checkpoint()
        if self.distributed:
            dist.barrier()
        if self.rank == 0:
            self._poll_worldtrack_eval_processes()
            if self._worldtrack_eval_processes:
                running_steps = ",".join(str(step) for step, _, _ in self._worldtrack_eval_processes)
                self.logger.info("WorldTrack auto-eval still running after training: steps=%s", running_steps)
        best_val_out = self.best_val if math.isfinite(self.best_val) else None
        best_metric_out = self.best_metric_value if math.isfinite(self.best_metric_value) else None
        self._update_run_metadata(
            {
                "status": "completed",
                "training_summary": {
                    "final_step": self.global_step,
                    "best_val_loss": best_val_out,
                    "best_metric_name": self.best_metric_name,
                    "best_metric_mode": self.best_metric_mode,
                    "best_metric_value": best_metric_out,
                    "running_supervision_stats": self._running_supervision_stats(),
                },
            }
        )
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
