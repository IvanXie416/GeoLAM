#!/usr/bin/env python3
"""Batch-evaluate step_*.ckpt files on WorldTrack and log metrics to TensorBoard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STEP_CKPT_RE = re.compile(r"^step_(?P<step>\d+)\.ckpt$")
EVAL_CASE_LOG_RE = re.compile(r"subset=(?P<subset>\S+)\s+seq=(?P<seq>\S+)\s+APD\(global\)=")
BEST_CKPT_STEP_RE = re.compile(r"(?:^|_)best_ckpt_step_(\d+)(?:_|$)")

SCALAR_KEYS = [
    "avg_pts_global",
    "avg_pts_pertraj",
    "avg_pts_sim3",
    "avg_pts_sim3_closed",
    "epe_global",
    "epe_pertraj",
    "epe_sim3",
    "epe_sim3_closed",
    "avg_pts_global_dyn",
    "epe_global_dyn",
    "avg_pts_sim3_closed_dyn",
    "epe_sim3_closed_dyn",
    "dyn_fraction",
]

CLIP64_TENSORBOARD_SCALARS = {
    ("po_mini", "avg_pts_global"),
    ("po_mini", "epe_global"),
    ("ds_mini", "avg_pts_global"),
    ("ds_mini", "epe_global"),
    ("adt_mini", "avg_pts_global"),
    ("adt_mini", "epe_global"),
    ("pstudio_mini", "avg_pts_global"),
    ("pstudio_mini", "epe_global"),
}

REQUIRED_SUMMARY_SCALAR_KEYS = [
    "avg_pts_global",
    "avg_pts_pertraj",
    "avg_pts_sim3",
    "avg_pts_sim3_closed",
    "epe_global",
    "epe_pertraj",
    "epe_sim3",
    "epe_sim3_closed",
    "total_queries",
    "num_sequences",
]


def _make_tqdm(*, total: int, desc: str) -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit="case", dynamic_ncols=True)


def _tqdm_write(progress: Any, message: str) -> None:
    if progress is None:
        print(message)
        return
    progress.write(message)


def _count_eval_cases(data_root: Path, subsets_csv: str, limit_seqs: int) -> int:
    total = 0
    for subset in [item.strip() for item in str(subsets_csv).split(",") if item.strip()]:
        subset_dir = data_root / subset
        if not subset_dir.exists():
            continue
        count = len(sorted(subset_dir.glob("*.npz")))
        if int(limit_seqs) > 0:
            count = min(count, int(limit_seqs))
        total += count
    return total


def _count_completed_cases_from_log(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    completed: set[tuple[str, str]] = set()
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return 0
    for line in lines:
        match = EVAL_CASE_LOG_RE.search(line)
        if match:
            completed.add((match.group("subset"), match.group("seq")))
    return len(completed)


@dataclass
class StepEvalRecord:
    exp_dir: str
    checkpoint_dir: str
    ckpt_path: str
    step: int
    model_config: str
    tensorboard_logdir: str
    tensorboard_latest_event: str | None
    eval_output_dir: str
    eval_summary_json: str
    eval_log_path: str
    has_existing_summary: bool
    tensorboard_written: bool = False
    eval_mode: str = "default"
    status: str = "pending"
    gpu_id: str | None = None
    returncode: int | None = None
    duration_sec: float | None = None
    error: str | None = None


def _summary_json_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    inputs = payload.get("inputs")
    subsets = payload.get("subsets")
    if not isinstance(inputs, dict) or not isinstance(subsets, dict):
        return False
    expected_subsets = inputs.get("subsets")
    if not isinstance(expected_subsets, list) or not expected_subsets:
        return False
    for subset in [str(item) for item in expected_subsets if str(item).strip()]:
        subset_payload = subsets.get(subset)
        if not isinstance(subset_payload, dict):
            return False
        for key in REQUIRED_SUMMARY_SCALAR_KEYS:
            if key not in subset_payload:
                return False
        try:
            num_sequences = int(subset_payload.get("num_sequences"))
            total_queries = int(subset_payload.get("total_queries"))
        except (TypeError, ValueError):
            return False
        sequences = subset_payload.get("sequences")
        if num_sequences <= 0 or total_queries <= 0:
            return False
        if not isinstance(sequences, list) or len(sequences) != num_sequences:
            return False
    return True


def _find_latest_event_file(tensorboard_logdir: Path) -> Path | None:
    if not tensorboard_logdir.exists():
        return None
    candidates = [
        path
        for path in tensorboard_logdir.rglob("events.out.tfevents.*")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.stat().st_mtime, str(item)))


def _step_from_ckpt(path: Path) -> int | None:
    match = STEP_CKPT_RE.match(path.name)
    if not match:
        return None
    return int(match.group("step"))


def _parse_steps(raw_steps: str) -> set[int]:
    steps: set[int] = set()
    for item in re.split(r"[\s,]+", str(raw_steps).strip()):
        if not item:
            continue
        match = re.search(r"(\d+)", item)
        if match is None:
            raise SystemExit(f"Invalid step value in --steps: {item}")
        steps.add(int(match.group(1)))
    return steps


def _summary_inputs_match(path: Path, args: argparse.Namespace) -> bool:
    if not _summary_json_complete(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return False
    try:
        if int(inputs.get("num_frames")) != int(args.num_frames):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _discover_step_ckpts(args: argparse.Namespace) -> list[StepEvalRecord]:
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint directory not found: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Expected checkpoint directory, got: {ckpt_dir}")

    exp_dir = ckpt_dir.parent
    model_config = Path(args.model_config).resolve() if args.model_config else exp_dir / "config" / "model_effective.yaml"
    if not model_config.exists():
        raise SystemExit(f"Model config not found: {model_config}")

    tensorboard_logdir = (
        Path(args.tensorboard_logdir).resolve()
        if args.tensorboard_logdir
        else exp_dir / "tensorboard"
    )
    latest_event = _find_latest_event_file(tensorboard_logdir)

    requested_steps = _parse_steps(str(args.steps)) if str(args.steps).strip() else set()
    step_ckpts: list[tuple[int, Path]] = []
    for ckpt_path in ckpt_dir.glob("step_*.ckpt"):
        step = _step_from_ckpt(ckpt_path)
        if step is None:
            continue
        if requested_steps and step not in requested_steps:
            continue
        if args.min_step is not None and step < int(args.min_step):
            continue
        if args.max_step is not None and step > int(args.max_step):
            continue
        step_ckpts.append((step, ckpt_path.resolve()))
    if requested_steps:
        found_steps = {step for step, _ in step_ckpts}
        missing_steps = sorted(requested_steps - found_steps)
        if missing_steps:
            missing_text = ",".join(str(step) for step in missing_steps)
            raise SystemExit(f"Requested step checkpoint(s) not found in {ckpt_dir}: {missing_text}")
    step_ckpts.sort(key=lambda item: item[0])
    if int(args.stride) > 1:
        step_ckpts = step_ckpts[:: int(args.stride)]
    if int(args.limit_ckpts) > 0:
        step_ckpts = step_ckpts[: int(args.limit_ckpts)]
    if not step_ckpts:
        raise SystemExit(f"No step_*.ckpt files found in {ckpt_dir}")

    output_root = Path(args.output_root).resolve() if args.output_root else exp_dir
    records: list[StepEvalRecord] = []
    for step, ckpt_path in step_ckpts:
        eval_output_dir = output_root / str(args.output_dir_name_template).format(step=f"{step:07d}", step_int=step)
        eval_summary_json = eval_output_dir / "summary.json"
        eval_log_path = eval_output_dir / "eval_worldtrack.log"
        records.append(
            StepEvalRecord(
                exp_dir=str(exp_dir),
                checkpoint_dir=str(ckpt_dir),
                ckpt_path=str(ckpt_path),
                step=int(step),
                model_config=str(model_config),
                tensorboard_logdir=str(tensorboard_logdir),
                tensorboard_latest_event=str(latest_event) if latest_event is not None else None,
                eval_output_dir=str(eval_output_dir),
                eval_summary_json=str(eval_summary_json),
                eval_log_path=str(eval_log_path),
                has_existing_summary=_summary_inputs_match(eval_summary_json, args),
                eval_mode=str(args.eval_mode),
            )
        )
    return records


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _best_step_suffix(step: int) -> str:
    return f"{int(step):04d}"


def _step_eval_dir(exp_dir: Path, step: int, eval_dir_suffix: str) -> Path:
    return exp_dir / f"eval_worldtrack_step_{int(step):07d}_{eval_dir_suffix}"


def _best_eval_dir(exp_dir: Path, step: int, eval_dir_suffix: str) -> Path:
    return exp_dir / f"eval_worldtrack_best_ckpt_step_{_best_step_suffix(step)}_{eval_dir_suffix}"


def _existing_64clip_summary(
    exp_dir: Path,
    step: int,
    args: argparse.Namespace,
) -> Path | None:
    suffix = str(args.eval_dir_suffix)
    candidates = [
        _step_eval_dir(exp_dir, step, suffix) / "summary.json",
        _best_eval_dir(exp_dir, step, suffix) / "summary.json",
    ]
    for summary_json in candidates:
        if _summary_inputs_match(summary_json, args):
            return summary_json
    return None


def _full_eval_best_ckpt_still_matches(row: dict[str, Any], best_ckpt: Path) -> bool:
    eval_dir_raw = row.get("eval_dir") or row.get("eval_output_dir")
    if not eval_dir_raw:
        return False
    metadata = _read_json_dict(Path(str(eval_dir_raw)) / "eval_metadata.json")
    if metadata is None:
        return False
    old_sig = metadata.get("best_ckpt")
    if not isinstance(old_sig, dict):
        return False
    try:
        stat = best_ckpt.stat()
    except OSError:
        return False
    return old_sig.get("size") == int(stat.st_size) and old_sig.get("mtime_ns") == int(stat.st_mtime_ns)


def _rank_summary_rows(rows: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    higher_better = _is_higher_better(sort_key)
    ranked = [row for row in rows if _sort_value(row, sort_key) == _sort_value(row, sort_key)]
    ranked.sort(
        key=lambda row: (
            -_sort_value(row, sort_key) if higher_better else _sort_value(row, sort_key),
            str(row.get("name", "")),
            str(row.get("eval_dir_name", "")),
        )
    )
    return ranked


def _discover_top_summary_ckpts(args: argparse.Namespace) -> tuple[list[StepEvalRecord], list[dict[str, Any]]]:
    summary_path = Path(str(args.top_summary_json)).resolve()
    payload = _read_json_dict(summary_path)
    if payload is None:
        raise SystemExit(f"Top summary JSON not found or invalid: {summary_path}")
    rows_raw = payload.get("records")
    if not isinstance(rows_raw, list):
        raise SystemExit(f"Top summary JSON missing records list: {summary_path}")
    rows = [row for row in rows_raw if isinstance(row, dict)]
    ranked_rows = _rank_summary_rows(rows, str(args.sort_key))

    rank_start = max(int(args.rank_start), 1)
    rank_end = int(args.rank_end)
    target_count = int(args.top_k)
    if rank_end > 0:
        candidate_rows = ranked_rows[rank_start - 1 : rank_end]
    else:
        if target_count <= 0:
            target_count = len(ranked_rows) - rank_start + 1
        candidate_rows = ranked_rows[rank_start - 1 : rank_start - 1 + target_count]

    selected_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    skipped: list[dict[str, Any]] = []
    for display_rank, row in enumerate(candidate_rows, start=rank_start):
        step_raw = row.get("step")
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            skipped.append({"rank": display_rank, "row": row.get("name"), "reason": f"invalid_step:{step_raw}"})
            continue
        exp_dir_raw = row.get("exp_dir")
        if not exp_dir_raw:
            skipped.append({"rank": display_rank, "row": row.get("name"), "step": step, "reason": "missing_exp_dir"})
            continue
        exp_dir = Path(str(exp_dir_raw)).resolve()
        key = (str(exp_dir), step)
        if key in seen_keys:
            skipped.append({"rank": display_rank, "row": row.get("name"), "step": step, "reason": "duplicate_exp_step"})
            continue
        seen_keys.add(key)
        row["_summary_rank"] = display_rank
        selected_rows.append(row)

    records: list[StepEvalRecord] = []
    planned: list[dict[str, Any]] = []
    for row in selected_rows:
        rank = int(row.get("_summary_rank", 0) or 0)
        step = int(row["step"])
        exp_dir = Path(str(row["exp_dir"])).resolve()
        ckpt_dir = exp_dir / "checkpoints"
        model_config = exp_dir / "config" / "model_effective.yaml"
        tensorboard_logdir = exp_dir / "tensorboard"
        latest_event = _find_latest_event_file(tensorboard_logdir)
        existing_summary = _existing_64clip_summary(exp_dir, step, args)

        ckpt_path = ckpt_dir / f"step_{step:07d}.ckpt"
        output_dir = _step_eval_dir(exp_dir, step, str(args.eval_dir_suffix))
        checkpoint_source = "step"
        if not ckpt_path.exists():
            best_ckpt = ckpt_dir / "best.ckpt"
            if (
                str(row.get("checkpoint_source")) == "best_ckpt"
                and best_ckpt.exists()
                and _full_eval_best_ckpt_still_matches(row, best_ckpt)
            ):
                ckpt_path = best_ckpt
                output_dir = _best_eval_dir(exp_dir, step, str(args.eval_dir_suffix))
                checkpoint_source = "best_ckpt"
            else:
                skipped.append(
                    {
                        "rank": rank,
                        "name": row.get("name"),
                        "step": step,
                        "exp_dir": str(exp_dir),
                        "reason": "exact_step_ckpt_missing_or_best_ckpt_changed",
                    }
                )
                continue

        if not model_config.exists():
            skipped.append(
                {
                    "rank": rank,
                    "name": row.get("name"),
                    "step": step,
                    "exp_dir": str(exp_dir),
                    "reason": "model_config_missing",
                }
            )
            continue

        if existing_summary is not None:
            output_dir = existing_summary.parent
        eval_summary_json = output_dir / "summary.json"
        eval_log_path = output_dir / "eval_worldtrack.log"
        record = StepEvalRecord(
            exp_dir=str(exp_dir),
            checkpoint_dir=str(ckpt_dir),
            ckpt_path=str(ckpt_path.resolve()),
            step=step,
            model_config=str(model_config.resolve()),
            tensorboard_logdir=str(tensorboard_logdir),
            tensorboard_latest_event=str(latest_event) if latest_event is not None else None,
            eval_output_dir=str(output_dir),
            eval_summary_json=str(eval_summary_json),
            eval_log_path=str(eval_log_path),
            has_existing_summary=existing_summary is not None,
            eval_mode=str(args.eval_mode),
        )
        records.append(record)
        planned.append(
            {
                "rank": rank,
                "name": row.get("name"),
                "step": step,
                "sort_key": str(args.sort_key),
                "sort_value": _sort_value(row, str(args.sort_key)),
                "checkpoint_source": checkpoint_source,
                "ckpt_path": str(ckpt_path),
                "eval_output_dir": str(output_dir),
                "has_existing_summary": existing_summary is not None,
            }
        )

    selection_report = Path(str(args.selection_report_path)).resolve() if str(args.selection_report_path).strip() else Path(str(args.report_path or "tmp/eval_worldtrack/top_full_missing_64clip_eval_report.json")).resolve().with_name("top_full_missing_64clip_selection_report.json")
    selection_report.parent.mkdir(parents=True, exist_ok=True)
    selection_report.write_text(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "sort_key": str(args.sort_key),
                "top_k": target_count,
                "rank_start": rank_start,
                "rank_end": rank_end,
                "eval_dir_suffix": str(args.eval_dir_suffix),
                "planned": planned,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved top-summary selection report to {selection_report}")
    if skipped:
        print(f"Skipped {len(skipped)} top-summary row(s); see selection report for reasons.")
    return records, skipped


def _build_eval_command(record: StepEvalRecord, args: argparse.Namespace) -> str:
    cmd = [
        "python",
        "eval_track3d_in_worldtrack.py",
        "--model-config",
        record.model_config,
        "--ckpt-path",
        record.ckpt_path,
        "--data-root",
        args.data_root,
        "--subsets",
        args.subsets,
        "--num-frames",
        str(int(args.num_frames)),
        "--query-chunk-size",
        str(int(args.query_chunk_size)),
        "--device",
        "cuda",
        "--output-dir",
        record.eval_output_dir,
        "--save-per-sequence",
    ]
    if int(args.limit_seqs) > 0:
        cmd.extend(["--limit-seqs", str(int(args.limit_seqs))])
    return " ".join(shlex.quote(part) for part in cmd)


def _to_float(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _is_higher_better(sort_key: str) -> bool:
    lowered = str(sort_key).lower()
    if "epe" in lowered or "loss" in lowered or lowered.endswith("_err"):
        return False
    return True


def _sort_value(row: dict[str, Any], sort_key: str) -> float:
    scalar = _to_float(row.get(sort_key))
    return float("nan") if scalar is None else float(scalar)


def _summary_num_frames(payload: dict[str, Any]) -> int | None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return None
    try:
        return int(inputs.get("num_frames"))
    except (TypeError, ValueError):
        return None


def _extract_tensorboard_scalars(summary_json: Path) -> dict[str, float]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    subsets = payload.get("subsets", {})
    if not isinstance(subsets, dict):
        raise ValueError(f"summary.json missing subsets: {summary_json}")

    num_frames = _summary_num_frames(payload)
    if num_frames == 64:
        tag_prefix = "eval_worldtrack_clip64"
        allowed_subset_metrics: set[tuple[str, str]] | None = CLIP64_TENSORBOARD_SCALARS
        include_counts = False
        include_overall = False
    else:
        tag_prefix = "eval_worldtrack"
        allowed_subset_metrics = None
        include_counts = True
        include_overall = True

    scalars: dict[str, float] = {}
    overall_accum: dict[str, list[float]] = {key: [] for key in SCALAR_KEYS}
    total_queries = 0
    total_dynamic_queries = 0
    total_sequences = 0

    for subset_name, subset_payload in subsets.items():
        if not isinstance(subset_payload, dict):
            continue
        prefix = str(subset_name)
        for key in SCALAR_KEYS:
            if allowed_subset_metrics is not None and (prefix, key) not in allowed_subset_metrics:
                continue
            scalar = _to_float(subset_payload.get(key))
            if scalar is None:
                continue
            scalars[f"{tag_prefix}/{prefix}/{key}"] = scalar
            if include_overall:
                overall_accum[key].append(scalar)

        subset_queries = int(subset_payload.get("total_queries", 0) or 0)
        subset_dynamic_queries = int(subset_payload.get("total_dynamic_queries", 0) or 0)
        subset_sequences = int(subset_payload.get("num_sequences", 0) or 0)
        if include_counts:
            scalars[f"{tag_prefix}/{prefix}/total_queries"] = float(subset_queries)
            scalars[f"{tag_prefix}/{prefix}/total_dynamic_queries"] = float(subset_dynamic_queries)
            scalars[f"{tag_prefix}/{prefix}/num_sequences"] = float(subset_sequences)
        total_queries += subset_queries
        total_dynamic_queries += subset_dynamic_queries
        total_sequences += subset_sequences

    if include_overall:
        scalars[f"{tag_prefix}/overall/total_queries"] = float(total_queries)
        scalars[f"{tag_prefix}/overall/total_dynamic_queries"] = float(total_dynamic_queries)
        scalars[f"{tag_prefix}/overall/num_sequences"] = float(total_sequences)
        for key, values in overall_accum.items():
            if values:
                scalars[f"{tag_prefix}/overall/{key}"] = float(sum(values) / len(values))
    return scalars


def _write_tensorboard(record: StepEvalRecord) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError("TensorBoard dependencies are missing; cannot write eval scalars.") from exc

    summary_json = Path(record.eval_summary_json)
    if not _summary_json_complete(summary_json):
        raise ValueError(f"Cannot write TensorBoard; incomplete summary: {summary_json}")

    tb_dir = Path(record.tensorboard_logdir)
    tb_dir.mkdir(parents=True, exist_ok=True)
    scalars = _extract_tensorboard_scalars(summary_json)
    writer = SummaryWriter(log_dir=str(tb_dir), filename_suffix=".eval_worldtrack")
    try:
        for tag, value in sorted(scalars.items()):
            writer.add_scalar(tag, value, int(record.step))
    finally:
        writer.flush()
        writer.close()
    record.tensorboard_written = True
    latest_event = _find_latest_event_file(tb_dir)
    record.tensorboard_latest_event = str(latest_event) if latest_event is not None else None


def _write_tensorboard_for_complete_records(records: list[StepEvalRecord], *, force: bool = False) -> None:
    for record in records:
        if record.tensorboard_written and not force:
            continue
        if not _summary_json_complete(Path(record.eval_summary_json)):
            continue
        _write_tensorboard(record)


def _write_report(report_path: Path, records: list[StepEvalRecord]) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = report_path.with_suffix(".csv")
    fieldnames = list(asdict(records[0]).keys()) if records else list(StepEvalRecord.__dataclass_fields__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="", help="Directory containing step_*.ckpt files.")
    parser.add_argument("--model-config", default="", help="Defaults to <experiment>/config/model_effective.yaml.")
    parser.add_argument("--tensorboard-logdir", default="", help="Defaults to <experiment>/tensorboard.")
    parser.add_argument("--output-root", default="", help="Defaults to the experiment directory.")
    parser.add_argument("--data-root", default="data/worldtrack_release")
    parser.add_argument("--subsets", default="adt_mini,po_mini,pstudio_mini,ds_mini")
    parser.add_argument("--num-frames", type=int, default=1000000)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU ids.")
    parser.add_argument("--procs-per-gpu", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--steps", default="", help="Comma- or whitespace-separated exact step numbers to evaluate.")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth discovered step checkpoint.")
    parser.add_argument("--limit-ckpts", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-tensorboard-only", action="store_true", help="Do not run eval; write TB from existing summaries.")
    parser.add_argument(
        "--report-path",
        default="",
        help="Aggregate JSON report path. Defaults to <experiment>/eval_worldtrack_step_ckpts_<eval-dir-suffix>_report.json.",
    )
    parser.add_argument(
        "--eval-dir-suffix",
        default="full_eval",
        help="Suffix used for default eval directory/report naming, e.g. full_eval or 64clip_eval.",
    )
    parser.add_argument(
        "--output-dir-name-template",
        default="",
        help="Per-step eval output directory name template. Supports {step} and {step_int}.",
    )
    parser.add_argument("--eval-mode", default="default")
    parser.add_argument(
        "--top-summary-json",
        default="",
        help="Scan a total-exp summary JSON, select top rows, and evaluate missing 64clip step ckpts.",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Top unique ckpts to select from --top-summary-json.")
    parser.add_argument("--rank-start", type=int, default=1, help="1-based inclusive start rank for --top-summary-json.")
    parser.add_argument("--rank-end", type=int, default=0, help="1-based inclusive end rank for --top-summary-json. <=0 uses --top-k.")
    parser.add_argument("--sort-key", default="overall.avg_pts_global", help="Metric used for --top-summary-json ranking.")
    parser.add_argument(
        "--selection-report-path",
        default="",
        help="Optional JSON report path for top-summary selection/skips.",
    )
    args = parser.parse_args()
    if not str(args.output_dir_name_template).strip():
        args.output_dir_name_template = f"eval_worldtrack_step_{{step}}_{str(args.eval_dir_suffix)}"

    repo_root = Path(__file__).resolve().parents[2]
    top_summary_mode = bool(str(args.top_summary_json).strip())
    if top_summary_mode:
        if not str(args.report_path).strip():
            args.report_path = "tmp/eval_worldtrack/top_full_missing_64clip_eval_report.json"
        records, _ = _discover_top_summary_ckpts(args)
        default_report = Path(args.report_path)
    else:
        if not str(args.checkpoint_dir).strip():
            raise SystemExit("--checkpoint-dir is required unless --top-summary-json is provided.")
        records = _discover_step_ckpts(args)
        default_report = Path(records[0].exp_dir) / f"eval_worldtrack_step_ckpts_{str(args.eval_dir_suffix)}_report.json"

    report_path = (Path(args.report_path).resolve() if args.report_path else default_report.resolve())
    if not records:
        csv_path = _write_report(report_path, records)
        print(f"No step checkpoint records discovered. Report: {report_path} and {csv_path}")
        return

    print(f"Discovered {len(records)} step checkpoint(s).")
    if top_summary_mode:
        print(f"Top summary: {args.top_summary_json}")
        print(f"sort_key={args.sort_key} top_k={args.top_k}")
    else:
        print(f"Experiment dir: {records[0].exp_dir}")
        print(f"TensorBoard logdir: {records[0].tensorboard_logdir}")
        print(f"Latest existing event: {records[0].tensorboard_latest_event or '-'}")
    for record in records:
        print(f"- step={record.step} existing_eval={record.has_existing_summary} ckpt={record.ckpt_path}")

    if args.dry_run:
        csv_path = _write_report(report_path, records)
        print(f"Saved dry-run report to {report_path} and {csv_path}")
        return

    if args.write_tensorboard_only:
        _write_tensorboard_for_complete_records(records, force=True)
        csv_path = _write_report(report_path, records)
        print(f"Wrote TensorBoard scalars from existing summaries. Report: {report_path} and {csv_path}")
        return

    pending: list[StepEvalRecord] = []
    for record in records:
        if record.has_existing_summary and not args.overwrite:
            record.status = "skipped_existing"
        else:
            pending.append(record)

    _write_tensorboard_for_complete_records(records)

    cases_per_eval = _count_eval_cases(
        data_root=(repo_root / args.data_root).resolve(),
        subsets_csv=str(args.subsets),
        limit_seqs=int(args.limit_seqs),
    )
    progress_total = cases_per_eval * len(pending)
    progress = _make_tqdm(total=progress_total, desc="WorldTrack step eval") if progress_total > 0 else None
    finished_case_count = 0

    gpu_ids = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU ids provided.")
    slot_count = min(int(args.max_parallel), len(gpu_ids) * int(args.procs_per_gpu))
    if slot_count <= 0:
        raise SystemExit("max_parallel must be positive.")
    gpu_slots: list[str] = []
    for gpu_id in gpu_ids:
        gpu_slots.extend([gpu_id] * int(args.procs_per_gpu))
    gpu_slots = gpu_slots[:slot_count]

    print(
        f"Launching up to {len(gpu_slots)} concurrent eval process(es) across GPUs {gpu_ids}. "
        f"cases_per_eval={cases_per_eval} total_cases={progress_total}"
    )
    running: list[tuple[subprocess.Popen[str], StepEvalRecord, float, str, Any]] = []
    idle_slots = gpu_slots[:]

    while pending or running:
        while pending and idle_slots:
            record = pending.pop(0)
            gpu_id = idle_slots.pop(0)
            output_dir = Path(record.eval_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            eval_command = _build_eval_command(record=record, args=args)
            (output_dir / "command.txt").write_text(eval_command + "\n", encoding="utf-8")
            log_handle = Path(record.eval_log_path).open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            cmd_lines = [f"cd {shlex.quote(str(repo_root))}"]
            conda_env = str(env.get("CONDA_ENV", "")).strip()
            conda_sh = str(env.get("CONDA_SH", "")).strip()
            if conda_env:
                if not conda_sh:
                    conda_exe = shutil.which("conda")
                    if conda_exe:
                        conda_base = subprocess.run(
                            [conda_exe, "info", "--base"],
                            check=False,
                            capture_output=True,
                            text=True,
                            env=env,
                        )
                        if conda_base.returncode == 0:
                            candidate = Path(conda_base.stdout.strip()) / "etc" / "profile.d" / "conda.sh"
                            if candidate.is_file():
                                conda_sh = str(candidate)
                if conda_sh:
                    cmd_lines.append(f"source {shlex.quote(conda_sh)}")
                    cmd_lines.append(f"conda activate {shlex.quote(conda_env)}")
            cmd_lines.append(eval_command)
            cmd = "\n".join(cmd_lines)
            _tqdm_write(progress, f"[launch] gpu={gpu_id} step={record.step}")
            proc = subprocess.Popen(
                ["/usr/bin/zsh", "-lc", cmd],
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            record.status = "running"
            record.gpu_id = gpu_id
            running.append((proc, record, time.time(), gpu_id, log_handle))

        if not running:
            break

        time.sleep(5.0)
        still_running: list[tuple[subprocess.Popen[str], StepEvalRecord, float, str, Any]] = []
        for proc, record, start_ts, gpu_id, log_handle in running:
            returncode = proc.poll()
            if returncode is None:
                still_running.append((proc, record, start_ts, gpu_id, log_handle))
                continue

            log_handle.close()
            record.returncode = int(returncode)
            record.duration_sec = float(time.time() - start_ts)
            completed_cases = min(cases_per_eval, _count_completed_cases_from_log(Path(record.eval_log_path)))
            finished_case_count += completed_cases
            if returncode == 0 and _summary_json_complete(Path(record.eval_summary_json)):
                record.status = "done"
                _write_tensorboard(record)
                _tqdm_write(
                    progress,
                    f"[done] gpu={gpu_id} step={record.step} "
                    f"cases={completed_cases}/{cases_per_eval} duration={record.duration_sec:.1f}s "
                    f"tb={record.tensorboard_latest_event}",
                )
            elif returncode == 0:
                record.status = "incomplete_summary"
                record.error = "summary_json_missing_or_incomplete"
                _tqdm_write(progress, f"[incomplete] gpu={gpu_id} step={record.step} cases={completed_cases}/{cases_per_eval}")
            else:
                record.status = "failed"
                record.error = f"exit_code={returncode}"
                _tqdm_write(progress, f"[fail] gpu={gpu_id} step={record.step} cases={completed_cases}/{cases_per_eval} exit_code={returncode}")
            idle_slots.append(gpu_id)
        running = still_running

        if progress is not None:
            running_case_count = sum(
                min(cases_per_eval, _count_completed_cases_from_log(Path(record.eval_log_path)))
                for _, record, _, _, _ in running
            )
            current = min(progress_total, finished_case_count + running_case_count)
            delta = current - int(progress.n)
            if delta > 0:
                progress.update(delta)
            progress.set_postfix(
                running=len(running),
                pending=len(pending),
                done_cases=f"{current}/{progress_total}",
            )

        _write_report(report_path, records)

    _write_tensorboard_for_complete_records(records)
    csv_path = _write_report(report_path, records)
    if progress is not None:
        progress.close()
    print(f"Saved final report to {report_path} and {csv_path}")


if __name__ == "__main__":
    main()
