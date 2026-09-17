"""Train/val query prediction visualization and TensorBoard image logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _safe_name(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:96]


def _line_ids(num_pairs: int, max_lines: int) -> np.ndarray:
    if num_pairs <= 0:
        return np.empty((0,), dtype=np.int64)
    if int(max_lines) <= 0 or int(max_lines) >= num_pairs:
        return np.arange(num_pairs, dtype=np.int64)
    return np.linspace(0, num_pairs - 1, num=int(max_lines), dtype=np.int64)


def _robust_vmax(errors: np.ndarray, q: float = 0.95) -> float:
    arr = np.asarray(errors, dtype=np.float64)
    if arr.size == 0 or (not np.isfinite(arr).any()):
        return 1e-6
    valid = arr[np.isfinite(arr)]
    vmax = float(np.quantile(valid, q))
    vmax = max(vmax, float(np.max(valid)), 1e-6)
    return vmax


def _set_3d_equal_axes(ax: Any, points_xyz: np.ndarray) -> None:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.nanmax(maxs - mins)) * 0.55
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _figure_to_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    rgb = np.ascontiguousarray(buf[:, :, :3])
    return rgb


def _meta_value(meta: dict[str, Any], key: str, index: int) -> str:
    value = meta.get(key)
    if value is None:
        return ""
    if isinstance(value, list):
        if 0 <= index < len(value):
            return str(value[index])
        return ""
    if torch.is_tensor(value):
        if value.ndim == 0:
            return str(value.item())
        if 0 <= index < value.shape[0]:
            return str(value[index].item())
        return ""
    return str(value)


def _select_query_indices(t_tgt: np.ndarray, uv_mask: np.ndarray) -> tuple[np.ndarray, int]:
    if uv_mask.any():
        valid_t = t_tgt[uv_mask]
        uniq, cnt = np.unique(valid_t, return_counts=True)
    else:
        uniq, cnt = np.unique(t_tgt, return_counts=True)
    if uniq.size == 0:
        return np.zeros_like(uv_mask, dtype=bool), 0
    frame = int(uniq[int(np.argmax(cnt))])
    selected = uv_mask & (t_tgt == frame)
    if not selected.any():
        selected = uv_mask.copy()
    return selected, frame


def _subsample_indices(mask: np.ndarray, max_points: int) -> np.ndarray:
    ids = np.flatnonzero(mask)
    if ids.size <= max_points:
        return ids
    pick = np.linspace(0, ids.size - 1, num=max_points, dtype=np.int64)
    return ids[pick]


def _solve_scale_only(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = float(np.dot(pred, pred))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(pred, gt) / denom)


def _solve_scale_shift(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    a = np.stack([pred, np.ones_like(pred)], axis=1)
    x, *_ = np.linalg.lstsq(a, gt, rcond=None)
    return float(x[0]), float(x[1])


def _align_pred_xyz_sequence(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    mask_xyz: np.ndarray,
    mode: str,
    min_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode_key = str(mode).strip().lower()
    aligned = np.asarray(pred_xyz).copy()
    info: dict[str, Any] = {
        "mode": mode_key,
        "applied": False,
        "scale": 1.0,
        "shift_z": 0.0,
        "num_points": 0,
    }
    if mode_key == "none":
        return aligned, info
    if mode_key not in {"scale", "scale_shift"}:
        info["mode"] = "none"
        return aligned, info

    pred = np.asarray(pred_xyz, dtype=np.float64)
    gt = np.asarray(gt_xyz, dtype=np.float64)
    mask = np.asarray(mask_xyz, dtype=bool)
    valid = mask & np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    count = int(valid.sum())
    info["num_points"] = count
    if count < max(2, int(min_points)):
        return aligned, info

    pred_z = pred[valid, 2]
    gt_z = gt[valid, 2]
    if mode_key == "scale":
        scale = _solve_scale_only(pred_z, gt_z)
        shift = 0.0
    else:
        scale, shift = _solve_scale_shift(pred_z, gt_z)
    if not (np.isfinite(scale) and np.isfinite(shift)):
        return aligned, info

    aligned64 = pred.copy()
    aligned64 *= float(scale)
    if mode_key == "scale_shift":
        aligned64[:, 2] = aligned64[:, 2] + float(shift)
    aligned = aligned64.astype(np.asarray(pred_xyz).dtype, copy=False)
    info["applied"] = True
    info["scale"] = float(scale)
    info["shift_z"] = float(shift)
    return aligned, info


def _plot_overlay_2d(
    image: np.ndarray,
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    errors_px: np.ndarray,
    hard_flags: np.ndarray | None,
    title: str,
    max_lines: int,
) -> Any:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(np.clip(image, 0.0, 1.0))

    ids = _line_ids(num_pairs=int(gt_xy.shape[0]), max_lines=max_lines)
    if ids.size > 0:
        segs = np.stack([gt_xy[ids], pred_xy[ids]], axis=1)
        line_err = np.asarray(errors_px[ids], dtype=np.float64)
        norm = mcolors.Normalize(vmin=0.0, vmax=_robust_vmax(np.asarray(errors_px, dtype=np.float64), q=0.95))
        lc = LineCollection(segs, cmap="turbo", norm=norm, linewidths=1.0, alpha=0.75)
        lc.set_array(line_err)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.01)
        cbar.set_label("pair distance (px)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax.scatter(gt_xy[:, 0], gt_xy[:, 1], c="lime", s=16, label="GT", alpha=0.95)
    ax.scatter(pred_xy[:, 0], pred_xy[:, 1], c="red", s=12, label="Pred", alpha=0.85)
    if hard_flags is not None and hard_flags.any():
        ax.scatter(
            gt_xy[hard_flags, 0],
            gt_xy[hard_flags, 1],
            s=36,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.8,
            label="Hard query",
        )

    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _project_pred_xyz_to_tgt_uv_np(
    *,
    pred_xyz_tcam: np.ndarray,
    k_seq: np.ndarray,
    t_wc_seq: np.ndarray,
    camera_valid: np.ndarray,
    t_cam: np.ndarray,
    t_tgt: np.ndarray,
    image_h: int,
    image_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_xyz = np.asarray(pred_xyz_tcam, dtype=np.float64)
    k_seq = np.asarray(k_seq, dtype=np.float64)
    t_wc_seq = np.asarray(t_wc_seq, dtype=np.float64)
    camera_valid = np.asarray(camera_valid, dtype=bool)
    t_cam = np.asarray(t_cam, dtype=np.int64)
    t_tgt = np.asarray(t_tgt, dtype=np.int64)
    num_queries = int(pred_xyz.shape[0])
    uv = np.full((num_queries, 2), np.nan, dtype=np.float64)
    valid = np.zeros((num_queries,), dtype=bool)
    for qi in range(num_queries):
        tc = int(t_cam[qi])
        tt = int(t_tgt[qi])
        if tc < 0 or tt < 0 or tc >= t_wc_seq.shape[0] or tt >= t_wc_seq.shape[0]:
            continue
        if not (bool(camera_valid[tc]) and bool(camera_valid[tt])):
            continue
        p = pred_xyz[qi]
        if not np.isfinite(p).all():
            continue
        p_h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        world_h = t_wc_seq[tc] @ p_h
        try:
            t_cw_tgt = np.linalg.inv(t_wc_seq[tt])
        except np.linalg.LinAlgError:
            continue
        p_tgt = (t_cw_tgt @ world_h)[:3]
        z = float(p_tgt[2])
        if not np.isfinite(z) or z <= 1e-6:
            continue
        proj = k_seq[tt] @ p_tgt
        uv[qi, 0] = float(proj[0] / z) / float(max(image_w - 1, 1))
        uv[qi, 1] = float(proj[1] / z) / float(max(image_h - 1, 1))
        valid[qi] = np.isfinite(uv[qi]).all()
    return uv.astype(np.float32), valid


def _plot_3d_on_image(image: np.ndarray, gt_xy: np.ndarray, errors_m: np.ndarray, title: str) -> Any:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(np.clip(image, 0.0, 1.0))
    err = np.asarray(errors_m, dtype=np.float64)
    norm = mcolors.Normalize(vmin=0.0, vmax=_robust_vmax(err, q=0.95))
    sc = ax.scatter(
        gt_xy[:, 0],
        gt_xy[:, 1],
        c=err,
        cmap="turbo",
        norm=norm,
        s=22,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.2,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.01)
    cbar.set_label("3D error (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _plot_3d_pairs(gt_xyz: np.ndarray, pred_xyz: np.ndarray, errors_m: np.ndarray, title: str, max_lines: int) -> Any:
    fig = plt.figure(figsize=(7, 6), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=9)

    gt = np.asarray(gt_xyz, dtype=np.float64)
    pred = np.asarray(pred_xyz, dtype=np.float64)
    err = np.asarray(errors_m, dtype=np.float64)

    ids = _line_ids(num_pairs=int(gt.shape[0]), max_lines=max_lines)
    if ids.size > 0:
        segs = np.stack([gt[ids], pred[ids]], axis=1)
        line_err = err[ids]
        norm = mcolors.Normalize(vmin=0.0, vmax=_robust_vmax(err, q=0.95))
        lc = Line3DCollection(segs, cmap="turbo", norm=norm, linewidths=1.1, alpha=0.75)
        lc.set_array(line_err)
        ax.add_collection3d(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("pair distance (m)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c="lime", s=10, alpha=0.9, label="GT xyz")
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="red", s=8, alpha=0.8, label="Pred xyz")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=22, azim=-58)
    _set_3d_equal_axes(ax, np.concatenate([gt, pred], axis=0))
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


class QueryPredictionVisualizer:
    """Periodic query prediction visualizer for train/val loop."""

    def __init__(self, train_cfg: Any, output_dir: Path, tb_writer: Any | None, logger: Any | None = None) -> None:
        vis_cfg = train_cfg.get_path("logging.visualization", {})
        if not isinstance(vis_cfg, dict):
            vis_cfg = {}

        self.enabled = bool(vis_cfg.get("enabled", tb_writer is not None))
        self.train_every_steps = int(vis_cfg.get("train_every_steps", 0))
        self.on_validation = bool(vis_cfg.get("on_validation", True))
        self.max_samples = max(1, int(vis_cfg.get("max_samples", 1)))
        self.max_points = max(1, int(vis_cfg.get("max_points", 128)))
        self.max_lines = int(vis_cfg.get("max_lines", -1))
        self.include_3d = bool(vis_cfg.get("include_3d", True))
        self.save_png = bool(vis_cfg.get("save_png", True))
        self.xyz_alignment_mode = str(vis_cfg.get("xyz_alignment_mode", "scale_shift")).strip().lower()
        self.xyz_alignment_min_points = max(2, int(vis_cfg.get("xyz_alignment_min_points", 16)))
        self.output_dir = Path(output_dir)
        self.tb_writer = tb_writer
        self.logger = logger
        self.root = self.output_dir / "visualizations"
        self._last_train_step = -1
        self._last_val_step = -1
        if self.xyz_alignment_mode not in {"none", "scale", "scale_shift"}:
            self.xyz_alignment_mode = "none"
            if self.logger is not None:
                self.logger.warning(
                    "logging.visualization.xyz_alignment_mode is invalid; fallback to 'none'."
                )

        if self.enabled and self.save_png:
            self.root.mkdir(parents=True, exist_ok=True)
        if self.enabled and self.tb_writer is None and self.logger is not None:
            self.logger.warning("logging.visualization enabled but TensorBoard writer is missing; only PNG files will be saved.")

    def should_log_train(self, step: int) -> bool:
        if not self.enabled:
            return False
        if self.train_every_steps <= 0:
            return False
        if int(step) == self._last_train_step:
            return False
        return int(step) % self.train_every_steps == 0

    def should_log_val(self, step: int) -> bool:
        if not self.enabled or not self.on_validation:
            return False
        if int(step) == self._last_val_step:
            return False
        return True

    def _save_fig_and_maybe_tb(self, fig: Any, save_path: Path | None, tb_tag: str | None, step: int) -> None:
        image = _figure_to_rgb(fig)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
        plt.close(fig)
        if self.tb_writer is not None and tb_tag:
            chw = np.transpose(image, (2, 0, 1))
            self.tb_writer.add_image(tb_tag, chw, int(step))

    def _tensor_sample(self, value: Any, batch_index: int) -> np.ndarray:
        if not torch.is_tensor(value):
            return np.asarray(value)
        sampled = value[batch_index].detach().cpu().numpy()
        return sampled

    def _event_dir(self, split: str, step: int) -> Path:
        return self.root / split / f"step_{int(step):07d}"

    def _log_case(self, batch: dict[str, Any], outputs: dict[str, Any], split: str, step: int, case_idx: int, batch_index: int) -> dict[str, Any] | None:
        video = self._tensor_sample(batch["video"], batch_index)
        query = batch.get("query", {})
        target = batch.get("target", {})
        mask = batch.get("mask", {})
        camera = batch.get("camera", {})
        query_stats = batch.get("query_stats", {})
        meta = batch.get("meta", {})

        t_tgt = self._tensor_sample(query["t_tgt"], batch_index).astype(np.int64)
        mask_uv = self._tensor_sample(mask["uv_2d"], batch_index).astype(bool)
        pred_uv = self._tensor_sample(outputs["uv_2d"], batch_index)
        tgt_uv = self._tensor_sample(target["uv_2d"], batch_index)
        hard_flags_all = None
        if isinstance(query_stats, dict) and "is_hard_query" in query_stats:
            hard_flags_all = self._tensor_sample(query_stats["is_hard_query"], batch_index).astype(bool)

        selected_mask, frame_idx = _select_query_indices(t_tgt, mask_uv)
        if not selected_mask.any():
            return None
        ids = _subsample_indices(selected_mask, max_points=self.max_points)
        if ids.size == 0:
            return None

        frame = np.asarray(video[frame_idx], dtype=np.float32).transpose(1, 2, 0)
        h, w = frame.shape[:2]
        gt = np.asarray(tgt_uv[ids], dtype=np.float64)
        pred = np.asarray(pred_uv[ids], dtype=np.float64)
        gt_xy = np.stack([np.clip(gt[:, 0], 0.0, 1.0) * (w - 1), np.clip(gt[:, 1], 0.0, 1.0) * (h - 1)], axis=-1)
        pred_xy = np.stack([np.clip(pred[:, 0], 0.0, 1.0) * (w - 1), np.clip(pred[:, 1], 0.0, 1.0) * (h - 1)], axis=-1)
        err_px = np.linalg.norm(pred_xy - gt_xy, axis=-1)
        hard_flags = hard_flags_all[ids] if hard_flags_all is not None else None

        dataset = _meta_value(meta, "dataset", batch_index) or "unknown_dataset"
        scene = _meta_value(meta, "scene_id", batch_index) or "unknown_scene"
        sample_key = _meta_value(meta, "sample_key", batch_index) or f"sample_{batch_index}"
        prefix = f"case{case_idx:02d}_{_safe_name(dataset)}_{_safe_name(scene)}_t{int(frame_idx)}"
        event_dir = self._event_dir(split=split, step=step) if self.save_png else None

        overlay_title = (
            f"{dataset}/{scene} t_tgt={int(frame_idx)} n={ids.size}\n"
            f"mean={float(np.mean(err_px)):.2f}px median={float(np.median(err_px)):.2f}px"
        )
        fig_overlay = _plot_overlay_2d(
            image=frame,
            gt_xy=gt_xy,
            pred_xy=pred_xy,
            errors_px=err_px,
            hard_flags=hard_flags,
            title=overlay_title,
            max_lines=self.max_lines,
        )
        overlay_path = (event_dir / f"{prefix}_overlay_2d.png") if event_dir is not None else None
        self._save_fig_and_maybe_tb(
            fig=fig_overlay,
            save_path=overlay_path,
            tb_tag=f"vis/{split}/case_{case_idx}/overlay_2d",
            step=step,
        )

        record: dict[str, Any] = {
            "split": split,
            "step": int(step),
            "dataset": dataset,
            "scene_id": scene,
            "sample_key": sample_key,
            "t_tgt": int(frame_idx),
            "num_points": int(ids.size),
            "uv_error_mean_px": float(np.mean(err_px)),
            "uv_error_median_px": float(np.median(err_px)),
            "uv_error_p90_px": float(np.quantile(err_px, 0.90)),
            "pck_2px": float(np.mean((err_px <= 2.0).astype(np.float64))),
            "pck_4px": float(np.mean((err_px <= 4.0).astype(np.float64))),
            "overlay_2d_path": str(overlay_path) if overlay_path is not None else None,
            "error_3d_on_image_path": None,
            "pairs_3d_path": None,
            "num_points_3d": 0,
            "xyz_error_mean_m": None,
            "xyz_error_median_m": None,
            "xyz_error_p90_m": None,
            "xyz_pck_2cm": None,
            "xyz_pck_5cm": None,
            "xyz_pck_10cm": None,
            "xyz_error_mean_m_raw": None,
            "xyz_error_median_m_raw": None,
            "xyz_error_p90_m_raw": None,
            "xyz_alignment_mode": self.xyz_alignment_mode,
            "xyz_alignment_applied": False,
            "xyz_alignment_scale": 1.0,
            "xyz_alignment_shift_z": 0.0,
            "xyz_alignment_num_points": 0,
        }

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f"vis/{split}/case_{case_idx}/uv_error_mean_px", record["uv_error_mean_px"], int(step))
            self.tb_writer.add_scalar(f"vis/{split}/case_{case_idx}/uv_error_median_px", record["uv_error_median_px"], int(step))

        record["overlay_2d_3d_proj_path"] = None
        record["reproj_uv_error_mean_px"] = None
        record["reproj_uv_error_median_px"] = None
        if (
            isinstance(camera, dict)
            and "xyz_3d" in outputs
            and all(key in camera for key in ("K", "T_wc", "camera_valid"))
            and all(key in query for key in ("t_cam", "t_tgt"))
        ):
            pred_xyz_all = self._tensor_sample(outputs["xyz_3d"], batch_index)
            k_seq = self._tensor_sample(camera["K"], batch_index)
            t_wc_seq = self._tensor_sample(camera["T_wc"], batch_index)
            camera_valid = self._tensor_sample(camera["camera_valid"], batch_index).astype(bool)
            t_cam_all = self._tensor_sample(query["t_cam"], batch_index).astype(np.int64)
            uv_proj_all, proj_valid_all = _project_pred_xyz_to_tgt_uv_np(
                pred_xyz_tcam=pred_xyz_all,
                k_seq=k_seq,
                t_wc_seq=t_wc_seq,
                camera_valid=camera_valid,
                t_cam=t_cam_all,
                t_tgt=t_tgt,
                image_h=h,
                image_w=w,
            )
            reproj_valid = proj_valid_all[ids] & np.isfinite(gt).all(axis=-1)
            if reproj_valid.any():
                uv_proj = np.asarray(uv_proj_all[ids], dtype=np.float64)
                proj_xy = np.stack(
                    [uv_proj[:, 0] * (w - 1), uv_proj[:, 1] * (h - 1)],
                    axis=-1,
                )
                proj_xy_v = proj_xy[reproj_valid]
                gt_xy_v = gt_xy[reproj_valid]
                err_proj_px = np.linalg.norm(proj_xy_v - gt_xy_v, axis=-1)
                fig_overlay_proj = _plot_overlay_2d(
                    image=frame,
                    gt_xy=gt_xy_v,
                    pred_xy=proj_xy_v,
                    errors_px=err_proj_px,
                    hard_flags=(hard_flags[reproj_valid] if hard_flags is not None else None),
                    title=(
                        f"{dataset}/{scene} xyz->gtcam reproj t_tgt={int(frame_idx)} n={int(reproj_valid.sum())}\n"
                        f"mean={float(np.mean(err_proj_px)):.2f}px median={float(np.median(err_proj_px)):.2f}px"
                    ),
                    max_lines=self.max_lines,
                )
                overlay_proj_path = (event_dir / f"{prefix}_overlay_2d_3d_proj.png") if event_dir is not None else None
                self._save_fig_and_maybe_tb(
                    fig=fig_overlay_proj,
                    save_path=overlay_proj_path,
                    tb_tag=f"vis/{split}/case_{case_idx}/overlay_2d_3d_proj",
                    step=step,
                )
                record["overlay_2d_3d_proj_path"] = str(overlay_proj_path) if overlay_proj_path is not None else None
                record["reproj_uv_error_mean_px"] = float(np.mean(err_proj_px))
                record["reproj_uv_error_median_px"] = float(np.median(err_proj_px))
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        f"vis/{split}/case_{case_idx}/reproj_uv_error_mean_px",
                        record["reproj_uv_error_mean_px"],
                        int(step),
                    )
                    self.tb_writer.add_scalar(
                        f"vis/{split}/case_{case_idx}/reproj_uv_error_median_px",
                        record["reproj_uv_error_median_px"],
                        int(step),
                    )

        if self.include_3d and "xyz_3d" in outputs and "xyz_3d" in target and "xyz_3d" in mask:
            pred_xyz_all_raw = self._tensor_sample(outputs["xyz_3d"], batch_index)
            tgt_xyz_all = self._tensor_sample(target["xyz_3d"], batch_index)
            mask_xyz_all = self._tensor_sample(mask["xyz_3d"], batch_index).astype(bool)
            pred_xyz_all, align_info = _align_pred_xyz_sequence(
                pred_xyz=pred_xyz_all_raw,
                gt_xyz=tgt_xyz_all,
                mask_xyz=mask_xyz_all,
                mode=self.xyz_alignment_mode,
                min_points=self.xyz_alignment_min_points,
            )
            record["xyz_alignment_mode"] = align_info["mode"]
            record["xyz_alignment_applied"] = bool(align_info["applied"])
            record["xyz_alignment_scale"] = float(align_info["scale"])
            record["xyz_alignment_shift_z"] = float(align_info["shift_z"])
            record["xyz_alignment_num_points"] = int(align_info["num_points"])

            pred_xyz_raw = pred_xyz_all_raw[ids]
            pred_xyz = pred_xyz_all[ids]
            tgt_xyz = tgt_xyz_all[ids]
            mask_xyz = mask_xyz_all[ids].astype(bool)
            valid3d = mask_xyz & np.isfinite(pred_xyz).all(axis=-1) & np.isfinite(tgt_xyz).all(axis=-1)
            if valid3d.any():
                pred_xyz_raw_v = pred_xyz_raw[valid3d]
                pred_xyz_v = pred_xyz[valid3d]
                tgt_xyz_v = tgt_xyz[valid3d]
                gt_xy_v = gt_xy[valid3d]
                err_m_raw = np.linalg.norm(np.asarray(pred_xyz_raw_v, dtype=np.float64) - np.asarray(tgt_xyz_v, dtype=np.float64), axis=-1)
                err_m = np.linalg.norm(np.asarray(pred_xyz_v, dtype=np.float64) - np.asarray(tgt_xyz_v, dtype=np.float64), axis=-1)

                fig_on_image = _plot_3d_on_image(
                    image=frame,
                    gt_xy=gt_xy_v,
                    errors_m=err_m,
                    title=f"{dataset}/{scene} 3D error on image ({align_info['mode']})",
                )
                on_image_path = (event_dir / f"{prefix}_3d_on_image.png") if event_dir is not None else None
                self._save_fig_and_maybe_tb(
                    fig=fig_on_image,
                    save_path=on_image_path,
                    tb_tag=f"vis/{split}/case_{case_idx}/error_3d_on_image",
                    step=step,
                )

                fig_pairs = _plot_3d_pairs(
                    gt_xyz=tgt_xyz_v,
                    pred_xyz=pred_xyz_v,
                    errors_m=err_m,
                    title=f"{dataset}/{scene} 3D pairs ({align_info['mode']})",
                    max_lines=self.max_lines,
                )
                pairs_path = (event_dir / f"{prefix}_3d_pairs.png") if event_dir is not None else None
                self._save_fig_and_maybe_tb(
                    fig=fig_pairs,
                    save_path=pairs_path,
                    tb_tag=f"vis/{split}/case_{case_idx}/pairs_3d",
                    step=step,
                )

                record["error_3d_on_image_path"] = str(on_image_path) if on_image_path is not None else None
                record["pairs_3d_path"] = str(pairs_path) if pairs_path is not None else None
                record["num_points_3d"] = int(err_m.shape[0])
                record["xyz_error_mean_m"] = float(np.mean(err_m))
                record["xyz_error_median_m"] = float(np.median(err_m))
                record["xyz_error_p90_m"] = float(np.quantile(err_m, 0.90))
                record["xyz_pck_2cm"] = float(np.mean((err_m <= 0.02).astype(np.float64)))
                record["xyz_pck_5cm"] = float(np.mean((err_m <= 0.05).astype(np.float64)))
                record["xyz_pck_10cm"] = float(np.mean((err_m <= 0.10).astype(np.float64)))
                record["xyz_error_mean_m_raw"] = float(np.mean(err_m_raw))
                record["xyz_error_median_m_raw"] = float(np.median(err_m_raw))
                record["xyz_error_p90_m_raw"] = float(np.quantile(err_m_raw, 0.90))
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(f"vis/{split}/case_{case_idx}/xyz_error_mean_m", record["xyz_error_mean_m"], int(step))
                    self.tb_writer.add_scalar(f"vis/{split}/case_{case_idx}/xyz_error_median_m", record["xyz_error_median_m"], int(step))
                    self.tb_writer.add_scalar(f"vis/{split}/case_{case_idx}/xyz_pck_5cm", record["xyz_pck_5cm"], int(step))
                    self.tb_writer.add_scalar(
                        f"vis/{split}/case_{case_idx}/xyz_error_mean_m_raw",
                        record["xyz_error_mean_m_raw"],
                        int(step),
                    )

        return record

    def log_batch(self, batch: dict[str, Any], outputs: dict[str, Any], split: str, step: int) -> None:
        if not self.enabled:
            return
        video = batch.get("video")
        if not torch.is_tensor(video) or video.ndim < 1:
            return
        bsz = int(video.shape[0])
        case_count = min(self.max_samples, bsz)
        records: list[dict[str, Any]] = []
        for i in range(case_count):
            try:
                rec = self._log_case(batch=batch, outputs=outputs, split=split, step=step, case_idx=i, batch_index=i)
            except Exception as exc:  # pragma: no cover - defensive
                if self.logger is not None:
                    self.logger.exception("Visualization failed at %s step=%d case=%d: %s", split, step, i, exc)
                continue
            if rec is not None:
                records.append(rec)

        if split == "train":
            self._last_train_step = int(step)
        elif split == "val":
            self._last_val_step = int(step)

        if self.save_png and records:
            event_dir = self._event_dir(split=split, step=step)
            event_dir.mkdir(parents=True, exist_ok=True)
            summary_path = event_dir / "summary.json"
            payload = {
                "split": split,
                "step": int(step),
                "num_records": len(records),
                "records": records,
            }
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if self.logger is not None and records:
            self.logger.info("Saved %d visualization case(s) for %s at step=%d", len(records), split, int(step))
