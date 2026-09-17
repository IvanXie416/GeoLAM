#!/usr/bin/env python3
"""Viser viewer for a generated vis_like_demo package."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a vis_like_demo package with viser.")
    parser.add_argument("--root", required=True, help="Demo package root directory.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--point-budget", type=int, default=30000, help="Max rendered cloud points per frame.")
    parser.add_argument("--track-history", type=int, default=0, help="0 means full history.")
    return parser.parse_args()


def _with_image_opacity(image_rgb_u8: np.ndarray, opacity: float) -> np.ndarray:
    img = np.asarray(image_rgb_u8, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    alpha = float(np.clip(opacity, 0.0, 1.0))
    if alpha >= 0.999:
        return img
    out = np.empty((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    out[:, :, :3] = img
    out[:, :, 3] = np.uint8(round(alpha * 255.0))
    return out


def _rotmat_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return tuple(float(v) for v in q.tolist())


def _jump_client_view_to_camera_pose(
    client: Any,
    t_wc: np.ndarray,
    *,
    fov: float | None = None,
    fallback_look_distance: float = 1.0,
) -> None:
    cam = getattr(client, "camera", None)
    if cam is None:
        return
    pose = np.asarray(t_wc, dtype=np.float64).reshape(4, 4)
    rot = pose[:3, :3]
    pos = pose[:3, 3]
    fwd = rot[:, 2]
    up = -rot[:, 1]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
    up = up / max(np.linalg.norm(up), 1e-12)
    look_distance = float(max(fallback_look_distance, 1e-3))
    try:
        cur_pos = np.asarray(cam.position, dtype=np.float64)
        cur_look = np.asarray(cam.look_at, dtype=np.float64)
        cur_dist = float(np.linalg.norm(cur_look - cur_pos))
        if np.isfinite(cur_dist) and cur_dist > 1e-3:
            look_distance = cur_dist
    except Exception:
        pass
    look_at = pos + fwd * look_distance
    try:
        if fov is not None and np.isfinite(float(fov)) and float(fov) > 1e-6:
            cam.fov = float(fov)
        cam.position = tuple(float(x) for x in pos.tolist())
        cam.look_at = tuple(float(x) for x in look_at.tolist())
        cam.up_direction = tuple(float(x) for x in up.tolist())
    except Exception:
        return


def _load_demo(root: Path) -> tuple[dict[str, Any], np.ndarray]:
    data = json.loads((root / "assets" / "demo_data.json").read_text())
    video_path = root / "assets" / "input_video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        poster = root / "assets" / "video_poster.jpg"
        if poster.exists():
            img = cv2.imread(str(poster), cv2.IMREAD_COLOR)
            if img is not None:
                frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    if not frames:
        raise RuntimeError(f"Failed to decode video frames from {video_path}")
    return data, np.stack(frames, axis=0)


def _np(x: Any, dtype: np.dtype | None = None) -> np.ndarray:
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def _track_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cols = []
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 1.0)
        cols.append([int(round(c * 255.0)) for c in rgb])
    return np.asarray(cols, dtype=np.uint8)


def _lighten_colors(colors: np.ndarray, amount: float = 0.4) -> np.ndarray:
    cols = np.asarray(colors, dtype=np.float32)
    mixed = cols * float(max(0.0, 1.0 - amount)) + 255.0 * float(np.clip(amount, 0.0, 1.0))
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _subsample_indices(total: int, keep: int) -> np.ndarray:
    keep = max(0, min(int(keep), int(total)))
    if keep <= 0:
        return np.zeros((0,), dtype=np.int64)
    if keep >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=keep, dtype=np.int64)


def _make_source_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _fov_from_k(k: np.ndarray, image_h: int) -> float:
    kk = np.asarray(k, dtype=np.float64).reshape(3, 3)
    fy = float(kk[1, 1])
    if not np.isfinite(fy) or abs(fy) < 1e-6:
        return math.radians(50.0)
    return float(2.0 * math.atan2(float(image_h) * 0.5, fy))


def _worldtrack_scale_from_meta(meta: dict[str, Any]) -> float | None:
    worldtrack_meta = meta.get("worldtrack", None)
    if not isinstance(worldtrack_meta, dict):
        return None
    track_alignment = worldtrack_meta.get("trackAlignment", None)
    if not isinstance(track_alignment, dict):
        return None
    scale = track_alignment.get("scale", None)
    if scale is None:
        return None
    scale = float(scale)
    return scale if np.isfinite(scale) and scale > 0.0 else None


def _pred_camera_translation_already_aligned(meta: dict[str, Any]) -> bool:
    worldtrack_meta = meta.get("worldtrack", None)
    if not isinstance(worldtrack_meta, dict):
        return False
    pred_alignment = worldtrack_meta.get("predCameraAlignment", None)
    return isinstance(pred_alignment, dict) and pred_alignment.get("scale", None) is not None


def _camera_pose_from_meta(
    meta: dict[str, Any],
    frame_idx: int,
    camera_source: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    order: list[tuple[str, Any]] = []
    if camera_source == "gt":
        order = [("gt", meta.get("camera", None)), ("pred", meta.get("cameraPred", None))]
    elif camera_source == "pred":
        order = [("pred", meta.get("cameraPred", None)), ("gt", meta.get("camera", None))]
    else:
        order = [("gt", meta.get("camera", None)), ("pred", meta.get("cameraPred", None))]

    for resolved_source, payload in order:
        if isinstance(payload, dict) and payload.get("K") is not None and payload.get("TRef0Cam") is not None:
            k_seq = _np(payload["K"], np.float32)
            t_seq = _np(payload["TRef0Cam"], np.float32).copy()
            if resolved_source == "pred" and not _pred_camera_translation_already_aligned(meta):
                legacy_scale = _worldtrack_scale_from_meta(meta)
                if legacy_scale is not None:
                    t_seq[:, :3, 3] *= float(legacy_scale)
            idx = int(np.clip(frame_idx, 0, max(k_seq.shape[0] - 1, 0)))
            return k_seq[idx], t_seq[idx], resolved_source
    return _np(meta.get("ref0K", np.eye(3, dtype=np.float32)), np.float32), _make_source_pose(), "ref0"


def main() -> int:
    args = parse_args()
    try:
        import viser
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: `viser` is not installed in current env. "
            "Install requirements.txt before running this viewer."
        ) from exc

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    data, video = _load_demo(root)
    meta = data["meta"]
    num_frames = int(meta["numFrames"])
    fps = float(meta["fps"])
    point_xyz = _np(data["points"]["xyzRef0"], np.float32)
    if "pointsRaw" not in data and meta.get("worldtrack", None) is not None:
        legacy_scale = _worldtrack_scale_from_meta(meta)
        if legacy_scale is not None:
            point_xyz = point_xyz * float(legacy_scale)
    point_vis = _np(data["points"]["visibility"], bool)
    point_rgb = _np(data["points"]["rgb"], np.uint8)
    point_dynamic = _np(data["points"].get("isDynamic", np.ones((point_xyz.shape[0], point_xyz.shape[1]), dtype=np.int32)), bool)
    track_xyz = _np(data["tracks"]["xyzRef0"], np.float32)
    track_vis = _np(data["tracks"]["visibility"], bool)
    track_colors = _track_colors(int(track_xyz.shape[0]))
    track_gt_meta = data.get("tracksGt", None)
    track_gt_xyz = _np(track_gt_meta["xyzRef0"], np.float32) if isinstance(track_gt_meta, dict) and track_gt_meta.get("xyzRef0") is not None else np.zeros((0, num_frames, 3), dtype=np.float32)
    track_gt_vis = _np(track_gt_meta["visibility"], bool) if isinstance(track_gt_meta, dict) and track_gt_meta.get("visibility") is not None else np.zeros((0, num_frames), dtype=bool)
    track_gt_colors = _lighten_colors(_track_colors(int(track_gt_xyz.shape[0])), amount=0.45)

    radius = float(meta["bounds"]["radius"])
    center = np.asarray(meta["bounds"]["center"], dtype=np.float32)
    frustum_aspect = float(meta["videoWidth"]) / max(float(meta["videoHeight"]), 1.0)
    frustum_scale = max(radius * 0.12, 0.25)

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction("+y")
    server.scene.add_grid(
        "/grid",
        width=max(radius * 2.4, 4.0),
        height=max(radius * 2.4, 4.0),
        width_segments=20,
        height_segments=20,
        plane="xz",
    )
    server.scene.add_frame(
        "/world",
        axes_length=max(radius * 0.35, 1.0),
        axes_radius=max(radius * 0.02, 0.01),
        position=tuple(float(x) for x in center.tolist()),
    )

    with server.gui.add_folder("Timeline", expand_by_default=True):
        frame_slider = server.gui.add_slider("frame_idx", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0)
        prev_btn = server.gui.add_button("prev_frame")
        next_btn = server.gui.add_button("next_frame")
        play_box = server.gui.add_checkbox("play", initial_value=False)
        loop_box = server.gui.add_checkbox("loop", initial_value=True)
        fps_slider = server.gui.add_slider("fps", min=1, max=30, step=1, initial_value=int(np.clip(round(fps), 1, 30)))

    with server.gui.add_folder("Display", expand_by_default=True):
        camera_source = server.gui.add_dropdown(
            "camera_source",
            options=("gt", "pred", "auto"),
            initial_value=("gt" if isinstance(meta.get("camera"), dict) else ("pred" if isinstance(meta.get("cameraPred"), dict) else "auto")),
        )
        show_cloud = server.gui.add_checkbox("show_cloud", initial_value=True)
        show_background = server.gui.add_checkbox("show_background", initial_value=True)
        cloud_size = server.gui.add_slider("cloud_size_scale", min=0.2, max=3.0, step=0.1, initial_value=1.0)
        show_pred_tracks = server.gui.add_checkbox("show_pred_tracks", initial_value=True)
        show_gt_tracks = server.gui.add_checkbox("show_gt_tracks", initial_value=bool(track_gt_xyz.shape[0] > 0))
        track_head_size = server.gui.add_slider("track_point_size_scale", min=0.2, max=3.0, step=0.1, initial_value=1.0)
        track_opacity = server.gui.add_slider("track_opacity", min=0.0, max=1.0, step=0.05, initial_value=0.9)
        show_frustum = server.gui.add_checkbox("show_frustum", initial_value=True)
        frustum_opacity = server.gui.add_slider("frustum_opacity", min=0.0, max=1.0, step=0.05, initial_value=0.55)
        click_jump = server.gui.add_checkbox("click_frustum_to_jump", initial_value=True)
        jump_btn = server.gui.add_button("jump_to_frustum_view")

    with server.gui.add_folder("Video", expand_by_default=True):
        frame_image = server.gui.add_image(video[min(0, video.shape[0] - 1)], label="rgb_frame")

    render_handles: list[Any] = []
    render_lock = threading.Lock()

    def clear_scene() -> None:
        nonlocal render_handles
        for handle in render_handles:
            try:
                handle.remove()
            except Exception:
                pass
        render_handles = []

    def add_handle(h: Any) -> None:
        render_handles.append(h)

    def _render_track_layer(
        *,
        name_prefix: str,
        xyz: np.ndarray,
        vis: np.ndarray,
        colors: np.ndarray,
        visible: bool,
        is_gt: bool,
        frame_idx: int,
    ) -> None:
        if not visible or int(xyz.shape[0]) <= 0:
            return
        hist = int(args.track_history)
        t0 = 0 if hist <= 0 else max(0, frame_idx - hist + 1)
        segs = []
        seg_cols = []
        head_pts = []
        head_cols = []
        for qi in range(xyz.shape[0]):
            pts = []
            for ti in range(t0, frame_idx + 1):
                if not bool(vis[qi, ti]):
                    continue
                p = xyz[qi, ti]
                if not np.isfinite(p).all():
                    continue
                pts.append(p.astype(np.float32))
            if len(pts) >= 2:
                pts_arr = np.asarray(pts, dtype=np.float32)
                seg = np.stack([pts_arr[:-1], pts_arr[1:]], axis=1)
                segs.append(seg)
                col = np.repeat(colors[qi][None, None, :], repeats=seg.shape[0], axis=0)
                seg_cols.append(np.repeat(col, repeats=2, axis=1))
            if len(pts) >= 1:
                head_pts.append(pts[-1])
                head_cols.append(colors[qi])
        if segs:
            add_handle(
                server.scene.add_line_segments(
                    f"/demo/{name_prefix}/lines",
                    points=np.concatenate(segs, axis=0).astype(np.float32),
                    colors=np.concatenate(seg_cols, axis=0).astype(np.uint8),
                    line_width=max(1.0, 4.0 * float(track_opacity.value) * (0.85 if is_gt else 1.0)),
                )
            )
        if head_pts:
            add_handle(
                server.scene.add_point_cloud(
                    f"/demo/{name_prefix}/heads",
                    points=np.asarray(head_pts, dtype=np.float32),
                    colors=np.asarray(head_cols, dtype=np.uint8),
                    point_size=float(max(radius * 0.008, 0.01) * float(track_head_size.value) * (0.85 if is_gt else 1.0)),
                    point_shape="sparkle",
                    precision="float32",
                )
            )

    def render() -> None:
        with render_lock:
            clear_scene()
            t = int(frame_slider.value)
            frame_image.image = video[min(t, video.shape[0] - 1)]

            if bool(show_cloud.value):
                valid = point_vis[t] & np.isfinite(point_xyz[t]).all(axis=-1)
                if not bool(show_background.value):
                    if point_dynamic.ndim == 2:
                        valid = valid & point_dynamic[t]
                    else:
                        valid = valid & point_dynamic
                idx = np.flatnonzero(valid)
                if idx.size > int(args.point_budget):
                    idx = idx[_subsample_indices(idx.size, int(args.point_budget))]
                if idx.size > 0:
                    add_handle(
                        server.scene.add_point_cloud(
                            "/demo/cloud",
                            points=point_xyz[t, idx].astype(np.float32),
                            colors=point_rgb[t, idx].astype(np.uint8),
                            point_size=float(max(radius * 0.0035, 0.003) * float(cloud_size.value)),
                            point_shape="circle",
                            precision="float32",
                        )
                    )

            _render_track_layer(
                name_prefix="tracks_pred",
                xyz=track_xyz,
                vis=track_vis,
                colors=track_colors,
                visible=bool(show_pred_tracks.value),
                is_gt=False,
                frame_idx=t,
            )
            _render_track_layer(
                name_prefix="tracks_gt",
                xyz=track_gt_xyz,
                vis=track_gt_vis,
                colors=track_gt_colors,
                visible=bool(show_gt_tracks.value),
                is_gt=True,
                frame_idx=t,
            )

            if bool(show_frustum.value):
                ref0_k, source_pose, resolved_camera_source = _camera_pose_from_meta(meta, t, str(camera_source.value))
                frustum_fov = _fov_from_k(ref0_k, int(meta["videoHeight"]))
                image_rgba = _with_image_opacity(video[min(t, video.shape[0] - 1)], float(frustum_opacity.value))
                frustum = server.scene.add_camera_frustum(
                    "/demo/source_frustum",
                    fov=float(frustum_fov),
                    aspect=float(frustum_aspect),
                    scale=float(frustum_scale),
                    color=(255, 255, 255),
                    image=image_rgba,
                    wxyz=_rotmat_to_wxyz(source_pose[:3, :3]),
                    position=tuple(float(x) for x in source_pose[:3, 3].tolist()),
                )
                add_handle(frustum)
                frustum_label = server.scene.add_label(
                    "/demo/source_frustum_label",
                    text=f"camera={resolved_camera_source}",
                    position=tuple(float(x) for x in source_pose[:3, 3].tolist()),
                )
                add_handle(frustum_label)
                add_handle(
                    server.scene.add_frame(
                        "/demo/source_frame",
                        axes_length=float(frustum_scale) * 0.35,
                        axes_radius=float(frustum_scale) * 0.03,
                        origin_color=(255, 255, 255),
                        wxyz=_rotmat_to_wxyz(source_pose[:3, :3]),
                        position=tuple(float(x) for x in source_pose[:3, 3].tolist()),
                    )
                )

                @frustum.on_click
                def _on_click(event: Any) -> None:
                    if bool(click_jump.value):
                        _jump_client_view_to_camera_pose(
                            event.client,
                            source_pose,
                            fov=float(frustum_fov),
                            fallback_look_distance=max(radius * 0.8, 1.0),
                        )

    def _step(delta: int) -> None:
        max_frame = int(frame_slider.max)
        cur = int(frame_slider.value)
        nxt = cur + int(delta)
        if nxt < 0:
            nxt = max_frame if bool(loop_box.value) else 0
        if nxt > max_frame:
            nxt = 0 if bool(loop_box.value) else max_frame
        frame_slider.value = nxt

    @prev_btn.on_click
    def _(_event: Any) -> None:
        _step(-1)

    @next_btn.on_click
    def _(_event: Any) -> None:
        _step(1)

    @jump_btn.on_click
    def _(_event: Any) -> None:
        _k0, source_pose, _resolved = _camera_pose_from_meta(meta, int(frame_slider.value), str(camera_source.value))
        frustum_fov = _fov_from_k(_k0, int(meta["videoHeight"]))
        for client in server.get_clients().values():
            _jump_client_view_to_camera_pose(
                client,
                source_pose,
                fov=float(frustum_fov),
                fallback_look_distance=max(radius * 0.8, 1.0),
            )

    widgets = [
        frame_slider, play_box, loop_box, fps_slider,
        camera_source,
        show_cloud, show_background, cloud_size,
        show_pred_tracks, show_gt_tracks, track_head_size, track_opacity,
        show_frustum, frustum_opacity, click_jump,
    ]
    for widget in widgets:
        widget.on_update(lambda _event: render())

    def playback_loop() -> None:
        while True:
            time.sleep(1.0 / max(float(fps_slider.value), 1.0))
            if not bool(play_box.value):
                continue
            _step(1)

    threading.Thread(target=playback_loop, daemon=True).start()

    print(f"[serve_demo_viser] http://{args.host}:{args.port}")
    print(f"[serve_demo_viser] root={root}")
    render()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
