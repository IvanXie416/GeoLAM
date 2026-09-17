# Unified Data Schema for D4RT Query Supervision

This document defines a single canonical schema so all source datasets are converted into one query-supervision format for D4RT.

## 1. Canonical Clip Package

Each sample is one fixed-length clip.

Directory layout:

```text
data/processed/{dataset_name}/{split}/{scene_id}/{clip_id}/
  meta.json
  frames.npz
  supervision.npz
  query_pool.npz
```

## 2. Coordinate and Index Conventions

- Time index: 0-based integer in `[0, T-1]`.
- Pixel coordinates:
  - `u, v` are normalized to `[0, 1]`.
  - `u` maps to width axis (x), `v` maps to height axis (y).
- Camera coordinates:
  - Right-handed.
  - `+x`: right, `+y`: down, `+z`: forward.
- Rigid transform naming:
  - `T_wc[t]`: 4x4 transform from camera at frame `t` to world.
  - `T_cw[t] = inv(T_wc[t])`.
- Units:
  - Depth and 3D coordinates are in meters.

## 3. File Schema

### 3.1 `meta.json`

Required fields:

```json
{
  "dataset": "scannet",
  "split": "train",
  "scene_id": "scene0000_00",
  "clip_id": "clip_000123",
  "T": 48,
  "H": 256,
  "W": 256,
  "fps": 30.0,
  "orig_height": 968,
  "orig_width": 1296,
  "frame_indices_in_source_video": [100, 102, 104, "..."],
  "has_depth": true,
  "has_normals": false,
  "has_visibility": true,
  "has_tracks": true,
  "has_camera": true,
  "camera_model": "pinhole",
  "distortion_model": "none"
}
```

### 3.2 `frames.npz`

Required arrays:

- `rgb_uint8`: shape `[T, H, W, 3]`, dtype `uint8`.
- `aspect_ratio_wh`: shape `[1]`, dtype `float32`, value `W / H` before resize.

### 3.3 `supervision.npz`

Dense supervision arrays:

- `depth_m`: `[T, H, W]`, `float32`, `NaN` for invalid.
- `depth_valid`: `[T, H, W]`, `bool`.
- `K`: `[T, 3, 3]`, `float32`, per-frame intrinsics (or `NaN` if unavailable).
- `T_wc`: `[T, 4, 4]`, `float32`, per-frame extrinsics (or `NaN` if unavailable).
- `camera_valid`: `[T]`, `bool`.

Optional dense arrays:

- `normal_cam`: `[T, H, W, 3]`, `float32`, unit vectors, `NaN` where invalid.
- `normal_valid`: `[T, H, W]`, `bool`.
- `motion_cam`: `[T, H, W, 3]`, `float32`, displacement vector in camera coordinates.
- `motion_valid`: `[T, H, W]`, `bool`.

Sparse correspondence arrays:

- `track_uv_norm`: `[N_track, T, 2]`, `float32`, normalized UV for each track and frame.
- `track_xyz_cam`: `[N_track, T, 3]`, `float32`, 3D points in each frame's own camera coordinates.
- `track_visible`: `[N_track, T]`, `bool`.
- `track_valid`: `[N_track, T]`, `bool`.

### 3.4 `query_pool.npz`

Canonical training/eval query-supervision table.

Query fields:

- `q_u_src`: `[M]`, `float32`, normalized.
- `q_v_src`: `[M]`, `float32`, normalized.
- `q_t_src`: `[M]`, `int16`.
- `q_t_tgt`: `[M]`, `int16`.
- `q_t_cam`: `[M]`, `int16`.

Target fields:

- `y_xyz_cam_tcam`: `[M, 3]`, `float32`.
- `y_uv_tgt`: `[M, 2]`, `float32`.
- `y_vis_tgt`: `[M]`, `float32` in `{0.0, 1.0}`.
- `y_disp_cam_tcam`: `[M, 3]`, `float32`, displacement from `(tsrc->ttgt)` represented in `tcam`.
- `y_normal_tgt_cam_tcam`: `[M, 3]`, `float32`, unit vector.

Validity masks:

- `m_xyz`: `[M]`, `bool`.
- `m_uv`: `[M]`, `bool`.
- `m_vis`: `[M]`, `bool`.
- `m_disp`: `[M]`, `bool`.
- `m_normal`: `[M]`, `bool`.

Hard-sampling helpers:

- `is_hard_boundary_query`: `[M]`, `bool`, computed from Sobel boundaries on depth/motion.

## 4. Query Construction Rules (Training)

Per clip:

- Sample `N=2048` queries each step.
- `30%` queries from boundary regions (`is_hard_boundary_query=true`).
- Sample `tsrc, ttgt, tcam` uniformly, except enforce `ttgt == tcam` with probability `0.4`.
- For depth-only training queries: enforce `tsrc == ttgt == tcam`.

## 5. Dataset Adapter Contract

Each raw dataset adapter must output canonical files above.

Required adapter interface:

```text
convert_raw_scene_to_clips(raw_scene_dir, out_dir, split, clip_len=48, out_hw=(256,256))
```

Adapter responsibilities:

- Decode raw frames and resize/crop to target resolution.
- Convert all geometry to meter scale.
- Convert camera poses to `T_wc`.
- Populate sparse tracks where available.
- Build `query_pool.npz` with valid masks.

## 6. Dataset-Specific Notes

- ScanNet / ScanNet++:
  - Rich depth and camera. Static-heavy scenes.
  - Use camera and depth as primary supervision.
- Sintel:
  - Dynamic scenes and full rendering effects.
  - Useful for depth and pose benchmarking.
- TartanAir / VirtualKITTI / Waymo / Kubric:
  - Synthetic/automotive style geometry is useful for motion and correspondences.
- Co3Dv2 / Dynamic Replica / PointOdyssey / MVS-Synth / BlendedMVS:
  - Convert whatever supervision exists into canonical sparse/dense fields.
  - Missing fields must be represented by masks, never by dropping samples.

## 7. Data Validation Checklist

A clip passes validation only if:

- `T=48`, `H=W=256`.
- `rgb_uint8` has no NaN, correct shape.
- All query indices are in range.
- All target arrays and masks have matching first dimension `M`.
- `y_xyz_cam_tcam` finite for entries where `m_xyz=true`.
- `track_visible=false` implies `track_valid=false`.
- Camera matrices have determinant checks when valid.

