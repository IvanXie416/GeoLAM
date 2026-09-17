# PointOdyssey

Loader: `src/data/pointodyssey_raw_dataset.py`

Default root in the reproduction config:

```text
data/pointodyssey/v2
```

## Expected Layout

The loader expects split folders under the root, with one scene directory per
sequence:

```text
data/pointodyssey/v2/
  train/
    <scene_name>/
      anno.npz
      rgbs/
        *.jpg or *.png
      depths/
        *.png
  test/
    <scene_name>/
      anno.npz
      rgbs/
      depths/
```

The split mapping used by default is:

- `train -> train`
- `val -> test`
- `test -> test`

## Required Files

- `anno.npz`
- `rgbs/*.jpg` or `rgbs/*.png`
- `depths/*.png`

The loader keeps the aligned prefix of RGB and depth frames and requires at
least `clip_frames` frames.

## `anno.npz` Fields Used By The Loader

- `trajs_2d`: `[T,N,2]`, 2D trajectories
- `valids`: `[T,N]`, track validity mask
- `visibs`: `[T,N]`, track visibility mask
- `intrinsics`: `[T,3,3]`, per-frame camera intrinsics
- `extrinsics`: `[T,4,4]`, per-frame `T_cw`
- `trajs_3d`:
  - optional `[T,N,3]`
  - if present, trajectory-based 3D supervision is used
  - if missing, the loader falls back to depth-based query building

## Depth Semantics

- `depths/*.png` are read as `uint16`
- the loader converts them to meters with `depth_m = depth_u16 / 65535 * 1000`
- valid depth is `depth_m > 0`

## Sample Fields Emitted To Training

- `video`: resized RGB clip, shape `[T,3,H,W]`
- `depth_m`, `depth_valid`
- `query`:
  - `u`, `v`: normalized source UV in `[0,1]`
  - `t_src`, `t_tgt`, `t_cam`: timestep indices
- `target`:
  - `xyz_3d`: target 3D point in `t_cam`
  - `uv_2d`: target UV in target frame
  - `visibility`
  - `displacement`
  - `normal`
- `mask`: validity mask for each target field
- `camera`:
  - `K`: `[T,3,3]`
  - `T_wc`: `[T,4,4]`
  - `camera_valid`: `[T]`
- `meta.source_mode`:
  - `pointodyssey_trajectory_tracks` when `trajs_3d` is available
  - `pointodyssey_depth_reproject_fallback` otherwise

For canonical field meanings, see [../data_schema.md](../data_schema.md).
