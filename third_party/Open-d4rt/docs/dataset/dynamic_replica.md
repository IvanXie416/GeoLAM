# Dynamic Replica

Loader: `src/data/dynamic_replica_raw_dataset.py`

Default root in the reproduction config:

```text
data/dynamic-replica/v2
```

## Expected Layout

The loader reads split directories directly from the dataset root:

```text
data/dynamic-replica/v2/
  train/
    frame_annotations_train.json
    <scene_name>/
      images/
        *.png
      depths/
        <scene_name>_####.geometric.png
      trajectories/
        *.pth
  valid/
    frame_annotations_valid.json
    <scene_name>/
      images/
      depths/
      trajectories/
  test/
    frame_annotations_test.json
    <scene_name>/
      images/
      depths/
      trajectories/
```

Default split mapping:

- `train -> train`
- `val -> valid`
- `test -> test`

## Required Files

- `frame_annotations_<split>.json`
- per-scene `images/*.png`
- per-scene `depths/*.geometric.png`
- per-scene `trajectories/*.pth`

The loader uses the minimum aligned count across image, depth, and trajectory
files.

## Camera Metadata

Frame camera parameters come from `frame_annotations_<split>.json`.

Fields used from each viewpoint entry:

- `R`, `T`
- `focal_length`
- `principal_point`
- `intrinsics_format`
- optional depth scale adjustment

The loader converts these into:

- `K`: OpenCV-style intrinsics
- `T_cw`
- `T_wc`

The reproduction config uses `camera_convention=dynamic_replica_v2`.

## Depth Semantics

`depths/*.geometric.png` are read as `uint16`, then decoded using one of:

- `float16_bitcast`
- `uint16_divisor`
- `auto` mode, which picks the more plausible decode

The default config uses:

```text
depth_decode_mode=auto
depth_divisor=10000.0
```

## Trajectory Files

Each `trajectories/*.pth` is loaded with `torch.load` and the loader expects:

- `traj_2d`
- `traj_3d_world`
- `verts_inds_vis`

If trajectory-based 3D supervision is unavailable for a scene, the loader
falls back to depth-based query construction and emits a warning.

## Sample Fields Emitted To Training

- `video`
- `depth_m`, `depth_valid`
- `query`, `target`, `mask`
- `camera.K`, `camera.T_wc`, `camera.camera_valid`
- `meta.source_mode`:
  - `dynamic_replica_trajectory_tracks`
  - `dynamic_replica_depth_reproject_fallback`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
