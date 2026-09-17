# TartanAir V2

Loader: `src/data/tartanair_raw_dataset.py`

Default root in the reproduction config:

```text
data/tartanair/v2
```

## Expected Layout

The loader scans each scene and the configured difficulty folders:

```text
data/tartanair/v2/
  <scene_name>/
    Data_easy/
      P000/
        image_lcam_front/
          *.png
        depth_lcam_front/
          *.png
        pose_lcam_front.txt
      P001/
        ...
    Data_hard/
      P000/
        ...
```

The default reproduction config uses:

- `camera_name = lcam_front`
- `difficulties = [Data_easy, Data_hard]`

## Required Files

For each trajectory directory:

- `image_<camera_name>/*.png`
- `depth_<camera_name>/*.png`
- `pose_<camera_name>.txt`

Frames are matched by the numeric prefix in the filename, and only frames with
RGB, depth, and pose are kept.

## Pose Semantics

`pose_<camera_name>.txt` is read as one 7-value pose per line:

```text
tx ty tz qx qy qz qw
```

These poses are interpreted in TartanAir's NED world frame and converted by the
loader into the standard OpenCV-like camera frame:

- `+x`: right
- `+y`: down
- `+z`: forward

The loader emits per-frame `T_wc`.

## Depth Semantics

`depth_<camera_name>/*.png` stores float32 depth packed into RGBA bytes.

The loader:

- reads the PNG with `cv2.IMREAD_UNCHANGED`
- reinterprets the BGRA bytes as little-endian float32
- resizes with nearest-neighbor if needed

Valid depth is:

- finite
- `> 0`
- `< max_depth_m`

## Split Rule

TartanAir trajectories are partitioned by a stable hash bucket using
`split_modulo`.

With the default `split_modulo = 20`:

- train uses buckets `0..17`
- val uses bucket `18`
- test uses bucket `19`

## Sample Fields Emitted To Training

- `video`
- `depth_m`, `depth_valid`
- `query`, `target`, `mask`
- `camera.K`, `camera.T_wc`, `camera.camera_valid`
- `meta.source_mode = tartanair_depth_reproject`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
