# ScanNet / ScanNet++

Loader: `src/data/scannet_raw_dataset.py`

Default root in the reproduction config:

```text
data/scannet/plus-v2/data
```

Default split files:

```text
data/scannet/plus-v2/splits/nvs_sem_train.txt
data/scannet/plus-v2/splits/nvs_sem_val.txt
data/scannet/plus-v2/splits/nvs_test.txt
```

## Supported Source Modes

The loader supports two source layouts:

1. `iphone_rgbd` (preferred in the docstring and default `source=auto`)
2. `dslr_colmap` fallback

## iPhone RGBD Layout

```text
data/scannet/plus-v2/data/
  <scene_id>/
    iphone/
      rgb.mkv
      depth.bin
      pose_intrinsic_imu.json
```

Files used:

- `rgb.mkv`: RGB video
- `depth.bin`: packed per-frame depth chunks
- `pose_intrinsic_imu.json`: per-frame `intrinsic` and `aligned_pose`

Depth chunks are decoded into meters. The loader handles both:

- `lz4`-compressed `uint16` millimeter depth
- zlib-compressed `float32` meter depth

## DSLR COLMAP Layout

```text
data/scannet/plus-v2/data/
  <scene_id>/
    dslr/
      resized_images/ or resized_undistorted_images/
        *.jpg or *.png
      colmap/
        cameras.txt
        images.txt
        points3D.txt
```

Files used:

- `cameras.txt`: intrinsics
- `images.txt`: poses and 2D observations
- `points3D.txt`: sparse world points

This mode builds sparse supervision from COLMAP tracks instead of dense depth.

## Sample Fields Emitted To Training

Common fields:

- `video`
- `query`, `target`, `mask`
- `camera.K`, `camera.T_wc`, `camera.camera_valid`

If iPhone mode is used:

- `depth_m`, `depth_valid`
- `meta.source_mode = iphone_rgbd`

If DSLR mode is used:

- `depth_m` is filled with `NaN`
- `depth_valid` is all false
- `meta.source_mode = dslr_colmap`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
