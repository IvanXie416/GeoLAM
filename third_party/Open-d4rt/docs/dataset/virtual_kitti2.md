# Virtual KITTI 2

Loader: `src/data/virtual_kitti2_raw_dataset.py`

Default root in the reproduction config:

```text
data/vitual-kitti-2/v2
```

Note the config path uses the existing directory spelling `vitual-kitti-2`.

## Expected Layout

The loader scans `Scene*` directories, then variants and camera ids:

```text
data/vitual-kitti-2/v2/
  Scene01/
    clone/
      intrinsic.txt
      extrinsic.txt
      frames/
        rgb/
          Camera_0/
            *.jpg or *.png
        depth/
          Camera_0/
            *.png
        forwardSceneFlow/
          Camera_0/
            *.png
    fog/
      ...
  Scene02/
    ...
```

The reproduction config uses:

- `variants = [clone]`
- `camera_ids = [0]`

## Required Files

- `intrinsic.txt`
- `extrinsic.txt`
- `frames/rgb/Camera_<id>/*`
- `frames/depth/Camera_<id>/*.png`

Optional but used when present:

- `frames/forwardSceneFlow/Camera_<id>/*.png`

## Geometry Semantics

- depth PNG is read as `uint16` and converted to meters using:

```text
depth_m = depth_u16 / 100.0
```

- `intrinsic.txt` provides per-frame intrinsics
- `extrinsic.txt` provides per-frame extrinsics
- forward scene flow is decoded from 16-bit RGB PNG into a 3D vector in meters

The loader anchors sparse points in world space from valid depth pixels, then
propagates them forward using the decoded scene flow to build sparse synthetic
tracks.

## Sample Fields Emitted To Training

- `video`
- `depth_m`, `depth_valid`
- `query`, `target`, `mask`
- `camera.K`, `camera.T_wc`, `camera.camera_valid`
- `meta.source_mode = vkitti2_scene_flow_tracks`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
