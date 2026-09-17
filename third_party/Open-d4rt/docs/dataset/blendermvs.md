# BlenderMVS

Loader: `src/data/blendermvs_raw_dataset.py`

Default roots in the reproduction config:

```text
data/blendermvs/base-low-res/BlendedMVS
data/blendermvs/blendermvs-plus/extracted
data/blendermvs/blendermvs-plusplus/extracted
```

## Expected Layout

Each root is scanned for scene directories with the following structure:

```text
<blendermvs_root>/
  <scene_name>/
    blended_images/
      *.jpg
    rendered_depth_maps/
      *.pfm
    cams/
      *_cam.txt
```

If `use_masked_images=true`, the loader selects `*_masked.jpg` instead of the
plain RGB frames.

## Required Files

- `blended_images/*.jpg`
- `rendered_depth_maps/*.pfm`
- `cams/*_cam.txt`

Frame ids are extracted from filenames and only the intersection of RGB, depth,
and camera files is used.

## Geometry Semantics

- depth is loaded from `.pfm` as float32 meters
- camera files provide both intrinsics and `T_wc`

The loader sanitizes depth using:

- `max_depth_m`
- `depth_clip_percentile`
- `min_depth_valid_ratio`

## Split Rule

Scenes are partitioned by stable hash bucket with `split_modulo`.

With modulo `M`:

- train uses buckets `< M-2`
- val uses bucket `M-2`
- test uses bucket `M-1`

## Sample Fields Emitted To Training

- `video`
- `depth_m`, `depth_valid`
- `query`, `target`, `mask`
- `camera.K`, `camera.T_wc`, `camera.camera_valid`
- `meta.source_mode = blendermvs_depth_reproject`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
