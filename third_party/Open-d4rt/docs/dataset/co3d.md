# CO3D v2

Loader: `src/data/co3d_raw_dataset.py`

Default root in the reproduction config:

```text
data/co3d/v2
```

## Expected Layout

The loader scans category directories under the root:

```text
data/co3d/v2/
  <category>/
    frame_annotations.jgz
    sequence_annotations.jgz
    set_lists/
      set_lists_fewview_train.json
      set_lists_fewview_dev.json
      set_lists_fewview_test.json
    ...
```

Referenced image, depth, and mask files are loaded via relative paths stored in
the annotation payloads.

## Files Used

- `frame_annotations.jgz`
- `sequence_annotations.jgz`
- one of the set-list files under `set_lists/`

Default split selection:

- train: `set_lists_fewview_train.json`
- val: `set_lists_fewview_dev.json`, fallback to train list
- test: `set_lists_fewview_test.json`, fallback to dev list

## Frame Fields Used

From frame annotations and viewpoint metadata, the loader consumes:

- image relative path
- depth relative path
- optional depth-mask relative path
- `R`, `T`
- `focal_length`
- `principal_point`
- `intrinsics_format`
- depth scale adjustment

The loader converts PyTorch3D camera convention into OpenCV-style:

- `K`
- `T_cw`
- `T_wc`

## Depth Semantics

CO3D depth PNG is decoded as:

- read `uint16`
- reinterpret bytes as `float16`
- multiply by the per-frame depth scale adjustment

If `use_depth_masks=true`, the optional depth mask is applied to the depth
validity mask.

## Sample Fields Emitted To Training

- `video`
- `depth_m`, `depth_valid`
- `query`, `target`, `mask`
- `camera.K`, `camera.T_wc`, `camera.camera_valid`
- `meta.source_mode = co3d_depth_reproject`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
