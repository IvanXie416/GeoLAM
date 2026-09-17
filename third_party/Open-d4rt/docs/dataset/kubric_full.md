# Kubric Full

Loader used by the reproduction script:

- `src/data/kubric_full_robust_preprocess_dataset.py`

Default root in the reproduction config:

```text
data/kubric_full/kubric_full_process_v1
```

The repro config uses `backend: preprocess`, so this document describes the
preprocessed `.npy` layout rather than the original TFDS release.

## Expected Layout

The loader supports either a manifest-based layout or plain split folders.

Manifest-based:

```text
data/kubric_full/kubric_full_process_v1/
  manifest.json
  <relative_dir_from_manifest>/
    rgb.npy
    depth_uint16.npy
    segmentation.npy
    normal_uint16.npy
    object_coordinates_uint16.npy
    camera_positions.npy
    camera_quaternions.npy
    instances_bboxes_3d.npy
    instances_positions.npy
    instances_quaternions.npy
    meta.json
```

Split-folder fallback:

```text
data/kubric_full/kubric_full_process_v1/
  train/
    <scene_name>/
      <required files>
  validation/
    <scene_name>/
      <required files>
```

Default split mapping:

- `train -> train`
- `val -> validation`
- `test -> validation`

## Required Files Per Scene

- `rgb.npy`
- `depth_uint16.npy`
- `segmentation.npy`
- `normal_uint16.npy`
- `object_coordinates_uint16.npy`
- `camera_positions.npy`
- `camera_quaternions.npy`
- `instances_bboxes_3d.npy`
- `instances_positions.npy`
- `instances_quaternions.npy`
- `meta.json`

## Metadata Fields Used

The loader reads these fields from `meta.json`:

- `camera_field_of_view`
- `depth_range`
- `num_frames`
- `num_instances`
- `video_name` or `scene_id`
- `height`
- `width`

## Array Meanings

- `rgb.npy`: RGB frames
- `depth_uint16.npy`: quantized depth representation used by the Kubric robust
  loader
- `segmentation.npy`: instance/semantic segmentation planes
- `normal_uint16.npy`: packed normals
- `object_coordinates_uint16.npy`: packed object-coordinate field
- `camera_positions.npy`: camera translations
- `camera_quaternions.npy`: camera orientations
- `instances_*`: per-instance 3D metadata

The robust parent loader reconstructs query supervision from these arrays and
converts the scene into the shared D4RT training schema.

## Sample Fields Emitted To Training

- `video`
- `query`, `target`, `mask`
- `camera`
- `meta.source_mode = kubric_movi_full_preprocessed_objectcoord`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
