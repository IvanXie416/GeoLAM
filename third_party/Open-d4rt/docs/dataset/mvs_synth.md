# MVS-Synth

Loader: `src/data/mvs_synth_raw_dataset.py`

Default root in the reproduction config:

```text
data/mvs-synth/v1
```

The loader additionally expects a clip subdirectory selected by
`data.mvs_synth.sequence_dir`, which defaults to:

```text
GTAV_540
```

## Expected Layout

```text
data/mvs-synth/v1/
  GTAV_540/
    <clip_name>/
      images/
        *.png
      depths/
        *.exr
      poses/
        *.json
```

## Required Files

- `images/*.png`
- `depths/*.exr`
- `poses/*.json`

Frame ids are matched by filename stem and only the common intersection is
used.

## Pose / Intrinsics Semantics

Each `poses/*.json` is parsed into:

- `K`
- `T_wc`

The loader rejects frames with non-finite intrinsics or transforms.

## Depth Semantics

- depth EXR is read as float32
- optional `depth_scale` is applied
- then depth is sanitized with:
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
- `meta.source_mode = mvs_synth_depth_reproject`

For canonical field meanings, see [../data_schema.md](../data_schema.md).
