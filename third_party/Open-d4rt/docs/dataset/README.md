# Dataset Notes

This directory documents the raw on-disk layouts expected by the dataset
adapters in `src/data/` for the 9Mix training recipe.

## Dataset Pages

- [PointOdyssey](pointodyssey.md)
- [Dynamic Replica](dynamic_replica.md)
- [Kubric Full](kubric_full.md)
- [TartanAir V2](tartanair.md)
- [Virtual KITTI 2](virtual_kitti2.md)
- [ScanNet / ScanNet++](scannet.md)
- [BlenderMVS](blendermvs.md)
- [CO3D v2](co3d.md)
- [MVS-Synth](mvs_synth.md)

## Common Training Sample Fields

All dataset adapters normalize their outputs into the same high-level training
sample structure:

- `video`: RGB clip tensor `[T,3,H,W]`
- `aspect_ratio`: original width/height ratio before resize/crop
- `depth_m`, `depth_valid`: dense depth and validity mask when available
- `query`: source UV/time query table with `u`, `v`, `t_src`, `t_tgt`, `t_cam`
- `target`: supervision targets such as `xyz_3d`, `uv_2d`, `visibility`
- `mask`: validity mask for each target field
- `camera`: per-frame `K`, `T_wc`, `camera_valid`
- `meta`: dataset name, scene id, clip start, and dataset-specific source mode

For the canonical meaning of these fields, see
[../data_schema.md](../data_schema.md).
