# Training Guide

Main training script:

```text
scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

## 1. Environment

Use either conda:

```bash
conda env create -f environment.yml
conda activate d4rt
```

or pip:

```bash
pip install -r requirements.txt
```


## 2. Required Checkpoints

The training script expects these files by default:

```text
checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
checkpoints/VideoMAE2/weights/mae-g/vit_g_hybrid_pt_1200e.pth
```

The VideoMAEv2 pre-train checkpoint should be downloaded from the official
VideoMAEv2 Model Zoo:

- <https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md>
- use the `vit_g_hybrid_pt_1200e` checkpoint entry

The VideoMAEv2 page notes that you need to submit their download request form
first, then download the checkpoint from the returned link.

If your local checkpoint paths differ, override them at launch time:

```bash
INIT_CKPT=checkpoints/opend4rt_32clip.ckpt \
VIDEOMAE2_CKPT=checkpoints/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

## 3. Required Datasets

The reproduction config uses the following dataset roots by default:

| Dataset | Config Key | Default Root | Notes |
| --- | --- | --- | --- |
| PointOdyssey | `data.pointodyssey.root` | `data/pointodyssey/v2` | [pointodyssey.md](dataset/pointodyssey.md) |
| Dynamic Replica | `data.dynamic_replica.root` | `data/dynamic-replica/v2` | [dynamic_replica.md](dataset/dynamic_replica.md) |
| Kubric Full (processed) | `data.kubric_full.processed_root` | `data/kubric_full/kubric_full_process_v1` | [kubric_full.md](dataset/kubric_full.md) |
| TartanAir V2 | `data.tartanair.root` | `data/tartanair/v2` | [tartanair.md](dataset/tartanair.md) |
| Virtual KITTI 2 | `data.virtual_kitti2.root` | `data/vitual-kitti-2/v2` | [virtual_kitti2.md](dataset/virtual_kitti2.md) |
| ScanNet / ScanNet++ | `data.scannet.root` | `data/scannet/plus-v2/data` | [scannet.md](dataset/scannet.md) |
| BlenderMVS | `data.blendermvs.roots` | `data/blendermvs/...` | [blendermvs.md](dataset/blendermvs.md) |
| CO3D v2 | `data.co3d.root` | `data/co3d/v2` | [co3d.md](dataset/co3d.md) |
| MVS-Synth | `data.mvs_synth.root` | `data/mvs-synth/v1` | [mvs_synth.md](dataset/mvs_synth.md) |

Dataset index:
[docs/dataset/README.md](dataset/README.md)

If your local paths differ from the defaults, override them with environment
variables before launching the script, for example:

```bash
POINTODYSSEY_ROOT=data/pointodyssey
DYNAMIC_REPLICA_ROOT=data/dynamic-replica
KUBRIC_FULL_PROCESSED_ROOT=data/kubric_processed
TARTANAIR_ROOT=data/tartanair
VIRTUAL_KITTI2_ROOT=data/vkitti2
SCANNET_ROOT=data/scannet_plus
BLENDERMVS_ROOTS=data/root_a,data/root_b,data/root_c
CO3D_ROOT=data/co3d
MVS_SYNTH_ROOT=data/mvs_synth
```

## 4. One-GPU Smoke Test

Run a short smoke test before the full job:

```bash
CUDA_VISIBLE_DEVICES=0 \
EXPECTED_WORLD_SIZE=1 \
NPROC_PER_NODE=1 \
TOTAL_STEPS=10 \
SAVE_EVERY_STEPS=10 \
STEP_SAVE_EVERY_STEPS=10 \
AUTO_EVAL_WORLDTRACK_ENABLED=false \
VIDEOMAE2_CKPT=checkpoints/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

## 6. Training Command Sample

Run the intended 8-GPU job:

```bash
VIDEOMAE2_CKPT=checkpoints/vit_g_hybrid_pt_1200e.pth \
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

## 7. Recommended Curriculum

For actual training, a more stable recipe is to use curriculum learning over
clip length instead of jumping directly to 48 frames:

```text
16 -> 32 -> 48 clips
```

The recommended practice is:

1. train a shorter-clip model first
2. initialize the next stage from the previous checkpoint
3. expand the timestep embedding with linear interpolation

This repository already supports that through the timestep embedding resize
path used during initialization. The intended mode here is `linear`.

## 8. Effective Configs

The training run uses:

```text
configs/model_effective.yaml
configs/train_effective.yaml
```
