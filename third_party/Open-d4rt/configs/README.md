# Configs

This directory contains the effective training configuration used by the
48-frame 9Mix training recipe.

`output/exp_worldtrack_sota_0512/worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_eval64clip`

Files:

- `model_effective.yaml`: 48-frame D4RT ViT-g model configuration.
- `train_effective.yaml`: 9-dataset mixture training recipe, no crop/color aug,
  hard query ratio 0.2, static local/global timestep sampling, 30000 total
  steps, peak lr `4e-6`, final lr `4e-7`.

The training run initializes from the 32-frame 9Mix checkpoint and resizes the
learned timestep embeddings with linear interpolation. Use:

```bash
bash scripts/train_worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_8gpu.sh
```

Required external files:

- `checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt`
- `checkpoints/VideoMAE2/weights/mae-g/vit_g_hybrid_pt_1200e.pth`
- the 9Mix training datasets at the paths listed in `train_effective.yaml`, or
  equivalent paths passed through the environment variables documented in the
  script.
