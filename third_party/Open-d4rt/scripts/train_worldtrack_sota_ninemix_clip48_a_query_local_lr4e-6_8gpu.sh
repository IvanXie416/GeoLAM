#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  _conda_sh="${CONDA_SH:-}"
  if [[ -z "${_conda_sh}" ]] && command -v conda >/dev/null 2>&1; then
    _conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${_conda_base}" ]] && [[ -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
      _conda_sh="${_conda_base}/etc/profile.d/conda.sh"
    fi
  fi
  if [[ -n "${_conda_sh}" ]] && [[ -f "${_conda_sh}" ]]; then
    source "${_conda_sh}"
    if [[ -n "${CONDA_ENV:-}" ]]; then
      set +u
      conda activate "${CONDA_ENV}"
      set -u
    fi
  fi
fi

EXP_NAME="${EXP_NAME:-d4rt_worldtrack_sota_ninemix_clip48_a_query_local_lr4e_6}"
EXP_OUTPUT="${EXP_OUTPUT:-worldtrack_sota_ninemix_clip48_a_query_local_lr4e-6_eval64clip}"
OUT_ROOT="${OUT_ROOT:-output/exp_worldtrack_sota_0512}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${EXP_OUTPUT}}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model_effective.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_effective.yaml}"
INIT_CKPT="${INIT_CKPT:-checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt}"
INIT_TIMESTEP_EMBED_RESIZE="${INIT_TIMESTEP_EMBED_RESIZE:-linear}"
VIDEOMAE2_CKPT="${VIDEOMAE2_CKPT:-checkpoints/VideoMAE2/weights/mae-g/vit_g_hybrid_pt_1200e.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#_gpus[@]}}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29713}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-8}"
WORLD_SIZE=$((NUM_MACHINES * NPROC_PER_NODE))

if [[ "$WORLD_SIZE" -ne "$EXPECTED_WORLD_SIZE" ]]; then
  echo "[ERROR] Expected WORLD_SIZE=${EXPECTED_WORLD_SIZE}, got ${WORLD_SIZE}." >&2
  echo "        Override EXPECTED_WORLD_SIZE=${WORLD_SIZE} if this is intentional." >&2
  exit 1
fi

for path in "$MODEL_CONFIG" "$TRAIN_CONFIG" "$INIT_CKPT" "$VIDEOMAE2_CKPT"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Required file not found: $path" >&2
    if [[ "$path" == "$VIDEOMAE2_CKPT" ]]; then
      echo "        Set VIDEOMAE2_CKPT=checkpoints/vit_g_hybrid_pt_1200e.pth or place it at the default path." >&2
    fi
    exit 1
  fi
done

export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export D4RT_CV2_WORKER_THREADS="${D4RT_CV2_WORKER_THREADS:-0}"
export D4RT_TORCH_WORKER_THREADS="${D4RT_TORCH_WORKER_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_OVERRIDES=(
  --override "experiment.name=${EXP_NAME}"
  --override "experiment.output_dir=${OUT_DIR}"
  --override "checkpoint.auto_eval_worldtrack_step.enabled=${AUTO_EVAL_WORLDTRACK_ENABLED:-true}"
  --override "checkpoint.auto_eval_worldtrack_step.num_frames=${AUTO_EVAL_WORLDTRACK_NUM_FRAMES:-64}"
  --override "checkpoint.auto_eval_worldtrack_step.env.GPUS=${WORLDTRACK_EVAL_GPUS:-4,5,6,7}"
  --override "checkpoint.auto_eval_worldtrack_step.env.MAX_PARALLEL=${WORLDTRACK_EVAL_MAX_PARALLEL:-4}"
  --override "checkpoint.auto_eval_worldtrack_step.env.PROCS_PER_GPU=${WORLDTRACK_EVAL_PROCS_PER_GPU:-1}"
)

[[ -n "${TOTAL_STEPS:-}" ]] && CONFIG_OVERRIDES+=(--override "schedule.total_steps=${TOTAL_STEPS}")
[[ -n "${WARMUP_STEPS:-}" ]] && CONFIG_OVERRIDES+=(--override "optimizer.learning_rate.warmup_steps=${WARMUP_STEPS}")
[[ -n "${PEAK_LR:-}" ]] && CONFIG_OVERRIDES+=(--override "optimizer.learning_rate.peak_lr=${PEAK_LR}")
[[ -n "${FINAL_LR:-}" ]] && CONFIG_OVERRIDES+=(--override "optimizer.learning_rate.final_lr=${FINAL_LR}")
[[ -n "${TRAIN_BATCH_SIZE:-}" ]] && CONFIG_OVERRIDES+=(--override "runtime.train_batch_size=${TRAIN_BATCH_SIZE}" --override "runtime.batch_size=${TRAIN_BATCH_SIZE}")
[[ -n "${VAL_BATCH_SIZE:-}" ]] && CONFIG_OVERRIDES+=(--override "runtime.val_batch_size=${VAL_BATCH_SIZE}")
[[ -n "${TRAIN_NUM_WORKERS:-}" ]] && CONFIG_OVERRIDES+=(--override "runtime.train_num_workers=${TRAIN_NUM_WORKERS}")
[[ -n "${VAL_NUM_WORKERS:-}" ]] && CONFIG_OVERRIDES+=(--override "runtime.val_num_workers=${VAL_NUM_WORKERS}")
[[ -n "${SAVE_EVERY_STEPS:-}" ]] && CONFIG_OVERRIDES+=(--override "checkpoint.save_every_steps=${SAVE_EVERY_STEPS}")
[[ -n "${STEP_SAVE_EVERY_STEPS:-}" ]] && CONFIG_OVERRIDES+=(--override "checkpoint.step_save_every_steps=${STEP_SAVE_EVERY_STEPS}")
[[ -n "${VALIDATE_EVERY_STEPS:-}" ]] && CONFIG_OVERRIDES+=(--override "logging.validate_every_steps=${VALIDATE_EVERY_STEPS}")
[[ -n "${VALIDATE_MAX_SAMPLES_GLOBAL:-}" ]] && CONFIG_OVERRIDES+=(--override "logging.validate_max_samples_global=${VALIDATE_MAX_SAMPLES_GLOBAL}")

[[ -n "${POINTODYSSEY_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.pointodyssey.root=${POINTODYSSEY_ROOT}")
[[ -n "${DYNAMIC_REPLICA_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.dynamic_replica.root=${DYNAMIC_REPLICA_ROOT}")
[[ -n "${KUBRIC_FULL_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.kubric_full.root=${KUBRIC_FULL_ROOT}")
[[ -n "${KUBRIC_FULL_PROCESSED_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.kubric_full.processed_root=${KUBRIC_FULL_PROCESSED_ROOT}")
[[ -n "${TARTANAIR_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.tartanair.root=${TARTANAIR_ROOT}")
[[ -n "${VIRTUAL_KITTI2_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.virtual_kitti2.root=${VIRTUAL_KITTI2_ROOT}")
[[ -n "${SCANNET_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.scannet.root=${SCANNET_ROOT}")
[[ -n "${BLENDERMVS_ROOTS:-}" ]] && CONFIG_OVERRIDES+=(--override "data.blendermvs.roots=${BLENDERMVS_ROOTS}")
[[ -n "${CO3D_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.co3d.root=${CO3D_ROOT}")
[[ -n "${MVS_SYNTH_ROOT:-}" ]] && CONFIG_OVERRIDES+=(--override "data.mvs_synth.root=${MVS_SYNTH_ROOT}")
CONFIG_OVERRIDES+=(--override "model.encoder.pretrained.path=${VIDEOMAE2_CKPT}")

echo "================================================================================"
echo "OpenD4RT WorldTrack SOTA 9Mix clip48 query-local lr4e-6 training"
echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "VIDEOMAE2_CKPT=${VIDEOMAE2_CKPT}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "AUTO_EVAL_WORLDTRACK_ENABLED=${AUTO_EVAL_WORLDTRACK_ENABLED:-true}"
echo "================================================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] Preflight passed; training not launched."
  exit 0
fi

torchrun \
  --nnodes="$NUM_MACHINES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$MACHINE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --tb_log \
  --model-config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG" \
  --init-model "$INIT_CKPT" \
  --init-timestep-embed-resize "$INIT_TIMESTEP_EMBED_RESIZE" \
  "${CONFIG_OVERRIDES[@]}" \
  "$@"
