#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Set DATA_ROOT to one dataset root, or DATA_ROOTS_OVERRIDE to a
# colon-separated list of roots. Keeping the dataset outside the repository
# makes the launcher portable across nodes and machines.
DATA_ROOTS=()
if [[ -n "${DATA_ROOTS_OVERRIDE:-}" ]]; then
  IFS=':' read -r -a DATA_ROOTS <<< "$DATA_ROOTS_OVERRIDE"
elif [[ -n "${DATA_ROOT:-}" ]]; then
  DATA_ROOTS=("$DATA_ROOT")
else
  echo "Set DATA_ROOT or DATA_ROOTS_OVERRIDE before starting training" >&2
  exit 2
fi
if [[ "${#DATA_ROOTS[@]}" -eq 0 ]]; then
  echo "At least one dataset root is required" >&2
  exit 2
fi
if [[ -n "${DATA_ROOT_SAMPLE_RATES_OVERRIDE:-}" ]]; then
  IFS=':' read -r -a DATA_ROOT_SAMPLE_RATES <<< "$DATA_ROOT_SAMPLE_RATES_OVERRIDE"
else
  DATA_ROOT_SAMPLE_RATES=()
  for _ in "${DATA_ROOTS[@]}"; do DATA_ROOT_SAMPLE_RATES+=(1); done
fi
if [[ "${#DATA_ROOT_SAMPLE_RATES[@]}" -ne "${#DATA_ROOTS[@]}" ]]; then
  echo "DATA_ROOT_SAMPLE_RATES must contain one value per data root" >&2
  exit 1
fi

LAM_PYTHON="${LAM_PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"

NNODES="${NNODES:-${NUM_MACHINES:-${PET_NNODES:-1}}}"
NODE_RANK="${NODE_RANK:-${MACHINE_RANK:-${PET_NODE_RANK:-0}}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPUS[@]}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
if [[ "$NNODES" == "1" ]]; then MASTER_ADDR=127.0.0.1; fi

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
export LAM_VIDEO_INDEX_CACHE_DIR="${LAM_VIDEO_INDEX_CACHE_DIR:-video_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/d4rt_pixel}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

echo "LAM distributed training"
echo "DATA_ROOTS=${DATA_ROOTS[*]}"
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

exec "$LAM_PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m lam.train "${DATA_ROOTS[@]}" \
  --config "${CONFIG:-configs/lam_3d_pixel.yaml}" --device cuda \
  --batch-size "${BATCH_SIZE:-32}" \
  --data-root-sample-rates "${DATA_ROOT_SAMPLE_RATES[@]}" \
  --num-workers "${NUM_WORKERS:-8}" --prefetch-factor "${PREFETCH_FACTOR:-2}" \
  --video-sampling "${VIDEO_SAMPLING:-sliding-window}" --pairs-per-video "${PAIRS_PER_VIDEO:-10}" \
  --steps "${STEPS:-200000}" --save-steps "${SAVE_STEPS:-10000}" \
  --val-steps "${VAL_STEPS:-5000}" --val-ratio "${VAL_RATIO:-0.01}" --val-batches "${VAL_BATCHES:-32}" \
  --output-dir "$OUTPUT_DIR" --log-file "$LOG_FILE" --log-steps "${LOG_STEPS:-100}" \
  --amp "${AMP:-bf16}" "$@"
