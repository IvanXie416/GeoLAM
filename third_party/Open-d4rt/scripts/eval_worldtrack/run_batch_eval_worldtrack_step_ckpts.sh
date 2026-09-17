#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
SUBSETS="${SUBSETS:-adt_mini,po_mini,pstudio_mini,ds_mini}"
NUM_FRAMES="${NUM_FRAMES:-1000000}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-4096}"
LIMIT_SEQS="${LIMIT_SEQS:-0}"
GPUS="${GPUS:-0,1,2,3}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
MIN_STEP="${MIN_STEP:-}"
MAX_STEP="${MAX_STEP:-}"
STRIDE="${STRIDE:-1}"
LIMIT_CKPTS="${LIMIT_CKPTS:-0}"
STEPS="${STEPS:-}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
WRITE_TENSORBOARD_ONLY="${WRITE_TENSORBOARD_ONLY:-0}"
REPORT_PATH="${REPORT_PATH:-}"
EVAL_DIR_SUFFIX="${EVAL_DIR_SUFFIX:-}"
if [[ -z "${EVAL_DIR_SUFFIX}" ]]; then
  if [[ "${NUM_FRAMES}" == "64" ]]; then
    EVAL_DIR_SUFFIX="64clip_eval"
  else
    EVAL_DIR_SUFFIX="full_eval"
  fi
fi
OUTPUT_DIR_NAME_TEMPLATE="${OUTPUT_DIR_NAME_TEMPLATE:-}"
if [[ -z "${OUTPUT_DIR_NAME_TEMPLATE}" ]]; then
  OUTPUT_DIR_NAME_TEMPLATE="eval_worldtrack_step_{step}_${EVAL_DIR_SUFFIX}"
fi
EVAL_MODE="${EVAL_MODE:-default}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] CHECKPOINT_DIR is required." >&2
  echo "Example: CHECKPOINT_DIR=output_read/exp/checkpoints bash $0" >&2
  exit 2
fi

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

CMD=(
  python scripts/eval_worldtrack/batch_eval_worldtrack_step_ckpts.py
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --data-root "${DATA_ROOT}"
  --subsets "${SUBSETS}"
  --num-frames "${NUM_FRAMES}"
  --query-chunk-size "${QUERY_CHUNK_SIZE}"
  --limit-seqs "${LIMIT_SEQS}"
  --gpus "${GPUS}"
  --procs-per-gpu "${PROCS_PER_GPU}"
  --max-parallel "${MAX_PARALLEL}"
  --stride "${STRIDE}"
  --limit-ckpts "${LIMIT_CKPTS}"
  --eval-dir-suffix "${EVAL_DIR_SUFFIX}"
  --output-dir-name-template "${OUTPUT_DIR_NAME_TEMPLATE}"
  --eval-mode "${EVAL_MODE}"
)

if [[ -n "${MODEL_CONFIG}" ]]; then
  CMD+=(--model-config "${MODEL_CONFIG}")
fi
if [[ -n "${TENSORBOARD_LOGDIR}" ]]; then
  CMD+=(--tensorboard-logdir "${TENSORBOARD_LOGDIR}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
  CMD+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ -n "${MIN_STEP}" ]]; then
  CMD+=(--min-step "${MIN_STEP}")
fi
if [[ -n "${MAX_STEP}" ]]; then
  CMD+=(--max-step "${MAX_STEP}")
fi
if [[ -n "${STEPS}" ]]; then
  CMD+=(--steps "${STEPS}")
fi
if [[ -n "${REPORT_PATH}" ]]; then
  CMD+=(--report-path "${REPORT_PATH}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${WRITE_TENSORBOARD_ONLY}" == "1" ]]; then
  CMD+=(--write-tensorboard-only)
fi
printf 'Running:'
for arg in "${CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${CMD[@]}"
