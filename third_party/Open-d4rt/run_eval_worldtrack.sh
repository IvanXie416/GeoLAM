#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

EXP="${EXP:-checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG}"
DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/eval_worldtrack}"
SUBSETS="${SUBSETS:-adt_mini,po_mini,pstudio_mini,ds_mini}"
NUM_FRAMES="${NUM_FRAMES:-64}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-4096}"
DEVICE="${DEVICE:-cuda}"

ARGS=()
if [[ -n "${LIMIT_SEQS:-}" ]]; then
  ARGS+=(--limit-seqs "$LIMIT_SEQS")
fi
if [[ "${SAVE_PER_SEQUENCE:-1}" != "0" ]]; then
  ARGS+=(--save-per-sequence)
fi

python eval_track3d_in_worldtrack.py \
  --model-config "$EXP/model.yaml" \
  --ckpt-path "$EXP/opend4rt.ckpt" \
  --data-root "$DATA_ROOT" \
  --subsets "$SUBSETS" \
  --num-frames "$NUM_FRAMES" \
  --query-chunk-size "$QUERY_CHUNK_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "${ARGS[@]}"
