#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-data/worldtrack_release}"
DEMO_CASE_RANK="${DEMO_CASE_RANK:-1}"

DEMO_CASES=(
  "pstudio_mini/juggle_5.npz"
  "ds_mini/fec654-3_obj_source_left_1.npz"
  "adt_mini/Apartment_release_meal_seq133_1.npz"
  "po_mini/cab_e_3rd_12.npz"
)

if [[ -z "${WORLDTRACK_NPZ:-}" ]]; then
  if [[ -n "${DEMO_CASE:-}" ]]; then
    WORLDTRACK_NPZ="$DATA_ROOT/$DEMO_CASE"
  else
    if ! [[ "$DEMO_CASE_RANK" =~ ^[0-9]+$ ]] || (( DEMO_CASE_RANK < 1 || DEMO_CASE_RANK > ${#DEMO_CASES[@]} )); then
      echo "DEMO_CASE_RANK must be between 1 and ${#DEMO_CASES[@]}." >&2
      exit 2
    fi
    WORLDTRACK_NPZ="$DATA_ROOT/${DEMO_CASES[$((DEMO_CASE_RANK - 1))]}"
  fi
fi

if [[ ! -f "$WORLDTRACK_NPZ" ]]; then
  echo "WorldTrack case not found: $WORLDTRACK_NPZ" >&2
  echo "Set DATA_ROOT, WORLDTRACK_NPZ, DEMO_CASE, or DEMO_CASE_RANK. Recommended cases:" >&2
  for case in "${DEMO_CASES[@]}"; do
    echo "  $DATA_ROOT/$case" >&2
  done
  exit 2
fi

echo "Using WorldTrack demo case: $WORLDTRACK_NPZ"

EXP="${EXP:-checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/worldtrack_demo}"
NUM_FRAMES="${NUM_FRAMES:-64}"
DEVICE="${DEVICE:-cuda}"
POINT_GRID_COLS="${POINT_GRID_COLS:-64}"
POINT_GRID_ROWS="${POINT_GRID_ROWS:-64}"
POINT_MAX_POINTS="${POINT_MAX_POINTS:-4096}"
TRACK_MAX_POINTS="${TRACK_MAX_POINTS:-256}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-1024}"
POINT_QUERY_CHUNK_SIZE="${POINT_QUERY_CHUNK_SIZE:-512}"
CAMERA_QUERY_CHUNK_SIZE="${CAMERA_QUERY_CHUNK_SIZE:-1024}"

python vis/build_like_demo_for_worldtrack.py \
  --config "$EXP/model.yaml" \
  --ckpt-path "$EXP/opend4rt.ckpt" \
  --worldtrack-npz "$WORLDTRACK_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --num-frames "$NUM_FRAMES" \
  --device "$DEVICE" \
  --point-grid-cols "$POINT_GRID_COLS" \
  --point-grid-rows "$POINT_GRID_ROWS" \
  --point-max-points "$POINT_MAX_POINTS" \
  --track-max-points "$TRACK_MAX_POINTS" \
  --query-chunk-size "$QUERY_CHUNK_SIZE" \
  --point-query-chunk-size "$POINT_QUERY_CHUNK_SIZE" \
  --camera-query-chunk-size "$CAMERA_QUERY_CHUNK_SIZE"
