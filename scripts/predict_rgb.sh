#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LAM_PYTHON="${LAM_PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$LAM_PYTHON" -m lam.predict_rgb "$@"
