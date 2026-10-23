#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

"$PYTHON_BIN" scripts/train_public_study.py \
  --data data/bbbc039 \
  --config configs/bbbc_primary_10seed_additional.json \
  --study bbbc_primary_10seed_additional

"$PYTHON_BIN" scripts/train_public_baselines.py \
  --data data/bbbc039 \
  --config configs/bbbc_xception_10seed_additional.json \
  --study bbbc_xception_10seed_additional

BREASTSEG_TNBC_STUDY=tnbc_grouped_cv_10seed_additional \
"$PYTHON_BIN" scripts/train_tnbc_grouped_cv.py \
  --data data \
  --config configs/tnbc_grouped_cv_10seed_additional.json
