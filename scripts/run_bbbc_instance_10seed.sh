#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

BREASTSEG_BBBC_INSTANCE_STUDY=bbbc_instance_10seed \
"$PYTHON_BIN" scripts/train_bbbc_instance.py \
  --data data/bbbc039 \
  --config configs/bbbc_instance_10seed.json
