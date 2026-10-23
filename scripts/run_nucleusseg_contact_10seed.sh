#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${BREASTSEG_PYTHON:-python}
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

"$PYTHON_BIN" scripts/train_nucleusseg_contact.py \
  --data data/nucleusseg \
  --config configs/nucleusseg_contact_10seed.json
