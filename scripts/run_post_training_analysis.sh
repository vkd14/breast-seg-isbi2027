#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PID_OF_FOLLOWUP_QUEUE" >&2
  exit 2
fi

FOLLOWUP_PID=$1
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${BREASTSEG_PYTHON:-python}

while kill -0 "$FOLLOWUP_PID" 2>/dev/null; do
  sleep 30
done

cd "$REPO_DIR"

# The primary queue may have been interrupted before this study by a runtime
# compatibility failure.  The trainer is resumable and skips complete runs, so
# invoking it here also acts as a completeness repair before aggregation.
BREASTSEG_TNBC_STUDY=tnbc_grouped_cv_10seed_additional \
"$PYTHON_BIN" scripts/train_tnbc_grouped_cv.py \
  --data data \
  --config configs/tnbc_grouped_cv_10seed_additional.json

mapfile -t BBBC_ADDED < <(find results/bbbc_primary_10seed_additional results/bbbc_xception_10seed_additional \
  -mindepth 1 -maxdepth 1 -type d -name '*_s*' | sort)
"$PYTHON_BIN" scripts/evaluate_bbbc_published_metrics.py "${BBBC_ADDED[@]}"

"$PYTHON_BIN" scripts/evaluate_tnbc_published_metrics.py \
  --study tnbc_grouped_cv_10seed_additional \
  --config configs/tnbc_grouped_cv_10seed_additional.json
"$PYTHON_BIN" scripts/evaluate_tnbc_published_metrics.py \
  --study tnbc_rgb_grouped_cv_10seed \
  --config configs/tnbc_rgb_grouped_cv_10seed.json

"$PYTHON_BIN" scripts/evaluate_bbbc038_grayscale.py
"$PYTHON_BIN" scripts/summarize_primary_10seed.py
"$PYTHON_BIN" scripts/summarize_published_10seed.py
