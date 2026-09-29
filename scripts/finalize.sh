#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
research_python="${1:-python}"
research_data="${2:-data}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OPENCV_LOG_LEVEL=ERROR
"$research_python" -m unittest discover -s tests -v > results/audit/regression_tests.txt 2>&1
"$research_python" scripts/validate_results.py --data "$research_data" > results/audit/validation_console.txt 2>&1
"$research_python" scripts/build_reports.py --data "$research_data" > results/report_build.log 2>&1
if [[ -x .tools/tectonic ]]; then
  (cd paper && ../.tools/tectonic first_draft_isbi.tex > ../results/latex_build.log 2>&1)
else
  printf '%s\n' 'Readable PDFs generated. Compile paper/first_draft_isbi.tex with a LaTeX installation for the official-template PDF.'
fi
