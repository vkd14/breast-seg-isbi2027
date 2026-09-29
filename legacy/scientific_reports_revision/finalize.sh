#!/bin/bash
# Regenerate stats, figures, manuscript, response letter and summary from whatever has finished.
cd "/home/vdasoju/Desktop/breast-3d/data&code"
PY=~/Desktop/breast-3d/breast_qubo/.venv/bin/python
$PY src/revision/stats.py 2>&1 | grep -v Warn
$PY src/revision/stats_clustered.py 2>&1 | grep -v Warn
$PY src/revision/figures.py 2>&1 | grep -vE "TIFF|Warn|warn"
$PY src/revision/make_manuscript.py 2>&1 | grep -v Warn
