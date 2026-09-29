#!/bin/bash
cd "/home/vdasoju/Desktop/breast-3d/data&code"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ~/Desktop/breast-3d/breast_qubo/.venv/bin/python src/revision/run_queue.py 1
