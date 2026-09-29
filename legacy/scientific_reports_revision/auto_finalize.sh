#!/bin/bash
cd "/home/vdasoju/Desktop/breast-3d/data&code"
until grep -q "QUEUE COMPLETE" outputs/revision/queue2.log 2>/dev/null; do sleep 120; done
src/revision/finalize.sh > outputs/revision/finalize.log 2>&1
echo "FINALIZED $(date)" >> outputs/revision/finalize.log
