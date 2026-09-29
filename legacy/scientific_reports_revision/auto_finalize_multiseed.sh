#!/bin/bash
# Wait for the active multiseed queue PID, then regenerate clustered statistics,
# figures and manuscript. Usage: auto_finalize_multiseed.sh QUEUE_PID
set -u
cd "/home/vdasoju/Desktop/breast-3d/data&code" || exit 1
queue_pid="${1:?queue PID required}"
while kill -0 "$queue_pid" 2>/dev/null; do
  sleep 30
done
src/revision/finalize.sh > outputs/revision/finalize_multiseed.log 2>&1
printf 'FINALIZED %s\n' "$(date --iso-8601=seconds)" >> outputs/revision/finalize_multiseed.log
