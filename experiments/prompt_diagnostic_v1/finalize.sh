#!/bin/bash
set -euo pipefail
cd /workspace/Cell-Free-DT-new
echo WAITING_FOR_DIAGNOSTIC_SIMULATIONS
while true; do
  pending=0
  for variant in dt deployment q25 q50 q75 mapped; do
    status=$(supervisorctl status "new_PDTdiag_B_${variant}" || true)
    case "$status" in
      *RUNNING*|*STARTING*) pending=1 ;;
      *EXITED*) grep -q "ALL_COMPLETE ${variant}" "/tmp/new_PDTdiag_B_${variant}.log" || { echo "DIAGNOSTIC_FAILED ${variant}"; exit 1; } ;;
      *) echo "DIAGNOSTIC_FAILED ${status}"; exit 1 ;;
    esac
  done
  if [ "$pending" -eq 0 ]; then break; fi
  sleep 30
done
/venv/main/bin/python -u experiments/prompt_diagnostic_v1/summarize.py
echo ALL_DIAGNOSTICS_VALIDATED
