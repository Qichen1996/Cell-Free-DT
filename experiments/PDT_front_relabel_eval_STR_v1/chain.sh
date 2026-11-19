#!/bin/bash
until [ -f /workspace/Cell-Free-DT-new/experiments/PDT_front_relabel_STR_v1/complete.json ]; do grep -q Traceback /workspace/Cell-Free-DT-new/experiments/PDT_front_relabel_STR_v1/PDT.log && exit 3; sleep 60; done
echo TRAINING_COMPLETE_STR
exec /venv/main/bin/python -u /workspace/Cell-Free-DT-new/experiments/PDT_front_relabel_eval_STR_v1/runner.py
