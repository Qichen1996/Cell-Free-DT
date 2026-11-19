#!/bin/bash
cd /workspace/Cell-Free-DT-new
until grep -q DT_TRAIN_DONE experiments/capacity_pdt_v1/train_DT_d256.log 2>/dev/null; do sleep 600; done
CAP_DIM=256 CAP_KINDS=DT /venv/main/bin/python -u experiments/capacity_pdt_v1/eval_cap.py > experiments/capacity_pdt_v1/eval_d256_DT.log 2>&1
