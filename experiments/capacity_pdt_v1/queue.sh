#!/bin/bash
# sequential queue (user: 不用并行这么多): wait for d448 chain, then 792 PDT, then 256 DT (+ their evals)
cd /workspace/Cell-Free-DT-new/experiments/capacity_pdt_v1
while pgrep -f "run.sh 448" > /dev/null; do sleep 300; done
./run.sh 792 1e-4 PDT
CAP_DIM=256 CAP_LR=5e-4 /venv/main/bin/python -u train_cap.py DT > train_DT_d256.log 2>&1 && echo DT_TRAIN_DONE >> train_DT_d256.log
