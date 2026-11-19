#!/bin/bash
# usage: run.sh DIM LR KINDS...  -> train each kind, then held-out eval on A,C (seeds 206-210, demos PPO 201-205)
cd /workspace/Cell-Free-DT-new; C=experiments/capacity_pdt_v1; D=$1; L=$2; shift 2
for k in "$@"; do CAP_DIM=$D CAP_LR=$L /venv/main/bin/python -u $C/train_cap.py $k > $C/train_${k}_d$D.log 2>&1 || { echo TRAIN_FAILED $k >> $C/train_${k}_d$D.log; exit 1; }; done
CAP_DIM=$D /venv/main/bin/python -u $C/eval_cap.py > $C/eval_d$D.log 2>&1
