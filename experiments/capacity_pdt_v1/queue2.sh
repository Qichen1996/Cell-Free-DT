#!/bin/bash
# replaces queue.sh (user: dt 的先不用跑): wait for d448 chain, then 792 PDT + eval only. No DT.
cd /workspace/Cell-Free-DT-new/experiments/capacity_pdt_v1
while pgrep -f "run.sh 448" > /dev/null; do sleep 300; done
./run.sh 792 1e-4 PDT
