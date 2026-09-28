#!/bin/sh
seed_max=1
log_level="NOTICE"


context_len=10
rtg_scale=1

wandb_user="qichenw"
wandb_api_key="e4bee9a24174f0daabc173c5384733a60c1ea981"

# for len in 15; do
#     for rtg in 1; do
#     python train.py --context_len ${len} --rtg_scale ${rtg} $@
#     python train.py --context_len ${len} --rtg_scale ${rtg} --norm $@
#     done
# done

for len in 5; do
    for rtg in 1; do
    python train_dt.py --context_len ${len} --rtg_scale ${rtg} $@
    done
done
