#!/bin/bash
acc=3000
w_pc=0.3
context_len=10
rtg_scale=1
rtg=-15

for seed in {1..5}; do
#     ./simulate.py -S A --w_str 60 --w_sen 0 --w_tol 0 --w_pc 1 --seed $seed -a $acc --model_index 0 $@
     ./simulate.py -S A --w_str 150 --w_sen 1.5 --w_tol 1.5 --w_pc 1 --seed $seed -a $acc --model_index 0 $@
#    ./simulate.py -S A --stochastic --w_str 200 --w_sen 15 --w_tol 0 --w_pc 0.1 --seed $seed -a $acc $@
#     ./simulate.py -S A --w_str 100 --w_sen 2 --w_tol 2 --w_pc 1 --seed $seed -a $acc $@
#     ./simulate.py -S B --w_str 300 --w_sen 2 --w_tol 0 --w_pc 0.6 --seed $seed -a $acc $@
#    ./simulate.py -S C --w_str 400 --w_sen 5 --w_tol 0 --w_pc 0.1 --seed $seed -a $acc $@
#    ./simulate.py -S B --w_str 400 --w_sen 5 --w_tol 0 --w_pc 0.1 --seed $seed -a $acc $@
#    ./simulate.py -S A --w_str 400 --w_sen 5 --w_tol 0 --w_pc 0.1 --seed $seed -a $acc $@
#    ./simulate.py -S C -A fixed --seed $seed -a $acc $@
#     for S in A B C; do
#       ./simulate.py -S $S --w_str 130 --w_sen 15 --w_tol 0 --w_pc 0.5 --seed $seed -a $acc $@
##         for w in 40; do
##             # MODEL_DIR="${MODEL_BASE}/${S}/mappo/check/run19/models"
##             # ./simulate.py -S $S -A dt --w_qos $w --w_pc $w_pc --context_len $context_len --rtg_scale $rtg_scale --target_rtg $rtg --seed $seed -a $acc $@
##             ./simulate.py -S $S --w_qos $w --w_pc $w_pc --seed $seed -a $acc $@
##             # ./simulate.py -S $S -A random --w_qos $w --w_pc $w_pc --seed $seed -a $acc $@
##             # ./simulate.py -S $S -A fixed --w_qos $w --w_pc $w_pc --seed $seed -a $acc $@
##             # ./simulate.py -S $S -A simple1 --w_qos $w --w_pc $w_pc --seed $seed -a $acc $@
##             # ./simulate.py -S $S --w_qos $w --w_pc $w_pc --seed $seed -a $acc --use_wandb --run_version run13 $@
##         #     # ./simulate.py -S $S --w_qos $w --seed $seed -a $acc $@
##         #     # ./simulate.py -S $S -A fixed --seed $seed  --model_dir "$MODEL_DIR" -a $acc $@
##         done
#     done
done