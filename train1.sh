#!/bin/sh
seed_max=1
log_level="NOTICE"

export DEBUG=0
export TRAIN=1
export EVAL=0

scenario="A"
accelerate=3000  # 1 step = 0.02 * 1200 = 24 s
n_training_threads=5
n_rollout_threads=21
num_env_steps=$((10080 * 150))  # steps_per_episode * episodes
experiment="check"

algo="mappo"
gamma=0.99
gain=0.01
lr=5e-4
critic_lr=5e-4
value_loss_coef=1
ppo_epoch=5
num_mini_batch=1

w_qos=40 # 60 0.4
w_str=300
w_sen=2
w_tol=0
w_xqos=0
w_pc=0.5

log_interval=1

wandb_user="qichenw"
wandb_api_key="e4bee9a24174f0daabc173c5384733a60c1ea981"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    for scenario in B; do
    CUDA_VISIBLE_DEVICES=0 python train.py --algorithm_name ${algo} --experiment_name ${experiment} --scenario ${scenario} --accelerate ${accelerate} --seed ${seed} --n_training_threads ${n_training_threads} --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --gamma ${gamma} --lr ${lr} --critic_lr ${critic_lr} --value_loss_coef ${value_loss_coef} --user_name ${wandb_user} --log_level ${log_level} --log_interval ${log_interval} --w_str ${w_str} --w_sen ${w_sen} --w_tol ${w_tol} --w_qos ${w_qos} --w_xqos ${w_xqos} --w_pc ${w_pc} $@
    done
done

w_str=300
w_sen=2
w_tol=0
w_pc=0.6

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    for scenario in B; do
    CUDA_VISIBLE_DEVICES=0 python train.py --algorithm_name ${algo} --experiment_name ${experiment} --scenario ${scenario} --accelerate ${accelerate} --seed ${seed} --n_training_threads ${n_training_threads} --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --gamma ${gamma} --lr ${lr} --critic_lr ${critic_lr} --value_loss_coef ${value_loss_coef} --user_name ${wandb_user} --log_level ${log_level} --log_interval ${log_interval} --w_str ${w_str} --w_sen ${w_sen} --w_tol ${w_tol} --w_qos ${w_qos} --w_xqos ${w_xqos} --w_pc ${w_pc} $@
    done
done

