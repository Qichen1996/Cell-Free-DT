#!/bin/bash
# Waits for each training's complete.json (no training launched), then runs its checkpoint sweep. Eval only.
cd /workspace/Cell-Free-DT-new
E=experiments/cp_sweep_v2
export ARCH_JSON=$PWD/experiments/tradeoff_selected_single_collect200_v1/models/pc0.6_trainseed1_cp240/architecture.json
go(){ # scene wdir name
  until [ -f $2/$1/pc_1.0/complete.json ]; do sleep 120; done
  echo "$(date) START $3"; SWEEP_SCENE=$1 SWEEP_W=$2 SWEEP_NAME=$3 /venv/main/bin/python -u $E/runner.py > $E/$3.main.log 2>&1; echo "$(date) END $3 rc=$?"
}
go B500x2 $PWD/results/weights/train_B500x2_w100_c90_v1 sweep_B500x2_w100 &
go A3x $PWD/results/weights/train_A3x_str80_c90_v1 sweep_A3x_str80 &
wait; echo "$(date) ALL_SWEEPS_DONE"; touch $E/ALL_SWEEPS_DONE
