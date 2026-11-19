#!/bin/bash
cd /workspace/Cell-Free-DT-new
/venv/main/bin/python -u experiments/heldout_seed_v1/ppo_runner.py > experiments/heldout_seed_v1/ppo.log 2>&1
/venv/main/bin/python -c "import json,sys;sys.exit(0 if json.load(open(\"experiments/heldout_seed_v1/complete.json\"))[\"failed\"]==[] else 1)" || { echo PPO_FAILED_STOP >> experiments/heldout_seed_v1/ppo.log; exit 1; }
/venv/main/bin/python -u experiments/heldout_seed_v1/pdt_eval.py > experiments/heldout_seed_v1/eval.log 2>&1
