cd /workspace/Cell-Free-DT-new
for s in 1 2 4 5; do echo "START DT $s $(date)"; /venv/main/bin/python -u experiments/new500_pdt_dt_B2000_cl90_v1/eval_b2000_cl90.py DT mean B2000 $s > experiments/new500_pdt_dt_B2000_cl90_v1/eval/logs_seq/DT-mean-B2000-$s.log 2>&1; echo "CASE DT $s rc $? $(date)"; done; echo SEQ_COMPLETE
