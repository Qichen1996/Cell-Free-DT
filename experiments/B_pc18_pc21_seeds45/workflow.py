from pathlib import Path
import subprocess,sys,json
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_pc18_pc21_seeds45';E=R/'experiments/B_pc18_pc21_seeds45_eval'
for script in [B/'runner.py',E/'prepare.py',E/'runner.py']:
 rc=subprocess.call([sys.executable,'-u',str(script)],cwd=R)
 if rc:raise SystemExit(rc)
(B/'workflow_complete.json').write_text(json.dumps(dict(models=4,eval_runs=100)))
print('ALL_COMPLETE_B_PC18_PC21_SEEDS45_WORKFLOW',flush=True)
