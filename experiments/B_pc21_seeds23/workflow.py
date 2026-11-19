import subprocess,sys
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new')
for name in ['B_pc21_seeds23','B_pc21_seeds23_checkpoint_eval']:
 rc=subprocess.call([sys.executable,'-u',str(R/'experiments'/name/'runner.py')],cwd=R)
 if rc:raise SystemExit(rc)
print('ALL_COMPLETE_B_PC21_TRAIN_AND_EVAL',flush=True)
