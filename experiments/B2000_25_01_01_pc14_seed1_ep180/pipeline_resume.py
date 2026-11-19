from pathlib import Path
import os,sys,subprocess,json
base=Path(__file__).resolve().parent
for script in ['evaluate_resume.py']:
 subprocess.run([sys.executable,'-u',str(base/script)],check=True)
eval_dir=base.with_name(base.name+'_eval')
assert json.loads((eval_dir/'complete.json').read_text())['runs']==40
rows=json.loads((eval_dir/'summary.json').read_text())
eligible=[r for r in rows if max(r['drop_pct'])<=.5]
winner=min(eligible,key=lambda r:r['power_w']) if eligible else None
(base/'selection.json').write_text(json.dumps(winner,indent=2))
if winner:
 print('SELECTED',winner['checkpoint'],winner['power_w'],winner['drop_pct'],flush=True)
 subprocess.run([sys.executable,'-u',str(base/'cross_resume.py')],env=dict(os.environ,WINNER_CP=winner['checkpoint']),check=True)
else:
 print('NO_QUALIFYING_CHECKPOINT',flush=True)
print('ALL_COMPLETE_B2000_PC14_PIPELINE',flush=True)
