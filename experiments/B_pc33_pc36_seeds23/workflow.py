from pathlib import Path
import sys,json,subprocess,concurrent.futures,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_pc33_pc36_seeds23'
for code in ['33','36']:
 out=R/f'results/weights/B_pc{code}_seeds23';assert not out.exists();out.mkdir()
 for seed in [2,3]:(out/f'seed{seed}').mkdir()
def job(task):
 code,pc,seed=task;assert shutil.disk_usage(R).free>4*1024**3
 exp=R/f'experiments/B_pc{code}_seeds23'
 with (exp/'logs'/f'train-seed{seed}.log').open('x') as f:
  rc=subprocess.call([sys.executable,'-u',str(exp/'runner.py'),'train',str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert rc==0,f'{task} failed; no automatic retry'
 done=json.loads((R/f'results/weights/B_pc{code}_seeds23/seed{seed}/pc_{pc}/complete.json').read_text());assert done['steps']==2520000
 print('TRAIN_COMPLETE',pc,seed,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for f in concurrent.futures.as_completed([pool.submit(job,(c,p,s)) for c,p in [('33',3.3),('36',3.6)] for s in [2,3]]):f.result()
for code in ['33','36']:(R/f'experiments/B_pc{code}_seeds23/complete.json').write_text(json.dumps(dict(seeds=[2,3],steps_each=2520000)))
(B/'complete.json').write_text(json.dumps(dict(models=4,steps_each=2520000)))
print('ALL_COMPLETE_B_PC33_PC36_SEEDS23',flush=True)
