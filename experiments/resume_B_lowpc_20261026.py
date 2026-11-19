import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,subprocess,concurrent.futures,shutil,csv,statistics
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_lowpc_seed1to5';H=B/'resume_20261026'
scope={'__name__':'recovery_helpers','__file__':str(R/'experiments/B_lowpc_seed1to5.py')}
exec(compile(Path(scope['__file__']).read_text(),scope['__file__'],'exec'),scope)
def main():
 assert not H.exists();(H/'logs').mkdir(parents=True)
 rows=scope['specs']();old=json.loads((B/'state.json').read_text())['paths'];paths={};pending=[]
 for i,s in enumerate(rows):
  for seed in range(1,6):
   key=f'{i}-{seed}';p=Path(old.get(key,str(scope['O']/s['tag']/f'B/B/SEED{seed}')))
   if (p/'verified.json').exists():
    scope['audit'](p,s,seed);paths[key]=str(p)
   else:
    if p.exists():
     dest=H/'interrupted_outputs'/s['tag']/f'SEED{seed}';dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(dest))
    pending.append((i,seed))
 (H/'plan.json').write_text(json.dumps(dict(total=150,verified_reused=len(paths),pending=pending,evaluation_seeds=[1,2,3,4,5],parallel=2,auto_retry=False),indent=2))
 print('RECOVERY_VERIFIED',len(paths),'PENDING',len(pending),flush=True)
 def save():
  t=H/'state.tmp';t.write_text(json.dumps(dict(complete=len(paths),total=150,paths=paths),indent=2));t.replace(H/'state.json')
 def task(x):
  i,seed=x;assert shutil.disk_usage(R).free>3*1024**3
  with (H/'logs'/f'{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',str(R/'experiments/B_lowpc_seed1to5.py'),'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failure {x}; no retry'
  return i,seed
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for i,seed in pool.map(task,pending):
   paths[f'{i}-{seed}']=str(scope['O']/rows[i]['tag']/f'B/B/SEED{seed}');save();print('COMPLETE',i,seed,len(paths),flush=True)
 summary=[]
 for i,s in enumerate(rows):
  powers=[];ds=[0,0,0];cs=[0,0,0]
  for seed in range(1,6):
   p=Path(paths[f'{i}-{seed}']);powers.append(scope['audit'](p,s,seed))
   with (p/'ue_stats.csv').open() as f:
    for r in csv.DictReader(f):
     j=[.05,.15,.3].index(float(r['delay_budget']));ds[j]+=int(float(r['dropped_packets']));cs[j]+=int(float(r['completed_packets']))
  drops=[100*d/(d+c) for d,c in zip(ds,cs)]
  summary.append(dict(**s,power_w=statistics.mean(powers),drop_pct=drops,dropped=ds,completed=cs,eligible=all(x<=.5 for x in drops)))
 selected=[]
 for pc in [1.8,2.1]:
  for ts in [1,2,3]:
   eligible=[r for r in summary if r['pc']==pc and r['training_seed']==ts and r['eligible']]
   selected.append(min(eligible,key=lambda r:(r['power_w'],r['checkpoint'])) if eligible else dict(pc=pc,training_seed=ts,eligible=False))
 (B/'summary.json').write_text(json.dumps(dict(results=summary,selected=selected,evaluation_seeds=[1,2,3,4,5],independent_test=False),indent=2))
 (B/'complete.json').write_text(json.dumps(dict(runs=150,recovered=True)));print('ALL_COMPLETE_B_LOWPC_SEED1TO5_150',flush=True)
if __name__=='__main__':main()
