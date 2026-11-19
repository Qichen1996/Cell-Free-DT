import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,csv,math,statistics,subprocess,concurrent.futures,hashlib,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/PDT_prompt_ABC_seed1to5';O=R/'analysis/sim_stats/PDT_prompt_ABC_seed1to5'
H='738f006d0ff4e3c1ed665509aa0613341c00ef8aa7ca62a9f127e4ed3c5118b6'
PROMPTS=[.2,.4,.6,.8,1.];ENVS=['A','B','C'];SEEDS=[1,2,3,4,5];PARALLEL=6
def path(name,prompt,seed):return O/f'prompt_{prompt:.2f}'/name/name/f'SEED{seed}'
def worker(name,prompt,seed,pre=False):
 assert prompt in PROMPTS and name in ENVS and seed in SEEDS
 assert hashlib.sha256((B/'model.pt').read_bytes()).hexdigest()==H
 source=(B/'evaluation_template.py').read_text().replace('1.8/3.3',repr(prompt)).replace('pc_preference=1.8',f'pc_preference={prompt*3.3!r}')
 scope={'__name__':'prompt_scan_helper','__file__':str(B/'evaluation_template.py')}
 exec(compile(source,str(B/'evaluation_template.py'),'exec'),scope)
 scope.update(BASE=B,MODEL=B/'model.pt',EXPECTED=H,STATS=O/f'prompt_{prompt:.2f}')
 scope['evaluate'](name,seed,pre)
 if not pre:audit(name,prompt,seed)
def audit(name,prompt,seed):
 p=path(name,prompt,seed);v=json.loads((p/'verified.json').read_text())
 assert v['seed']==seed and v['steps']==10080 and v['model_sha256']==H
 assert abs(v['prompt']-prompt)<1e-12 and abs(v['pc_preference']-prompt*3.3)<1e-12
 target=json.loads((B/'rtg.json').read_text())['targets'][name]
 assert v['rtg_target']==target and v['reward']==[40,.5,.5,3]
 assert v['environment']==dict(scenario=name,area=1000,aps=16,spacing_m=250.,neighbor_km=.25,cluster=.98 if name=='A' else .9,min_ap=1)
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  with (p/f'{n}.csv').open() as f:reader=csv.reader(f);assert next(reader,None) and next(reader,None)
 with (p/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080 and all(math.isfinite(float(r[k])) for r in rows for k in ['pc_kw','reward'])
 assert not list(p.glob('*.pkl'))
 return p,statistics.mean(float(r['pc_kw']) for r in rows)*1000
def main():
 assert not O.exists() and not (B/'plan.json').exists()
 targets=json.loads((B/'rtg.json').read_text())['targets']
 (B/'plan.json').write_text(json.dumps(dict(model_sha256=H,scenarios=ENVS,prompts=PROMPTS,corresponding_pc_weights=[p*3.3 for p in PROMPTS],evaluation_seeds=SEEDS,steps=10080,total=75,new_runs=75,reused_runs=0,parallel=PARALLEL,rtg_targets={n:targets[n] for n in ENVS},evaluation_weights=[40,.5,.5,3],single_frozen_model=True,save_traj_pkl=False,auto_retry=False),indent=2))
 def pre(task):
  n,p=task
  with (B/'logs'/f'preflight-{n}-{p:.2f}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',n,str(p),'1'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Preflight failed {n} {p}'
  print('PREFLIGHT_OK',n,p,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(pre,[(n,p) for n in ENVS for p in PROMPTS]))
 print('ALL_15_PREFLIGHTS_OK',flush=True)
 done=[]
 def save():
  t=B/'state.tmp';t.write_text(json.dumps(dict(complete=len(done),total=75,done=done),indent=2));t.replace(B/'state.json')
 save()
 def job(task):
  n,p,s=task;assert shutil.disk_usage(R).free>4*1024**3
  with (B/'logs'/f'{n}-{p:.2f}-{s}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',n,str(p),str(s)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {task}; no automatic retry'
  audit(n,p,s);return task
 tasks=[(n,p,s) for s in SEEDS for p in PROMPTS for n in ENVS]
 with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as pool:
  for future in concurrent.futures.as_completed([pool.submit(job,t) for t in tasks]):
   task=future.result();done.append(task);save();print('CASE_COMPLETE',task,len(done),flush=True)
 results=[]
 for n in ENVS:
  for p in PROMPTS:
   power=[];ds=[0]*3;cs=[0]*3
   for s in SEEDS:
    dest,w=audit(n,p,s);power.append(w)
    with (dest/'ue_stats.csv').open() as f:
     for r in csv.DictReader(f):
      j=[.05,.15,.3].index(float(r['delay_budget']));ds[j]+=int(float(r['dropped_packets']));cs[j]+=int(float(r['completed_packets']))
   drops=[100*d/(d+c) for d,c in zip(ds,cs)]
   results.append(dict(scenario=n,prompt=p,pc_preference=p*3.3,rtg_target=targets[n],power_w=statistics.mean(power),drop_pct=drops,dropped=ds,completed=cs,eligible=all(v<=.5 for v in drops)))
 selected=[]
 for n in ENVS:
  ok=[r for r in results if r['scenario']==n and r['eligible']]
  selected.append(min(ok,key=lambda r:r['power_w']) if ok else dict(scenario=n,eligible=False))
 (B/'summary.json').write_text(json.dumps(dict(results=results,selected=selected,selection_uses_evaluation_seeds=True),indent=2))
 (B/'complete.json').write_text(json.dumps(dict(runs=75,model_sha256=H)))
 print('ALL_COMPLETE_PDT_PROMPT_ABC_75',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[2],float(sys.argv[3]),int(sys.argv[4]),sys.argv[1]=='preflight')
 else:main()
