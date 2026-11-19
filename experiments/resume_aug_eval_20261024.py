import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,importlib.util,subprocess,concurrent.futures,shutil,csv,math
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(R))
B=R/'experiments/selected_seven_baseline_aug_eval_v1'
spec=importlib.util.spec_from_file_location('original',B/'runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RES=B/'resume_20261024'
def dest(s,seed):
 scenario='A' if s['env']=='A' else 'C' if s['env']=='C' else 'B'
 return m.OUT/s['tag']/s['env']/scenario/f'SEED{seed}'
def check(s,seed):
 d=dest(s,seed);v=json.loads((d/'verified.json').read_text())
 assert v['seed']==seed and v['steps']==10080 and v['model_sha256']==m.HASHES[s['kind']] and v['rtg_target']==s['rtg']
 assert v['reward']==[40,.5,.5,3] and v['prompt_enabled']==(s['kind']=='PDT')
 if s['kind']=='PDT':assert abs(v['pc_preference']-s['pc'])<1e-12
 for n in ['reward','ue_stats','trajectory','bs_stats','net_stats']:
  with (d/f'{n}.csv').open() as f:assert next(f,None) and next(f,None)
 with (d/'reward.csv').open() as f:vals=[float(x['pc_kw']) for x in csv.DictReader(f)]
 assert len(vals)==10080 and all(math.isfinite(x) for x in vals)
def main():
 assert not RES.exists(),'No duplicate resume';RES.mkdir();(RES/'logs').mkdir()
 p=json.loads((B/'plan.json').read_text());assert p['specs']==m.specs() and p['source_sha256']==m.sha(m.SOURCE)
 for k,h in m.HASHES.items():assert m.sha(m.MODELS/k/'model.pt')==h
 state=json.loads((B/'state.json').read_text());done=[];todo=[]
 for seed in m.SEEDS:
  for i,s in enumerate(m.specs()):
   d=dest(s,seed)
   if (d/'verified.json').exists():check(s,seed);done.append([i,seed])
   else:
    assert state['jobs'].get(f'{i}-{seed}')!='complete'
    if d.exists():
     target=RES/'interrupted_outputs'/str(i)/f'SEED{seed}';target.parent.mkdir(parents=True,exist_ok=True)
     assert m.OUT in d.resolve().parents and not d.is_symlink();d.rename(target)
    todo.append([i,seed])
 assert len(done)+len(todo)==185
 (RES/'plan.json').write_text(json.dumps(dict(reused=done,pending=todo,total=185,parallel=3),indent=2))
 state=dict(phase='simulating',complete=len(done),total=185,jobs={f'{i}-{s}':'complete' for i,s in done})
 def update():
  t=B/'state.tmp';t.write_text(json.dumps(state,indent=2));t.replace(B/'state.json')
 update();print('REUSED_VERIFIED',len(done),'PENDING',len(todo),flush=True)
 def task(pair):
  i,seed=pair;assert shutil.disk_usage(R).free>4*1024**3
  with (RES/'logs'/f'{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',str(B/'runner.py'),'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {i} {seed}; no retry';check(m.specs()[i],seed);return pair
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  it=iter(todo);pending={}
  def submit():
   try:x=next(it)
   except StopIteration:return
   pending[pool.submit(task,x)]=x
  for _ in range(3):submit()
  while pending:
   fs,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in fs:
    i,seed=pending.pop(f)
    try:f.result()
    except Exception:state.update(phase='failed',failed_job=[i,seed]);update();raise
    state['jobs'][f'{i}-{seed}']='complete';state['complete']+=1;update();print('CASE_COMPLETE',i,seed,flush=True);submit()
 (B/'complete.json').write_text(json.dumps(dict(runs=185,groups=37,seeds=m.SEEDS)))
 state['phase']='complete';update();print('ALL_COMPLETE_BASELINE_AUG_EVAL_185',flush=True)
if __name__=='__main__':main()
