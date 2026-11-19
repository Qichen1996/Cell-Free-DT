import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
# Evaluate PDT_front_relabel_{MODE}_v1 on scenario B: prompt grid x seeds 1-10, fixed B RTG (-426.00145441593037), same env/worker as single200 evals.
# Prompt p is passed as pc_preference=p*3.3 (worker computes pc/3.3). trajectory.csv deleted after checks (as for the other PDT evals).
from pathlib import Path
import json,sys,subprocess,concurrent.futures,importlib.util,hashlib,shutil
MODE=Path(__file__).parent.name.split('_')[4]  # experiments/PDT_front_relabel_eval_{MODE}_v1
R=Path('/workspace/Cell-Free-DT-new');B=Path(__file__).parent;O=R/f'analysis/sim_stats/PDT_front_relabel_eval_{MODE}_v1'
BASE=R/'experiments/PDT_Bsingle200_prompt_ABC_v1'
M=R/f'dt_results/PDT_front_relabel_{MODE}_v1/PDT/model.pt';H=json.loads((R/f'experiments/PDT_front_relabel_{MODE}_v1/complete.json').read_text())['model_sha256']
PROMPTS=json.loads((B/'prompts.json').read_text());SEEDS=list(range(1,11))
S=[dict(env='B',pc=p*3.3,prompt=p,tag=f'prompt_{p:.3f}') for p in PROMPTS]
FREE=0.7*1024**3
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(i,seed,pre=False):
 spec=importlib.util.spec_from_file_location('base_eval',BASE/'runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.B=B;m.O=O;m.M=M;m.H=H;m.specs=lambda:S
 m.worker(i,seed,pre)
def finish(i,seed):
 s=S[i];d=O/s['tag']/s['env']/s['env']/f'SEED{seed}'
 for f in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  p=d/f'{f}.csv';assert p.is_file() and p.stat().st_size>0,p
 v=json.loads((d/'verified.json').read_text());assert v['seed']==seed and v['model_sha256']==H and abs(v['prompt']-s['prompt'])<1e-9
 t=d/'trajectory.csv';(d/'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv',bytes=t.stat().st_size,sha256=sha(t),reason='not needed for PC/drop; disk limit; user decision 2026-11-04')))
 t.unlink()
def main():
 assert not (B/'plan.json').exists() and not O.exists()
 assert sha(M)==H and shutil.disk_usage(R).free>1.1*1024**3
 (B/'logs').mkdir(parents=True,exist_ok=True);n=len(S)*len(SEEDS)
 (B/'plan.json').write_text(json.dumps(dict(mode=MODE,model=str(M),sha256=H,specs=S,seeds=SEEDS,runs=n,steps=10080,rtg='fixed B -426.00145441593037',reward=[40,.5,.5,3],parallel=12,keep_trajectory_csv=False,auto_retry=False),indent=2))
 state=dict(phase='preflight',complete=0,total=n,jobs=[])
 def save():
  p=B/'state.tmp';p.write_text(json.dumps(state));p.replace(B/'state.json')
 def job(i,seed,pre=False):
  assert shutil.disk_usage(R).free>FREE
  with (B/'logs'/f'{"pre" if pre else "eval"}-{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'pre' if pre else 'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,(i,seed,rc)
  if not pre:finish(i,seed)
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p:list(p.map(lambda i:job(i,1,True),range(len(S))))
 print('PREFLIGHT_OK_ALL',flush=True)
 state['phase']='simulating';save()
 tasks=[(i,seed) for seed in SEEDS for i in range(len(S))]
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p:
  it=iter(tasks);pending={}
  def submit():
   try:i,s=next(it)
   except StopIteration:return
   pending[p.submit(job,i,s)]=(i,s)
  for _ in range(12):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    pair=pending.pop(f)
    try:f.result()
    except Exception:state.update(phase='failed',failed=pair);save();raise
    state['complete']+=1;state['jobs'].append(pair);save();print('COMPLETE',state['complete'],flush=True);submit()
 state['phase']='complete';save();(B/'complete.json').write_text(json.dumps(dict(mode=MODE,runs=n,model_sha256=H)))
 print(f'ALL_COMPLETE_PDT_FRONT_RELABEL_EVAL_{MODE}',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='pre')
 else:main()
