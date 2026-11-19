import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,sys,subprocess,concurrent.futures,importlib.util,hashlib,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/PDT_refresh150_single200_A_seed20_v1';O=R/'analysis/sim_stats/PDT_refresh150_single200_A_seed20_v1'
S=json.loads((B/'specs.json').read_text())
def worker(i,seed,pre=False):
 spec=importlib.util.spec_from_file_location('base_eval',R/'experiments/PDT_Bsingle200_prompt_ABC_v1/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 s=S[i];m.B=B;m.O=O;m.M=Path(s['model_path']);m.H=s['sha'];m.specs=lambda:S
 # Resource guard at 3 GiB; new output forecast <1 GiB, no PKL.
 m.SOURCE=m.SOURCE
 m.worker(i,seed,pre)
def main():
 assert not (B/'plan.json').exists()
 (B/'plan.json').write_text(json.dumps(dict(models=['refresh150','single200'],env='A',prompt_count=10,evaluation_seeds=list(range(1,21)),new_seeds=list(range(6,21)),new_runs=300,reused_runs=100,total=400,parallel=6,rtg=-288.18793981031047,reward=[40,.5,.5,3],save_traj_pkl=False,auto_retry=False),indent=2))
 state=dict(phase='preflight',complete=0,new_total=300,reused=100,jobs=[])
 def save():
  p=B/'state.tmp';p.write_text(json.dumps(state));p.replace(B/'state.json')
 def job(i,seed,pre=False):
  assert shutil.disk_usage(R).free>3*1024**3
  with (B/'logs'/f'{"pre" if pre else "eval"}-{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'pre' if pre else 'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,(i,seed,rc)
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:list(p.map(lambda i:job(i,6,True),range(20)))
 state['phase']='simulating';save()
 tasks=[(i,seed) for seed in range(6,21) for i in range(20)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:
  it=iter(tasks);pending={}
  def submit():
   try:i,s=next(it)
   except StopIteration:return
   pending[p.submit(job,i,s)]=(i,s)
  for _ in range(6):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    pair=pending.pop(f)
    try:f.result()
    except Exception:state.update(phase='failed',failed=pair);save();raise
    state['complete']+=1;state['jobs'].append(pair);save();print('COMPLETE',state['complete'],flush=True);submit()
 state['phase']='complete';save();(B/'complete.json').write_text(json.dumps(dict(new_runs=300,reused_runs=100,total=400)));print('ALL_COMPLETE_PDT_TWO_MODELS_A_SEED20_400',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='pre')
 else:main()
