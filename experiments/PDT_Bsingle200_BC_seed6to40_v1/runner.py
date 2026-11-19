import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
# single200 PDT, scenarios B and C, 10 prompts, NEW eval seeds 6-40 (seeds 1-5 already in PDT_Bsingle200_prompt_ABC_v1).
# Reuses the ABC_v1 worker unchanged (same model, fixed RTG per scenario, same env); only output root and spec list differ.
# trajectory.csv is deleted after each run passes the CSV checks (user decision 2026-11-04); its size/sha256 are kept in trajectory_deleted.json.
from pathlib import Path
import json,sys,subprocess,concurrent.futures,importlib.util,hashlib,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/PDT_Bsingle200_BC_seed6to40_v1';O=R/'analysis/sim_stats/PDT_Bsingle200_BC_seed6to40_v1'
BASE=R/'experiments/PDT_Bsingle200_prompt_ABC_v1'
M=BASE/'model.pt';H='00c227342a337f9239a3572e387b1cd7696e084cfede9acc443d04bd6a190514'
PCS=[.6,.9,1.2,1.5,1.8,2.1,2.4,2.7,3.,3.3];SEEDS=list(range(6,41))
S=[dict(env=e,pc=p,tag=f'pc{p:.1f}') for e in ['B','C'] for p in PCS]
FREE=1.5*1024**3
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(i,seed,pre=False):
 spec=importlib.util.spec_from_file_location('base_eval',BASE/'runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.B=B;m.O=O;m.M=M;m.H=H;m.specs=lambda:S
 m.worker(i,seed,pre)
def finish(i,seed):
 s=S[i];d=O/s['tag']/s['env']/s['env']/f'SEED{seed}'
 for f in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  p=d/f'{f}.csv';assert p.is_file() and p.stat().st_size>0,p
 v=json.loads((d/'verified.json').read_text());assert v['seed']==seed and v['model_sha256']==H
 t=d/'trajectory.csv';(d/'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv',bytes=t.stat().st_size,sha256=sha(t),reason='not needed for PC/drop; disk limit; user decision 2026-11-04')))
 t.unlink()
def main():
 assert not (B/'plan.json').exists() and not O.exists()
 assert sha(M)==H and shutil.disk_usage(R).free>3*1024**3
 (B/'logs').mkdir(parents=True,exist_ok=True)
 (B/'plan.json').write_text(json.dumps(dict(model=str(M),sha256=H,specs=S,env=['B','C'],new_seeds=SEEDS,reused_seeds_1to5='analysis/sim_stats/PDT_Bsingle200_prompt_ABC_v1',new_runs=700,steps=10080,rtg='fixed per scenario from selected_seven_pc33_eval/rtg.json (B -426.00145441593037, C -849.5081207960611)',reward=[40,.5,.5,3],parallel=6,save_traj_pkl=False,keep_trajectory_csv=False,auto_retry=False),indent=2))
 state=dict(phase='preflight',complete=0,total=700,jobs=[])
 def save():
  p=B/'state.tmp';p.write_text(json.dumps(state));p.replace(B/'state.json')
 def job(i,seed,pre=False):
  assert shutil.disk_usage(R).free>FREE
  with (B/'logs'/f'{"pre" if pre else "eval"}-{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'pre' if pre else 'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,(i,seed,rc)
  if not pre:finish(i,seed)
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:list(p.map(lambda i:job(i,6,True),range(20)))
 print('PREFLIGHT_OK_ALL',flush=True)
 state['phase']='simulating';save()
 tasks=[(i,seed) for seed in SEEDS for i in range(20)]
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
 state['phase']='complete';save();(B/'complete.json').write_text(json.dumps(dict(new_runs=700,reused_runs_seed1to5=100,total=800,model_sha256=H)))
 print('ALL_COMPLETE_PDT_BSINGLE200_BC_SEED6TO40_700',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='pre')
 else:main()
