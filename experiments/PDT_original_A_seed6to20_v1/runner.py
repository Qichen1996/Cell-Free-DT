import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
# Original PDT (selected_seven_pc33_v1, sha 738f006d...), scenario A only, 10 prompts, NEW eval seeds 6-20.
# Seeds 1-5 already exist (PDT_prompt_ABC_original_grid_seed1to5 + PDT_prompt_ABC_seed1to5 for prompt 1.00).
# Same evaluation_template / model / fixed RTG as the seed1-5 grid; only output root and seeds differ.
# trajectory.csv deleted after checks (user decision 2026-11-04); size/sha256 kept in trajectory_deleted.json.
from pathlib import Path
import sys,json,csv,math,subprocess,concurrent.futures,hashlib,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/PDT_original_A_seed6to20_v1';O=R/'analysis/sim_stats/PDT_original_A_seed6to20_v1'
SRC=R/'experiments/PDT_prompt_ABC_original_grid_seed1to5'
H='738f006d0ff4e3c1ed665509aa0613341c00ef8aa7ca62a9f127e4ed3c5118b6'
PROMPTS=[pc/3.3 for pc in [.6,.9,1.2,1.5,1.8,2.1,2.4,2.7,3.,3.3]];SEEDS=list(range(6,21));PARALLEL=6
RTG=json.loads((SRC/'rtg.json').read_text())['targets']['A']
ENV_A=dict(scenario='A',area=1000,aps=16,spacing_m=250.,neighbor_km=.25,cluster=.98,min_ap=1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dest(prompt,seed):return O/f'prompt_{prompt:.2f}'/'A'/'A'/f'SEED{seed}'
def worker(prompt,seed,pre=False):
 assert any(abs(prompt-p)<1e-12 for p in PROMPTS) and seed in SEEDS
 assert sha(SRC/'model.pt')==H
 source=(SRC/'evaluation_template.py').read_text().replace('1.8/3.3',repr(prompt)).replace('pc_preference=1.8',f'pc_preference={prompt*3.3!r}')
 scope={'__name__':'prompt_scan_helper','__file__':str(SRC/'evaluation_template.py')}
 exec(compile(source,str(SRC/'evaluation_template.py'),'exec'),scope)
 scope.update(BASE=SRC,MODEL=SRC/'model.pt',EXPECTED=H,STATS=(B/'preflight_stats' if pre else O)/f'prompt_{prompt:.2f}')
 scope['evaluate']('A',seed,pre)
def finish(prompt,seed):
 p=dest(prompt,seed);v=json.loads((p/'verified.json').read_text())
 assert v['seed']==seed and v['steps']==10080 and v['model_sha256']==H and abs(v['prompt']-prompt)<1e-12
 assert abs(v['pc_preference']-prompt*3.3)<1e-12 and v['rtg_target']==RTG and v['reward']==[40,.5,.5,3] and v['environment']==ENV_A
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  f=p/f'{n}.csv';assert f.is_file() and f.stat().st_size>0,f
 with (p/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080 and all(math.isfinite(float(r[k])) for r in rows for k in ['pc_kw','reward'])
 t=p/'trajectory.csv';(p/'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv',bytes=t.stat().st_size,sha256=sha(t),reason='not needed for PC/drop; disk limit; user decision 2026-11-04')))
 t.unlink()
def main():
 assert not (B/'plan.json').exists() and not O.exists()
 assert sha(SRC/'model.pt')==H and shutil.disk_usage(R).free>3*1024**3
 (B/'logs').mkdir(parents=True,exist_ok=True)
 (B/'plan.json').write_text(json.dumps(dict(model=str(SRC/'model.pt'),model_sha256=H,scenario='A',prompts=PROMPTS,corresponding_pc_weights=[p*3.3 for p in PROMPTS],new_seeds=SEEDS,reused_seeds_1to5=['analysis/sim_stats/PDT_prompt_ABC_original_grid_seed1to5','analysis/sim_stats/PDT_prompt_ABC_seed1to5 (prompt 1.00)'],new_runs=150,steps=10080,rtg_target=RTG,evaluation_weights=[40,.5,.5,3],parallel=PARALLEL,keep_trajectory_csv=False,save_traj_pkl=False,auto_retry=False),indent=2))
 state=dict(phase='preflight',complete=0,total=150,jobs=[])
 def save():
  t=B/'state.tmp';t.write_text(json.dumps(state));t.replace(B/'state.json')
 save()
 def run(prompt,seed,pre):
  assert shutil.disk_usage(R).free>1.5*1024**3
  with (B/'logs'/f'{"pre" if pre else "eval"}-{prompt:.2f}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'pre' if pre else 'job',repr(prompt),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,(prompt,seed,rc)
  if not pre:finish(prompt,seed)
 with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as pool:list(pool.map(lambda p:run(p,6,True),PROMPTS))
 print('PREFLIGHT_OK_ALL',flush=True)
 state['phase']='simulating';save()
 tasks=[(p,s) for s in SEEDS for p in PROMPTS]
 with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as pool:
  futs={pool.submit(run,p,s,False):(p,s) for p,s in tasks}
  for f in concurrent.futures.as_completed(futs):
   try:f.result()
   except Exception:state.update(phase='failed',failed=futs[f]);save();raise
   state['complete']+=1;state['jobs'].append(futs[f]);save();print('COMPLETE',state['complete'],flush=True)
 state['phase']='complete';save()
 (B/'complete.json').write_text(json.dumps(dict(new_runs=150,reused_runs_seed1to5=50,total=200,model_sha256=H)))
 print('ALL_COMPLETE_PDT_ORIGINAL_A_SEED6TO20_150',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(float(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='pre')
 else:main()
