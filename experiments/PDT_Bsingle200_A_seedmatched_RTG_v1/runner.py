import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,subprocess,concurrent.futures,shutil,hashlib
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new'); B=R/'experiments/PDT_Bsingle200_A_seedmatched_RTG_v1'; O=R/'analysis/sim_stats/PDT_Bsingle200_A_seedmatched_RTG_v1'
M=B/'model.pt'; H='00c227342a337f9239a3572e387b1cd7696e084cfede9acc443d04bd6a190514'
SOURCE=R/'experiments/selected_seven_pc33_eval/eval_pc33.py'
SEED_TARGET=json.loads((B/'seed_targets.json').read_text())
TARGET=json.loads((R/'experiments/selected_seven_pc33_eval/rtg.json').read_text())['targets']
def specs():
 rows=[dict(env=e,pc=p,tag=f'pc{p:.1f}') for e in ['A'] for p in [.6,.9,1.2,1.5,1.8,2.1,2.4,2.7,3.,3.3]]
 return rows
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(i,seed,pre=False):
 s=specs()[i];assert sha(M)==H
 code=SOURCE.read_text();dest=B/'preflight_stats'/str(i) if pre else O/s['tag']
 replacements={"STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33'":f'STATS=Path({str(dest)!r})',"MODEL=ROOT/'dt_results/selected_seven_pc33_v1/PDT/model.pt'":f'MODEL=Path({str(M)!r})','00c227342a337f9239a3572e387b1cd7696e084cfede9acc443d04bd6a190514':H,"target=json.loads((BASE/'rtg.json').read_text())['targets'][name]":f'target={SEED_TARGET[s["env"]][str(seed)]!r}','pc_preference=1.8,':f'pc_preference={s["pc"]!r},','1.8/3.3':f'({s["pc"]!r})/3.3'}
 for old,new in replacements.items():assert old in code,old;code=code.replace(old,new)
 code=code.replace(" model.load_state_dict("," assert checkpoint['prompt_enabled'] is True and checkpoint['context']==8 and checkpoint['seed']==1\n model.load_state_dict(")
 if pre:
  old=" print('SEED_START',name,seed,flush=True)";assert old in code
  code=code.replace(old," states[:,0]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);rtg[:,0]=running\n heads=model(ts[:,:8],states[:,:8],actions[:,:8],rtg[:,:8],prompt)\n assert all(torch.isfinite(h).all() for h in heads)\n print('PREFLIGHT_OK',name,seed,flush=True)\n return")
 scope={'__name__':'worker','__file__':str(SOURCE)};exec(compile(code,str(B/'worker_in_memory.py'),'exec'),scope);scope['evaluate'](s['env'],seed)
def main():
 assert not (B/'plan.json').exists() and not O.exists()
 assert sha(M)==H and shutil.disk_usage(R).free>4*1024**3
 plan=dict(model=str(M),sha256=H,specs=specs(),seeds=[1,2,3,4,5],steps=10080,unique_runs=50,seven_scene_runs=0,prompt_A_runs=50,overlap_runs=0,parallel=6,rtg_targets=SEED_TARGET,rtg_method='Same-seed PPO AP-mean actual initial RTG; shared scalar across APs; diagnostic reference',reward=[40,.5,.5,3],auto_retry=False,save_traj_pkl=False)
 (B/'plan.json').write_text(json.dumps(plan,indent=2));state=dict(phase='preflight',complete=0,total=50,jobs=[])
 def save():
  p=B/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(B/'state.json')
 def job(i,seed,pre=False):
  assert shutil.disk_usage(R).free>4*1024**3
  with (B/'logs'/f'{"pre" if pre else "eval"}-{i}-{seed}.log').open('x') as f: rc=subprocess.call([sys.executable,'-u',__file__,'pre' if pre else 'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {i} {seed}; no retry'
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for i in pool.map(lambda i:(job(i,1,True),i)[1],range(10)):print('PREFLIGHT_OK',i,flush=True)
 state['phase']='simulating';save()
 tasks=[(i,s) for s in range(1,6) for i in range(10)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  it=iter(tasks);pending={}
  def submit():
   try:i,s=next(it)
   except StopIteration:return
   pending[pool.submit(job,i,s)]=(i,s)
  for _ in range(6):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    i,s=pending.pop(f)
    try:f.result()
    except Exception:state.update(phase='failed',failed=[i,s]);save();raise
    state['complete']+=1;state['jobs'].append([i,s]);save();print('CASE_COMPLETE',i,s,flush=True);submit()
 (B/'complete.json').write_text(json.dumps(dict(unique_runs=50,seven_scene_runs=0,prompt_A_runs=50,overlap_runs=0,model_sha256=H)))
 state['phase']='complete';save();print('ALL_COMPLETE_PDT_BSINGLE200_A_SEEDMATCHED_RTG_50',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='pre')
 else:main()
