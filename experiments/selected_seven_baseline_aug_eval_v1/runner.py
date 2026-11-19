"""Matched original evaluation suite for both baseline-augmented models."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,subprocess,concurrent.futures,shutil
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new');BASE=ROOT/'experiments/selected_seven_baseline_aug_eval_v1'
OUT=ROOT/'analysis/sim_stats/selected_seven_baseline_aug_eval_v1'
SOURCE=ROOT/'experiments/selected_seven_pc33_eval/eval_pc33.py'
MODELS=ROOT/'dt_results/selected_seven_baseline_aug_v1'
HASHES={'PDT':'273cf98f0b2e45f2d46154898ddd519b0f3b8e6060e3ecf9507833d5163bf812','DT':'4b412b86eda211aaba551da78311bf9a76c4ec6aa5919c8f00c0a5fa945d026e'}
OLD_TARGETS=json.loads((ROOT/'experiments/selected_seven_pc33_eval/rtg.json').read_text())['targets']
ENVS=['A','B','C','B4','B36','B500','B2000'];SEEDS=list(range(101,106))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2)
def specs():
 rows=[]
 for env in ENVS:
  for kind in ['PDT','DT']:rows.append(dict(kind=kind,env=env,pc=1.8 if kind=='PDT' else None,rtg=OLD_TARGETS[env],tag=f'{kind}_baseline'))
 for pc in [.6,.9,1.2,1.5,2.1,2.4,2.7,3.,3.3]:rows.append(dict(kind='PDT',env='B',pc=pc,rtg=OLD_TARGETS['B'],tag=f'PDT_pc{pc:.1f}'))
 for delta in [-60,-30,30,60]:rows.append(dict(kind='PDT',env='B',pc=1.8,rtg=OLD_TARGETS['B']+delta,tag=f'PDT_rtg_delta{delta:+d}'))
 for delta in [-240,-210,-180,-150,-120,-90,-60,-30,30,60]:rows.append(dict(kind='DT',env='B',pc=None,rtg=OLD_TARGETS['B']+delta,tag=f'DT_rtg_delta{delta:+d}'))
 assert len(rows)==37
 return rows
def worker(index,seed,preflight=False):
 s=specs()[index];kind=s['kind'];model=MODELS/kind/'model.pt'
 assert sha(model)==HASHES[kind]
 source=SOURCE.read_text()
 destroot=BASE/'preflight_stats'/str(index) if preflight else OUT/s['tag']
 replacements={"STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33'":f'STATS=Path({str(destroot)!r})',
 "dt_results/selected_seven_pc33_v1/PDT/model.pt":f'dt_results/selected_seven_baseline_aug_v1/{kind}/model.pt',
 '738f006d0ff4e3c1ed665509aa0613341c00ef8aa7ca62a9f127e4ed3c5118b6':HASHES[kind],
 "target=json.loads((BASE/'rtg.json').read_text())['targets'][name]":f'target={s["rtg"]!r}',
 'model=training.Model(True)':f'model=training.Model({kind=="PDT"!r})'}
 for old,new in replacements.items():assert old in source,old;source=source.replace(old,new)
 if kind=='DT':
  source=source.replace('pc_preference=1.8,prompt=1.8/3.3,','prompt_enabled=False,')
  source=source.replace("'PROMPT',1.8/3.3,","'PROMPT_ENABLED',False,")
 else:
  source=source.replace('pc_preference=1.8,',f'prompt_enabled=True,pc_preference={s["pc"]!r},')
  source=source.replace('1.8/3.3',f'({s["pc"]!r})/3.3')
 source=source.replace(' model.load_state_dict(',f" assert checkpoint['prompt_enabled'] is {kind=='PDT'!r}\n model.load_state_dict(")
 if preflight:
  old=" print('SEED_START',name,seed,flush=True)";assert old in source
  source=source.replace(old," with torch.no_grad():\n  states[:,0]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);rtg[:,0]=running\n  heads=model(ts[:,:8],states[:,:8],actions[:,:8],rtg[:,:8],prompt)\n  assert len(heads)==2 and all(torch.isfinite(h).all() for h in heads)\n  assert torch.stack([h[:,0].argmax(-1) for h in heads],dim=-1).shape==(n,2)\n print('PREFLIGHT_OK',name,seed,flush=True)\n return")
 scope={'__name__':'augmented_eval_worker','__file__':str(SOURCE)}
 exec(compile(source,str(BASE/'worker_in_memory.py'),'exec'),scope)
 scope['evaluate'](s['env'],seed)
def main():
 assert not (BASE/'plan.json').exists() and not OUT.exists(),'No duplicate/retry/overwrite'
 assert shutil.disk_usage(ROOT).free>4*1024**3
 assert sha(SOURCE)=='01f8530803c60a630ed14e9dcdecc96c2dd5e20846f372699cf9d45e201d279c'
 for k,h in HASHES.items():assert sha(MODELS/k/'model.pt')==h and json.loads((MODELS/k/'complete.json').read_text())['updates']==270000
 save(BASE/'plan.json',dict(models={k:dict(path=str(MODELS/k/'model.pt'),sha256=h) for k,h in HASHES.items()},specs=specs(),seeds=SEEDS,runs=185,groups=37,steps=10080,parallel=3,reward=[40,.5,.5,3],old_rtg_targets=OLD_TARGETS,rtg_method='Use original absolute RTG values for matched comparison; no retuning',baseline_reuse='PPO and rule-policy evaluations retained, not repeated',save_traj_pkl=False,auto_retry=False,source_sha256=sha(SOURCE)))
 state=dict(phase='preflight',complete=0,total=185,jobs={})
 def update():
  t=BASE/'state.tmp';t.write_text(json.dumps(state,indent=2));t.replace(BASE/'state.json')
 update()
 def preflight_task(index):
  s=specs()[index]
  with (BASE/'logs'/f'preflight-{index}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',str(index),'101'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Preflight {index} failed'
  print('PREFLIGHT_OK',index,s['kind'],s['env'],s['rtg'],flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(preflight_task,range(37)))
 state['phase']='simulating';update()
 def task(index,seed):
  assert shutil.disk_usage(ROOT).free>4*1024**3
  with (BASE/'logs'/f'eval-{index}-seed{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(index),str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {index} {seed}; no automatic retry'
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  it=iter([(i,s) for s in SEEDS for i in range(37)]);pending={}
  def submit():
   try:i,s=next(it)
   except StopIteration:return
   pending[pool.submit(task,i,s)]=(i,s)
  for _ in range(3):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    i,s=pending.pop(f)
    try:f.result()
    except Exception:
     state.update(phase='failed',failed_job=[i,s]);update();raise
    state['jobs'][f'{i}-{s}']='complete';state['complete']+=1;update();print('CASE_COMPLETE',i,s,flush=True);submit()
 save(BASE/'complete.json',dict(runs=185,groups=37,seeds=SEEDS));state['phase']='complete';update()
 print('ALL_COMPLETE_BASELINE_AUG_EVAL_185',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
