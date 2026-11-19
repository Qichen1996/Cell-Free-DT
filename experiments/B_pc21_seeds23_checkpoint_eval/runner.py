"""Wait for seven authorized training runs, then evaluate five checkpoints each."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,time,json,hashlib,shutil,subprocess,concurrent.futures,csv,math
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/B_pc21_seeds23_checkpoint_eval'
OUT=ROOT/'analysis/sim_stats/B_pc21_seeds23_checkpoint_eval'
SEEDS=list(range(101,106))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_new(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2)
def specs():
 return [dict(tag=f'B_pc2.1_trainseed{seed}',scenario='B',pc=2.1,training_seed=seed,training_dir=str(ROOT/f'results/weights/B_pc21_seeds23/seed{seed}/pc_2.1'),checkpoints=[180,200,220,230,240]) for seed in [2,3]]
def jobs():return [(s,cp) for s in specs() for cp in s['checkpoints']]
def worker(index,seed,preflight=False):
 spec,cp=jobs()[index];folder=BASE/'models'/spec['tag']/f'eps{cp}'
 manifest=json.loads((folder/'manifest.json').read_text())
 for n,h in manifest.items():assert sha(folder/n)==h
 source=(ROOT/'experiments/IPPO_pc21_cp200_cross/runner.py').read_text()
 source=source.replace('pc_weight=2.1,checkpoint=200',f"pc_weight={spec['pc']},checkpoint={cp}")
 scope={'__name__':'checkpoint_eval_worker','__file__':str(BASE/'runner.py')}
 exec(compile(source,str(BASE/'worker_in_memory.py'),'exec'),scope)
 scope.update(SOURCE=folder,AH=manifest['actor.pt'],CH=manifest['critic.pt'],OUT=OUT/spec['tag']/f'eps{cp}')
 scope['worker'](spec['scenario'],seed,preflight)
def main():
 assert not OUT.exists() and not (BASE/'plan.json').exists()
 plan=dict(models=specs(),evaluation_seeds=SEEDS,steps=10080,evaluation_weights=[40,.5,.5,3],parallel=2,new_runs=50,wait_for_all_training=True,auto_retry=False,save_traj_pkl=False)
 save_new(BASE/'plan.json',plan)
 previous=None
 while True:
  pending=[s['tag'] for s in specs() if not (Path(s['training_dir'])/'complete.json').is_file()]
  if pending!=previous:print('WAIT_TRAINING',json.dumps(pending),flush=True);previous=pending
  if not pending:break
  assert shutil.disk_usage(ROOT).free>3*1024**3,'Disk reserve below3GiB'
  for name in ['B_pc21_seeds23']:
   status=subprocess.run(['supervisorctl','status',name],capture_output=True,text=True).stdout
   assert 'RUNNING' in status or 'EXITED' in status,status
   if 'EXITED' in status:
    marker=ROOT/('experiments/A_pc_sweep_12_16/complete.json' if name.startswith('A_') else 'experiments/B_pc21_seeds23/complete.json')
    assert marker.is_file(),f'Training exited without complete marker: {name}'
  time.sleep(60)
 print('ALL_TRAINING_READY',flush=True)
 import yaml
 keys=list(json.loads((ROOT/'results/weights/B1000_pc_sweep/pc_1.8/architecture.json').read_text()))
 for index,(spec,cp) in enumerate(jobs()):
  trained=Path(spec['training_dir']);completion=json.loads((trained/'complete.json').read_text())
  src=Path(completion['actor_path']).parent
  assert sha(src/'actor.pt')==completion['actor_sha256']
  dest=BASE/'models'/spec['tag']/f'eps{cp}';assert not dest.exists();dest.mkdir(parents=True)
  pair={}
  for kind in ['actor','critic']:
   p=src/f'{kind}_eps{cp}.pt';assert p.is_file()
   target=dest/f'{kind}.pt';shutil.copy2(p,target);assert sha(p)==sha(target)
   pair[target.name]=sha(target)
  cfg={k:v['value'] for k,v in yaml.safe_load((src/'config.yaml').read_text()).items() if isinstance(v,dict) and 'value' in v}
  assert cfg['seed']==spec['training_seed'] and cfg['w_pc']==spec['pc']
  assert [cfg['w_str'],cfg['w_sen'],cfg['w_tol']]==([400,1,1] if spec['scenario']=='A' else [80,.2,.2])
  save_new(dest/'architecture.json',{k:cfg[k] for k in keys})
  save_new(dest/'manifest.json',pair)
  save_new(dest/'provenance.json',dict(**spec,checkpoint=cp,original_model_dir=str(src),original_actor=str(src/f'actor_eps{cp}.pt'),manifest=pair))
  with (BASE/'logs'/f'preflight-{index}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',str(index),'101'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Preflight failed index{index}'
  print('PREFLIGHT_OK',spec['tag'],cp,flush=True)
 state=dict(status='simulating',complete_runs=0,total_runs=50,jobs={})
 def state_save():
  temp=BASE/'state.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(BASE/'state.json')
 def task(index,seed):
  assert shutil.disk_usage(ROOT).free>3*1024**3
  logfile=BASE/'logs'/f'eval-{index}-seed{seed}.log'
  with logfile.open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(index),str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed index{index} seed{seed}; no automatic retry'
  return index,seed
 state_save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(task,index,seed) for seed in SEEDS for index in range(10)]
  for future in concurrent.futures.as_completed(futures):
   index,seed=future.result();state['jobs'][f'{index}-{seed}']='complete';state['complete_runs']+=1;state_save();print('CASE_COMPLETE',index,seed,flush=True)
 summary=[]
 for spec,cp in jobs():
  ds=[0,0,0];cs=[0,0,0];ps=0.;pn=0
  for seed in SEEDS:
   p=OUT/spec['tag']/f'eps{cp}'/spec['scenario']/spec['scenario']/f'SEED{seed}'
   with (p/'reward.csv').open() as f:
    rows=list(csv.DictReader(f));assert len(rows)==10080
    vals=[float(r['pc_kw']) for r in rows];assert all(math.isfinite(x) for x in vals);ps+=sum(vals);pn+=len(vals)
   with (p/'ue_stats.csv').open() as f:
    for r in csv.DictReader(f):
     j=[.05,.15,.3].index(float(r['delay_budget']));ds[j]+=int(float(r['dropped_packets']));cs[j]+=int(float(r['completed_packets']))
  summary.append(dict(tag=spec['tag'],checkpoint=cp,pc_kw=ps/pn,drop_percent=[100*d/(d+c) for d,c in zip(ds,cs)],dropped=ds,completed=cs))
 save_new(BASE/'summary.json',dict(results=summary,evaluation_seeds=SEEDS,method='Pooled UE packet counts by delay class; PC mean reward.pc_kw'))
 state['status']='complete';state_save();save_new(BASE/'complete.json',dict(runs=50,groups=10))
 print('ALL_COMPLETE_AB_CHECKPOINT_EVAL_50',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
