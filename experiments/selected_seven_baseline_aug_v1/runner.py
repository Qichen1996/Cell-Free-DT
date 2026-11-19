"""Collect missing rule-policy trajectories, then matched PDT/DT retraining."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,shutil,subprocess,concurrent.futures,pickle,time
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(ROOT))
BASE=ROOT/'experiments/selected_seven_baseline_aug_v1'
DATA=ROOT/'analysis/trajectory/selected_seven_baselines_v1'
MODELS=ROOT/'dt_results/selected_seven_baseline_aug_v1'
TEMPLATE=ROOT/'experiments/selected_seven_pc33_v1/train_mixed.py'
ENV_SOURCE=ROOT/'experiments/selected_seven_pc33_eval/eval_pc33.py'
ENVS=['A','B','C','B4','B36','B500','B2000'];KINDS=['Always_on','Auto_SM1'];SEEDS=list(range(1,6));STEPS=10080
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2)
def space():assert shutil.disk_usage(ROOT).free>4*1024**3,'Disk reserve below4GiB; no deletion or retry'
def environment(name,seed,stats):
 scope={'__name__':'baseline_collection_env','__file__':str(ENV_SOURCE)}
 exec(compile(ENV_SOURCE.read_text(),str(ENV_SOURCE),'exec'),scope)
 return scope['environment'](name,seed,stats)
def action(kind,bss):
 import numpy as np
 actions=[]
 for bs in bss:
  assert bs.num_ant==bs.max_antennas==8
  sm=0
  if kind=='Auto_SM1':
   assert bs.sleep in [0,1]
   sm=(0 if sum(u.required_rate for u in bs.queue)>0 else 1) if bs.sleep else (1 if sum(u.required_rate for u in bs.ues.values())==0 else 0)
  # Fixed-policy no-change antenna action; original Auto-SM1 increments at max.
  actions.append([1 if kind=='Always_on' else 2,sm])
 return np.asarray(actions,dtype=np.int64)
def check(tr,spec,kind,seed):
 import numpy as np
 n=spec['aps']
 assert np.array_equal(tr['prompt'],[{'A':0,'B':1,'C':2}[spec['scenario']],n,spec['area']])
 for k,shape in [('observations',(STEPS,n,110)),('actions',(STEPS,n,2)),('rewards',(STEPS,n,1))]:assert tr[k].shape==shape and np.isfinite(tr[k]).all(),(k,tr[k].shape)
 assert tr['seed']==seed and tr['policy']==kind and tr['weights']==[40,.5,.5,3]
 assert np.equal(tr['actions'],np.floor(tr['actions'])).all()
 assert (tr['actions'][...,0]==(1 if kind=='Always_on' else 2)).all()
 assert np.isin(tr['actions'][...,1],[0] if kind=='Always_on' else [0,1]).all()
def collect(kind,name,preflight=False):
 import numpy as np,torch,zstandard
 torch.set_num_threads(1)
 target=DATA/kind/name/'trajectories.pkl.zst';assert not target.exists()
 trajectories=[];spec=None
 for seed in ([1] if preflight else SEEDS):
  space();torch.manual_seed(seed);np.random.seed(seed)
  stats=BASE/('preflight_stats' if preflight else 'collection_stats')/kind/name
  env,obs,spec=environment(name,seed,stats);bss=list(env.net.bss.values());n=spec['aps']
  assert len(bss)==n and np.asarray(obs).shape==(n,110)
  print('ENV_OK',kind,name,seed,json.dumps(spec),flush=True)
  if preflight:
   act=action(kind,bss);obs,_,reward,done,_,_=env.step(act)
   assert act.shape==(n,2) and np.asarray(reward).shape==(n,1) and np.isfinite(reward).all() and not done
   print('PREFLIGHT_OK',kind,name,flush=True);return
  dest=stats/spec['scenario']/f'SEED{seed}';assert not dest.exists()
  observations=np.empty((STEPS,n,110),dtype=np.float32);actions=np.empty((STEPS,n,2),dtype=np.float32);rewards=np.empty((STEPS,n,1),dtype=np.float32)
  print('SEED_START',kind,name,seed,flush=True)
  for step in range(STEPS):
   act=action(kind,bss);observations[step]=np.asarray(obs);actions[step]=act
   obs,_,reward,done,_,_=env.step(act);rewards[step]=np.asarray(reward).reshape(n,1)
   assert bool(done)==(step==STEPS-1)
   assert all(bs.sleep in ([0] if kind=='Always_on' else [0,1]) for bs in bss)
   if (step+1)%1440==0:space();print('DAY_COMPLETE',kind,name,seed,(step+1)//1440,flush=True)
  env.close()
  for f in ['reward','ue_stats','trajectory','net_stats','bs_stats']:assert (dest/f'{f}.csv').is_file()
  tr=dict(prompt=np.array([{'A':0,'B':1,'C':2}[spec['scenario']],n,spec['area']],dtype=np.float32),observations=observations,actions=actions,rewards=rewards,seed=seed,policy=kind,weights=[40,.5,.5,3],environment=spec,reference_pc=1.8,reference_pc_semantics='Inherited PC1.8 reference tag; rule policy has no optimized PC weight',rule_source='agents/fixed.py AlwaysOnPolicy; agents/simple.py SimplePolicySM1Only',rule_adaptation='Current simulator required-rate sums; inactive third connection action omitted; maximum8 antennas')
  check(tr,spec,kind,seed);trajectories.append(tr)
  save(dest/'collection_verified.json',dict(policy=kind,env=name,seed=seed,environment=spec,steps=STEPS,weights=[40,.5,.5,3]))
  print('SEED_COMPLETE',kind,name,seed,flush=True);del env
 target.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(target)+'.partial')
 with temp.open('xb') as f,zstandard.ZstdCompressor(level=3,write_checksum=True).stream_writer(f,closefd=False) as writer:pickle.dump(trajectories,writer,protocol=pickle.HIGHEST_PROTOCOL)
 with temp.open('rb') as f,zstandard.ZstdDecompressor().stream_reader(f) as reader:loaded=pickle.load(reader)
 assert len(loaded)==5
 for expected,actual in zip(trajectories,loaded):
  check(actual,spec,kind,actual['seed'])
  for k in ['observations','actions','rewards','prompt']:assert np.array_equal(expected[k],actual[k])
 assert not target.exists();os.link(temp,target);temp.unlink()
 save(target.parent/'complete.json',dict(policy=kind,env=name,environment=spec,seeds=SEEDS,episodes=5,ap_trajectories=5*n,steps=STEPS,path=str(target),sha256=sha(target),compressed_bytes=target.stat().st_size))
 print('COLLECTION_COMPLETE',kind,name,flush=True)
def training_source():
 source=TEMPLATE.read_text()
 for old,new in [("BASE=ROOT/'experiments/selected_seven_pc33_v1';MODELS=ROOT/'dt_results/selected_seven_pc33_v1'",f'BASE=Path({str(BASE)!r});MODELS=Path({str(MODELS)!r})'),('==382','==452'),('episodes=382 AP=4772','episodes=452 AP=5972'),('episodes=382','episodes=452'),('ap_trajectories=4772','ap_trajectories=5972'),('uniform 11 B1000 source groups','uniform source groups within each environment')]:
  assert old in source,old;source=source.replace(old,new)
 compile(source,str(BASE/'training_in_memory.py'),'exec')
 return source
def train(kind):
 plan=json.loads((BASE/'workflow-plan.json').read_text());assert sha(TEMPLATE)==plan['training_template_sha256']
 manifest=json.loads((BASE/'baseline-dataset-manifest.json').read_text());assert len(manifest['files'])==14
 source=training_source()
 scope={'__name__':'augmented_training','__file__':__file__};exec(compile(source,str(BASE/'training_in_memory.py'),'exec'),scope)
 old_entries=scope['entries'];OldDataset=scope['MixedDataset']
 def entries():
  rows=old_entries()
  for x in manifest['files']:
   assert sha(x['path'])==x['sha256'];s=x['environment']
   rows.append(dict(env=x['env'],path=x['path'],traffic={'A':0,'B':1,'C':2}[s['scenario']],aps=s['aps'],area=s['area'],spacing=s['spacing_m'],cluster=s['cluster'],episodes=5,preference=None,expected_sha=x['sha256'],baseline_policy=x['policy']))
  return rows
 class AugmentedDataset(OldDataset):
  def __init__(self):
   super().__init__()
   assert len(self.base.trajectories)==452
   assert sum(t['observations'].shape[1] for t in self.base.trajectories)==5972
   for spec,(a,b),record in zip(self.specs,self.ranges,self.manifest):
    if 'baseline_policy' in spec:
     record['prompt_assignment']='Inherited PC1.8 reference label; baseline has no optimized PC preference'
     for tr in self.base.trajectories[a:b]:check(tr,tr['environment'],spec['baseline_policy'],tr['seed']);assert tr['seed'] in SEEDS
 scope['entries']=entries;scope['MixedDataset']=AugmentedDataset
 scope['run'](kind)
def pipeline():
 assert not (BASE/'state.json').exists() and not DATA.exists() and not MODELS.exists(),'No duplicate or overwrite'
 space();training_source()
 import torch
 assert torch.cuda.is_available() and torch.isfinite(torch.ones(1,device='cuda')).all()
 state=dict(phase='preflight',groups_complete=0,groups_total=14)
 def update():
  p=BASE/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(BASE/'state.json')
 save(BASE/'workflow-plan.json',dict(environments=ENVS,policies=KINDS,collection_seeds=SEEDS,heldout_seeds_excluded=list(range(101,121)),episodes_new=70,episodes_total=452,ap_trajectories_total=5972,steps=STEPS,collection_parallel=3,reward=[40,.5,.5,3],models=['PDT','DT'],from_scratch=True,training_seed=1,updates_each=270000,rounds=1800,batch=256,context=8,blocks=3,dimension=128,heads=1,lr=.0005,weight_decay=.0001,dropout=.1,prompt='PC/3.3 unchanged; baselines inherit1.8 reference, not optimized PC',sampling='uniform seven environments; uniform source groups within environment, then episode/AP/window',training_template_sha256=sha(TEMPLATE),environment_helper_sha256=sha(ENV_SOURCE),auto_retry=False))
 update()
 def launch(args,label):
  space()
  with (BASE/'logs'/f'{label}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__]+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {label}; no automatic retry'
 tasks=[(k,n) for n in ENVS for k in KINDS]
 for k,n in tasks:launch(['preflight',k,n],f'preflight-{k}-{n}');print('PREFLIGHT_OK',k,n,flush=True)
 state['phase']='collecting';update()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  iterator=iter(tasks);pending={}
  def submit():
   try:k,n=next(iterator)
   except StopIteration:return
   pending[pool.submit(launch,['collect',k,n],f'collect-{k}-{n}')]=(k,n)
  for _ in range(3):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    k,n=pending.pop(f)
    try:f.result()
    except Exception:
     state.update(phase='failed',failed_group=[k,n]);update();raise
    state['groups_complete']+=1;update();print('GROUP_COMPLETE',k,n,flush=True);submit()
 files=[]
 for k,n in tasks:
  m=json.loads((DATA/k/n/'complete.json').read_text());assert sha(m['path'])==m['sha256'];files.append(m)
 save(BASE/'baseline-dataset-manifest.json',dict(files=files,episodes=70,ap_trajectories=1200,reward=[40,.5,.5,3]))
 save(BASE/'collection-complete.json',dict(groups=14,episodes=70,ap_trajectories=1200))
 for kind in ['PDT','DT']:
  state['phase']='training_'+kind;update();launch(['train',kind],'train-'+kind)
  m=json.loads((MODELS/kind/'complete.json').read_text());assert m['updates']==270000 and sha(MODELS/kind/'model.pt')==m['sha256']
 save(BASE/'complete.json',dict(models=['PDT','DT'],updates_each=270000,episodes=452,new_baseline_episodes=70))
 state['phase']='complete';update();print('ALL_COMPLETE_BASELINE_AUG_PDT_DT',flush=True)
if __name__=='__main__':
 if len(sys.argv)==1:pipeline()
 elif sys.argv[1] in ['collect','preflight']:collect(sys.argv[2],sys.argv[3],sys.argv[1]=='preflight')
 elif sys.argv[1]=='train':train(sys.argv[2])
 else:raise ValueError(sys.argv)
