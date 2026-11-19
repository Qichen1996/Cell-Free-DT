import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,csv,math,hashlib,statistics,subprocess,concurrent.futures,pickle,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/tradeoff_selected_single_collect200_v1';O=R/'analysis/sim_stats/tradeoff_selected_single_collect200_v1';D=R/'analysis/trajectory/tradeoff_selected_single_seed1to20_v1'
sys.path.insert(0,str(R))
ENVS={'A':('A',1000,16,250.,.25,.98),'B':('B',1000,16,250.,.25,.9),'C':('C',1000,16,250.,.25,.9),'B4':('B',1000,4,500.,.5,.9),'B36':('B',1000,36,1000/6,1/6,.9),'B500':('B',500,16,125.,.125,.9),'B2000':('B',2000,16,500.,.5,.95)}
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def specs():return json.loads((B/'specs.json').read_text())
def verify(s):
 p=Path(s['source'])
 for n,h in s['manifest'].items():assert sha(p/n)==h
def expected(name):
 scenario,area,n,dist,near,cluster=ENVS[name]
 return dict(scenario=scenario,area=area,aps=n,spacing_m=dist,neighbor_km=near,cluster=cluster,min_ap=1)
def setup(s,name,seed,stats):
 import numpy as np,torch
 verify(s);torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 f=R/'experiments/selected_seven_pc33_eval/eval_pc33.py';scope={'__name__':'collection_environment','__file__':str(f)}
 exec(compile(f.read_text(),str(f),'exec'),scope)
 env,obs,params=scope['environment'](name,seed,stats);assert params==expected(name)
 from arguments import get_config
 from agents.mappo.policy import MappoPolicy
 args=get_config().parse_args([])
 for k,v in json.loads((Path(s['source'])/'architecture.json').read_text()).items():setattr(args,k,v)
 args.model_version='';args.use_wandb=False
 policy=MappoPolicy(args,env.observation_space[0],env.cent_observation_space[0],env.action_space[0])
 for k in ['actor','critic']:getattr(policy,k).load_state_dict(torch.load(Path(s['source'])/f'{k}.pt',map_location='cpu',weights_only=False),strict=True)
 policy.prep_rollout()
 with torch.no_grad():assert policy.act(obs,deterministic=True).shape==(params['aps'],2)
 return env,obs,policy,params
def audit(p,s,name,seed,reused=False):
 verify(s);v=json.loads((p/'verified.json').read_text())
 assert v['seed']==seed and v['checkpoint']==s['checkpoint'] and v['pc_weight']==s['pc']
 assert v['actor_sha256']==s['manifest']['actor.pt'] and v['critic_sha256']==s['manifest']['critic.pt'] and v['evaluation_weights']==[40,.5,.5,3]
 if 'environment' in v:assert v['environment']==expected(name)
 else:
  assert name=='B' and [v[k] for k in ['spacing_m','neighbor_km','cluster']]==[250,.25,.9]
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  with (p/f'{n}.csv').open() as f:reader=csv.reader(f);assert next(reader,None) and next(reader,None)
 with (p/'reward.csv').open() as f:rr=list(csv.DictReader(f))
 assert len(rr)==10080 and all(math.isfinite(float(v[k])) for v in rr for k in ['reward','pc_kw'])
 ds=[0]*3;cs=[0]*3
 with (p/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 return dict(power_w=statistics.mean(float(v['pc_kw']) for v in rr)*1000,rtg0=sum(float(v['reward']) for v in rr),dropped=ds,completed=cs)
def worker(mode,i,name,seed):
 import numpy as np,torch,zstandard
 s=specs()[i];pre=mode=='preflight';collect=mode=='collect';folder='preflight' if pre else 'collection' if collect else 'cross'
 stats=O/folder/s['tag']/name;scenario=expected(name)['scenario'];dest=stats/scenario/f'SEED{seed}'
 assert not dest.exists();env,obs,policy,params=setup(s,name,seed,stats)
 evidence=dict(seed=seed,steps=10080,actor_sha256=s['manifest']['actor.pt'],critic_sha256=s['manifest']['critic.pt'],model_source=s['source'],pc_weight=s['pc'],training_seed=s['training_seed'],checkpoint=s['checkpoint'],evaluation_weights=[40,.5,.5,3],environment=params,obs_shape=list(np.asarray(obs).shape))
 print('ENV_OK',json.dumps(evidence),flush=True)
 if pre:print('PREFLIGHT_OK',i,name,flush=True);return
 print('SEED_START',mode,i,name,seed,flush=True)
 n=params['aps']
 if collect:observations=np.empty((10080,n,110),np.float32);actions=np.empty((10080,n,2),np.float32);rewards=np.empty((10080,n,1),np.float32)
 with torch.no_grad():
  for step in range(10080):
   act=policy.act(obs,deterministic=True)
   if collect:observations[step]=obs;actions[step]=act
   obs,_,reward,done,_,_=env.step(act)
   if collect:rewards[step]=np.asarray(reward).reshape(n,1)
   assert bool(done)==(step==10079)
   if (step+1)%1440==0:
    assert shutil.disk_usage(R).free>4*1024**3
    print('DAY_COMPLETE',mode,i,name,seed,(step+1)//1440,flush=True)
 env.close();verify(s);save(dest/'verified.json',evidence);metric=audit(dest,s,name,seed)
 if collect:
  assert name=='B' and n==16
  for x in [observations,actions,rewards]:assert np.isfinite(x).all()
  with (dest/'reward.csv').open() as f:csvreward=np.array([float(v['reward']) for v in csv.DictReader(f)])
  assert np.allclose(rewards.mean(axis=(1,2)),csvreward,rtol=1e-5,atol=1e-6)
  tr=dict(prompt=np.array([1,16,1000],np.float32),observations=observations,actions=actions,rewards=rewards,seed=seed,training_seed=s['training_seed'],checkpoint=s['checkpoint'],preference_pc=s['pc'],weights=[40,.5,.5,3],training_weights=[80,.2,.2,s['pc']],environment=params,actor_sha256=s['manifest']['actor.pt'],critic_sha256=s['manifest']['critic.pt'])
  target=D/s['tag']/f'SEED{seed}.pkl.zst';target.parent.mkdir(parents=True,exist_ok=True)
  temp=target.with_name(target.stem+'.partial.pkl.zst')
  with temp.open('xb') as f,zstandard.ZstdCompressor(level=3,write_checksum=True).stream_writer(f,closefd=False) as w:pickle.dump([tr],w,protocol=pickle.HIGHEST_PROTOCOL)
  from trajectory_io import load_trajectory_pickle
  loaded=load_trajectory_pickle(temp)[0]
  for k in ['observations','actions','rewards','prompt']:assert np.array_equal(tr[k],loaded[k])
  temp.rename(target);metric.update(path=str(target),sha256=sha(target),compressed_bytes=target.stat().st_size,episode=1,ap_trajectories=16)
 save(dest/'result.json',metric);print('SEED_COMPLETE',mode,i,name,seed,flush=True)

def main():
 rows=specs();assert len(rows)==10 and not (B/'plan.json').exists() and not O.exists() and not D.exists()
 assert shutil.disk_usage(R).free>4*1024**3
 save(B/'plan.json',dict(models=rows,seeds=list(range(1,21)),episodes=200,ap_trajectories=3200,steps=10080,days=7,parallel=6,environment=expected('B'),reward=[40,.5,.5,3],deterministic=True,dataset_dir=str(D),auto_retry=False,auto_train=False,selection='User-confirmed single-model tradeoff curve, PC2.7 seed1 CP220'))
 def launch(task):
  mode,i,name,seed=task
  with (B/'logs'/f'{mode}-{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,mode,str(i),name,str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'FAILED {task}; no automatic retry'
  return task
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for t in pool.map(launch,[('preflight',i,'B',1) for i in range(10)]):print('PREFLIGHT_OK',t,flush=True)
 print('ALL_10_PREFLIGHTS_OK',flush=True);completed=[]
 def state(phase='collecting'):
  p=B/'state.tmp';p.write_text(json.dumps(dict(phase=phase,complete=len(completed),total=200,completed=completed),indent=2));p.replace(B/'state.json')
 state();tasks=[('collect',i,'B',seed) for seed in range(1,21) for i in range(10)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  it=iter(tasks);pending={}
  def submit():
   try:t=next(it)
   except StopIteration:return
   pending[pool.submit(launch,t)]=t
  for _ in range(6):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    t=pending.pop(f)
    try:f.result()
    except Exception:state('failed');raise
    completed.append(t);state();print('COMPLETE',t,len(completed),flush=True);submit()
 manifest=[]
 for _,i,e,seed in tasks:
  p=O/'collection'/rows[i]['tag']/'B'/'B'/f'SEED{seed}';metric=audit(p,rows[i],'B',seed);record=json.loads((p/'result.json').read_text());assert sha(Path(record['path']))==record['sha256'];manifest.append(dict(model=rows[i],seed=seed,**record))
 save(B/'dataset-manifest.json',dict(episodes=200,ap_trajectories=3200,collection_seeds=list(range(1,21)),files=manifest))
 save(B/'complete.json',dict(collection_episodes=200,models=10,seeds=list(range(1,21))));state('complete');print('ALL_COMPLETE_TRADEOFF_SELECTED_SINGLE_COLLECT200',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
 else:main()
