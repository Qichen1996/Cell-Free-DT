import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,csv,math,hashlib,statistics,subprocess,concurrent.futures,pickle,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/tradeoff_selected123_collect_cross';O=R/'analysis/sim_stats/tradeoff_selected123_collect_cross';D=R/'analysis/trajectory/tradeoff_selected123_v1'
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
 rows=specs();assert len(rows)==30 and not (B/'plan.json').exists() and not O.exists() and not D.exists()
 cross=[i for i,s in enumerate(rows) if s['pc']==2.1];assert len(cross)==3
 paths={}
 for i in cross:
  for seed,path in enumerate(rows[i]['existing_B_paths'],1):audit(Path(path),rows[i],'B',seed,True);paths[f'{i}-B-{seed}']=path
 save(B/'plan.json',dict(models=rows,collection=dict(models=30,episodes=150,seeds=[11,12,13,14,15],days=7,parallel=6,reward=[40,.5,.5,3],dataset_dir=str(D),evaluation_seeds_excluded=[1,2,3,4,5],heldout101plus_excluded=True),cross=dict(models=cross,environments=list(ENVS),seeds=[1,2,3,4,5],total=105,new=90,reused=15,parallel=4),auto_retry=False,auto_train=False))
 def launch(task):
  mode,i,name,seed=task;log=B/'logs'/f'{mode}-{i}-{name}-{seed}.log'
  with log.open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,mode,str(i),name,str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'FAILED {task}; no retry. See {log}'
  return task
 pres=[('preflight',i,'B',11) for i in range(30)]+[('preflight',i,e,1) for i in cross for e in ENVS if e!='B']
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for result in pool.map(launch,pres):print('PREFLIGHT_OK',result,flush=True)
 print('ALL_48_PREFLIGHTS_OK',flush=True)
 ct=[('collect',i,'B',seed) for i in range(30) for seed in range(11,16)]
 xt=[('cross',i,e,seed) for i in cross for e in ENVS if e!='B' for seed in range(1,6)]
 completed=[]
 def state():
  tmp=B/'state.tmp';tmp.write_text(json.dumps(dict(collection_complete=sum(x[0]=='collect' for x in completed),collection_total=150,cross_complete=len(paths),cross_total=105,completed=completed,cross_paths=paths),indent=2));tmp.replace(B/'state.json')
 state()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as cp,concurrent.futures.ThreadPoolExecutor(max_workers=4) as xp:
  futures=[cp.submit(launch,t) for t in ct]+[xp.submit(launch,t) for t in xt]
  for f in concurrent.futures.as_completed(futures):
   t=f.result();completed.append(t);mode,i,e,seed=t
   if mode=='cross':paths[f'{i}-{e}-{seed}']=str(O/'cross'/rows[i]['tag']/e/expected(e)['scenario']/f'SEED{seed}')
   state();print('COMPLETE',t,'collection',sum(x[0]=='collect' for x in completed),'cross',len(paths),flush=True)
 manifest=[]
 for mode,i,e,seed in ct:
  q=O/'collection'/rows[i]['tag']/e/expected(e)['scenario']/f'SEED{seed}'
  record=json.loads((q/'result.json').read_text());assert sha(Path(record['path']))==record['sha256'];manifest.append(dict(model=rows[i],seed=seed,**record))
 summary=[]
 for i in cross:
  for e in ENVS:
   runs=[audit(Path(paths[f'{i}-{e}-{seed}']),rows[i],e,seed) for seed in range(1,6)];ds=[sum(v['dropped'][j] for v in runs) for j in range(3)];cs=[sum(v['completed'][j] for v in runs) for j in range(3)]
   summary.append(dict(model=rows[i],environment=e,power_w=statistics.mean(v['power_w'] for v in runs),rtg0=statistics.mean(v['rtg0'] for v in runs),drop_pct=[100*d/(d+c) for d,c in zip(ds,cs)],dropped=ds,completed=cs))
 save(B/'dataset-manifest.json',dict(episodes=150,ap_trajectories=2400,collection_seeds=list(range(11,16)),files=manifest))
 save(B/'cross-summary.json',dict(evaluation_seeds=list(range(1,6)),results=summary))
 save(B/'complete.json',dict(collection_episodes=150,cross_runs=105,new_cross_runs=90,reused_cross_runs=15))
 print('ALL_COMPLETE_TRADEOFF_SELECTED123_COLLECT150_CROSS105',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
 else:main()

