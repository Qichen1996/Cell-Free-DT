"""Eval SHPDT (self-history prompt, own history at test time) and DT (no prompt) on 7 scenes, seeds 101-105.
RTG target = mean initial AP return of that scene's training data. trajectory.csv deleted after verification (recorded)."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,csv,subprocess,concurrent.futures,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(ROOT/'experiments/selfhist_pdt_v1'));sys.path.insert(0,str(ROOT))
import train_selfhist as T
BASE=ROOT/'experiments/selfhist_pdt_v1/eval';STATS=ROOT/'analysis/sim_stats/selfhist_pdt_v1'
SEEDS=[101,102,103,104,105];KINDS=['SHPDT','DT']
SPEC={s[0]:s for s in T.SCENES}
def save(p,d):T.save(p,d)
def environment(name,seed,stats):
 env_,files,traffic,aps,area,cluster,_=SPEC[name]
 grid=int(np.sqrt(aps));n=grid**2;dist=area/grid
 from network import config as nc
 nc.areaSize=np.array([area,area],dtype=float);nc.numBS=n;nc.interBSDist=dist
 nc.bsPositions=nc.square_grid_centers(grid,grid,area,area);nc.coverageGainFraction=cluster;nc.minCandidateBS=1
 from network.base_station import BaseStation
 from network.network import MultiCellNetwork
 from network.user_equipment import UserEquipment
 from env import MultiCellNetEnv
 BaseStation.inter_dist=dist/1000;MultiCellNetwork.inter_bs_dist=dist
 MultiCellNetwork.default_area=nc.areaSize;MultiCellNetwork.default_bs_poses=nc.bsPositions
 MultiCellNetEnv.bs_poses=nc.bsPositions;MultiCellNetEnv.num_agents=n
 UserEquipment.coverage_gain_fraction=cluster;UserEquipment.min_candidate_bs=1
 from utils import set_log_level
 set_log_level('ERROR')
 scenario=['A','B','C'][traffic]
 env=MultiCellNetEnv(seed=seed,scenario=scenario,area_size=nc.areaSize,accelerate=3000,episode_len=10080,w_str=40,w_sen=.5,w_tol=.5,w_qos=40,w_xqos=0,w_pc=3,save_trajectory=True,stats_dir=str(stats))
 obs,_,_=env.reset()
 assert np.asarray(obs).shape==(n,110) and np.allclose(np.array([b.pos[:2] for b in env.net.bss.values()]),nc.bsPositions)
 assert UserEquipment.coverage_gain_fraction==cluster
 return env,obs,scenario,dict(scenario=scenario,area=area,aps=n,spacing_m=dist,cluster=cluster)
def prepare():
 from trajectory_io import load_trajectory_pickle
 out={}
 for name,files,*_ in T.SCENES:
  v=[]
  for p in files:
   for tr in load_trajectory_pickle(p):v.append(float(tr['rewards'].sum(axis=0,dtype=np.float64).mean()))
  out[name]=float(np.mean(v));print('RTG',name,len(v),out[name],flush=True)
 save(BASE/'rtg.json',dict(targets=out,method='mean initial AP return over all training episodes of the scene',scale=1))
@torch.no_grad()
def evaluate(kind,name,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 mp=ROOT/'dt_results/selfhist_pdt_v1'/kind/'model.pt';ck=torch.load(mp,map_location='cpu',weights_only=False)
 assert ck['updates']==270000
 model=T.Model(kind=='SHPDT');model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=json.loads((BASE/'rtg.json').read_text())['targets'][name]
 stats=STATS/kind/name;env,obs,scenario,params=environment(name,seed,stats);n=params['aps']
 dest=stats/scenario/f'SEED{seed}'
 C,H=T.CTX,T.H
 ts=torch.arange(10080).repeat(n,1);S=torch.zeros(n,10080,110);A=torch.zeros(n,10080,2);R=torch.zeros(n,10080,1);RW=torch.zeros(n,10080,1)
 running=torch.full((n,1),target)
 for step in range(10080):
  S[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);R[:,step]=running
  left=max(0,step-C+1);right=max(C,step+1)
  hi=T.history_index(left);valid=torch.from_numpy(hi>=0);hc=torch.from_numpy(np.where(hi>=0,hi,0))
  hs=S[:,hc]*valid[None,:,None];ha=A[:,hc]*valid[None,:,None];hr=RW[:,hc]*valid[None,:,None]
  heads=model(hs,ha,hr,valid.expand(n,H),ts[:,left:right],S[:,left:right],A[:,left:right],R[:,left:right])
  idx=step if step<C else -1
  act=torch.stack([h[:,idx].argmax(-1) for h in heads],dim=-1);A[:,step]=act
  obs,_,reward,done,_,_=env.step(act.numpy());rw=torch.as_tensor(np.asarray(reward).reshape(n,1),dtype=torch.float32)
  RW[:,step]=rw;running-=rw
  assert bool(done)==(step==10079)
 env.close()
 for f in ['reward','ue_stats','trajectory']:assert (dest/(f+'.csv')).stat().st_size>0
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000
 ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 drop=[100*d/max(1,d+c) for d,c in zip(ds,cs)]
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 save(dest/'verified.json',dict(kind=kind,env=name,seed=seed,steps=10080,model_sha256=T.sha(mp),rtg_target=target,power_w=pc,drop_pct=drop,dropped=ds,completed=cs,environment=params,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE',kind,name,seed,pc,drop,flush=True)
def main():
 if len(sys.argv)>1:evaluate(sys.argv[1],sys.argv[2],int(sys.argv[3]));return
 assert not (BASE/'plan.json').exists() and not STATS.exists()
 for k in KINDS:assert (ROOT/'dt_results/selfhist_pdt_v1'/k/'complete.json').is_file()
 save(BASE/'plan.json',dict(kinds=KINDS,seeds=SEEDS,envs=list(SPEC),steps=10080,reward=[40,.5,.5,3]))
 prepare()
 def job(t):
  k,name,seed=t;lf=BASE/'logs'/f'{k}-{name}-{seed}.log';lf.parent.mkdir(exist_ok=True)
  with lf.open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,k,name,str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',k,name,seed,'rc',rc,flush=True);return (k,name,seed,rc)
 tasks=[(k,e,s) for s in SEEDS for e in SPEC for k in KINDS]
 with concurrent.futures.ThreadPoolExecutor(max_workers=35) as pool:res=list(pool.map(job,tasks))
 failed=[r for r in res if r[3]!=0]
 summ={}
 for k in KINDS:
  for e in SPEC:
   vs=[json.loads((STATS/k/e/['A','B','C'][SPEC[e][2]]/f'SEED{s}'/'verified.json').read_text()) for s in SEEDS if (k,e,s,0) in res]
   if not vs:continue
   ds=np.sum([v['dropped'] for v in vs],0);cs=np.sum([v['completed'] for v in vs],0)
   summ[f'{k}/{e}']=dict(n=len(vs),power_w=float(np.mean([v['power_w'] for v in vs])),drop_pct=[float(100*d/max(1,d+c)) for d,c in zip(ds,cs)])
 save(BASE/'complete.json',dict(runs=len(res),failed=failed,summary=summ))
 print('ALL_COMPLETE',json.dumps(summ),flush=True)
if __name__=='__main__':main()
