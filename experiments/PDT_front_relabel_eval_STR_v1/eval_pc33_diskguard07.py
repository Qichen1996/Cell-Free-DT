"""Seven environments, fixed PC1.8 prompt, training-only mean RTG, seeds101..105."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,time,csv,subprocess,concurrent.futures,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(ROOT/'experiments/selected_seven_pc33_v1'))
import train_mixed as training
BASE=ROOT/'experiments/selected_seven_pc33_eval'
STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33'
MODEL=ROOT/'dt_results/selected_seven_pc33_v1/PDT/model.pt'
EXPECTED='738f006d0ff4e3c1ed665509aa0613341c00ef8aa7ca62a9f127e4ed3c5118b6'
def save(p,d):training.save(p,d)
def environment(name,seed,stats):
 spec=next(e for e in training.entries() if e['env']==name)
 area=spec['area'];grid=int(np.sqrt(spec['aps']));n=grid**2;dist=area/grid
 from network import config as nc
 nc.areaSize=np.array([area,area],dtype=float);nc.numBS=n;nc.interBSDist=dist
 nc.bsPositions=nc.square_grid_centers(grid,grid,area,area);nc.coverageGainFraction=spec['cluster'];nc.minCandidateBS=1
 from network.base_station import BaseStation
 from network.network import MultiCellNetwork
 from network.user_equipment import UserEquipment
 from env import MultiCellNetEnv
 BaseStation.inter_dist=dist/1000;MultiCellNetwork.inter_bs_dist=dist
 MultiCellNetwork.default_area=nc.areaSize;MultiCellNetwork.default_bs_poses=nc.bsPositions
 MultiCellNetEnv.bs_poses=nc.bsPositions;MultiCellNetEnv.num_agents=n
 UserEquipment.coverage_gain_fraction=spec['cluster'];UserEquipment.min_candidate_bs=1
 from utils import set_log_level
 set_log_level('ERROR')
 scenario=['A','B','C'][spec['traffic']]
 env=MultiCellNetEnv(seed=seed,scenario=scenario,area_size=nc.areaSize,accelerate=3000,episode_len=10080,w_str=40,w_sen=.5,w_tol=.5,w_qos=40,w_xqos=0,w_pc=3,save_trajectory=True,stats_dir=str(stats))
 obs,_,_=env.reset()
 assert np.asarray(obs).shape==(n,110)
 assert np.array_equal(env.net.area,nc.areaSize)
 assert np.allclose(np.array([b.pos[:2] for b in env.net.bss.values()]),nc.bsPositions)
 assert BaseStation.inter_dist==dist/1000 and UserEquipment.coverage_gain_fraction==spec['cluster']
 return env,obs,dict(scenario=scenario,area=area,aps=n,spacing_m=dist,neighbor_km=dist/1000,cluster=spec['cluster'],min_ap=1)
def prepare():
 from trajectory_io import load_trajectory_pickle
 manifest=json.loads((ROOT/'dt_results/selected_seven_pc33_v1/PDT/dataset-manifest.json').read_text())
 groups={s[0]:[] for s in training.SPECS};records=[]
 for spec in manifest['files']:
  assert training.sha(spec['path'])==spec['sha256']
  data=load_trajectory_pickle(spec['path'])
  value=float(np.mean([tr['rewards'].sum(axis=0,dtype=np.float64).mean() for tr in data]))
  groups[spec['env']].append(value);records.append(dict(path=spec['path'],env=spec['env'],mean_initial_ap_return=value))
  del data
 targets={env:float(np.mean(vals)) for env,vals in groups.items()}
 save(BASE/'rtg.json',dict(targets=targets,source_groups=records,method='Mean initial AP return per source group, equal source-group weighting within environment; training data only',scale=1))
 print('RTG_READY',targets,flush=True)
@torch.no_grad()
def evaluate(name,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 assert training.sha(MODEL)==EXPECTED
 model=training.Model(True);checkpoint=torch.load(MODEL,map_location='cpu',weights_only=False)
 assert checkpoint['prompt_normalization']=='PC/3.3; no clipping' and checkpoint['prompt_dim']==1 and checkpoint['updates']==270000
 model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
 target=json.loads((BASE/'rtg.json').read_text())['targets'][name]
 stats=STATS/name;scenario='A' if name=='A' else 'C' if name=='C' else 'B'
 dest=stats/scenario/f'SEED{seed}';assert not dest.exists(),str(dest)
 env,obs,params=environment(name,seed,stats);n=params['aps']
 print('ENV_OK',name,seed,json.dumps(params),'PROMPT',1.8/3.3,'RTG',target,flush=True)
 ts=torch.arange(10080).repeat(n,1);states=torch.zeros(n,10080,110);actions=torch.zeros(n,10080,2);rtg=torch.zeros(n,10080,1)
 running=torch.full((n,1),target);prompt=torch.full((n,1),1.8/3.3)
 print('SEED_START',name,seed,flush=True)
 for step in range(10080):
  states[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);rtg[:,step]=running
  left=max(0,step-7);right=max(8,step+1)
  heads=model(ts[:,left:right],states[:,left:right],actions[:,left:right],rtg[:,left:right],prompt)
  idx=step if step<8 else -1
  act=torch.stack([h[:,idx].argmax(-1) for h in heads],dim=-1);actions[:,step]=act
  obs,_,reward,done,_,_=env.step(act.numpy());running-=torch.as_tensor(np.asarray(reward).reshape(n,1),dtype=torch.float32)
  assert bool(done)==(step==10079)
  if (step+1)%1440==0:
   assert shutil.disk_usage(ROOT).free>0.7*1024**3
   print('DAY_COMPLETE',name,seed,(step+1)//1440,flush=True)
 env.close()
 for file in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  path=dest/(file+'.csv');assert path.is_file() and path.stat().st_size>0
 with (dest/'reward.csv').open() as f:
  rows=list(csv.DictReader(f));assert len(rows)==10080 and all(np.isfinite(float(r['pc_kw'])) for r in rows)
 save(dest/'verified.json',dict(seed=seed,steps=10080,model_sha256=EXPECTED,pc_preference=1.8,prompt=1.8/3.3,rtg_target=target,reward=[40,.5,.5,3],environment=params))
 print('SEED_COMPLETE',name,seed,flush=True)
def main():
 if len(sys.argv)>1:evaluate(sys.argv[1],int(sys.argv[2]));return
 assert not (BASE/'plan.json').exists() and not STATS.exists()
 save(BASE/'plan.json',dict(model=str(MODEL),sha256=EXPECTED,seeds=[101,102,103,104,105],environments=[s[0] for s in training.SPECS],steps=10080,pc_preference=1.8,prompt=1.8/3.3,parallel=5,automatic_rtg_sweep=False))
 prepare()
 def job(task):
  name,seed=task;logfile=BASE/'logs'/f'{name}-{seed}.log';logfile.parent.mkdir(exist_ok=True)
  with logfile.open('x') as log:rc=subprocess.call([sys.executable,'-u',str(Path(__file__)),name,str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  assert rc==0,f'{name} seed{seed} rc={rc}; {logfile}'
  print('CASE_COMPLETE',name,seed,flush=True)
 tasks=[(name,seed) for seed in range(101,106) for name in [s[0] for s in training.SPECS]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(job,tasks))
 save(BASE/'complete.json',dict(runs=35,model_sha256=EXPECTED))
 print('ALL_COMPLETE_PDT_PC33_SEVEN_35',flush=True)
if __name__=='__main__':main()
