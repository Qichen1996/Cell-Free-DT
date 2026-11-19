"""Collect user-selected PPO teachers; reuse verified collection implementation."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,subprocess,concurrent.futures,time
from pathlib import Path
PROJECT=Path('/workspace/Cell-Free-DT-new')
BASE=PROJECT/'experiments/B1000_selected_traj'
sys.path.insert(0,str(PROJECT/'experiments/preference_prompt_v2'))
import workflow as w
import torch
POINTS=json.loads((BASE/'SELECTED_POINTS_FINAL.json').read_text())['results']
BYPC={r['pc_weight']:r for r in POINTS}
PCS=list(BYPC)
DATA=PROJECT/'analysis/trajectory/B1000_selected'
STATS=PROJECT/'analysis/sim_stats/B1000_selected_collection'
def folder(pc):
 r=BYPC[pc]
 return Path(r['source']) if r.get('source') else PROJECT/'results/weights/B1000_pc_sweep_extension'/f'pc_{pc:.1f}'
# The final point's historical plot row lacks hash/source; verified existing source.
BYPC[2.7]['actor_sha256']='3dbad535ee3f6bf239ad83f98d8c15aac47007fbee162e6f33ab80aacd874a98'
EXPECTED={pc:(BYPC[pc]['actor_sha256'],w.sha(folder(pc)/'critic.pt')) for pc in PCS}
def validate():
 for pc in PCS:
  assert w.sha(folder(pc)/'actor.pt')==EXPECTED[pc][0],pc
  assert w.sha(folder(pc)/'critic.pt')==EXPECTED[pc][1],pc
  assert (folder(pc)/'architecture.json').is_file(),str(folder(pc))
def teacher(pc,env):
 from arguments import get_config
 from agents.mappo.policy import MappoPolicy
 config=get_config().parse_args([])
 for key,value in json.loads((folder(pc)/'architecture.json').read_text()).items(): setattr(config,key,value)
 config.model_version='';config.use_wandb=False
 policy=MappoPolicy(config,env.observation_space[0],env.cent_observation_space[0],env.action_space[0])
 for name in ['actor','critic']:
  getattr(policy,name).load_state_dict(torch.load(folder(pc)/(name+'.pt'),map_location='cpu',weights_only=False),strict=True)
 policy.prep_rollout()
 return policy
old_check=w.check_trajectory
def check(tr):
 old_check(tr)
 pc=float(tr['preference_pc'])
 assert tr['actor_sha256']==EXPECTED[pc][0]
 tr['model_source']=str(folder(pc));tr['checkpoint']=BYPC[pc]['checkpoint']
 tr['critic_sha256']=EXPECTED[pc][1]
w.BASE=BASE;w.DATA=DATA;w.STATS=STATS;w.PCS=PCS;w.EXPECTED=EXPECTED
w.validate_sources=validate;w.teacher=teacher;w.check_trajectory=check
# original check function lives in common and has its own preference list
import common
common.PCS=PCS
REUSED=PROJECT/'analysis/trajectory/B1000_preference_v2/pc_2.7/B.pkl.zst'
def verify_reuse():
 from trajectory_io import load_trajectory_pickle
 meta=json.loads(REUSED.with_name('complete.json').read_text())
 assert w.sha(REUSED)==meta['sha256'] and meta['actor_sha256']==EXPECTED[2.7][0]
 data=load_trajectory_pickle(REUSED)
 assert len(data)==5 and [tr['seed'] for tr in data]==[1,2,3,4,5]
 for tr in data: check(tr)
 return dict(pc=2.7,path=str(REUSED),sha256=meta['sha256'],episodes=5,ap_trajectories=80,reused=True)
def main():
 validate();w.space()
 reused=verify_reuse()
 if len(sys.argv)>1:
  w.collect(float(sys.argv[1]));return
 assert not (BASE/'collection-plan.json').exists(),'Do not duplicate/restart collection'
 plan=dict(points=[dict(pc=pc,checkpoint=BYPC[pc]['checkpoint'],model_source=str(folder(pc)),actor_sha256=EXPECTED[pc][0],critic_sha256=EXPECTED[pc][1]) for pc in PCS],seeds=[1,2,3,4,5],steps=10080,parallel=9,reward=[40,.5,.5,3],environment=dict(scenario='B',area=1000,aps=16,spacing_m=250,neighbor_km=.25,cluster=.9,min_ap=1),reused=reused)
 w.save_new(BASE/'collection-plan.json',plan)
 def job(pc):
  logfile=BASE/'logs'/f'pc_{pc:.1f}.log';logfile.parent.mkdir(exist_ok=True)
  with logfile.open('x') as log:
   proc=subprocess.Popen([sys.executable,str(Path(__file__)),str(pc)],stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT)
   print('GROUP_START',pc,proc.pid,flush=True)
   rc=proc.wait()
  if rc: raise RuntimeError(f'Collection pc={pc} failed rc={rc}; inspect {logfile}')
  print('GROUP_COMPLETE',pc,flush=True)
 started=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
  list(pool.map(job,[pc for pc in PCS if pc!=2.7]))
 files=[]
 for pc in PCS:
  if pc==2.7: files.append(reused);continue
  path=DATA/f'pc_{pc:.1f}'/'B.pkl.zst'
  meta=json.loads(path.with_name('complete.json').read_text())
  assert meta['sha256']==w.sha(path) and meta['actor_sha256']==EXPECTED[pc][0]
  files.append(dict(pc=pc,path=str(path),**meta))
 w.save_new(BASE/'dataset-manifest.json',dict(files=files,episodes=50,ap_trajectories=800,steps=10080,reward=[40,.5,.5,3]))
 w.save_new(BASE/'collection-complete.json',dict(episodes=50,new_episodes=45,ap_trajectories=800,seconds=time.monotonic()-started))
 print('ALL_COMPLETE_SELECTED_TRAJ_50',flush=True)
if __name__=='__main__': main()
