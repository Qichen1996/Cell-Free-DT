"""Matched new-model B evaluation. No training, overwrite, or automatic retry."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,subprocess,concurrent.futures,csv,shutil
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/selected_seven_pc33_matched_B'
STATS=ROOT/'analysis/sim_stats/selected_seven_pc33_matched_B'
SOURCE=ROOT/'experiments/selected_seven_pc33_eval/eval_pc33.py'
TARGET=-426.00145441593037
TARGETS=[TARGET-60,TARGET-30,TARGET,TARGET+30,TARGET+60]
PPO=ROOT/'results/weights/B1000_checkpoint230_test/by_tag/eps230/pc_1.8'
AH='f9561880b46706fd583b2e64b501c23ebdbb9b4e0349c0a3785ace4e830a4cd5'
CH='c65e07ce75adc296a1dd0bd3776987056c1f359d9f65cff57562f90a0b8c732a'
DH='239d7f9fbec9285dbfa2c8cf2aee0b0df09d94560e2a451405a9824a577f36f7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(kind,index,seed):
 source=SOURCE.read_text()
 destroot=STATS/kind/('baseline' if index==2 else f'test{index+1:02}')
 source=source.replace("STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33'",'STATS=Path('+repr(str(destroot))+')')
 source=source.replace("target=json.loads((BASE/'rtg.json').read_text())['targets'][name]",'target='+repr(TARGETS[index]))
 if kind=='DT':
  source=source.replace("dt_results/selected_seven_pc33_v1/PDT/model.pt","dt_results/selected_seven_pc33_v1/DT/model.pt")
  source=source.replace('738f006d0ff4e3c1ed665509aa0613341c00ef8aa7ca62a9f127e4ed3c5118b6',DH)
  source=source.replace('model=training.Model(True)','model=training.Model(False)')
 scope={'__name__':'matched_worker','__file__':str(SOURCE)}
 exec(compile(source,str(SOURCE),'exec'),scope)
 if kind!='IPPO':
  scope['evaluate']('B',seed)
  return
 import numpy as np,torch
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 assert sha(PPO/'actor.pt')==AH and sha(PPO/'critic.pt')==CH
 stats=destroot/'B';dest=stats/'B'/f'SEED{seed}';assert not dest.exists()
 env,obs,params=scope['environment']('B',seed,stats)
 from arguments import get_config
 from agents.mappo.policy import MappoPolicy
 args=get_config().parse_args([])
 for k,v in json.loads((PPO/'architecture.json').read_text()).items():setattr(args,k,v)
 args.model_version='';args.use_wandb=False
 policy=MappoPolicy(args,env.observation_space[0],env.cent_observation_space[0],env.action_space[0])
 policy.actor.load_state_dict(torch.load(PPO/'actor.pt',map_location='cpu',weights_only=False),strict=True)
 policy.critic.load_state_dict(torch.load(PPO/'critic.pt',map_location='cpu',weights_only=False),strict=True)
 policy.prep_rollout()
 print('ENV_OK IPPO',seed,json.dumps(params),flush=True)
 print('SEED_START IPPO',seed,flush=True)
 with torch.no_grad():
  for step in range(10080):
   act=policy.act(obs,deterministic=True);assert act.shape==(16,2)
   obs,_,_,done,_,_=env.step(act);assert bool(done)==(step==10079)
   if (step+1)%1440==0:
    assert shutil.disk_usage(ROOT).free>2*1024**3
    print('DAY_COMPLETE IPPO',seed,(step+1)//1440,flush=True)
 env.close()
 for file in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  p=dest/(file+'.csv');assert p.is_file() and p.stat().st_size>0
 with (dest/'reward.csv').open() as f:
  rows=list(csv.DictReader(f));assert len(rows)==10080 and all(np.isfinite(float(r['pc_kw'])) for r in rows)
 scope['save'](dest/'verified.json',dict(seed=seed,steps=10080,actor_sha256=AH,critic_sha256=CH,environment=params,reward=[40,.5,.5,3],source=str(PPO)))
 print('SEED_COMPLETE IPPO',seed,flush=True)
def main():
 if len(sys.argv)==4:return worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
 assert not (BASE/'plan.json').exists() and not STATS.exists()
 (BASE/'logs').mkdir(parents=True)
 # Baseline comparisons first. PDT baseline already exists and is reused.
 tasks=[(kind,2,seed) for seed in range(101,106) for kind in ['DT','IPPO']]
 tasks += [(kind,i,seed) for seed in range(101,106) for i in [1,3,0,4] for kind in ['DT','PDT']]
 plan=dict(tasks=tasks,rtg_targets=TARGETS,seeds=list(range(101,106)),steps=10080,parallel=3,prompt_pc=1.8,prompt=1.8/3.3,pdt_baseline=str(ROOT/'analysis/sim_stats/PDT_selected_seven_pc33/B/B'),ppo_source=str(PPO),source_sha256=sha(SOURCE),notes='Same B1000 environment and evaluation reward. DT contains no prompt tokens. IPPO RTG not applicable.')
 with (BASE/'plan.json').open('x') as f:json.dump(plan,f,indent=2)
 def job(task):
  kind,i,seed=task
  with (BASE/'logs'/f'{kind}-test{i+1:02}-seed{seed}.log').open('x') as f:
   rc=subprocess.call([sys.executable,'-u',str(Path(__file__)),kind,str(i),str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'FAILED {task} rc={rc}'
  print('CASE_COMPLETE',kind,i,seed,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(job,tasks))
 with (BASE/'complete.json').open('x') as f:json.dump(dict(runs=len(tasks)),f)
 print('ALL_COMPLETE_PC33_MATCHED_B_50',flush=True)
if __name__=='__main__':main()
