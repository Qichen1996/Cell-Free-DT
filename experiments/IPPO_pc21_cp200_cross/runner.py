"""Fixed B-trained IPPO policy cross-environment evaluation, CSV only."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,csv,hashlib,subprocess,concurrent.futures,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/IPPO_pc21_cp200_cross'
OUT=ROOT/'analysis/sim_stats/IPPO_pc21_cp200_cross'
SOURCE=ROOT/'results/weights/B1000_checkpoint_test/by_tag/eps200/pc_2.1'
AH='ba618fc08fdbc719527fa58bf21857b66221861cda1e80ff55929b62067f6886'
CH='9c89f9b2bcc11798e7a67fe92a3432e22cfa84e40b7edd8e5957d3f93f6c2004'
NAMES=['A','C','B','B36','B500','B2000']
EXPECTED={'A':('A',1000,16,250.,.25,.98),'B':('B',1000,16,250.,.25,.90),'C':('C',1000,16,250.,.25,.90),'B36':('B',1000,36,1000/6,1/6,.90),'B500':('B',500,16,125.,.125,.90),'B2000':('B',2000,16,500.,.5,.95)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 assert sha(SOURCE/'actor.pt')==AH and sha(SOURCE/'critic.pt')==CH
 for n,h in json.loads((SOURCE/'manifest.json').read_text()).items():assert sha(SOURCE/n)==h
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def worker(name,seed,preflight=False):
 verify();torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 helper=ROOT/'experiments/selected_seven_pc33_eval/eval_pc33.py'
 scope={'__name__':'cross_environment_helper','__file__':str(helper)}
 exec(compile(helper.read_text(),str(helper),'exec'),scope)
 stats=OUT/('preflight' if preflight else name)
 scenario,area,n,dist,neighbor,cluster=EXPECTED[name]
 dest=stats/scenario/f'SEED{seed}';assert not dest.exists()
 env,obs,params=scope['environment'](name,seed,stats)
 assert params==dict(scenario=scenario,area=area,aps=n,spacing_m=dist,neighbor_km=neighbor,cluster=cluster,min_ap=1),params
 from arguments import get_config
 from agents.mappo.policy import MappoPolicy
 args=get_config().parse_args([])
 for k,v in json.loads((SOURCE/'architecture.json').read_text()).items():setattr(args,k,v)
 args.model_version='';args.use_wandb=False
 policy=MappoPolicy(args,env.observation_space[0],env.cent_observation_space[0],env.action_space[0])
 policy.actor.load_state_dict(torch.load(SOURCE/'actor.pt',map_location='cpu',weights_only=False),strict=True)
 policy.critic.load_state_dict(torch.load(SOURCE/'critic.pt',map_location='cpu',weights_only=False),strict=True);policy.prep_rollout()
 with torch.no_grad():assert policy.act(obs,deterministic=True).shape==(n,2)
 evidence=dict(seed=seed,steps=10080,actor_sha256=AH,critic_sha256=CH,model_source=str(SOURCE),pc_weight=2.1,checkpoint=200,evaluation_weights=[40,.5,.5,3],environment=params,obs_shape=list(np.asarray(obs).shape))
 print('ENV_OK',name,seed,json.dumps(evidence),flush=True)
 if preflight:print('PREFLIGHT_OK',name,flush=True);return
 print('SEED_START',name,seed,flush=True)
 with torch.no_grad():
  for step in range(10080):
   act=policy.act(obs,deterministic=True);assert act.shape==(n,2)
   obs,_,_,done,_,_=env.step(act);assert bool(done)==(step==10079)
   if (step+1)%1440==0:
    assert shutil.disk_usage(ROOT).free>2*1024**3
    print('DAY_COMPLETE',name,seed,(step+1)//1440,flush=True)
 env.close()
 for file in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  p=dest/(file+'.csv');assert p.is_file() and p.stat().st_size>0
  with p.open() as f:assert next(f,None) and next(f,None)
 with (dest/'reward.csv').open() as f:
  rows=list(csv.DictReader(f));assert len(rows)==10080 and all(np.isfinite(float(r['pc_kw'])) for r in rows)
 assert not list(dest.glob('*.pkl'));verify();save(dest/'verified.json',evidence)
 print('SEED_COMPLETE',name,seed,flush=True)
def main():
 verify();assert not OUT.exists();assert not (BASE/'plan.json').exists()
 save(BASE/'plan.json',dict(model_source=str(SOURCE),actor_sha256=AH,critic_sha256=CH,environments=NAMES,seeds=list(range(1,6)),steps=10080,parallel=3,evaluation_weights=[40,.5,.5,3],new_runs=30))
 for name in NAMES:
  p=subprocess.run([sys.executable,'-u',__file__,'preflight',name,'1'],cwd=ROOT,capture_output=True,text=True)
  with (BASE/'logs'/f'preflight-{name}.log').open('x') as f:f.write(p.stdout+p.stderr)
  assert p.returncode==0,p.stdout+p.stderr
  print('PREFLIGHT_OK',name,flush=True)
 def job(task):
  name,seed=task
  with (BASE/'logs'/f'{name}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',name,str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'FAILED {name} {seed}; no automatic retry'
  print('CASE_COMPLETE',name,seed,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(job,[(name,seed) for seed in range(1,6) for name in NAMES]))
 save(BASE/'complete.json',dict(runs=30,actor_sha256=AH,critic_sha256=CH))
 print('ALL_COMPLETE_IPPO_PC21_CP200_CROSS_30',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[2],int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
