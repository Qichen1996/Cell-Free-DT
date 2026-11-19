"""PDT B500 seeds 1-5 re-sim with per-step action/BS-state log (actions, actual num_ant, actual sleep mode)
for action analysis vs PPO B500 cp290. Same model/demos/RTG(mean) as eval_pdt. Saves actlog.npz per seed."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,csv,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new');D=ROOT/'experiments/selfhist_pdt_v1';sys.path.insert(0,str(D));sys.path.insert(0,str(ROOT))
import train_selfhist as T
import eval_selfhist as E
import eval_pdt as P
OUT=D/'actlog_B500';STATS=ROOT/'analysis/sim_stats/selfhist_pdt_v1_actlog';NAME='B500';SEEDS=[1,2,3,4,5]
@torch.no_grad()
def run(seed):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 ck=torch.load(P.MP,map_location='cpu',weights_only=False);model=T.Model(True);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=json.loads((E.BASE/'rtg.json').read_text())['targets'][NAME]
 d=np.load(P.BASE/f'demo_{NAME}_{seed}.npz');hs,ha,hr=[torch.from_numpy(d[k]) for k in ('hs','ha','hr')]
 stats=STATS/'PDT'/NAME;env,obs,scenario,params=E.environment(NAME,seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}'
 bss=list(env.net.bss.values());hv=torch.ones(n,T.H,dtype=torch.bool);C=T.CTX
 ts=torch.arange(10080).repeat(n,1);S=torch.zeros(n,10080,110);A=torch.zeros(n,10080,2);R=torch.zeros(n,10080,1)
 ant=np.zeros((n,10080),np.int8);slp=np.zeros((n,10080),np.int8);running=torch.full((n,1),target)
 for step in range(10080):
  S[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);R[:,step]=running
  left=max(0,step-C+1);right=max(C,step+1)
  heads=model(hs,ha,hr,hv,ts[:,left:right],S[:,left:right],A[:,left:right],R[:,left:right])
  idx=step if step<C else -1
  act=torch.stack([h[:,idx].argmax(-1) for h in heads],dim=-1);A[:,step]=act
  obs,_,reward,done,_,_=env.step(act.numpy());running-=torch.as_tensor(np.asarray(reward).reshape(n,1),dtype=torch.float32)
  ant[:,step]=[b.num_ant for b in bss];slp[:,step]=[b.sleep for b in bss]
 env.close()
 np.savez_compressed(dest/'actlog.npz',actions=A.numpy().astype(np.int8),num_ant=ant,sleep=slp)
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 E.save(dest/'verified.json',dict(seed=seed,model_sha256=T.sha(P.MP),rtg_target=target,power_w=float(np.mean([float(r['pc_kw']) for r in rows]))*1000,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE',seed,flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:run(int(sys.argv[1]));sys.exit()
 OUT.mkdir(exist_ok=False)
 def job(s):
  with (OUT/f'{s}.log').open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,str(s)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',s,'rc',rc,flush=True);return s,rc
 res=list(concurrent.futures.ThreadPoolExecutor(5).map(job,SEEDS))
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[1]!=0]))
 print('ALL_COMPLETE',flush=True)
