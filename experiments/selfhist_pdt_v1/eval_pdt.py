"""Eval Prompt-DT (fixed expert-demo prompt from the scene's training data) on 7 scenes, seeds 101-105.
Per seed/AP: one demo (random episode, AP, end step; rng=seed) held fixed for the whole episode.
RTG targets reused from eval/rtg.json. trajectory.csv deleted after verification (recorded)."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,csv,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(ROOT/'experiments/selfhist_pdt_v1'));sys.path.insert(0,str(ROOT))
import train_selfhist as T
import eval_selfhist as E
BASE=ROOT/'experiments/selfhist_pdt_v1/eval_pdt';STATS=ROOT/'analysis/sim_stats/selfhist_pdt_v1'
MP=ROOT/'dt_results/selfhist_pdt_v1/PDT/model.pt'
def prepare():
 from trajectory_io import load_trajectory_pickle
 BASE.mkdir(parents=True,exist_ok=True);rec={}
 for name,files,traffic,aps,*_ in T.SCENES:
  refs=[(p,i) for p in files for i in range(len(load_trajectory_pickle(p)))] if len(files)==1 else [(p,0) for p in files]
  for seed in E.SEEDS:
   rng=np.random.default_rng(seed);hs=[];ha=[];hr=[];src=[]
   cache={}
   for ap in range(aps):
    p,i=refs[rng.integers(len(refs))];a2=int(rng.integers(aps));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081))
    if p not in cache:cache.clear();cache[p]=load_trajectory_pickle(p)
    tr=cache[p][i];hi=end-1-T.STRIDE*np.arange(T.H)
    hs.append(tr['observations'][hi,a2]);ha.append(tr['actions'][hi,a2]);hr.append(tr['rewards'][hi,a2]);src.append([p,i,a2,end])
   np.savez(BASE/f'demo_{name}_{seed}.npz',hs=np.array(hs,np.float32),ha=np.array(ha,np.float32),hr=np.array(hr,np.float32))
   rec[f'{name}/{seed}']=src
  print('DEMO_READY',name,flush=True)
 E.save(BASE/'demos.json',rec)
@torch.no_grad()
def evaluate(name,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 ck=torch.load(MP,map_location='cpu',weights_only=False);assert ck['updates']==270000
 model=T.Model(True);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=json.loads((E.BASE/'rtg.json').read_text())['targets'][name]
 d=np.load(BASE/f'demo_{name}_{seed}.npz');hs,ha,hr=[torch.from_numpy(d[k]) for k in ('hs','ha','hr')]
 stats=STATS/'PDT'/name;env,obs,scenario,params=E.environment(name,seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}'
 assert hs.shape==(n,T.H,110);hv=torch.ones(n,T.H,dtype=torch.bool);C=T.CTX
 ts=torch.arange(10080).repeat(n,1);S=torch.zeros(n,10080,110);A=torch.zeros(n,10080,2);R=torch.zeros(n,10080,1)
 running=torch.full((n,1),target)
 for step in range(10080):
  S[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);R[:,step]=running
  left=max(0,step-C+1);right=max(C,step+1)
  heads=model(hs,ha,hr,hv,ts[:,left:right],S[:,left:right],A[:,left:right],R[:,left:right])
  idx=step if step<C else -1
  act=torch.stack([h[:,idx].argmax(-1) for h in heads],dim=-1);A[:,step]=act
  obs,_,reward,done,_,_=env.step(act.numpy());running-=torch.as_tensor(np.asarray(reward).reshape(n,1),dtype=torch.float32)
  assert bool(done)==(step==10079)
 env.close()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000;ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 drop=[100*a/max(1,a+b) for a,b in zip(ds,cs)]
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 E.save(dest/'verified.json',dict(kind='PDT',env=name,seed=seed,steps=10080,model_sha256=T.sha(MP),rtg_target=target,power_w=pc,drop_pct=drop,dropped=ds,completed=cs,environment=params,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE PDT',name,seed,pc,drop,flush=True)
def main():
 if len(sys.argv)>1:evaluate(sys.argv[1],int(sys.argv[2]));return
 assert (ROOT/'dt_results/selfhist_pdt_v1/PDT/complete.json').is_file() and not (STATS/'PDT').exists() and (E.BASE/'rtg.json').is_file()
 prepare()
 def job(t):
  name,seed=t;lf=BASE/'logs'/f'PDT-{name}-{seed}.log';lf.parent.mkdir(exist_ok=True)
  with lf.open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,name,str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE PDT',name,seed,'rc',rc,flush=True);return (name,seed,rc)
 tasks=[(e,s) for s in E.SEEDS for e in E.SPEC]
 with concurrent.futures.ThreadPoolExecutor(max_workers=35) as pool:res=list(pool.map(job,tasks))
 summ={}
 for e in E.SPEC:
  vs=[json.loads((STATS/'PDT'/e/['A','B','C'][E.SPEC[e][2]]/f'SEED{s}'/'verified.json').read_text()) for s in E.SEEDS if (e,s,0) in res]
  if not vs:continue
  ds=np.sum([v['dropped'] for v in vs],0);cs=np.sum([v['completed'] for v in vs],0)
  summ[f'PDT/{e}']=dict(n=len(vs),power_w=float(np.mean([v['power_w'] for v in vs])),drop_pct=[float(100*a/max(1,a+b)) for a,b in zip(ds,cs)])
 E.save(BASE/'complete.json',dict(runs=len(res),failed=[r for r in res if r[2]!=0],summary=summ))
 print('ALL_COMPLETE',json.dumps(summ),flush=True)
if __name__=='__main__':main()
