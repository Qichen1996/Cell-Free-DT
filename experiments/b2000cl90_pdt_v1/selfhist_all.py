"""[b2000cl90_pdt_v1 selfhist all scenes] (user: 用自己历史跑一下所有场景). New PDT, prompt = own history in this episode (lags left-1-6k, k=0..15, as prompt_sources_B.py),
slots without history filled from the scene's main-eval demo. Scenes A,C,B4,B36,B500,B2000 (B already done in eval_B_src: 272.10 W). Seeds 1-5, RTG = same as main evals
(eval_other/prep.json; B2000 eval/prep.json -1580.84). Fallback demos: selfhist_pdt_v1/eval_pdt/demo_{e}_{s}; B500 new500_cl98_pdt_v1/eval/demo_B500_{s}; B2000 eval/demo_own_{s}.
30 runs parallel, watchdog 7200 s. Out eval_selfhist_all/, sims analysis/sim_stats/b2000cl90_pdt_v1_selfhist_all. trajectory.csv deleted, size recorded."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,csv,subprocess,concurrent.futures,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new');D=ROOT/'experiments/b2000cl90_pdt_v1'
sys.path.insert(0,str(D));sys.path.insert(0,str(ROOT/'experiments/selfhist_pdt_v1'));sys.path.insert(0,str(ROOT))
import train_b2000cl90 as T
import eval_selfhist as E
E.SPEC={s[0]:s for s in T.SCENES}
OUT=D/'eval_selfhist_all';STATS=ROOT/'analysis/sim_stats/b2000cl90_pdt_v1_selfhist_all';SEEDS=[1,2,3,4,5];ENVS=['A','C','B4','B36','B500','B2000'];LIMIT=7200
MP=ROOT/'dt_results/b2000cl90_pdt_v1/PDT/model.pt';HI=lambda end:end-1-T.STRIDE*np.arange(T.H)
def rtg(e):return json.loads((D/('eval/prep.json' if e=='B2000' else 'eval_other/prep.json')).read_text())['rtg']['mean'][e]
def demo(e,s):
 if e=='B500':return ROOT/f'experiments/new500_cl98_pdt_v1/eval/demo_B500_{s}.npz'
 if e=='B2000':return D/f'eval/demo_own_{s}.npz'
 return ROOT/f'experiments/selfhist_pdt_v1/eval_pdt/demo_{e}_{s}.npz'
@torch.no_grad()
def evaluate(name,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 ck=torch.load(MP,map_location='cpu',weights_only=False);assert ck['updates']==270000
 model=T.Model(True);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=rtg(name);d=np.load(demo(name,seed));hs0,ha0,hr0=[torch.from_numpy(d[k]) for k in ('hs','ha','hr')]
 stats=STATS/name;env,obs,scenario,params=E.environment(name,seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}';assert hs0.shape[0]==n
 hv=torch.ones(n,T.H,dtype=torch.bool);C=T.CTX;hs,ha,hr=hs0,ha0,hr0
 ts=torch.arange(10080).repeat(n,1);S=torch.zeros(n,10080,110);A=torch.zeros(n,10080,2);R=torch.zeros(n,10080,1);RW=torch.zeros(n,10080,1);running=torch.full((n,1),target)
 t0=time.time()
 for step in range(10080):
  S[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);R[:,step]=running
  left=max(0,step-C+1);right=max(C,step+1)
  hi=HI(left);m=np.where(hi>=0)[0]
  if len(m):
   hs,ha,hr=hs0.clone(),ha0.clone(),hr0.clone();ix=torch.as_tensor(hi[m]);jm=torch.as_tensor(m)
   hs[:,jm]=S[:,ix];ha[:,jm]=A[:,ix];hr[:,jm]=RW[:,ix]
  heads=model(hs,ha,hr,hv,ts[:,left:right],S[:,left:right],A[:,left:right],R[:,left:right])
  idx=step if step<C else -1
  act=torch.stack([h[:,idx].argmax(-1) for h in heads],dim=-1);A[:,step]=act
  obs,_,reward,done,_,_=env.step(act.numpy());rw=torch.as_tensor(np.asarray(reward).reshape(n,1),dtype=torch.float32);RW[:,step]=rw;running-=rw
  assert bool(done)==(step==10079)
  if time.time()-t0>LIMIT:
   nt=getattr(env,'net',None) or env.env.net;print('HANG_TIMEOUT',name,seed,'step',step,'ues',len(nt.ues),flush=True);sys.exit(3)
 env.close()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000;ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 a=A.numpy().astype(int);sw=float((a[...,0]!=1).mean());deep=float((a[...,1]==3).mean())
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 E.save(dest/'verified.json',dict(kind='PDT',prompt='selfhist',seconds=time.time()-t0,env=name,seed=seed,model_sha256=T.sha(MP),rtg_target=target,power_w=pc,dropped=ds,completed=cs,
  return_ap_mean=float(target-running.mean()),ant_switch=sw,deep_sleep=deep,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE',name,seed,pc,ds,flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:evaluate(sys.argv[1],int(sys.argv[2]));sys.exit()
 OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir()
 for e in ENVS:
  for s in SEEDS:assert demo(e,s).is_file(),demo(e,s)
 print('RTG',{e:rtg(e) for e in ENVS},flush=True)
 def job(t):
  with (OUT/'logs'/f'{t[0]}-{t[1]}.log').open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,t[0],str(t[1])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',*t,'rc',rc,flush=True);return (*t,rc)
 tasks=[(e,s) for e in ENVS for s in SEEDS];res=list(concurrent.futures.ThreadPoolExecutor(30).map(job,tasks));summ={}
 for e in ENVS:
  vs=[json.loads(p.read_text()) for p in sorted((STATS/e).glob('*/SEED*/verified.json'))]
  if not vs:continue
  dd=np.sum([v['dropped'] for v in vs],0);c=np.sum([v['completed'] for v in vs],0)
  summ[e]=dict(n=len(vs),power_w=round(float(np.mean([v['power_w'] for v in vs])),2),power_seeds=[round(v['power_w'],1) for v in vs],drop_pct=[round(float(100*x/max(1,x+y)),4) for x,y in zip(dd,c)],
   str_seeds=[round(100*v['dropped'][0]/max(1,v['dropped'][0]+v['completed'][0]),3) for v in vs])
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[-1]!=0],summary=summ));print('ALL_COMPLETE',flush=True)
