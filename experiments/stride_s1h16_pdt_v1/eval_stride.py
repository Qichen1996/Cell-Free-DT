"""[prompt stride ablation eval] usage: eval_stride.py <exp> (exp = stride_s1h16_pdt_v1 | stride_s1h96_pdt_v1). user: 测一下不跨步采样而是连续采样; 确认.
Waits for dt_results/<exp>/PDT/complete.json. 7 scenes x seeds 1-5 x prompt {demo, self} = 70 runs, parallel, watchdog 7200 s.
demo: per seed rng=default_rng(seed), for each AP draw (episode of the scene's training data, AP, end) uniformly, segment = end-1-STRIDE*k, k<H
      (same rule as the main evals, but with this model's H/STRIDE). Fixed for the whole episode.
self: own history lags left-1-STRIDE*k; unavailable slots from the demo above.
RTG = same as main evals (b2000cl90_pdt_v1 eval_other/prep.json + eval/prep.json for B2000).
Out experiments/<exp>/eval/, sims analysis/sim_stats/<exp>/PDT_<prompt>/<scene>. trajectory.csv deleted, size recorded."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,csv,subprocess,concurrent.futures,time,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new');EXP=sys.argv[1];D=ROOT/'experiments'/EXP;OUT=D/'eval'
sys.path.insert(0,str(ROOT/'experiments/selfhist_pdt_v1'));sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('T',D/'train.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
import eval_selfhist as E
E.SPEC={s[0]:s for s in T.SCENES}
REF=ROOT/'experiments/b2000cl90_pdt_v1'
STATS=ROOT/'analysis/sim_stats'/EXP;SEEDS=[1,2,3,4,5];ENVS=['A','B','C','B4','B36','B500','B2000'];PROMPTS=['demo','self'];LIMIT=7200
MP=ROOT/f'dt_results/{EXP}/PDT/model.pt';HI=lambda end:end-1-T.STRIDE*np.arange(T.H)
def rtg(e):return json.loads((REF/('eval/prep.json' if e=='B2000' else 'eval_other/prep.json')).read_text())['rtg']['mean'][e]
def prepare():
 from trajectory_io import load_trajectory_pickle
 OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir();rec={}
 for name,files,traffic,aps,*_ in T.SCENES:
  eps=[(p,i,tr) for p in files for i,tr in enumerate(load_trajectory_pickle(p))];rec[name]={}
  for s in SEEDS:
   rng=np.random.default_rng(s);hs=[];ha=[];hr=[];src=[]
   for _ in range(aps):
    k=int(rng.integers(len(eps)));p,i,tr=eps[k];a=int(rng.integers(aps));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081));hi=HI(end)
    hs.append(tr['observations'][hi,a]);ha.append(tr['actions'][hi,a]);hr.append(tr['rewards'][hi,a]);src.append([p,i,a,end])
   np.savez(OUT/f'demo_{name}_{s}.npz',hs=np.array(hs,np.float32),ha=np.array(ha,np.float32),hr=np.array(hr,np.float32));rec[name][s]=src
  del eps;print('PREP',name,flush=True)
 E.save(OUT/'prep.json',dict(H=T.H,stride=T.STRIDE,rtg={e:rtg(e) for e in ENVS},demos=rec))
@torch.no_grad()
def evaluate(name,ptag,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 ck=torch.load(MP,map_location='cpu',weights_only=False);assert ck['updates']==270000 and ck['H']==T.H and ck['stride']==T.STRIDE
 model=T.Model(True);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=rtg(name);d=np.load(OUT/f'demo_{name}_{seed}.npz');hs0,ha0,hr0=[torch.from_numpy(d[k]) for k in ('hs','ha','hr')]
 stats=STATS/f'PDT_{ptag}'/name;env,obs,scenario,params=E.environment(name,seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}';assert hs0.shape[:2]==(n,T.H)
 hv=torch.ones(n,T.H,dtype=torch.bool);C=T.CTX;hs,ha,hr=hs0,ha0,hr0
 ts=torch.arange(10080).repeat(n,1);S=torch.zeros(n,10080,110);A=torch.zeros(n,10080,2);R=torch.zeros(n,10080,1);RW=torch.zeros(n,10080,1);running=torch.full((n,1),target)
 t0=time.time()
 for step in range(10080):
  S[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);R[:,step]=running
  left=max(0,step-C+1);right=max(C,step+1)
  if ptag=='self':
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
   nt=getattr(env,'net',None) or env.env.net;print('HANG_TIMEOUT',name,ptag,seed,'step',step,'ues',len(nt.ues),flush=True);sys.exit(3)
 env.close()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000;ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 a=A.numpy().astype(int);sw=float((a[...,0]!=1).mean());deep=float((a[...,1]==3).mean())
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 E.save(dest/'verified.json',dict(kind='PDT',exp=EXP,H=T.H,stride=T.STRIDE,prompt=ptag,seconds=time.time()-t0,env=name,seed=seed,model_sha256=T.sha(MP),rtg_target=target,power_w=pc,dropped=ds,completed=cs,
  return_ap_mean=float(target-running.mean()),ant_switch=sw,deep_sleep=deep,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE',name,ptag,seed,pc,ds,flush=True)
if __name__=='__main__':
 if len(sys.argv)>2:evaluate(sys.argv[2],sys.argv[3],int(sys.argv[4]));sys.exit()
 while not (MP.parent/'complete.json').is_file():time.sleep(300)
 print('TRAIN_DONE_START',time.ctime(),flush=True)
 prepare()
 def job(t):
  with (OUT/'logs'/('-'.join(map(str,t))+'.log')).open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,EXP,*map(str,t)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',*t,'rc',rc,flush=True);return (*t,rc)
 tasks=[(e,p,s) for p in PROMPTS for e in ENVS for s in SEEDS];res=list(concurrent.futures.ThreadPoolExecutor(70).map(job,tasks));summ={}
 for p in PROMPTS:
  for e in ENVS:
   vs=[json.loads(x.read_text()) for x in sorted((STATS/f'PDT_{p}'/e).glob('*/SEED*/verified.json'))]
   if not vs:continue
   dd=np.sum([v['dropped'] for v in vs],0);cc=np.sum([v['completed'] for v in vs],0)
   summ[f'{p}/{e}']=dict(n=len(vs),seeds=[v['seed'] for v in vs],power_w=round(float(np.mean([v['power_w'] for v in vs])),2),power_seeds=[round(v['power_w'],1) for v in vs],
    drop_pct=[round(float(100*x/max(1,x+y)),4) for x,y in zip(dd,cc)],str_seeds=[round(100*v['dropped'][0]/max(1,v['dropped'][0]+v['completed'][0]),3) for v in vs])
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[-1]!=0],summary=summ));print('ALL_COMPLETE',flush=True)
