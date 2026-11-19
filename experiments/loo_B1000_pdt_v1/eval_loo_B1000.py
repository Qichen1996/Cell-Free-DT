"""[loo_B1000_pdt_v1 eval] Models trained WITHOUT scene B (B1000, 16 AP, traffic B, cl .9); evaluated on B, seeds 1-5 (user: 试一下1).
Configs (kind, prompt, rtg):
  PDT demo  true : B teacher demos selfhist_pdt_v1/eval_pdt/demo_B_<s> (same as main eval), RTG = B training mean -415.70 (optimistic: uses B data)
  PDT demo  est  : same demos, RTG estimated from the prompt only = mean(demo rewards) * 10080
  PDT self  true / est : own-history prompt (fallback = same demo), RTG true / est
  PDT b500  true : neighbour-scene prompt (B500 demos new500_cl98_pdt_v1/eval/demo_B500_<s>), RTG true
  DT  none  true / est
Waits for both dt_results/loo_B1000_pdt_v1/{PDT,DT}/complete.json. 35 runs parallel, watchdog 7200 s.
Out experiments/loo_B1000_pdt_v1/eval/, sims analysis/sim_stats/loo_B1000_pdt_v1/<kind>_<prompt>_<rtg>/. trajectory.csv deleted, size recorded."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,csv,subprocess,concurrent.futures,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/Cell-Free-DT-new');D=ROOT/'experiments/loo_B1000_pdt_v1';OUT=D/'eval'
sys.path.insert(0,str(ROOT/'experiments/b2000cl90_pdt_v1'));sys.path.insert(0,str(ROOT/'experiments/selfhist_pdt_v1'));sys.path.insert(0,str(ROOT))
import train_b2000cl90 as T   # architecture identical to the LOO script; SCENES used only for env spec of B
import eval_selfhist as E
E.SPEC={s[0]:s for s in T.SCENES};assert E.SPEC['B'][5]==.9 and E.SPEC['B'][4]==1000 and E.SPEC['B'][3]==16
STATS=ROOT/'analysis/sim_stats/loo_B1000_pdt_v1';SEEDS=[1,2,3,4,5];LIMIT=7200;RTG_TRUE=-415.7040981077988
CFG=[('PDT','demo','true'),('PDT','demo','est'),('PDT','self','true'),('PDT','self','est'),('PDT','b500','true'),('DT','none','true'),('DT','none','est')]
MP=lambda k:ROOT/f'dt_results/loo_B1000_pdt_v1/{k}/model.pt';HI=lambda end:end-1-T.STRIDE*np.arange(T.H)
DEMO=lambda s:ROOT/f'experiments/selfhist_pdt_v1/eval_pdt/demo_B_{s}.npz'
B500=lambda s:ROOT/f'experiments/new500_cl98_pdt_v1/eval/demo_B500_{s}.npz'
def rtg_est(s):return float(np.load(DEMO(s))['hr'].astype(np.float64).mean()*10080)
@torch.no_grad()
def evaluate(kind,ptag,rtag,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);name='B'
 ck=torch.load(MP(kind),map_location='cpu',weights_only=False);assert ck['updates']==270000
 model=T.Model(kind=='PDT');model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=RTG_TRUE if rtag=='true' else rtg_est(seed)
 d=np.load(B500(seed) if ptag=='b500' else DEMO(seed));hs0,ha0,hr0=[torch.from_numpy(d[k]) for k in ('hs','ha','hr')]
 tag=f'{kind}_{ptag}_{rtag}';stats=STATS/tag;env,obs,scenario,params=E.environment(name,seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}'
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
   nt=getattr(env,'net',None) or env.env.net;print('HANG_TIMEOUT',tag,seed,'step',step,'ues',len(nt.ues),flush=True);sys.exit(3)
 env.close()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000;ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 a=A.numpy().astype(int);sw=float((a[...,0]!=1).mean());deep=float((a[...,1]==3).mean())
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 E.save(dest/'verified.json',dict(kind=kind,prompt=ptag,rtg=rtag,seconds=time.time()-t0,env=name,seed=seed,model_sha256=T.sha(MP(kind)),rtg_target=target,power_w=pc,dropped=ds,completed=cs,
  return_ap_mean=float(target-running.mean()),ant_switch=sw,deep_sleep=deep,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE',tag,seed,pc,ds,flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:evaluate(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]));sys.exit()
 while not all((MP(k).parent/'complete.json').is_file() for k in ('PDT','DT')):time.sleep(300)
 print('TRAIN_DONE_START',time.ctime(),flush=True)
 OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir()
 for s in SEEDS:assert DEMO(s).is_file() and B500(s).is_file()
 E.save(OUT/'prep.json',dict(rtg_true=RTG_TRUE,rtg_est={s:rtg_est(s) for s in SEEDS},configs=CFG));print('RTG_EST',{s:rtg_est(s) for s in SEEDS},flush=True)
 def job(t):
  with (OUT/'logs'/('-'.join(map(str,t))+'.log')).open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,*map(str,t)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',*t,'rc',rc,flush=True);return (*t,rc)
 tasks=[(*c,s) for c in CFG for s in SEEDS];res=list(concurrent.futures.ThreadPoolExecutor(35).map(job,tasks));summ={}
 for c in CFG:
  tag='_'.join(c);vs=[json.loads(p.read_text()) for p in sorted((STATS/tag).glob('*/SEED*/verified.json'))]
  if not vs:continue
  dd=np.sum([v['dropped'] for v in vs],0);cc=np.sum([v['completed'] for v in vs],0)
  summ[tag]=dict(n=len(vs),seeds=[v['seed'] for v in vs],power_w=round(float(np.mean([v['power_w'] for v in vs])),2),power_seeds=[round(v['power_w'],1) for v in vs],
   drop_pct=[round(float(100*x/max(1,x+y)),4) for x,y in zip(dd,cc)],str_seeds=[round(100*v['dropped'][0]/max(1,v['dropped'][0]+v['completed'][0]),3) for v in vs])
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[-1]!=0],summary=summ));print('ALL_COMPLETE',flush=True)
