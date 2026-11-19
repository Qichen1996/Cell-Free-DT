"""Paper-style PDT eval (Prompt-DT for wireless): top-1 prompt + max RTG. No retraining.
Prompt: for each scene, the (episode, AP) with the highest AP return in training data; the window
(16 tokens, stride 6) with the highest summed reward. Same prompt for all APs and seeds.
Variants: top1_mean (top-1 prompt, RTG = scene mean) and top1_max (top-1 prompt, RTG = max episode mean AP return).
Scenes B500 and B (B1000), seeds 1-5."""
import os,sys,json,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/Cell-Free-DT-new');D=ROOT/'experiments/selfhist_pdt_v1';sys.path.insert(0,str(D));sys.path.insert(0,str(ROOT))
import train_selfhist as T
import eval_selfhist as E
import eval_pdt as P
OUT=D/'eval_pdt_top1';SEEDS=[1,2,3,4,5];ENVS=['B500','B'];VARS=['top1_mean','top1_max']
def setup(var):
 P.BASE=OUT/'demos';E.BASE=OUT/var;P.STATS=ROOT/f'analysis/sim_stats/selfhist_pdt_v1_{var}'
def prepare():
 from trajectory_io import load_trajectory_pickle
 OUT.mkdir(exist_ok=False);(OUT/'demos').mkdir();rec={};tm={};tx={};mean={}
 for name,files,traffic,aps,*_ in T.SCENES:
  if name not in ENVS:continue
  best=None;eps=[]
  for p in files:
   for i,tr in enumerate(load_trajectory_pickle(p)):
    r=tr['rewards'].astype(np.float64);ret=r.sum(0);eps.append(float(ret.mean()))
    a=int(ret.argmax())
    if best is None or ret[a]>best[0]:best=(float(ret[a]),p,i,a)
  _,p,i,a=best;tr=load_trajectory_pickle(p)[i];r=tr['rewards'][:,a].astype(np.float64)
  ends=np.arange(T.STRIDE*(T.H-1)+1,10081);sc=[r[e-1-T.STRIDE*np.arange(T.H)].sum() for e in ends];end=int(ends[int(np.argmax(sc))])
  hi=end-1-T.STRIDE*np.arange(T.H)
  one=lambda k:np.repeat(tr[k][hi,a][None],aps,0).astype(np.float32)
  for s in SEEDS:np.savez(OUT/'demos'/f'demo_{name}_{s}.npz',hs=one('observations'),ha=one('actions'),hr=one('rewards'))
  rec[name]=dict(file=p,episode=i,ap=a,ap_return=best[0],end=end)
  mean[name]=float(np.mean(eps));tm[name]=mean[name];tx[name]=float(np.max(eps))
  print('PROMPT',name,rec[name],'RTG mean',tm[name],'max',tx[name],flush=True)
 E.save(OUT/'demos.json',rec)
 for v,t in [('top1_mean',tm),('top1_max',tx)]:
  (OUT/v).mkdir();E.save(OUT/v/'rtg.json',dict(targets=t,mean=mean,method=v))
if __name__=='__main__':
 if len(sys.argv)>1:setup(sys.argv[1]);P.evaluate(sys.argv[2],int(sys.argv[3]));sys.exit()
 prepare()
 def job(t):
  v,e,s=t;lf=OUT/v/f'PDT-{e}-{s}.log'
  with lf.open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,v,e,str(s)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',v,e,s,'rc',rc,flush=True);return v,e,s,rc
 res=list(concurrent.futures.ThreadPoolExecutor(20).map(job,[(v,e,s) for v in VARS for e in ENVS for s in SEEDS]))
 summ={}
 for v in VARS:
  setup(v)
  for e in ENVS:
   vs=[json.loads((P.STATS/'PDT'/e/'B'/f'SEED{s}'/'verified.json').read_text()) for s in SEEDS if (v,e,s,0) in res]
   if not vs:continue
   d=np.sum([x['dropped'] for x in vs],0);c=np.sum([x['completed'] for x in vs],0)
   summ[f'{v}/{e}']=dict(n=len(vs),power_w=float(np.mean([x['power_w'] for x in vs])),drop_pct=[float(100*a/max(1,a+b)) for a,b in zip(d,c)],str=f'{int(d[0])}/{int(d[0]+c[0])}')
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[3]!=0],summary=summ))
 print('ALL_COMPLETE',json.dumps(summ),flush=True)
