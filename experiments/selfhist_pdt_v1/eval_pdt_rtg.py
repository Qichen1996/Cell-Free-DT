"""PDT eval with a higher RTG target: per-scene 75th percentile (instead of mean) of training-episode
initial AP returns. Same model, same seed 1-5 demos (eval_pdt/demo_*), only RTG changes."""
import os,sys,json,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/Cell-Free-DT-new');D=ROOT/'experiments/selfhist_pdt_v1';sys.path.insert(0,str(D));sys.path.insert(0,str(ROOT))
import train_selfhist as T
import eval_selfhist as E
import eval_pdt as P
Q=int(os.environ.get('RTG_Q','75'));ONLY=os.environ.get('RTG_ENVS');TAG=f'rtgp{Q}'+(f'_{ONLY}' if ONLY else '');OUT=D/f'eval_pdt_{TAG}';SEEDS=[1,2,3,4,5]
P.STATS=ROOT/f'analysis/sim_stats/selfhist_pdt_v1_{TAG}';E.BASE=OUT
def prepare():
 from trajectory_io import load_trajectory_pickle
 OUT.mkdir(exist_ok=False);t={};mean={}
 for name,files,*_ in T.SCENES:
  v=[float(tr['rewards'].sum(axis=0,dtype=np.float64).mean()) for p in files for tr in load_trajectory_pickle(p)]
  t[name]=float(np.percentile(v,Q));mean[name]=float(np.mean(v));print('RTG',name,mean[name],'->',t[name],flush=True)
 E.save(OUT/'rtg.json',dict(targets=t,mean=mean,method=f'{Q}th percentile of training-episode mean initial AP return',scale=1))
if __name__=='__main__':
 if len(sys.argv)>1:P.evaluate(sys.argv[1],int(sys.argv[2]));sys.exit()
 prepare()
 for e in E.SPEC:
  if ONLY and e not in ONLY.split(','):continue
  for s in SEEDS:assert (P.BASE/f'demo_{e}_{s}.npz').is_file(),(e,s)
 def job(t):
  e,s=t;lf=OUT/f'PDT-{e}-{s}.log'
  with lf.open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,e,str(s)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',e,s,'rc',rc,flush=True);return e,s,rc
 res=list(concurrent.futures.ThreadPoolExecutor(35).map(job,[(e,s) for s in SEEDS for e in E.SPEC if not ONLY or e in ONLY.split(',')]))
 summ={}
 for e in E.SPEC:
  vs=[json.loads((P.STATS/'PDT'/e/['A','B','C'][E.SPEC[e][2]]/f'SEED{s}'/'verified.json').read_text()) for s in SEEDS if (e,s,0) in res]
  if not vs:continue
  d=np.sum([v['dropped'] for v in vs],0);c=np.sum([v['completed'] for v in vs],0)
  summ[e]=dict(n=len(vs),power_w=float(np.mean([v['power_w'] for v in vs])),drop_pct=[float(100*a/max(1,a+b)) for a,b in zip(d,c)],str_dropped=int(d[0]),str_total=int(d[0]+c[0]))
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[2]!=0],summary=summ))
 print('ALL_COMPLETE',json.dumps(summ),flush=True)
