"""Re-run B500 eval with seeds 1-5 for SHPDT / PDT / DT (user: 以后仿真都只用seed1-5; 先把500场景的重跑了)."""
import os,sys,json,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/Cell-Free-DT-new');D=ROOT/'experiments/selfhist_pdt_v1';sys.path.insert(0,str(D));sys.path.insert(0,str(ROOT))
import train_selfhist as T
import eval_selfhist as E
import eval_pdt as P
SEEDS=[1,2,3,4,5];NAME=sys.argv[1] if len(sys.argv)>1 else 'B500';OUT=D/('rerun_b500_s1to5' if NAME=='B500' else f'rerun_{NAME}_s1to5');OUT.mkdir(exist_ok=False)
STATS=ROOT/'analysis/sim_stats/selfhist_pdt_v1'
# PDT demos for B500 seeds 1-5, same rule as eval_pdt.prepare
from trajectory_io import load_trajectory_pickle
name,files,traffic,aps,*_=next(s for s in T.SCENES if s[0]==NAME);rec={};SC=['A','B','C'][traffic]
refs=[(p,i) for p in files for i in range(len(load_trajectory_pickle(p)))] if len(files)==1 else [(p,0) for p in files];cache={}
for seed in SEEDS:
 rng=np.random.default_rng(seed);hs=[];ha=[];hr=[];src=[]
 for ap in range(aps):
  p,i=refs[rng.integers(len(refs))];a2=int(rng.integers(aps));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081))
  if p not in cache:cache.clear();cache[p]=load_trajectory_pickle(p)
  tr=cache[p][i];hi=end-1-T.STRIDE*np.arange(T.H)
  hs.append(tr['observations'][hi,a2]);ha.append(tr['actions'][hi,a2]);hr.append(tr['rewards'][hi,a2]);src.append([p,i,a2,end])
 f=P.BASE/f'demo_{NAME}_{seed}.npz';assert not f.exists()
 np.savez(f,hs=np.array(hs,np.float32),ha=np.array(ha,np.float32),hr=np.array(hr,np.float32));rec[f'{NAME}/{seed}']=src
E.save(OUT/'demos.json',rec)
for k in ['SHPDT','PDT','DT']:
 for s_ in SEEDS:assert not (STATS/k/NAME/SC/f'SEED{s_}').exists()
def job(t):
 k,s=t;lf=OUT/f'{k}-{NAME}-{s}.log'
 cmd=[sys.executable,'-u',str(D/'eval_pdt.py'),NAME,str(s)] if k=='PDT' else [sys.executable,'-u',str(D/'eval_selfhist.py'),k,NAME,str(s)]
 with lf.open('x') as log:rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 print('CASE',k,s,'rc',rc,flush=True);return k,s,rc
res=list(concurrent.futures.ThreadPoolExecutor(15).map(job,[(k,s) for k in ['SHPDT','PDT','DT'] for s in SEEDS]))
summ={}
for k in ['SHPDT','PDT','DT']:
 vs=[json.loads((STATS/k/NAME/SC/f'SEED{s}'/'verified.json').read_text()) for s in SEEDS if (k,s,0) in res]
 if not vs:continue
 d=np.sum([v['dropped'] for v in vs],0);c=np.sum([v['completed'] for v in vs],0)
 summ[k]=dict(n=len(vs),power_w=float(np.mean([v['power_w'] for v in vs])),drop_pct=[float(100*a/max(1,a+b)) for a,b in zip(d,c)])
E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[2]!=0],summary=summ))
print('ALL_COMPLETE',json.dumps(summ),flush=True)
