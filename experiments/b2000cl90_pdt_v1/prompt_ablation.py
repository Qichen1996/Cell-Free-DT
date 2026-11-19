"""[b2000cl90_pdt_v1 prompt ablation] (user: 跑完之后验证prompt). New PDT on B2000 cluster .90, seeds 1-5, RTG -1580.8 (same as main eval).
Prompt variants (each 16 APs x H=16 transitions, stride 6, same shape as the main 'own' demos):
  own       : main eval demos (already run, reused for comparison)
  own_d2/d3 : same rule, different draw (rng = seed+1000 / seed+2000) -> prompt-draw variance
  wrongA    : drawn from scene A training data (traffic A, 1000 m, cl .98)
  wrongB500 : drawn from scene B500 training data (500 m, cl .98)
  zero      : all-zero s/a/r prompt
  shuffle   : own demos with the 16 history steps randomly permuted per AP (breaks temporal order)
  hiq / loq : own B2000 cl90 data, segments from top / bottom 10% of 4000 random candidates by segment reward sum
Waits for the DT 6-scene eval (eval_other_DT.log ALL_COMPLETE) before starting. 40 PDT runs, 40 parallel, watchdog in eval_b2000cl90.py.
Out: eval/demo_<tag>_<seed>.npz, eval/ablation_prep.json, sims analysis/sim_stats/b2000cl90_pdt_v1/PDT_mean_prompt_<tag>, eval/ablation_complete.json."""
import os,sys,json,time,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/Cell-Free-DT-new');D=ROOT/'experiments/b2000cl90_pdt_v1';OUT=D/'eval'
sys.path.insert(0,str(D));sys.path.insert(0,str(ROOT))
import train_b2000cl90 as T
from trajectory_io import load_trajectory_pickle
SEEDS=[1,2,3,4,5];TAGS=['own_d2','own_d3','wrongA','wrongB500','zero','shuffle','hiq','loq'];STATS=ROOT/'analysis/sim_stats/b2000cl90_pdt_v1'
SC={s[0]:s for s in T.SCENES};HI=lambda end:end-1-T.STRIDE*np.arange(T.H)
_cache={}
def tr(p):
 if p not in _cache:_cache[p]=load_trajectory_pickle(p)[0]
 return _cache[p]
def seg(p,a,end):
 t=tr(p);hi=HI(end);return t['observations'][hi,a],t['actions'][hi,a],t['rewards'][hi,a],[p,0,int(a),int(end)]
def draw(env,rng,aps=16):
 files=SC[env][1];n=SC[env][3];out=[]
 for _ in range(aps):
  p=files[rng.integers(len(files))];a=int(rng.integers(n));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081));out.append(seg(p,a,end))
 return out
def save(tag,seed,segs,extra=None):
 hs,ha,hr=[np.array([s[i] for s in segs],np.float32) for i in range(3)]
 np.savez(OUT/f'demo_{tag}_{seed}.npz',hs=hs,ha=ha,hr=hr);return [s[3] for s in segs]
def prepare():
 assert not (OUT/'ablation_prep.json').exists()
 rec={}
 files=SC['B2000'][1];rng=np.random.default_rng(12345);cand=[]
 for _ in range(4000):
  p=files[rng.integers(len(files))];a=int(rng.integers(16));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081))
  cand.append((float(tr(p)['rewards'][HI(end),a].astype(np.float64).sum()),p,a,end))
 cand.sort(key=lambda x:x[0]);k=len(cand)//10;lo,hi=cand[:k],cand[-k:]
 rec['quality']=dict(candidates=len(cand),lo_max=lo[-1][0],hi_min=hi[0][0],median=cand[len(cand)//2][0])
 for seed in SEEDS:
  r=rec[str(seed)]={}
  r['own_d2']=save('own_d2',seed,draw('B2000',np.random.default_rng(seed+1000)))
  r['own_d3']=save('own_d3',seed,draw('B2000',np.random.default_rng(seed+2000)))
  r['wrongA']=save('wrongA',seed,draw('A',np.random.default_rng(seed)))
  r['wrongB500']=save('wrongB500',seed,draw('B500',np.random.default_rng(seed)))
  d=np.load(OUT/f'demo_own_{seed}.npz');np.savez(OUT/f'demo_zero_{seed}.npz',**{x:np.zeros_like(d[x]) for x in ('hs','ha','hr')});r['zero']='zeros'
  g=np.random.default_rng(seed+3000);perm=np.stack([g.permutation(T.H) for _ in range(d['hs'].shape[0])])
  np.savez(OUT/f'demo_shuffle_{seed}.npz',**{x:np.take_along_axis(d[x],perm[...,None],1) for x in ('hs','ha','hr')});r['shuffle']=perm.tolist()
  g=np.random.default_rng(seed+4000)
  for tag,pool in (('hiq',hi),('loq',lo)):
   pick=[pool[i] for i in g.choice(len(pool),16,replace=False)];r[tag]=save(tag,seed,[seg(p,a,e) for _,p,a,e in pick])
 (OUT/'ablation_prep.json').write_text(json.dumps(rec,indent=1));print('ABL_PREP',rec['quality'],flush=True)
def main():
 while 'ALL_COMPLETE' not in (D/'eval_other_DT.log').read_text():time.sleep(300)
 print('DT_EVAL_DONE_START',time.ctime(),flush=True)
 prepare();(OUT/'logs_ablation').mkdir()
 def job(t):
  with (OUT/'logs_ablation'/('-'.join(map(str,t))+'.log')).open('x') as log:
   rc=subprocess.call([sys.executable,'-u',str(D/'eval_b2000cl90.py'),'PDT','mean','B2000',str(t[1]),t[0]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',*t,'rc',rc,flush=True);return (*t,rc)
 tasks=[(g,s) for g in TAGS for s in SEEDS]
 res=list(concurrent.futures.ThreadPoolExecutor(40).map(job,tasks));summ={}
 for g in ['own']+TAGS:
  base=STATS/('PDT_mean' if g=='own' else f'PDT_mean_prompt_{g}')/'B2000'/'B'
  vs=[json.loads((base/f'SEED{s}'/'verified.json').read_text()) for s in SEEDS if (base/f'SEED{s}'/'verified.json').is_file()]
  if not vs:continue
  d=np.sum([v['dropped'] for v in vs],0);c=np.sum([v['completed'] for v in vs],0)
  summ[g]=dict(n=len(vs),power_w=round(float(np.mean([v['power_w'] for v in vs])),2),power_seeds=[round(v['power_w'],1) for v in vs],
   drop_pct=[round(float(100*x/max(1,x+y)),4) for x,y in zip(d,c)],str_seeds=[round(100*v['dropped'][0]/max(1,v['dropped'][0]+v['completed'][0]),3) for v in vs])
 (OUT/'ablation_complete.json').write_text(json.dumps(dict(runs=len(res),failed=[r for r in res if r[-1]!=0],summary=summ),indent=1))
 print('ABLATION_COMPLETE',flush=True)
if __name__=='__main__':main()
