"""[b2000cl90_pdt_v1 eval] New PDT/DT (B2000 data recollected at cluster .90) on B2000 cluster .90, seeds 1-5 (user: 要重新训练根据更新traj的pdt和dt).
RTG = mean initial AP return of the new B2000 cl90 data; PDT demos re-drawn from the new data (same rule as eval_new500 B500: rng=seed, random file/AP/end, H=16 stride 6).
PDT runs 5 in parallel, DT one at a time (user: 一个一个跑不要并行); per-run watchdog 7200 s (DT hang = UE backlog collapse, see record).
Arg 5 = prompt tag (default own) so prompt ablations reuse this script. Based on new500_cl98_pdt_v1/eval_new500.py."""
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
E.SPEC={s[0]:s for s in T.SCENES};assert E.SPEC['B2000'][5]==.9 and 'B2000cl90' in E.SPEC['B2000'][1][0]
OUT=D/'eval';STATS=ROOT/'analysis/sim_stats/b2000cl90_pdt_v1';SEEDS=[1,2,3,4,5];ENVS=['B2000'];LIMIT=7200;KINDS=['PDT','DT'];QS=['mean']
MP=lambda k:ROOT/f'dt_results/b2000cl90_pdt_v1/{k}/model.pt'
def prepare():
 from trajectory_io import load_trajectory_pickle
 OUT.mkdir(exist_ok=False);rec={};tg={q:{} for q in QS}
 for name,files,traffic,aps,*_ in T.SCENES:
  if name not in ENVS:continue
  best=None;eps=[]
  for p in files:
   for i,tr in enumerate(load_trajectory_pickle(p)):
    ret=tr['rewards'][...,0].astype(np.float64).sum(0);eps.append(float(ret.mean()));a=int(ret.argmax())
    if best is None or ret[a]>best[0]:best=(float(ret[a]),p,i,a)
  _,p,i,a=best;tr=load_trajectory_pickle(p)[i];r=tr['rewards'][:,a,0].astype(np.float64)
  ends=np.arange(T.STRIDE*(T.H-1)+1,10081);end=int(ends[int(np.argmax([r[e-1-T.STRIDE*np.arange(T.H)].sum() for e in ends]))]);hi=end-1-T.STRIDE*np.arange(T.H)
  one=lambda k:np.repeat(tr[k][hi,a][None],aps,0).astype(np.float32)
  np.savez(OUT/f'demo_{name}.npz',hs=one('observations'),ha=one('actions'),hr=one('rewards'))
  rec[name]=dict(file=p,episode=i,ap=a,ap_return=best[0],end=end,episodes=len(eps))
  tg['p25'][name]=float(np.percentile(eps,25));tg['mean'][name]=float(np.mean(eps));tg['max'][name]=float(np.max(eps))
  print('PREP',name,rec[name],{q:tg[q][name] for q in QS},flush=True)
 E.save(OUT/'prep.json',dict(prompt=rec,rtg=tg))
@torch.no_grad()
def evaluate(kind,q,name,seed,ptag='own'):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 ck=torch.load(MP(kind),map_location='cpu',weights_only=False);assert ck['updates']==270000
 model=T.Model(kind=='PDT');model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=json.loads((OUT/'prep.json').read_text())['rtg'][q][name]
 d=np.load(OUT/f'demo_{ptag}_{seed}.npz');hs,ha,hr=[torch.from_numpy(d[k]) for k in ('hs','ha','hr')]
 stats=STATS/(f'{kind}_{q}' if ptag=='own' else f'{kind}_{q}_prompt_{ptag}')/name;env,obs,scenario,params=E.environment(name,seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}'
 hv=torch.ones(n,T.H,dtype=torch.bool);C=T.CTX
 ts=torch.arange(10080).repeat(n,1);S=torch.zeros(n,10080,110);A=torch.zeros(n,10080,2);R=torch.zeros(n,10080,1);running=torch.full((n,1),target)
 t0=time.time()
 for step in range(10080):
  S[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);R[:,step]=running
  left=max(0,step-C+1);right=max(C,step+1)
  heads=model(hs,ha,hr,hv,ts[:,left:right],S[:,left:right],A[:,left:right],R[:,left:right])
  idx=step if step<C else -1
  act=torch.stack([h[:,idx].argmax(-1) for h in heads],dim=-1);A[:,step]=act
  obs,_,reward,done,_,_=env.step(act.numpy());running-=torch.as_tensor(np.asarray(reward).reshape(n,1),dtype=torch.float32)
  assert bool(done)==(step==10079)
  if time.time()-t0>LIMIT:
   nt=getattr(env,'net',None) or env.env.net;print('HANG_TIMEOUT',kind,ptag,seed,'step',step,'ues',len(nt.ues),flush=True);sys.exit(3)
 env.close()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000;ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 a=A.numpy().astype(int);sw=float((a[...,0]!=1).mean());deep=float((a[...,1]==3).mean())
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 E.save(dest/'verified.json',dict(kind=kind,prompt=ptag,seconds=time.time()-t0,rtg=q,env=name,seed=seed,model_sha256=T.sha(MP(kind)),rtg_target=target,power_w=pc,dropped=ds,completed=cs,
  return_ap_mean=float(target-running.mean()),ant_switch=sw,deep_sleep=deep,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE',kind,q,name,seed,pc,ds,flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:evaluate(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),*(sys.argv[5:6]));sys.exit()
 for k in KINDS:assert (ROOT/f'dt_results/b2000cl90_pdt_v1/{k}/complete.json').is_file()
 OUT.mkdir(exist_ok=False);from trajectory_io import load_trajectory_pickle
 name,files,traffic,aps,*_=next(x for x in T.SCENES if x[0]=='B2000');eps=[];demos={}
 for p in files:eps.append(float(load_trajectory_pickle(p)[0]['rewards'][...,0].astype(np.float64).sum(0).mean()))
 for seed in SEEDS:
  rng=np.random.default_rng(seed);hs=[];ha=[];hr=[];src=[]
  for ap in range(aps):
   p=files[rng.integers(len(files))];a2=int(rng.integers(aps));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081))
   tr=load_trajectory_pickle(p)[0];hi=end-1-T.STRIDE*np.arange(T.H)
   hs.append(tr['observations'][hi,a2]);ha.append(tr['actions'][hi,a2]);hr.append(tr['rewards'][hi,a2]);src.append([p,0,a2,end])
  np.savez(OUT/f'demo_own_{seed}.npz',hs=np.array(hs,np.float32),ha=np.array(ha,np.float32),hr=np.array(hr,np.float32));demos[str(seed)]=src
 E.save(OUT/'prep.json',dict(rtg=dict(mean={'B2000':float(np.mean(eps))}),source='B2000 cl90 new data mean of %d eps (old .95 was -1445.6)'%len(eps),own_demos=demos));(OUT/'logs').mkdir()
 print('PREP',json.loads((OUT/'prep.json').read_text())['rtg'],flush=True)
 def job(t):
  with (OUT/'logs'/('-'.join(map(str,t))+'.log')).open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,*map(str,t)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',*t,'rc',rc,flush=True);return (*t,rc)
 tasks=[(k,q,e,s) for k in KINDS for q in QS for e in ENVS for s in SEEDS]
 res=list(concurrent.futures.ThreadPoolExecutor(5).map(job,[t for t in tasks if t[0]=='PDT']))+[job(t) for t in tasks if t[0]=='DT'];summ={}
 for k in KINDS:
  for q in QS:
   for e in ENVS:
    vs=[json.loads((STATS/f'{k}_{q}'/e/'B'/f'SEED{s}'/'verified.json').read_text()) for s in SEEDS if (k,q,e,s,0) in res]
    if not vs:continue
    d=np.sum([v['dropped'] for v in vs],0);c=np.sum([v['completed'] for v in vs],0)
    summ[f'{k}/{q}/{e}']=dict(n=len(vs),power_w=round(float(np.mean([v['power_w'] for v in vs])),2),ret=round(float(np.mean([v['return_ap_mean'] for v in vs])),2),
     drop_pct=[round(float(100*x/max(1,x+y)),4) for x,y in zip(d,c)],str=f'{int(d[0])}/{int(d[0]+c[0])}',sw=round(float(np.mean([v['ant_switch'] for v in vs])),3),deep=round(float(np.mean([v['deep_sleep'] for v in vs])),3))
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[-1]!=0],summary=summ))
 print('ALL_COMPLETE',flush=True)
