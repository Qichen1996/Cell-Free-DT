"""[b2000cl90_pdt_v1 prompt sources, basic scene B1000] (user: 试一下来源不同的; 都只在basic 场景试)
New PDT (dt_results/b2000cl90_pdt_v1/PDT) on B (1000 m, 16 AP, traffic B, cluster .9), seeds 1-5, RTG -415.70 (selfhist rtg.json, same as main eval).
Baseline 'own' = main eval_other result (teacher pc2.1 demos selfhist_pdt_v1/eval_pdt/demo_B_<seed>.npz): 272.17 W, str .485% (not rerun).
Prompt sources (16 APs x H=16 transitions, stride 6; draw rule rng=seed, random file/AP/end like the main eval):
  ppo_pc12 : another PPO expert, PPO_front_B_seed1to12_v1/pc1.2_trainseed2_cp180 (12 eps; 290.5 W in its collection)
  ppo_pc36 : another PPO expert, PPO_front_B_seed1to12_v1/pc3.6_trainseed3_cp200 (12 eps; 272.1 W)
  allon    : rule baseline, all APs fixed action (antenna keep=1, sleep 0) on B1000 cl .9, episodes seed 101,102 collected here
  selfhist : the model's own history in this episode (lags 1,7,...,91 before the context window, as SHPDT training);
             slots with no history yet (first ~100 steps) filled from the own demo
PDT 20 runs in parallel, watchdog 7200 s. Out eval_B_src/, sim_stats/b2000cl90_pdt_v1_Bsrc. trajectory.csv deleted with size recorded."""
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
E.SPEC={s[0]:s for s in T.SCENES};assert E.SPEC['B'][5]==.9 and E.SPEC['B'][4]==1000 and E.SPEC['B'][3]==16
OUT=D/'eval_B_src';STATS=ROOT/'analysis/sim_stats/b2000cl90_pdt_v1_Bsrc';SEEDS=[1,2,3,4,5];TAGS=['ppo_pc12','ppo_pc36','allon','selfhist'];LIMIT=7200
MP=lambda k:ROOT/f'dt_results/b2000cl90_pdt_v1/{k}/model.pt'
TRJ=ROOT/'analysis/trajectory/PPO_front_B_seed1to12_v1';HI=lambda end:end-1-T.STRIDE*np.arange(T.H)
def collect_allon(seed):
 stats=STATS/'allon_collect';env,obs,scenario,params=E.environment('B',seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}'
 O=np.zeros((10080,n,110),np.float32);Ac=np.zeros((10080,n,2),np.float32);Rw=np.zeros((10080,n,1),np.float32);act=np.tile(np.array([1,0]),(n,1))
 for step in range(10080):
  O[step]=np.asarray(obs);Ac[step]=act;obs,_,reward,done,_,_=env.step(act);Rw[step]=np.asarray(reward).reshape(n,1)
 env.close()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000;ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 t=dest/'trajectory.csv';tsz=t.stat().st_size if t.exists() else 0
 if t.exists():t.unlink()
 np.savez(OUT/f'allon_traj_{seed}.npz',observations=O,actions=Ac,rewards=Rw)
 r=dict(seed=seed,power_w=pc,dropped=ds,completed=cs,return_ap_mean=float(Rw[...,0].sum(0).mean()),trajectory_csv_deleted_bytes=tsz);E.save(dest/'verified.json',r);print('ALLON',r,flush=True);return r
def draw(trajs,seed,n=16):
 rng=np.random.default_rng(seed);hs=[];ha=[];hr=[];src=[]
 for _ in range(n):
  k=int(rng.integers(len(trajs)));name,t=trajs[k];a=int(rng.integers(t['observations'].shape[1]));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081));hi=HI(end)
  hs.append(t['observations'][hi,a]);ha.append(t['actions'][hi,a]);hr.append(t['rewards'][hi,a]);src.append([name,a,end])
 return [np.array(x,np.float32) for x in (hs,ha,hr)],src
def prepare():
 from trajectory_io import load_trajectory_pickle
 OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir()
 allon=list(concurrent.futures.ThreadPoolExecutor(2).map(collect_allon,[101,102]))
 pools={'allon':[(f'allon_traj_{s}',dict(np.load(OUT/f'allon_traj_{s}.npz'))) for s in (101,102)]}
 for tag,m in (('ppo_pc12','pc1.2_trainseed2_cp180'),('ppo_pc36','pc3.6_trainseed3_cp200')):
  pools[tag]=[(str(p),load_trajectory_pickle(p)[0]) for p in sorted((TRJ/m).glob('SEED*.pkl.zst'))];assert len(pools[tag])==12
 rec={}
 for seed in SEEDS:
  rec[seed]={}
  for tag,pool in pools.items():
   (hs,ha,hr),src=draw(pool,seed);np.savez(OUT/f'demo_{tag}_{seed}.npz',hs=hs,ha=ha,hr=hr);rec[seed][tag]=src
  d=np.load(ROOT/f'experiments/selfhist_pdt_v1/eval_pdt/demo_B_{seed}.npz');np.savez(OUT/f'demo_selfhist_{seed}.npz',**{k:d[k] for k in ('hs','ha','hr')});rec[seed]['selfhist']='fallback = own demo selfhist_pdt_v1/eval_pdt/demo_B_%d.npz'%seed
 E.save(OUT/'prep.json',dict(rtg=-415.7040981077988,allon=allon,demos=rec))
@torch.no_grad()
def evaluate(seed,ptag):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);name='B';q='mean'
 ck=torch.load(MP('PDT'),map_location='cpu',weights_only=False);assert ck['updates']==270000
 model=T.Model(True);model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
 target=json.loads((OUT/'prep.json').read_text())['rtg']
 d=np.load(OUT/f'demo_{ptag}_{seed}.npz');hs0,ha0,hr0=[torch.from_numpy(d[k]) for k in ('hs','ha','hr')]
 stats=STATS/f'PDT_prompt_{ptag}'/name;env,obs,scenario,params=E.environment(name,seed,stats);n=params['aps'];dest=stats/scenario/f'SEED{seed}'
 hv=torch.ones(n,T.H,dtype=torch.bool);C=T.CTX;hs,ha,hr=hs0,ha0,hr0
 ts=torch.arange(10080).repeat(n,1);S=torch.zeros(n,10080,110);A=torch.zeros(n,10080,2);R=torch.zeros(n,10080,1);RW=torch.zeros(n,10080,1);running=torch.full((n,1),target)
 t0=time.time()
 for step in range(10080):
  S[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32);R[:,step]=running
  left=max(0,step-C+1);right=max(C,step+1)
  if ptag=='selfhist':
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
   nt=getattr(env,'net',None) or env.env.net;print('HANG_TIMEOUT',ptag,seed,'step',step,'ues',len(nt.ues),flush=True);sys.exit(3)
 env.close()
 with (dest/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080
 pc=float(np.mean([float(r['pc_kw']) for r in rows]))*1000;ds=[0,0,0];cs=[0,0,0]
 with (dest/'ue_stats.csv').open() as f:
  for v in csv.DictReader(f):
   j=[.05,.15,.3].index(float(v['delay_budget']));ds[j]+=int(float(v['dropped_packets']));cs[j]+=int(float(v['completed_packets']))
 a=A.numpy().astype(int);sw=float((a[...,0]!=1).mean());deep=float((a[...,1]==3).mean())
 tsz=(dest/'trajectory.csv').stat().st_size;(dest/'trajectory.csv').unlink()
 E.save(dest/'verified.json',dict(kind='PDT',prompt=ptag,seconds=time.time()-t0,env=name,seed=seed,model_sha256=T.sha(MP('PDT')),rtg_target=target,power_w=pc,dropped=ds,completed=cs,
  return_ap_mean=float(target-running.mean()),ant_switch=sw,deep_sleep=deep,trajectory_csv_deleted_bytes=tsz))
 print('SEED_COMPLETE',ptag,seed,pc,ds,flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:evaluate(int(sys.argv[1]),sys.argv[2]);sys.exit()
 prepare();print('PREP_DONE',flush=True)
 def job(t):
  with (OUT/'logs'/f'{t[0]}-{t[1]}.log').open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,str(t[1]),t[0]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  print('CASE',*t,'rc',rc,flush=True);return (*t,rc)
 tasks=[(g,s) for g in TAGS for s in SEEDS];res=list(concurrent.futures.ThreadPoolExecutor(20).map(job,tasks));summ={}
 for g in TAGS:
  vs=[json.loads(p.read_text()) for p in sorted((STATS/f'PDT_prompt_{g}'/'B'/'B').glob('SEED*/verified.json'))]
  if not vs:continue
  dd=np.sum([v['dropped'] for v in vs],0);c=np.sum([v['completed'] for v in vs],0)
  summ[g]=dict(n=len(vs),power_w=round(float(np.mean([v['power_w'] for v in vs])),2),power_seeds=[round(v['power_w'],1) for v in vs],drop_pct=[round(float(100*x/max(1,x+y)),4) for x,y in zip(dd,c)],
   str_seeds=[round(100*v['dropped'][0]/max(1,v['dropped'][0]+v['completed'][0]),3) for v in vs])
 E.save(OUT/'complete.json',dict(runs=len(res),failed=[r for r in res if r[-1]!=0],summary=summ));print('ALL_COMPLETE',flush=True)
