"""[stride_s1h16_pdt_v1] Prompt stride ablation (user: 测一下不跨步采样而是连续采样; 确认): identical to b2000cl90_pdt_v1/train_b2000cl90.py except STRIDE=1, H=16 (orig 6, 16).
[b2000cl90_pdt_v1] Same as new500_cl98_pdt_v1 except B2000 data replaced by zdovv6uu cp170 recollected at cluster .90, seeds 1-35
(analysis/trajectory/newmodels_traj_collect_B2000cl90_v1; 795.9 W, str .460%). (user: 要重新训练根据更新traj的pdt和dt)
[new500_cl98_pdt_v1] Same as bigmodel_pdt_v1 except B500 data replaced by B500 PPO bbcyap5g cpfinal under cluster .98, seeds 1-35
(analysis/trajectory/B500_evalw_final_cluster98_collect_v1; 189.7 W, str .517%). (user: 就用这个500的final作为新的500来收集traj再训练dt和pdt)
Base was bigmodel_pdt_v1: identical data/settings to selfhist_pdt_v1 (expert data), only bigger model
(256 dim, 6 blocks, 4 heads), checkpoints every 600 rounds.
Original doc: Self-history prompt DT (Meta-DT style): the prompt is the AP's own recent transitions
(s, a, r) before the context window, no explicit scene id. Test time uses the model's own history.
Seven scenes, new-model trajectories (newmodels_traj_collect_v1) + reused C/B4/B36."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='5',MKL_NUM_THREADS='5',OPENBLAS_NUM_THREADS='1')
import sys,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(ROOT))
from decision_transformer.model import DecisionTransformer
from trajectory_io import load_trajectory_pickle
BASE=ROOT/'experiments/stride_s1h16_pdt_v1';MODELS=ROOT/'dt_results/stride_s1h16_pdt_v1'
TR=ROOT/'analysis/trajectory';NEW=TR/'newmodels_traj_collect_v1';OLD21=TR/'tradeoff_selected_single_seed1to20_v1/pc2.1_trainseed5_cpfinal'
CTX=8;H=16;STRIDE=1;TRUNC_P=.1;ROUNDS=1800;UPD=150;BATCH=256
def seeds(d,a,b):return [str(d/f'SEED{k}.pkl.zst') for k in range(a,b+1)]
SCENES=[  # env, files, traffic, aps, area, cluster, episodes
 ('A',seeds(NEW/'A_pc1.5_trainseed1_cp440',1,35),0,16,1000,.98,35),
 ('B',seeds(OLD21,1,20)+seeds(NEW/'pc2.1_trainseed5_cpfinal',21,35),1,16,1000,.9,35),
 ('C',[str(TR/'uniform/C.pkl.zst')],2,16,1000,.9,35),
 ('B4',[str(TR/'B40-05-05-3/B4/B.pkl.zst')],1,4,1000,.9,140),
 ('B36',[str(TR/'B40-05-05-3/B36/B.pkl.zst')],1,36,1000,.9,17),
 ('B500',seeds(TR/'B500_evalw_final_cluster98_collect_v1/B500evalw_bbcyap5g_cpfinal_cl98',1,35),1,16,500,.98,35),
 ('B2000',seeds(TR/'newmodels_traj_collect_B2000cl90_v1/B2000_zdovv6uu_pc1.2_cp170',1,35),1,16,2000,.9,35)]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2)
def history_index(left):
 """indices of the H transitions before window start `left` (lag 1, 1+S, ...); -1 = unavailable"""
 idx=left-1-STRIDE*np.arange(H)
 return np.where(idx>=0,idx,-1)
class SelfHistDataset(Dataset):
 def __init__(self,scenes=SCENES,mode='SHPDT'):
  self.mode=mode;self.trajs=[];self.groups={};self.manifest=[]
  for env,files,traffic,aps,area,cluster,n in scenes:
   start=len(self.trajs)
   for p in files:
    items=load_trajectory_pickle(p)
    for tr in items:
     assert np.array_equal(tr['prompt'],[traffic,aps,area]),(env,p)
     for k,shape in [('observations',(10080,aps,110)),('actions',(10080,aps,2)),('rewards',(10080,aps,1))]:
      assert tr[k].shape==shape and np.isfinite(tr[k]).all(),(env,k)
     a=tr['actions'];assert ((a[...,0]>=0)&(a[...,0]<3)&(a[...,1]>=0)&(a[...,1]<4)).all()
     r=tr['rewards'].astype(np.float32)
     rtg=np.flip(np.cumsum(np.flip(r,0),0,dtype=np.float64),0).astype(np.float32)
     self.trajs.append(dict(obs=tr['observations'].astype(np.float32),act=a.astype(np.float32),rew=r,rtg=rtg))
    self.manifest.append(dict(env=env,path=p,episodes=len(items),sha256=sha(p)))
    del items
   assert len(self.trajs)-start==n,(env,len(self.trajs)-start)
   self.groups[env]=(start,len(self.trajs))
  self.envs=list(self.groups)
 def __len__(self):return ROUNDS*UPD*BATCH
 def sample(self,ti,ap,si,truncate=None):
  tr=self.trajs[ti];sl=slice(si,si+CTX)
  hi=history_index(si)
  if truncate is not None:hi[truncate:]=-1
  valid=hi>=0;hc=np.where(valid,hi,0)
  hs=tr['obs'][hc,ap];ha=tr['act'][hc,ap];hr=tr['rew'][hc,ap]
  hs[~valid]=0;ha[~valid]=0;hr[~valid]=0
  return (torch.from_numpy(hs),torch.from_numpy(ha),torch.from_numpy(hr),torch.from_numpy(valid),
          torch.arange(si,si+CTX),torch.from_numpy(tr['obs'][sl,ap].copy()),torch.from_numpy(tr['act'][sl,ap].copy()),torch.from_numpy(tr['rtg'][sl,ap].copy()))
 def demo(self,ti,ap,end):
  """Prompt-DT demo: H transitions (stride S) ending at `end` of expert episode ti, AP ap"""
  tr=self.trajs[ti];hi=end-1-STRIDE*np.arange(H)
  return torch.from_numpy(tr['obs'][hi,ap]),torch.from_numpy(tr['act'][hi,ap]),torch.from_numpy(tr['rew'][hi,ap]),torch.ones(H,dtype=torch.bool)
 def __getitem__(self,index):
  env=self.envs[int(torch.randint(len(self.envs),()))];a,b=self.groups[env]
  ti=int(torch.randint(a,b,()));ap=int(torch.randint(self.trajs[ti]['obs'].shape[1],()))
  si=int(torch.randint(10080-CTX+1,()))
  if self.mode=='PDT':
   pool=[k for k in range(a,b) if k!=ti] or [ti];pj=pool[int(torch.randint(len(pool),()))]
   pap=int(torch.randint(self.trajs[pj]['obs'].shape[1],()));end=int(torch.randint(STRIDE*(H-1)+1,10081,()))
   return self.demo(pj,pap,end)+self.sample(ti,ap,si)[4:]
  trunc=int(torch.randint(H+1,())) if float(torch.rand(()))<TRUNC_P else None
  return self.sample(ti,ap,si,trunc)
class Model(DecisionTransformer):
 def __init__(self,use_hist=True):
  super().__init__(110,2,6,256,CTX,4,.1,max_timestep=10080,no_prompt=True)
  self.use_hist=use_hist
  if not use_hist:return
  self.h_state=nn.Linear(110,256);self.h_a1=nn.Embedding(3,256);self.h_a2=nn.Embedding(4,256)
  self.h_rew=nn.Linear(1,256);self.h_lag=nn.Embedding(H,256);self.h_pad=nn.Parameter(torch.zeros(256))
 def forward(self,hs,ha,hr,hv,ts,s,a,r):
  B,L,_=s.shape;t=self.embed_timestep(ts)
  tok=torch.stack((self.embed_rtg(r)+t,self.embed_state(s)+t,self.embed_action_1(a[...,0].long())+self.embed_action_2(a[...,1].long())+t),dim=2).reshape(B,3*L,256)
  if not self.use_hist:
   h=self.transformer(self.embed_ln(tok)).reshape(B,L,3,256)[:,:,1]
   return [self.predict_action_1(h),self.predict_action_2(h)]
  hist=self.h_state(hs)+self.h_a1(ha[...,0].long())+self.h_a2(ha[...,1].long())+self.h_rew(hr)
  # oldest first so causal attention lets every main token see the whole history
  hist=hist+self.h_lag.weight.unsqueeze(0);hist=torch.where(hv.unsqueeze(-1),hist,self.h_pad.expand_as(hist)).flip(1)
  tok=torch.stack((self.embed_rtg(r)+t,self.embed_state(s)+t,self.embed_action_1(a[...,0].long())+self.embed_action_2(a[...,1].long())+t),dim=2).reshape(B,3*L,256)
  h=self.transformer(self.embed_ln(torch.cat((hist,tok),1)))[:,H:]
  h=h.reshape(B,L,3,256)[:,:,1]
  return [self.predict_action_1(h),self.predict_action_2(h)]
def loss_of(model,batch):
 hs,ha,hr,hv,t,s,a,r=batch;heads=model(hs,ha,hr,hv,t,s,a,r)
 return sum(nn.functional.cross_entropy(h.reshape(-1,h.shape[-1]),a[...,i].long().reshape(-1)) for i,h in enumerate(heads))/2
def run(smoke=False,kind='SHPDT'):
 use_hist=kind in('SHPDT','PDT')
 torch.set_num_threads(5);torch.manual_seed(1);np.random.seed(1)
 scenes=[(e,f[:1],tr,ap,ar,c,(35 if e in('C',) else 140 if e=='B4' else 17 if e=='B36' else 1)) for e,f,tr,ap,ar,c,n in SCENES] if smoke else SCENES
 data=SelfHistDataset(scenes,kind)
 # history must be strictly before the window; check alignment on real data
 x=data.sample(0,0,100);assert x[3].all() and torch.equal(x[0][0],torch.from_numpy(data.trajs[0]['obs'][99,0]))
 x=data.sample(0,0,0);assert not x[3].any()
 m=Model(use_hist);m.train();b=[v.unsqueeze(0) for v in data[0]];loss=loss_of(m,b);loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);del m
 rounds=5 if smoke else ROUNDS
 if smoke:folder=BASE/'smoke';folder.mkdir(parents=True,exist_ok=True)
 else:
  folder=MODELS/kind;folder.mkdir(parents=True,exist_ok=False)
  save(folder/'dataset-manifest.json',dict(files=data.manifest,episodes=len(data.trajs),reward=[40,.5,.5,3],sampling='uniform seven environments; uniform episode/AP/window'))
  save(folder/'preflight.json',dict(finite_gradients=True,history_alignment=True,H=H,stride=STRIDE,trunc_p=TRUNC_P,context=CTX))
 torch.manual_seed(1);np.random.seed(1)
 loader=DataLoader(data,batch_size=BATCH,shuffle=False,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(1))
 it=iter(loader);model=Model(use_hist).cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
 sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1-step/(ROUNDS*UPD)/4)
 start=time.monotonic();count=0
 print('TRAIN_START',kind,'episodes',len(data.trajs),flush=True)
 with (folder/'loss.jsonl').open('w' if smoke else 'x') as log:
  for rn in range(1,rounds+1):
   ls=[]
   for _ in range(UPD):
    batch=[v.cuda(non_blocking=True) for v in next(it)]
    loss=loss_of(model,batch);assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.25);opt.step();sch.step()
    ls.append(loss.item());count+=1
   row=dict(round=rn,updates=count,loss=float(np.mean(ls)),seconds=time.monotonic()-start)
   log.write(json.dumps(row)+'\n');log.flush()
   if smoke or rn==1 or rn%20==0:print('TRAIN_PROGRESS',json.dumps(row),flush=True)
   if not smoke and rn%600==0:torch.save(dict(model_state_dict=model.state_dict(),updates=count),folder/f'model_r{rn}.pt')
 if smoke:print('SMOKE_OK',flush=True);return
 with (folder/'model.pt').open('xb') as f:torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),prompt={'SHPDT':'self-history','PDT':'expert demo from another episode of same scene','DT':'none'}[kind],H=H,stride=STRIDE,trunc_p=TRUNC_P,updates=count,context=CTX,rtg_scale=1,seed=1),f)
 save(folder/'complete.json',dict(updates=count,seconds=time.monotonic()-start,sha256=sha(folder/'model.pt')))
 print('TRAIN_COMPLETE',kind,flush=True)
if __name__=='__main__':
 a=sys.argv[1:];run(smoke='smoke' in a,kind='DT' if 'DT' in a else 'PDT' if 'PDT' in a else 'SHPDT')
