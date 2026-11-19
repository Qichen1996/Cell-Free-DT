"""Matched multi-environment PDT/DT with PC-only normalized preference."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='5',MKL_NUM_THREADS='5',OPENBLAS_NUM_THREADS='1')
import sys,json,time,hashlib,subprocess,bisect
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(ROOT))
BASE=ROOT/'experiments/selected_seven_pc_only_v1';MODELS=ROOT/'dt_results/selected_seven_pc_only_v1'
from decision_transformer.model import DecisionTransformer
from decision_transformer.utils import TrajectoryDataset
from torch.utils.data import Dataset,DataLoader
SPECS=[('A','uniform/A.pkl.zst',0,16,1000,250,.98,35),('B','uniform/B.pkl.zst',1,16,1000,250,.9,35),('C','uniform/C.pkl.zst',2,16,1000,250,.9,35),('B4','B40-05-05-3/B4/B.pkl.zst',1,4,1000,500,.9,140),('B36','B40-05-05-3/B36/B.pkl.zst',1,36,1000,1000/6,.9,17),('B500','B500_e450_uniform/B.pkl.zst',1,16,500,125,.9,35),('B2000','B2000_e180_uniform/B.pkl.zst',1,16,2000,500,.95,35)]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2)
def entries():
 out=[]
 for env,rel,traffic,aps,area,spacing,cluster,n in SPECS:
  out.append(dict(env=env,path=str(ROOT/'analysis/trajectory'/rel),traffic=traffic,aps=aps,area=area,spacing=spacing,cluster=cluster,episodes=n,preference=None))
 manifest=json.loads((ROOT/'experiments/B1000_selected_traj/dataset-manifest.json').read_text())
 for row in manifest['files']:
  out.append(dict(env='B',path=row['path'],traffic=1,aps=16,area=1000,spacing=250,cluster=.9,episodes=5,preference=row['pc'],expected_sha=row['sha256']))
 return out
class MixedDataset(Dataset):
 def __init__(self):
  self.specs=entries();self.base=TrajectoryDataset([e['path'] for e in self.specs],8,1,seed_fraction=1,single_ap=True)
  assert len(self.base.trajectories)==382
  self.groups={e[0]:[] for e in SPECS};self.prompts=[];self.ranges=[];self.manifest=[]
  ti=0
  for spec in self.specs:
   start=ti;pref=spec['preference']
   vec=np.array([(1.8 if pref is None else pref)/2.5],dtype=np.float32)
   digest=sha(spec['path'])
   if 'expected_sha' in spec:assert digest==spec['expected_sha']
   for tr in self.base.trajectories[start:start+spec['episodes']]:
    n=spec['aps'];assert np.array_equal(tr['prompt'],[spec['traffic'],n,spec['area']])
    for k,shape in [('observations',(10080,n,110)),('actions',(10080,n,2)),('rewards',(10080,n,1))]:
     assert tr[k].shape==shape and np.isfinite(tr[k]).all(),(spec['env'],k)
    if 'weights' in tr:assert np.array_equal(tr['weights'],[40,.5,.5,3])
    if pref is not None:assert tr['preference_pc']==pref
    assert ((tr['actions'][...,0]>=0)&(tr['actions'][...,0]<3)&(tr['actions'][...,1]>=0)&(tr['actions'][...,1]<4)).all()
    self.prompts.append(torch.tensor(vec));ti+=1
   self.ranges.append((start,ti));self.groups[spec['env']].append(len(self.ranges)-1)
   self.manifest.append(dict(spec,sha256=digest,prompt_pc=1.8 if pref is None else pref,normalized_prompt=float(vec[0]),prompt_assignment='user-assigned PC1.8 reference, not source training weight' if pref is None else 'actual source PPO PC weight'))
  self.envs=list(self.groups)
 def __len__(self):return 1800*150*256
 def __getitem__(self,index):
  env=self.envs[int(torch.randint(7,()))];groups=self.groups[env]
  group=groups[int(torch.randint(len(groups),()))];a,b=self.ranges[group]
  ti=int(torch.randint(a,b,()));tr=self.base.trajectories[ti];ap=int(torch.randint(tr['observations'].shape[1],()))
  si=int(torch.randint(10080-8+1,()));offset=(self.base.window_ends[ti-1] if ti else 0)+ap*self.base.window_counts[ti]+si
  _,ts,s,a,r=self.base[offset]
  return self.prompts[ti],ts,s.float().squeeze(1),a.float().squeeze(1),r.float().squeeze(1)
class Model(DecisionTransformer):
 def __init__(self,enabled):
  super().__init__(110,2,3,128,8,1,.1,max_timestep=10080,no_prompt=True)
  self.enabled=enabled
  if enabled:self.prompt_projection=nn.Linear(1,128)
 def forward(self,ts,s,a,r,p):
  batch,length,_=s.shape;t=self.embed_timestep(ts)
  tokens=torch.stack((self.embed_rtg(r)+t,self.embed_state(s)+t,self.embed_action_1(a[...,0].long())+self.embed_action_2(a[...,1].long())+t),dim=2).reshape(batch,3*length,128)
  if self.enabled:tokens=torch.cat((self.prompt_projection(p).unsqueeze(1),tokens),dim=1)
  h=self.transformer(self.embed_ln(tokens))
  if self.enabled:h=h[:,1:]
  h=h.reshape(batch,length,3,128)[:,:,1]
  return [self.predict_action_1(h),self.predict_action_2(h)]
def loss_of(model,batch):
 p,t,s,a,r=batch;heads=model(t,s,a,r,p)
 return sum(nn.functional.cross_entropy(h.reshape(-1,h.shape[-1]),a[...,i].long().reshape(-1)) for i,h in enumerate(heads))/2
def run(kind):
 torch.set_num_threads(5);torch.manual_seed(1);np.random.seed(1)
 data=MixedDataset();batch=[x.unsqueeze(0) for x in data[0]]
 # Faithful no-prompt model and finite real-data gradients before training.
 for enabled in [False,True]:
  torch.manual_seed(1);m=Model(enabled)
  if not enabled:
   ref=DecisionTransformer(110,2,3,128,8,1,.1,max_timestep=10080,no_prompt=True);ref.load_state_dict(m.state_dict());m.eval();ref.eval()
   p,t,s,a,r=batch
   with torch.no_grad():assert all(torch.equal(x,y) for x,y in zip(m(t,s,a,r,p),ref(t,s,a,r,p)))
  m.train();loss=loss_of(m,batch);loss.backward()
  assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 del m,ref
 folder=MODELS/kind;folder.mkdir(parents=True,exist_ok=False)
 save(folder/'dataset-manifest.json',dict(files=data.manifest,episodes=382,ap_trajectories=4772,reward=[40,.5,.5,3],sampling='uniform seven environments; uniform 11 B1000 source groups; uniform episode/AP/window'))
 save(folder/'preflight.json',dict(finite_gradients=True,no_prompt_fidelity=True,prompt_dimensions=1,scene_prompt=False,old_seven_normalized_prompt=.72))
 # Reset RNG after preflight so both models receive identical sampling sequence.
 torch.manual_seed(1);np.random.seed(1)
 loader=DataLoader(data,batch_size=256,shuffle=False,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(1))
 iterator=iter(loader);model=Model(kind=='PDT').cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1-step/270000/4)
 start=time.monotonic();count=0
 print('TRAIN_START',kind,'episodes=382 AP=4772',flush=True)
 with (folder/'loss.jsonl').open('x') as log:
  for roundno in range(1,1801):
   losses=[]
   for update in range(150):
    batch=[x.cuda(non_blocking=True) for x in next(iterator)]
    loss=loss_of(model,batch);assert torch.isfinite(loss)
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.25);optimizer.step();scheduler.step()
    losses.append(loss.item());count+=1
   row=dict(round=roundno,updates=count,loss=float(np.mean(losses)),seconds=time.monotonic()-start)
   log.write(json.dumps(row)+'\n');log.flush()
   if roundno==1 or roundno%20==0:print('TRAIN_PROGRESS',kind,json.dumps(row),flush=True)
 with (folder/'model.pt').open('xb') as f:torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),prompt_enabled=kind=='PDT',prompt_dim=1,prompt_normalization='PC/2.5; no clipping',old_seven_prompt=.72,updates=count,context=8,rtg_scale=1,seed=1),f)
 save(folder/'complete.json',dict(updates=count,seconds=time.monotonic()-start,sha256=sha(folder/'model.pt')))
 print('TRAIN_COMPLETE',kind,flush=True)
def main():
 if len(sys.argv)>1:run(sys.argv[1]);return
 assert not MODELS.exists() and not (BASE/'training-plan.json').exists()
 save(BASE/'training-plan.json',dict(models=['PDT','DT'],rounds=1800,updates_each=270000,batch=256,context=8,blocks=3,dimension=128,heads=1,lr=.0005,weight_decay=.0001,dropout=.1,seed=1,observation_normalization=False,rtg_scale=1,prompt=['PC/2.5'],scene_prompt=False,prompt_clipping=False,old_seven_prompt=dict(assigned_pc=1.8,normalized=.72,reason='explicit user reference assignment, not source training weight'),old_B='original uniform/B; not multicheckpoint replacement',data=entries()))
 for kind in ['PDT','DT']:
  with (BASE/(kind+'.log')).open('x') as log:
   rc=subprocess.call([sys.executable,'-u',str(Path(__file__)),kind],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  assert rc==0,f'{kind} failed rc={rc}; inspect {BASE/kind}.log'
 save(BASE/'training-complete.json',dict(models=['PDT','DT'],updates_each=270000))
 print('ALL_TRAIN_COMPLETE_SELECTED_SEVEN',flush=True)
if __name__=='__main__':main()
