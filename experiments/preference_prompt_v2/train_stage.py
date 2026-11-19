from workflow import *
import workflow as w
import gc

def mixed_path(pc):
 root=PROJECT/'analysis/trajectory'/('B1000_preference_v1' if pc<2.4 else 'B1000_preference_v2')
 return root/f'pc_{pc:.1f}'/'B.pkl.zst'

class BalancedDataset:
 def __init__(self):
  from decision_transformer.utils import TrajectoryDataset
  self.base=TrajectoryDataset([str(mixed_path(pc)) for pc in PCS],8,1,seed_fraction=1,single_ap=True)
  assert len(self.base.trajectories)==35
 def __len__(self):return len(self.base)
 def __getitem__(self,index):
  _,ts,states,actions,rtg=self.base[index]
  ti=bisect.bisect_right(self.base.window_ends,index)
  return torch.tensor(self.base.trajectories[ti]['preference_pc'],dtype=torch.float32),ts,states,actions,rtg

def preflight():
 from trajectory_io import load_trajectory_pickle
 from preference_model import PreferenceTransformer
 from decision_transformer.model import DecisionTransformer
 from torch.utils.data import DataLoader
 import torch.nn.functional as F
 validate_sources();space();manifest=[]
 for pc in PCS:
  p=mixed_path(pc);done=json.loads((p.parent/'complete.json').read_text())
  assert sha(p)==done['sha256']
  data=load_trajectory_pickle(p);assert len(data)==5 and sorted(x['seed'] for x in data)==[1,2,3,4,5]
  for tr in data:
   check_trajectory(tr);assert tr['preference_pc']==pc and tr['actor_sha256']==done['actor_sha256']
  manifest.append(dict(pc=pc,path=str(p),sha256=sha(p),episodes=5,actor_sha256=done['actor_sha256'],return_min=float(min(x['rewards'].sum(0).min() for x in data)),return_max=float(max(x['rewards'].sum(0).max() for x in data))))
 save_new(BASE/'dataset-manifest.json',dict(files=manifest,episodes=35,ap_trajectories=560,reward=COMMON_WEIGHTS))
 data=BalancedDataset();batch=next(iter(DataLoader(data,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(1))))
 pref,ts,states,actions,rtg=batch;states=states.squeeze(2);actions=actions.squeeze(2);rtg=rtg.squeeze(2)
 assert all(torch.isfinite(x).all() for x in batch)
 for enabled in [False,True]:
  torch.manual_seed(1);model=PreferenceTransformer(enabled)
  if not enabled:
   ref=DecisionTransformer(110,2,3,128,8,1,.1,max_timestep=10080,no_prompt=True);ref.load_state_dict(model.state_dict(),strict=True);model.eval();ref.eval()
   with torch.no_grad():assert all(torch.equal(a,b) for a,b in zip(model(ts,states,actions,rtg,pref),ref(ts,states,actions,rtg,pref)))
  model.train();heads=model(ts,states,actions,rtg,pref)
  loss=sum(F.cross_entropy(h.reshape(-1,h.shape[-1]),actions[...,i].long().reshape(-1)) for i,h in enumerate(heads))/2
  loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 save_new(BASE/'real-data-preflight.json',dict(episodes=35,ap_trajectories=560,batch=256,finite_gradients=True,no_prompt_fidelity=True,preference_scale=2.5,clipping=False))
 save_new(BASE/'rtg.json',dict(target=-435.27698835441845,scale=1,method='Fixed v1 target for matched comparisons'))

if __name__=='__main__':
 assert (BASE/'collection-complete.json').exists() and not MODELS.exists()
 w.dataset_path=mixed_path;w.PreferenceDataset=BalancedDataset
 preflight()
 for kind in ['PDT','DT']:
  print('TRAIN_STAGE',kind,flush=True);w.train(kind);gc.collect();torch.cuda.empty_cache()
 save_new(BASE/'training-complete.json',dict(models=['PDT','DT'],updates_each=270000))
