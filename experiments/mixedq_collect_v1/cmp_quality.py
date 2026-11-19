import sys,json,glob
from pathlib import Path
import numpy as np
R=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(R))
from trajectory_io import load_trajectory_pickle as L
TR=R/'analysis/trajectory'
def recon(a):
 n,T=a.shape[1],a.shape[0];cur=np.full(n,8);ant=np.zeros((T,n),int)
 for t in range(T):
  nw=cur+np.array([-1,0,1])[a[t,:,0]];ok=(nw>=0)&(nw<=8);cur=np.where(ok,nw,cur);ant[t]=cur
 return ant
def stats(d,seeds):
 out=dict(ret=[],ant=[],busy_ant=[],night_ant=[],sleep0=[],deep=[],busy_deep=[],sw=[],inval=[])
 for s in seeds:
  tr=L(d/f'SEED{s}.pkl.zst')[0];a=tr['actions'].astype(int);r=tr['rewards'][...,0]
  ant=recon(a);h=(np.arange(a.shape[0])%1440)//60;busy=(h>=10)&(h<20);night=(h<6)
  nw=None
  out['ret'].append(r.sum(0).mean());out['ant'].append(ant.mean());out['busy_ant'].append(ant[busy].mean());out['night_ant'].append(ant[night].mean())
  out['sleep0'].append((a[...,1]==0).mean());out['deep'].append((a[...,1]==3).mean());out['busy_deep'].append((a[busy][...,1]==3).mean())
  out['sw'].append((a[...,0]!=1).mean())
  # invalid antenna requests (at 0 asking -1 or at 8 asking +1)
  prev=np.vstack([np.full((1,a.shape[1]),8),ant[:-1]]);out['inval'].append((((prev==0)&(a[...,0]==0))|((prev==8)&(a[...,0]==2))).mean())
 return {k:round(float(np.mean(v)),3) for k,v in out.items()}
rows={}
for c in (100,150,200,250):rows[f'B500 cp{c}']=stats(TR/f'mixedq_collect_v1/B500_5m820ao4_cp{c}',range(1,11))
rows['B500 cp290']=stats(TR/'newmodels_traj_collect_v1/B500_5m820ao4_cp290',range(1,11))
for p in ['pc0.6_trainseed1_cp240','pc1.5_trainseed3_cp240','pc2.1_trainseed5_cpfinal','pc3.3_trainseed1_cp200']:
 rows['B1000 '+p.split('_')[0]]=stats(TR/'tradeoff_selected_single_seed1to20_v1'/p,range(1,6))
for k,v in rows.items():print(k.ljust(16),v)
