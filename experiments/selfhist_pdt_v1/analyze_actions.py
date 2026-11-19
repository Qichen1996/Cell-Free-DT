"""Compare PDT vs PPO(B500 cp290) actions on B500 seeds 1-5: per-hour antennas / sleep / power,
and behaviour in the 30 steps before each str drop. PPO antenna count reconstructed from its actions
(start 8, +-1, invalid ignored, same rule as BaseStation.switch_antennas); sleep = requested mode for both."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(ROOT))
from trajectory_io import load_trajectory_pickle
PDT=ROOT/'analysis/sim_stats/selfhist_pdt_v1_actlog/PDT/B500/B'
PPO=ROOT/'analysis/sim_stats/newmodels_traj_collect_v1/collection/B500_5m820ao4_cp290/B500/B'
PKL=ROOT/'analysis/trajectory/newmodels_traj_collect_v1/B500_5m820ao4_cp290'
OUT=ROOT/'experiments/selfhist_pdt_v1/actlog_B500/analysis.json'
def ppo(seed):
 a=load_trajectory_pickle(PKL/f'SEED{seed}.pkl.zst')[0]['actions'].astype(int).transpose(1,0,2)  # n,T,2
 n,T,_=a.shape;ant=np.zeros((n,T),int);cur=np.full(n,8)
 for t in range(T):
  nw=cur+np.array([-1,0,1])[a[:,t,0]];ok=(nw>=0)&(nw<=8);cur=np.where(ok,nw,cur);ant[:,t]=cur
 return a,ant,a[:,:,1]
def recon(a):
 n,T,_=a.shape;ant=np.zeros((n,T),int);cur=np.full(n,8)
 for t in range(T):
  nw=cur+np.array([-1,0,1])[a[:,t,0]];ok=(nw>=0)&(nw<=8);cur=np.where(ok,nw,cur);ant[:,t]=cur
 return ant
def pdt(seed):
 d=np.load(PDT/f'SEED{seed}/actlog.npz');a=d['actions'].astype(int)
 if seed==1:print('antenna reconstruction check (PDT logged vs rule):',float((recon(a)==d['num_ant']).mean()),'| requested vs actual sleep agree:',float((a[:,:,1]==d['sleep']).mean()))
 return a,d['num_ant'].astype(int),a[:,:,1]
res={}
for name,src,root in [('PDT',pdt,PDT),('PPO',ppo,PPO)]:
 H=dict(ant=np.zeros(24),sleep0=np.zeros(24),deep=np.zeros(24),pc=np.zeros(24),antsw=np.zeros(24),drops=np.zeros(24,int))
 pre=dict(ant=[],sleep0=[],deep=[],antsw=[],dant=[]);base=dict(ant=[],sleep0=[]);cnt=0;str_drop=0;str_tot=0
 for s in range(1,6):
  a,ant,slp=src(s);r=pd.read_csv(root/f'SEED{s}/reward.csv');T=len(r);hour=(np.arange(T)%1440)//60
  ue=pd.read_csv(root/f'SEED{s}/ue_stats.csv');m=ue.delay_budget==.05;str_drop+=int(ue.dropped_packets[m].sum());str_tot+=int((ue.dropped_packets+ue.completed_packets)[m].sum())
  sw=(a[:,:,0]!=1)
  for h in range(24):
   k=hour==h;H['ant'][h]+=ant[:,k].mean()/5;H['sleep0'][h]+=(slp[:,k]==0).mean()/5;H['deep'][h]+=(slp[:,k]==3).mean()/5
   H['pc'][h]+=r.pc_kw.values[k].mean()*1000/5;H['antsw'][h]+=sw[:,k].mean()/5
  busy=(hour>=10)&(hour<20);base['ant'].append(ant[:,busy].mean());base['sleep0'].append((slp[:,busy]==0).mean())
  for t in np.where(r.str_drop.values>0)[0]:
   H['drops'][hour[t]]+=1;w=slice(max(0,t-30),t+1);cnt+=1
   pre['ant'].append(ant[:,w].mean());pre['sleep0'].append((slp[:,w]==0).mean());pre['deep'].append((slp[:,w]==3).mean())
   pre['antsw'].append(sw[:,w].mean());pre['dant'].append(float(ant[:,t].mean()-ant[:,max(0,t-30)].mean()))
 res[name]=dict(str=f'{str_drop}/{str_tot}',drop_steps=cnt,hourly={k:np.round(v,3).tolist() for k,v in H.items()},
  busy_baseline={k:float(np.mean(v)) for k,v in base.items()},pre_drop_30={k:float(np.mean(v)) if v else None for k,v in pre.items()})
OUT.write_text(json.dumps(res,indent=1))
for k,v in res.items():
 print('==',k,'str',v['str'],'drop steps',v['drop_steps'])
 for h in ['pc','ant','sleep0','deep','antsw','drops']:print(f' {h:7s}',v['hourly'][h])
 print(' busy 10-20h baseline',v['busy_baseline']);print(' 30 steps before drop',v['pre_drop_30'])
