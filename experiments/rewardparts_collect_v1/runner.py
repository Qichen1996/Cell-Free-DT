# Re-collect ALL DT training episodes of the 7 scenes with per-AP reward components (multi-objective DT data).
# Same policies + same env seeds as the existing datasets (deterministic), plus reward_parts[t,ap] =
# [str_drop_ratio, sen_drop_ratio, tol_drop_ratio, pc_kw] captured inside BaseStation.get_reward.
# Check per step: reward == -(40,.5,.5).drop - 3*pc (eval weights). trajectory.csv deleted per run with record.
# No training, no retries, no other deletions. (user: 直接做拆分权重这一步，但是需要把所有场景都做了)
import os,sys,json,hashlib,subprocess,concurrent.futures,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=Path(__file__).parent
SRC=R/'experiments/tradeoff_selected_single_collect200_v1/runner.py'
O=R/'analysis/sim_stats/rewardparts_collect_v1';D=R/'analysis/trajectory/rewardparts_collect_v1'
NEWM=R/'experiments/newmodels_traj_collect_v1/models';TOS=R/'experiments/tradeoff_selected_single_collect200_v1'
ARCH=NEWM/'B500_5m820ao4_cp290/architecture.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def make_specs():
 sp=[];old={s['tag']:s for s in json.loads((TOS/'specs.json').read_text())}
 new={s['tag']:s for s in json.loads((R/'experiments/newmodels_traj_collect_v1/specs.json').read_text())}
 def add(tag,env,src,pc,ck,tw,seeds,ts=1,orig=None):
  m=Path(src);man={n:sha(m/n) for n in ['actor.pt','critic.pt','architecture.json']}
  sp.append(dict(pc=pc,training_seed=ts,checkpoint=ck,tag=tag,env=env,training_weights=tw,source=str(m),original_source=orig or str(m),manifest=man,seeds=seeds))
 s=new['A_pc1.5_trainseed1_cp440'];add(s['tag'],'A',s['source'],s['pc'],s['checkpoint'],s['training_weights'],[1,35])
 for t,s in old.items():add(t,'B',s['source'],s['pc'],s['checkpoint'],[80,.2,.2,s['pc']],[1,35] if t=='pc2.1_trainseed5_cpfinal' else [1,20],s['training_seed'])
 s=new['B500_5m820ao4_cp290'];add(s['tag'],'B500',s['source'],s['pc'],s['checkpoint'],s['training_weights'],[1,35])
 s=new['B2000_zdovv6uu_pc1.2_cp170'];add(s['tag'],'B2000',s['source'],s['pc'],s['checkpoint'],s['training_weights'],[1,35])
 for tag,env,cfg,seeds,tw in [('B4_k164wiqz_final','B4','B4_k164wiqz_final',[1,140],[40,.5,.5,1]),('B36_jywudswq_final','B36','B36_jywudswq_final',[1,17],[150,.3,.3,1])]:
  m=B/'models'/cfg;shutil.copy(ARCH,m/'architecture.json');add(tag,env,m,tw[3],'final',tw,seeds,orig=str(m/'config.yaml'))
 m=B/'models'/'C_run1';m.mkdir(exist_ok=True)
 for n in ['actor.pt','critic.pt']:shutil.copy(R/f'results/DT/C/mappo/check/run1/models/{n}',m/n)
 shutil.copy(ARCH,m/'architecture.json');add('C_run1','C',m,None,'run1',None,[1,35],orig=str(R/'results/DT/C/mappo/check/run1/models'))
 (B/'specs.json').write_text(json.dumps(sp,indent=1))
HOOK = (
 " from network.base_station import BaseStation as _BS\n"
 " if not getattr(_BS,'_rp_patched',False):\n"
 "  _orig=_BS.get_reward\n"
 "  def _gr(self,w_str,w_sen,w_tol,w_pc):\n"
 "   self._rp=np.array(list(np.asarray(self.drop_ratio,np.float64))+[self.power_consumption*1e-3],np.float64);return _orig(self,w_str,w_sen,w_tol,w_pc)\n"
 "  _BS.get_reward=_gr;_BS._rp_patched=True\n")
def load():
 code=SRC.read_text()
 rep={"B=R/'experiments/tradeoff_selected_single_collect200_v1'":f'B=Path({str(B)!r})',
      "O=R/'analysis/sim_stats/tradeoff_selected_single_collect200_v1'":f'O=Path({str(O)!r})',
      "D=R/'analysis/trajectory/tradeoff_selected_single_seed1to20_v1'":f'D=Path({str(D)!r})',
      " import numpy as np,torch,zstandard\n":" import numpy as np,torch,zstandard\n"+HOOK,
      "observations=np.empty((10080,n,110),np.float32);":"observations=np.empty((10080,n,110),np.float32);parts=np.empty((10080,n,4),np.float64);",
      "   if collect:rewards[step]=np.asarray(reward).reshape(n,1)\n":"   if collect:rewards[step]=np.asarray(reward).reshape(n,1);parts[step]=np.stack([b._rp for b in env.net.bss.values()])\n",
      "assert name=='B' and n==16":"assert name==s['env'] and n==expected(name)['aps']",
      "for x in [observations,actions,rewards]:assert np.isfinite(x).all()":"for x in [observations,actions,rewards,parts]:assert np.isfinite(x).all()\n  recon=-(parts[...,:3]@np.array([40,.5,.5]))-3*parts[...,3];assert np.allclose(recon,rewards[...,0],rtol=1e-4,atol=1e-5),float(np.abs(recon-rewards[...,0]).max())",
      "prompt=np.array([1,16,1000],np.float32)":"prompt=np.array([{'A':0,'B':1,'C':2}[scenario],n,expected(name)['area']],np.float32)",
      "training_weights=[80,.2,.2,s['pc']]":"training_weights=s['training_weights'],reward_parts=parts.astype(np.float32),reward_parts_order=['str_drop_ratio','sen_drop_ratio','tol_drop_ratio','pc_kw'],reward_parts_weights=[40,.5,.5,3]",
      "for k in ['observations','actions','rewards','prompt']:assert":"for k in ['observations','actions','rewards','prompt','reward_parts']:assert"}
 for a,b in rep.items():assert code.count(a)==1,a;code=code.replace(a,b)
 assert code.count('4*1024**3')==2;code=code.replace('4*1024**3','1.5*1024**3')
 sc={'__name__':'collect_rp','__file__':str(SRC)};exec(compile(code,str(B/'collect_in_memory.py'),'exec'),sc);return sc
def cases():
 sp=json.loads((B/'specs.json').read_text());return [(i,s['tag'],s['env'],seed) for i,s in enumerate(sp) for seed in range(s['seeds'][0],s['seeds'][1]+1)]
def worker(k):
 i,tag,env,seed=cases()[k];sc=load();sc['worker']('collect',i,env,seed)
 d=O/'collection'/tag/env/sc['expected'](env)['scenario']/f'SEED{seed}'
 t=d/'trajectory.csv';(d/'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv',bytes=t.stat().st_size,sha256=sha(t),reason='disk; user rule')));t.unlink()
 r=json.loads((d/'result.json').read_text());r.update(tag=tag,env=env,seed=seed)
 (B/'done').mkdir(exist_ok=True);(B/'done'/f'{tag}__{env}__{seed}.json').write_text(json.dumps(r))
 print('CASE_DONE',tag,env,seed,r['power_w'],r['dropped'],flush=True)
def main(only=None):
 if not (B/'specs.json').exists():make_specs()
 C=cases();print('cases',len(C),flush=True)
 if only=='test':
  K=[k for k,c in enumerate(C) if c[3]==1 and c[1] in('C_run1','B4_k164wiqz_final','B36_jywudswq_final','A_pc1.5_trainseed1_cp440')]
 else:
  assert not (B/'plan.json').exists();(B/'plan.json').write_text(json.dumps(dict(cases=C,runs=len(C),parallel=60,dataset_dir=str(D)),indent=1))
  K=[k for k in range(len(C)) if not (B/'done'/f'{C[k][1]}__{C[k][2]}__{C[k][3]}.json').exists()]
 (B/'logs').mkdir(exist_ok=True)
 def job(k):
  assert shutil.disk_usage(R).free>1.5*1024**3,'disk'
  with (B/'logs'/f'case-{k}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(k)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  print('JOB',k,C[k][1:],rc,flush=True);return rc
 with concurrent.futures.ThreadPoolExecutor(max_workers=60) as p:rcs=list(p.map(job,K))
 if only=='test':print('TEST_DONE',rcs,flush=True);return
 (B/'complete.json').write_text(json.dumps(dict(runs=len(K),rc=rcs,failed=[C[k] for k,r in zip(K,rcs) if r])));print('ALL_COMPLETE',sum(r!=0 for r in rcs),'failed',flush=True)
if __name__=='__main__':
 if len(sys.argv)>2 and sys.argv[1]=='job':worker(int(sys.argv[2]))
 else:main(sys.argv[1] if len(sys.argv)>1 else None)
