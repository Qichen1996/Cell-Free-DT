# [B500_evalw_final_cluster98_collect_v1] Collect trajectories for PDT: B500 PPO bbcyap5g cpfinal (trained w/ eval weights) under CLUSTER .98, seeds 1-35. (user: 把他多跑一些seed 跑到35 这次要收集traj以备pdt来运行)
# Based on B500_evalw_ckpt_eval_v1/runner.py (which was:) Eval + collect (seeds 1-5) checkpoints of B500 PPO trained with evaluation weights: cp200/250/290/final.
# Based on mixedq_collect_v1 runner. Old header: mixed-quality B500 data. Weaker checkpoints of the same B500 PPO run (5m820ao4, w=[150,1,1,.7])
# cp100/150/200/250, seeds 1-10 each (40 episodes). cp290 (35 eps) already exists. B1000 reuses existing 10-policy x 20-seed set.
# Same worker as newmodels_traj_collect_v1 (collect mode, pkl round-trip check). trajectory.csv deleted after each run with record.
# No training, no retries, no other deletions. Disk guard 0.5G.
import os,sys,json,hashlib,subprocess,concurrent.futures,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=Path(__file__).parent
SRC=R/'experiments/tradeoff_selected_single_collect200_v1/runner.py'
O=R/'analysis/sim_stats/B500_evalw_final_cluster98_collect_v1';D=R/'analysis/trajectory/B500_evalw_final_cluster98_collect_v1'
RUN=R/'results/DT/B/mappo/B500_evalw_40_05_05_pc3_seed1_ep300/wandb/run-20261112_015715-bbcyap5g/files'
CPS=['final'];SEEDS=range(1,36)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def make_specs():
 arch=R/'experiments/newmodels_traj_collect_v1/models/B500_5m820ao4_cp290/architecture.json';sp=[]
 for cp in CPS:
  tag=f'B500evalw_bbcyap5g_cp{cp}_cl98';m=B/'models'/tag;m.mkdir(parents=True)
  sfx='' if cp=='final' else f'_eps{cp}';shutil.copy(RUN/f'actor{sfx}.pt',m/'actor.pt');shutil.copy(RUN/f'critic{sfx}.pt',m/'critic.pt');shutil.copy(arch,m/'architecture.json')
  sp.append(dict(pc=3.0,training_seed=1,checkpoint=cp,tag=tag,env='B500',training_weights=[40,0.5,0.5,3],source=str(m),
   original_source=str(RUN/f'actor{sfx}.pt'),manifest={n:sha(m/n) for n in ['actor.pt','critic.pt','architecture.json']}))
 (B/'specs.json').write_text(json.dumps(sp,indent=1))
def load():
 code=SRC.read_text()
 rep={"B=R/'experiments/tradeoff_selected_single_collect200_v1'":f'B=Path({str(B)!r})',
      "O=R/'analysis/sim_stats/tradeoff_selected_single_collect200_v1'":f'O=Path({str(O)!r})',
      "D=R/'analysis/trajectory/tradeoff_selected_single_seed1to20_v1'":f'D=Path({str(D)!r})',
      "assert name=='B' and n==16":"assert name==s['env'] and n==16",
      "prompt=np.array([1,16,1000],np.float32)":"prompt=np.array([{'A':0,'B':1,'C':2}[scenario],n,expected(name)['area']],np.float32)",
      "training_weights=[80,.2,.2,s['pc']]":"training_weights=s['training_weights']",
      "'B500':('B',500,16,125.,.125,.9)":"'B500':('B',500,16,125.,.125,.98)",
      "exec(compile(f.read_text(),str(f),'exec'),scope)\n":"exec(compile(f.read_text(),str(f),'exec'),scope)\n _e=scope['training'].entries;scope['training'].entries=lambda:[dict(e,cluster=.98) if e['env']=='B500' else e for e in _e()]\n"}
 for a,b in rep.items():assert code.count(a)==1,a;code=code.replace(a,b)
 assert code.count('4*1024**3')==2;code=code.replace('4*1024**3','0.5*1024**3')
 sc={'__name__':'collect_mixedq','__file__':str(SRC)};exec(compile(code,str(B/'collect_in_memory.py'),'exec'),sc);return sc
def cases():
 sp=json.loads((B/'specs.json').read_text());return [(i,s['tag'],s['env'],seed) for i,s in enumerate(sp) for seed in SEEDS]
def worker(k):
 i,tag,env,seed=cases()[k];sc=load();sc['worker']('collect',i,env,seed)
 d=O/'collection'/tag/env/sc['expected'](env)['scenario']/f'SEED{seed}'
 t=d/'trajectory.csv';(d/'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv',bytes=t.stat().st_size,sha256=sha(t),reason='disk; user rule')));t.unlink()
 r=json.loads((d/'result.json').read_text());r.update(tag=tag,env=env,seed=seed)
 (B/'done').mkdir(exist_ok=True);(B/'done'/f'{tag}__{env}__{seed}.json').write_text(json.dumps(r))
 print('CASE_DONE',tag,env,seed,r['power_w'],r['dropped'],flush=True)
def main():
 assert not O.exists() and not D.exists() and not (B/'plan.json').exists()
 make_specs();C=cases();assert len(C)==35,len(C)
 (B/'logs').mkdir(exist_ok=True);(B/'plan.json').write_text(json.dumps(dict(cases=C,runs=len(C),parallel=35,cluster=.98,dataset_dir=str(D)),indent=1))
 def job(k):
  assert shutil.disk_usage(R).free>0.5*1024**3,'disk'
  with (B/'logs'/f'case-{k}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(k)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  print('JOB',k,C[k][1:],rc,flush=True);return rc
 with concurrent.futures.ThreadPoolExecutor(max_workers=35) as p:rcs=list(p.map(job,range(len(C))))
 (B/'complete.json').write_text(json.dumps(dict(runs=len(C),rc=rcs,failed=[C[k] for k,r in enumerate(rcs) if r])));print('ALL_COMPLETE',sum(r!=0 for r in rcs),'failed',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]))
 else:main()
