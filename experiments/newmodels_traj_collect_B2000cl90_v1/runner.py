# [B2000 cl90 recollect] copy of newmodels_traj_collect_v1/runner.py, only B2000 zdovv6uu cp170 seeds1-35, cluster .95->.90 (user: 2000的traj重新用90%收集).
# Trajectory (pkl.zst) collection for DT/PDT with the NEW per-scene models, original episode counts:
# A pc1.5 cp440 seeds1-35 (cluster .98); B500 cp290 seeds1-35 (.9); B2000 pc1.2 cp170 seeds1-35 (.95); B1000 pc2.1 final seeds21-35 (.9; seeds1-20 already in tradeoff_selected_single_seed1to20_v1).
# C/B4/B36 reuse existing pkls (same models). Reuses tradeoff_selected_single_collect200_v1 worker (collect mode, pkl round-trip check) with:
# specs/output redirected, B-only assert lifted, prompt=[traffic,aps,area] per scene, training_weights per model, disk guard 4G->0.5G.
# trajectory.csv deleted after each run with a record (user rule). No training, no retries, no other deletions.
import os,sys,json,hashlib,subprocess,concurrent.futures,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=Path(__file__).parent
SRC=R/'experiments/tradeoff_selected_single_collect200_v1/runner.py'
O=R/'analysis/sim_stats/newmodels_traj_collect_B2000cl90_v1';D=R/'analysis/trajectory/newmodels_traj_collect_B2000cl90_v1'
SEEDS={'B2000':range(1,36)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load():
 code=SRC.read_text()
 rep={"B=R/'experiments/tradeoff_selected_single_collect200_v1'":f'B=Path({str(B)!r})',
      "O=R/'analysis/sim_stats/tradeoff_selected_single_collect200_v1'":f'O=Path({str(O)!r})',
      "D=R/'analysis/trajectory/tradeoff_selected_single_seed1to20_v1'":f'D=Path({str(D)!r})',
      "assert name=='B' and n==16":"assert name==s['env'] and n==16",
      "prompt=np.array([1,16,1000],np.float32)":"prompt=np.array([{'A':0,'B':1,'C':2}[scenario],n,expected(name)['area']],np.float32)",
      "training_weights=[80,.2,.2,s['pc']]":"training_weights=s['training_weights']"}
 rep["'B2000':('B',2000,16,500.,.5,.95)"]="'B2000':('B',2000,16,500.,.5,.9)"
 rep["exec(compile(f.read_text(),str(f),'exec'),scope)"]="exec(compile(f.read_text(),str(f),'exec'),scope);_oe=scope['training'].entries;scope['training'].entries=lambda:[dict(e,cluster=.9) if e['env']=='B2000' else e for e in _oe()]"
 for a,b in rep.items():assert code.count(a)==1,a;code=code.replace(a,b)
 assert code.count('4*1024**3')==2;code=code.replace('4*1024**3','0.5*1024**3')
 sc={'__name__':'collect_new','__file__':str(SRC)};exec(compile(code,str(B/'collect_in_memory.py'),'exec'),sc);return sc
def cases():
 sp=load()['specs']();return [(i,s['tag'],s['env'],seed) for i,s in enumerate(sp) for seed in SEEDS[s['env']]]
def worker(k):
 i,tag,env,seed=cases()[k];sc=load();sc['worker']('collect',i,env,seed)
 d=O/'collection'/tag/env/sc['expected'](env)['scenario']/f'SEED{seed}'
 t=d/'trajectory.csv';(d/'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv',bytes=t.stat().st_size,sha256=sha(t),reason='disk; user rule')));t.unlink()
 r=json.loads((d/'result.json').read_text());r.update(tag=tag,env=env,seed=seed)
 (B/'done').mkdir(exist_ok=True);(B/'done'/f'{tag}__{env}__{seed}.json').write_text(json.dumps(r))
 print('CASE_DONE',tag,env,seed,r['power_w'],r['dropped'],flush=True)
def main():
 assert not O.exists() and not D.exists() and not (B/'plan.json').exists()
 C=cases();assert len(C)==35,len(C)
 (B/'logs').mkdir(exist_ok=True);(B/'plan.json').write_text(json.dumps(dict(cases=C,runs=len(C),parallel=35,dataset_dir=str(D)),indent=1))
 def job(k):
  assert shutil.disk_usage(R).free>0.5*1024**3,'disk'
  with (B/'logs'/f'case-{k}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(k)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  print('JOB',k,C[k][1:],rc,flush=True);return rc
 with concurrent.futures.ThreadPoolExecutor(max_workers=35) as p:rcs=list(p.map(job,range(len(C))))
 (B/'complete.json').write_text(json.dumps(dict(runs=len(C),rc=rcs,failed=[C[k] for k,r in enumerate(rcs) if r])));print('ALL_COMPLETE',sum(r!=0 for r in rcs),'failed',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]))
 else:main()
