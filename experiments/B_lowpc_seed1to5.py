import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,csv,math,statistics,subprocess,concurrent.futures,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_lowpc_seed1to5';O=R/'analysis/sim_stats/B_lowpc_seed1to5'
def specs():
 rows=[]
 for pc in [1.8,2.1]:
  for ts in [1,2,3]:
   for cp in [180,200,220,230,240]:
    if ts==1:
     exp='B1000_checkpoint230_test' if cp==230 else 'B1000_checkpoint_test'
     src=R/f'results/weights/{exp}/by_tag/eps{cp}/pc_{pc}'
    else:
     exp='AB_new_training_checkpoint_eval' if pc==1.8 else 'B_pc21_seeds23_checkpoint_eval'
     src=R/f'experiments/{exp}/models/B_pc{pc}_trainseed{ts}/eps{cp}'
    rows.append(dict(pc=pc,training_seed=ts,checkpoint=cp,source=str(src),tag=f'pc{pc}_trainseed{ts}_cp{cp}'))
 return rows
def manifest(s):
 src=Path(s['source']);m=json.loads((src/'manifest.json').read_text())
 for n,h in m.items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h
 return m
def audit(p,s,seed):
 m=manifest(s);v=json.loads((p/'verified.json').read_text())
 assert v['seed']==seed and v['checkpoint']==s['checkpoint'] and v['pc_weight']==s['pc'] and v['steps']==10080
 assert v['actor_sha256']==m['actor.pt'] and v['critic_sha256']==m['critic.pt']
 assert v['evaluation_weights']==[40,.5,.5,3] and v['environment']==dict(scenario='B',area=1000,aps=16,spacing_m=250.,neighbor_km=.25,cluster=.9,min_ap=1)
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:assert (p/f'{n}.csv').stat().st_size>0
 with (p/'reward.csv').open() as f:v=[float(r['pc_kw']) for r in csv.DictReader(f)]
 assert len(v)==10080 and all(math.isfinite(x) for x in v)
 return statistics.mean(v)*1000
def worker(i,seed,pre=False):
 s=specs()[i];m=manifest(s)
 code=(R/'experiments/IPPO_pc21_cp200_cross/runner.py').read_text().replace('pc_weight=2.1,checkpoint=200',f"pc_weight={s['pc']},checkpoint={s['checkpoint']}")
 scope={'__name__':'lowpc_worker','__file__':__file__};exec(compile(code,str(B/'worker_memory.py'),'exec'),scope)
 scope.update(SOURCE=Path(s['source']),AH=m['actor.pt'],CH=m['critic.pt'],OUT=O/s['tag'])
 scope['worker']('B',seed,pre)
 if not pre:audit(O/s['tag']/f'B/B/SEED{seed}',s,seed)
def main():
 assert not B.exists() and not O.exists(),'No duplicate launch'
 (B/'logs').mkdir(parents=True);rows=specs();paths={};todo=[]
 for i,s in enumerate(rows):
  manifest(s)
  for seed in range(1,6):
   candidates=[]
   if s['training_seed']==1:
    exp='B1000_checkpoint230_test' if s['checkpoint']==230 else 'B1000_checkpoint_test'
    candidates.append(R/f"analysis/sim_stats/{exp}/eps{s['checkpoint']}/pc_{s['pc']}/B/SEED{seed}")
    if s['pc']==2.1 and s['checkpoint']==200:candidates.append(R/f'analysis/sim_stats/IPPO_pc21_cp200_cross/B/B/SEED{seed}')
   found=None
   for p in candidates:
    if (p/'verified.json').is_file():
     try:audit(p,s,seed);found=p;break
     except (AssertionError,KeyError):pass
   if found:paths[f'{i}-{seed}']=str(found)
   else:todo.append((i,seed))
 plan=dict(models=rows,evaluation_seeds=[1,2,3,4,5],steps=10080,new_runs=len(todo),reused_runs=len(paths),parallel=2,selection='Per training seed: lowest mean reward.pc_kw among checkpoints with all three pooled packet drop percentages <=0.5; ties choose earlier checkpoint',independent_test=False,auto_retry=False,save_traj_pkl=False)
 (B/'plan.json').write_text(json.dumps(plan,indent=2));print('PLAN',len(todo),'NEW',len(paths),'REUSED',flush=True)
 for i in range(len(rows)):
  with (B/'logs'/f'preflight-{i}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',str(i),'1'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Preflight {i} failed'
 print('ALL_30_PREFLIGHTS_OK',flush=True)
 def task(x):
  i,seed=x;assert shutil.disk_usage(R).free>3*1024**3
  with (B/'logs'/f'{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failure {x}; no retry'
  return i,seed
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for i,seed in pool.map(task,todo):
   paths[f'{i}-{seed}']=str(O/rows[i]['tag']/f'B/B/SEED{seed}')
   (B/'state.json').write_text(json.dumps(dict(complete=len(paths),total=150,paths=paths),indent=2));print('COMPLETE',i,seed,len(paths),flush=True)
 summary=[]
 for i,s in enumerate(rows):
  powers=[];ds=[0,0,0];cs=[0,0,0]
  for seed in range(1,6):
   p=Path(paths[f'{i}-{seed}']);powers.append(audit(p,s,seed))
   with (p/'ue_stats.csv').open() as f:
    for r in csv.DictReader(f):
     j=[.05,.15,.3].index(float(r['delay_budget']));ds[j]+=int(float(r['dropped_packets']));cs[j]+=int(float(r['completed_packets']))
  drops=[100*d/(d+c) for d,c in zip(ds,cs)]
  summary.append(dict(**s,power_w=statistics.mean(powers),drop_pct=drops,dropped=ds,completed=cs,eligible=all(x<=.5 for x in drops)))
 selected=[]
 for pc in [1.8,2.1]:
  for ts in [1,2,3]:
   eligible=[r for r in summary if r['pc']==pc and r['training_seed']==ts and r['eligible']]
   selected.append(min(eligible,key=lambda r:(r['power_w'],r['checkpoint'])) if eligible else dict(pc=pc,training_seed=ts,eligible=False))
 (B/'summary.json').write_text(json.dumps(dict(results=summary,selected=selected,evaluation_seeds=[1,2,3,4,5],independent_test=False),indent=2))
 (B/'complete.json').write_text(json.dumps(dict(runs=150,new_runs=len(todo),reused=150-len(todo))))
 print('ALL_COMPLETE_B_LOWPC_SEED1TO5_150',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
