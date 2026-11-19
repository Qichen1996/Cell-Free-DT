import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,csv,math,statistics,subprocess,concurrent.futures,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_pc33_early_checkpoint_eval';O=R/'analysis/sim_stats/B_pc33_early_checkpoint_eval'
def specs():
 return json.loads((B/'specs.json').read_text())

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

def audit_old(p,s,seed):
 m=manifest(s);v=json.loads((p/'verified.json').read_text())
 assert v['seed']==seed and v['pc_weight']==s['pc'] and v['actor_sha256']==m['actor.pt']
 assert v['evaluation_weights']==[40,.5,.5,3] and v['complete']
 assert [v[k] for k in ['spacing_m','neighbor_km','cluster']]==[250,.25,.9]
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:assert (p/f'{n}.csv').stat().st_size>0
 with (p/'reward.csv').open() as f:rr=list(csv.DictReader(f))
 assert len(rr)==10080 and all(math.isfinite(float(r[k])) for r in rr for k in ['pc_kw','reward'])
 return statistics.mean(float(r['pc_kw']) for r in rr)*1000
def main():
 assert not (B/'plan.json').exists() and not O.exists()
 rows=specs();paths={};todo=[]
 for i,s in enumerate(rows):
  manifest(s)
  for seed in range(1,6):
   oldexp='B1000_checkpoint230_test' if s['checkpoint']==230 else 'B1000_checkpoint_test'
   p=R/f"analysis/sim_stats/{oldexp}/eps{s['checkpoint']}/pc_{s['pc']}/B/SEED{seed}"
   if s['training_seed']==1 and (p/'verified.json').is_file():
    audit_old(p,s,seed);paths[f'{i}-{seed}']=str(p)
   else:todo.append((i,seed))
 plan=dict(models=rows,evaluation_seeds=[1,2,3,4,5],steps=10080,total=60,new_runs=len(todo),reused_runs=len(paths),parallel=8,preflight_groups=12,auto_retry=False,save_traj_pkl=False,evaluation_weights=[40,.5,.5,3])
 (B/'plan.json').write_text(json.dumps(plan,indent=2));print('PLAN',len(todo),'NEW',len(paths),'REUSED',flush=True)
 def save():
  t=B/'state.tmp';t.write_text(json.dumps(dict(complete=len(paths),total=60,paths=paths),indent=2));t.replace(B/'state.json')
 save()
 def pre(i):
  with (B/'logs'/f'preflight-{i}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',str(i),'1'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Preflight {i} failed'
  print('PREFLIGHT_OK',i,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(pre,range(len(rows))))
 print('ALL_12_PREFLIGHTS_OK',flush=True)
 def task(x):
  i,seed=x;assert shutil.disk_usage(R).free>4*1024**3
  with (B/'logs'/f'{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failure {x}; no automatic retry'
  return i,seed
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(task,x) for x in todo]
  for future in concurrent.futures.as_completed(futures):
   i,seed=future.result();paths[f'{i}-{seed}']=str(O/rows[i]['tag']/f'B/B/SEED{seed}');save();print('COMPLETE',i,seed,len(paths),flush=True)
 summary=[]
 for i,s in enumerate(rows):
  powers=[];ds=[0]*3;cs=[0]*3
  for seed in range(1,6):
   p=Path(paths[f'{i}-{seed}'])
   powers.append(audit(p,s,seed) if p.is_relative_to(O) else audit_old(p,s,seed))
   with (p/'ue_stats.csv').open() as f:
    for r in csv.DictReader(f):
     j=[.05,.15,.3].index(float(r['delay_budget']));ds[j]+=int(float(r['dropped_packets']));cs[j]+=int(float(r['completed_packets']))
  drops=[100*d/(d+c) for d,c in zip(ds,cs)]
  summary.append(dict(**s,power_w=statistics.mean(powers),drop_pct=drops,dropped=ds,completed=cs,eligible=all(x<=.5 for x in drops)))
 (B/'summary.json').write_text(json.dumps(dict(results=summary,evaluation_seeds=[1,2,3,4,5]),indent=2))
 (B/'complete.json').write_text(json.dumps(dict(runs=60,new_runs=len(todo),reused=60-len(todo))))
 print('ALL_COMPLETE_B_PC33_EARLY_EVAL_60',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
