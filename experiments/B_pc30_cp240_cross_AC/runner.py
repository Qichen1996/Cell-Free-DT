import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,subprocess,concurrent.futures,csv,math,statistics,shutil,hashlib
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_pc30_cp240_cross_AC';O=R/'analysis/sim_stats/B_pc30_cp240_cross_AC'
def worker(ts,name,seed,pre=False):
 spec=next(s for s in json.loads((B/'specs.json').read_text()) if s['training_seed']==ts)
 source=(R/'experiments/IPPO_pc21_cp200_cross/runner.py').read_text().replace('pc_weight=2.1,checkpoint=200',f'pc_weight=3.0,checkpoint=240,training_seed={ts}')
 scope={'__name__':'cross_worker','__file__':__file__};exec(compile(source,str(B/'worker_memory.py'),'exec'),scope)
 scope.update(BASE=B,SOURCE=Path(spec['source']),AH=spec['manifest']['actor.pt'],CH=spec['manifest']['critic.pt'],OUT=O/f'trainseed{ts}')
 scope['worker'](name,seed,pre)
def audit(ts,name,seed):
 p=O/f'trainseed{ts}/{name}/{name}/SEED{seed}';v=json.loads((p/'verified.json').read_text())
 spec=next(s for s in json.loads((B/'specs.json').read_text()) if s['training_seed']==ts)
 assert v['seed']==seed and v['training_seed']==ts and v['checkpoint']==240 and v['pc_weight']==3.0 and v['steps']==10080
 assert v['actor_sha256']==spec['manifest']['actor.pt'] and v['critic_sha256']==spec['manifest']['critic.pt']
 assert v['evaluation_weights']==[40,.5,.5,3]
 assert v['environment']==dict(scenario=name,area=1000,aps=16,spacing_m=250.,neighbor_km=.25,cluster=.98 if name=='A' else .9,min_ap=1)
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:assert (p/f'{n}.csv').stat().st_size>0
 with (p/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080 and all(math.isfinite(float(r[k])) for r in rows for k in ['pc_kw','reward'])
 return p,statistics.mean(float(r['pc_kw']) for r in rows)*1000
def main():
 assert not O.exists() and not (B/'plan.json').exists()
 jobs=[(ts,name,seed) for seed in range(1,6) for ts in [1,2,3] for name in ['A','C']]
 (B/'plan.json').write_text(json.dumps(dict(pc=3.,checkpoint=240,training_seeds=[1,2,3],evaluation_seeds=[1,2,3,4,5],scenarios=['A','C'],steps=10080,total=30,parallel=4,evaluation_weights=[40,.5,.5,3],save_traj_pkl=False,auto_retry=False),indent=2))
 for ts in [1,2,3]:
  for name in ['A','C']:
   with (B/'logs'/f'preflight-{ts}-{name}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',str(ts),name,'1'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
   assert rc==0,f'Preflight failed {ts} {name}'
   print('PREFLIGHT_OK',ts,name,flush=True)
 print('ALL_6_PREFLIGHTS_OK',flush=True)
 done=[]
 def save():
  t=B/'state.tmp';t.write_text(json.dumps(dict(complete=len(done),total=30,done=done),indent=2));t.replace(B/'state.json')
 save()
 def job(task):
  ts,name,seed=task;assert shutil.disk_usage(R).free>4*1024**3
  with (B/'logs'/f'{ts}-{name}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(ts),name,str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {task}; no automatic retry'
  audit(ts,name,seed);return task
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for future in concurrent.futures.as_completed([pool.submit(job,t) for t in jobs]):
   task=future.result();done.append(task);save();print('COMPLETE',task,len(done),flush=True)
 summary=[]
 for ts in [1,2,3]:
  for name in ['A','C']:
   powers=[];ds=[0]*3;cs=[0]*3
   for seed in range(1,6):
    p,power=audit(ts,name,seed);powers.append(power)
    with (p/'ue_stats.csv').open() as f:
     for r in csv.DictReader(f):
      j=[.05,.15,.3].index(float(r['delay_budget']));ds[j]+=int(float(r['dropped_packets']));cs[j]+=int(float(r['completed_packets']))
   summary.append(dict(training_seed=ts,scenario=name,pc=3.,checkpoint=240,power_w=statistics.mean(powers),drop_pct=[100*d/(d+c) for d,c in zip(ds,cs)],dropped=ds,completed=cs))
 (B/'summary.json').write_text(json.dumps(dict(results=summary),indent=2));(B/'complete.json').write_text(json.dumps(dict(runs=30)))
 print('ALL_COMPLETE_B_PC30_CP240_CROSS_AC_30',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),sys.argv[1]=='preflight')
 else:main()
