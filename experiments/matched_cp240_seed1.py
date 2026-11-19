import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,csv,math,subprocess,concurrent.futures,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_matched_cp240_seed1';O=R/'analysis/sim_stats/B_matched_cp240_seed1'
def audit(p,seed,pc,src):
 v=json.loads((p/'verified.json').read_text());m=json.loads((src/'manifest.json').read_text())
 assert v['seed']==seed and v['checkpoint']==240 and v['pc_weight']==pc
 assert v['actor_sha256']==m['actor.pt'] and v['critic_sha256']==m['critic.pt']
 assert v['evaluation_weights']==[40,.5,.5,3] and v['environment']==dict(scenario='B',area=1000,aps=16,spacing_m=250.,neighbor_km=.25,cluster=.9,min_ap=1)
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:assert (p/f'{n}.csv').stat().st_size>0
 with (p/'reward.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==10080 and all(math.isfinite(float(r['pc_kw'])) for r in rows)
def worker(i,seed,pre=False):
 pc=[1.8,2.1][i];src=R/f'results/weights/B1000_checkpoint_test/by_tag/eps240/pc_{pc}'
 m=json.loads((src/'manifest.json').read_text())
 for n,h in m.items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h
 code=(R/'experiments/IPPO_pc21_cp200_cross/runner.py').read_text().replace('pc_weight=2.1,checkpoint=200',f'pc_weight={pc},checkpoint=240')
 scope={'__name__':'matched_worker','__file__':__file__};exec(compile(code,str(B/'worker_in_memory.py'),'exec'),scope)
 scope.update(SOURCE=src,AH=m['actor.pt'],CH=m['critic.pt'],OUT=O/f'pc_{pc}_trainseed1')
 scope['worker']('B',seed,pre)
 if not pre:audit(O/f'pc_{pc}_trainseed1/B/B/SEED{seed}',seed,pc,src)
def main():
 assert not B.exists() and not O.exists(),'No duplicate launch'
 (B/'logs').mkdir(parents=True)
 # Independently check the ten existing PC1.8 CP240 evaluations before reuse.
 for ts in [2,3]:
  src=R/f'experiments/AB_new_training_checkpoint_eval/models/B_pc1.8_trainseed{ts}/eps240'
  for seed in range(101,106):audit(R/f'analysis/sim_stats/AB_new_training_checkpoint_eval/B_pc1.8_trainseed{ts}/eps240/B/B/SEED{seed}',seed,1.8,src)
 plan=dict(weights=[1.8,2.1],checkpoint=240,training_seeds=[1,2,3],evaluation_seeds=list(range(101,106)),new_runs=10,reused_pc18_runs=10,pc21_seed23_runs='existing training/evaluation workflow',steps=10080,parallel=2,auto_retry=False,save_traj_pkl=False)
 (B/'plan.json').write_text(json.dumps(plan,indent=2))
 for i in range(2):
  with (B/'logs'/f'preflight-{i}.log').open('x') as f:r=subprocess.call([sys.executable,'-u',__file__,'preflight',str(i),'101'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert r==0
  print('PREFLIGHT_OK',i,flush=True)
 def task(x):
  i,s=x;assert shutil.disk_usage(R).free>3*1024**3
  with (B/'logs'/f'{i}-{s}.log').open('x') as f:r=subprocess.call([sys.executable,'-u',__file__,'job',str(i),str(s)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert r==0,f'Failed {x}; no retry'
  return x
 done=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for x in pool.map(task,[(i,s) for s in range(101,106) for i in range(2)]):
   done.append(x);(B/'state.json').write_text(json.dumps(dict(complete=len(done),total=10,jobs=done)));print('COMPLETE',x,flush=True)
 (B/'complete.json').write_text(json.dumps(dict(runs=10)))
 print('ALL_COMPLETE_B_MATCHED_CP240_SEED1',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
