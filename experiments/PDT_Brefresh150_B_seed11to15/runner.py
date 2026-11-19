import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,subprocess,concurrent.futures,runpy,csv,statistics,hashlib
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/PDT_Brefresh150_B_seed11to15';O=R/'analysis/sim_stats/PDT_Brefresh150_B_seed11to15';H='bf4dd6575b1ebd6c600915fb33f65027665a8256cd79e3e1c598dda53da30f39'
def worker(seed,pre=False):
 scope=runpy.run_path(str(R/'experiments/PDT_Brefresh150_seven_prompt_v1/runner.py'),run_name='helper')
 g=scope['worker'].__globals__;g.update(B=B,O=O,M=B/'model.pt',specs=lambda:[dict(env='B',pc=1.8,tag='baseline')]);scope['worker'](0,seed,pre)
def main():
 assert not (B/'plan.json').exists() and not O.exists()
 assert hashlib.sha256((B/'model.pt').read_bytes()).hexdigest()==H
 (B/'plan.json').write_text(json.dumps(dict(model_sha256=H,environment='B',prompt=1.8/3.3,pc_preference=1.8,seeds=[11,12,13,14,15],steps=10080,parallel=3,reward=[40,.5,.5,3],rtg=-426.00145441593037,comparison='Training collection seeds; diagnostic, not held-out generalization',auto_retry=False),indent=2))
 def job(seed,pre=False):
  with (B/'logs'/f'{"pre" if pre else "eval"}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'pre' if pre else 'job',str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed seed{seed}; no retry'
 job(11,True);print('PREFLIGHT_OK',flush=True);done=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for seed in pool.map(lambda seed:(job(seed),seed)[1],range(11,16)):
   done.append(seed);(B/'state.json').write_text(json.dumps(dict(complete=len(done),total=5,seeds=done)));print('SEED_COMPLETE',seed,flush=True)
 d=[0]*3;c=[0]*3;powers=[];per=[]
 for seed in range(11,16):
  p=O/f'baseline/B/B/SEED{seed}';v=json.loads((p/'verified.json').read_text());assert v['model_sha256']==H and v['seed']==seed
  with (p/'reward.csv').open() as f:rows=list(csv.DictReader(f))
  assert len(rows)==10080;powers.append(statistics.mean(float(x['pc_kw']) for x in rows)*1000)
  sd=[0]*3;sc=[0]*3
  with (p/'ue_stats.csv').open() as f:
   for x in csv.DictReader(f):
    j=[.05,.15,.3].index(float(x['delay_budget']));sd[j]+=int(float(x['dropped_packets']));sc[j]+=int(float(x['completed_packets']))
  d=[a+b for a,b in zip(d,sd)];c=[a+b for a,b in zip(c,sc)];per.append(dict(seed=seed,power_w=powers[-1],dropped=sd,completed=sc))
 (B/'summary.json').write_text(json.dumps(dict(power_w=statistics.mean(powers),drop_pct=[100*a/(a+b) for a,b in zip(d,c)],dropped=d,completed=c,seeds=per),indent=2));(B/'complete.json').write_text(json.dumps(dict(runs=5,model_sha256=H)));print('ALL_COMPLETE_PDT_B_SEED11TO15',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),sys.argv[1]=='pre')
 else:main()
