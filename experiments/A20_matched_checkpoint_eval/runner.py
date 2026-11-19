"""26 fixed checkpoints on A, matched seeds101..120. No training/retries."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,subprocess,shutil,concurrent.futures,math,csv
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/A20_matched_checkpoint_eval'
OUT=ROOT/'analysis/sim_stats/A20_matched_checkpoint_eval'
SEEDS=list(range(101,121))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2)
def jobs():
 rows=[]
 for pc in [1.2,1.3,1.4,1.5,1.6]:
  tag=f'A_pc{pc:.1f}_trainseed1'
  for cp in [400,420,440,460,480]:
   rows.append(dict(tag=tag,pc=pc,checkpoint=cp,training_seed=1,source=str(ROOT/'experiments/AB_new_training_checkpoint_eval/models'/tag/f'eps{cp}'),output=str(ROOT/'analysis/sim_stats/AB_new_training_checkpoint_eval'/tag/f'eps{cp}')))
 rows.append(dict(tag='B_pc1.8_trainseed1_on_A',pc=1.8,checkpoint=230,training_seed=1,source=str(ROOT/'results/weights/B1000_checkpoint230_test/by_tag/eps230/pc_1.8'),output=str(OUT/'B_pc1.8_trainseed1_on_A/eps230')))
 return rows
def verify_model(s):
 src=Path(s['source']);m=json.loads((src/'manifest.json').read_text())
 for n,h in m.items():assert sha(src/n)==h
 if s['tag'].startswith('B_'):
  assert m['actor.pt']=='f9561880b46706fd583b2e64b501c23ebdbb9b4e0349c0a3785ace4e830a4cd5'
  assert m['critic.pt']=='c65e07ce75adc296a1dd0bd3776987056c1f359d9f65cff57562f90a0b8c732a'
 else:
  p=json.loads((src/'provenance.json').read_text())
  assert p['tag']==s['tag'] and p['training_seed']==1 and p['checkpoint']==s['checkpoint'] and p['pc']==s['pc']
 return m
def dest(s,seed):return Path(s['output'])/'A/A'/f'SEED{seed}'
def check_existing(s,seed):
 d=dest(s,seed);v=json.loads((d/'verified.json').read_text());m=verify_model(s)
 assert v['seed']==seed and v['checkpoint']==s['checkpoint'] and v['pc_weight']==s['pc'] and v['steps']==10080
 assert v['actor_sha256']==m['actor.pt'] and v['critic_sha256']==m['critic.pt']
 assert v['evaluation_weights']==[40,.5,.5,3] and v['obs_shape']==[16,110]
 assert v['environment']==dict(scenario='A',area=1000,aps=16,spacing_m=250,neighbor_km=.25,cluster=.98,min_ap=1)
 for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
  with (d/f'{n}.csv').open() as f:assert next(f,None) and next(f,None)
 with (d/'reward.csv').open() as f:vals=[float(x['pc_kw']) for x in csv.DictReader(f)]
 assert len(vals)==10080 and all(math.isfinite(x) for x in vals)
 assert not list(d.glob('*.pkl'))
def worker(index,seed,preflight=False):
 s=jobs()[index];m=verify_model(s)
 source=(ROOT/'experiments/IPPO_pc21_cp200_cross/runner.py').read_text().replace('pc_weight=2.1,checkpoint=200',f"pc_weight={s['pc']},checkpoint={s['checkpoint']}")
 scope={'__name__':'A20_worker','__file__':str(BASE/'runner.py')}
 exec(compile(source,str(BASE/'worker_in_memory.py'),'exec'),scope)
 scope.update(SOURCE=Path(s['source']),AH=m['actor.pt'],CH=m['critic.pt'],OUT=Path(s['output']) if not preflight else BASE/'preflight'/str(index))
 scope['worker']('A',seed,preflight)
def main():
 assert not (BASE/'state.json').exists(),'No automatic resume or duplicate launch'
 assert shutil.disk_usage(ROOT).free>6*1024**3
 specs=jobs();new=[];reuse=[]
 for i,s in enumerate(specs):
  m=verify_model(s)
  save(BASE/'models'/s['tag']/f"eps{s['checkpoint']}"/'provenance.json',dict(**s,manifest=m))
  for seed in SEEDS:
   if i<25 and seed<=105:check_existing(s,seed);reuse.append([i,seed])
   else:assert not dest(s,seed).exists();new.append([i,seed])
 assert len(new)==395 and len(reuse)==125
 save(BASE/'plan.json',dict(models=specs,seeds=SEEDS,new_runs=395,reused_runs=125,total_runs=520,parallel=3,scenario='A',evaluation_weights=[40,.5,.5,3],steps=10080,auto_retry=False,save_traj_pkl=False,old_B_seeds_1_to_5='reference only, excluded'))
 for i,s in enumerate(specs):
  with (BASE/'logs'/f'preflight-{i}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',str(i),'101'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Preflight failed {i}'
  print('PREFLIGHT_OK',i,s['tag'],s['checkpoint'],flush=True)
 state=dict(status='simulating',new_complete=0,new_total=395,reused=125,total=520,jobs={})
 def update():
  t=BASE/'state.tmp';t.write_text(json.dumps(state,indent=2));t.replace(BASE/'state.json')
 def task(i,seed):
  assert shutil.disk_usage(ROOT).free>4*1024**3,'Disk reserve below4GiB'
  with (BASE/'logs'/f'eval-{i}-seed{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(i),str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {i} {seed}; no retry'
  return i,seed
 update()
 # B counterpoint first, then the A checkpoints, interleaved by seed.
 new.sort(key=lambda x:(x[1],0 if x[0]==25 else x[0]+1))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  pending={};it=iter(new)
  def submit():
   try:i,seed=next(it)
   except StopIteration:return
   pending[pool.submit(task,i,seed)]=(i,seed)
  for _ in range(3):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    i,seed=pending.pop(f)
    try:f.result()
    except Exception:
     state['status']='failed';state['failed_job']=[i,seed];update();raise
    state['new_complete']+=1;state['jobs'][f'{i}-{seed}']='complete';update();print('CASE_COMPLETE',i,seed,flush=True);submit()
 summary=[]
 for s in specs:
  counts=[[0.,0.] for _ in range(3)];powers=[]
  for seed in SEEDS:
   check_existing(s,seed);d=dest(s,seed)
   with (d/'reward.csv').open() as f:powers.extend(float(x['pc_kw']) for x in csv.DictReader(f))
   with (d/'ue_stats.csv').open() as f:
    for x in csv.DictReader(f):
     j=[.05,.15,.3].index(float(x['delay_budget']))
     for k,n in enumerate(['dropped_packets','completed_packets']):counts[j][k]+=float(x[n])
  summary.append(dict(**s,pc_kw=math.fsum(powers)/len(powers),packet_counts=counts,drop_percent=[100*d/(d+c) for d,c in counts]))
 save(BASE/'summary.json',dict(results=summary,seeds=SEEDS,method='pooled counts; reward.pc_kw mean'))
 state['status']='complete';update();save(BASE/'complete.json',dict(new_runs=395,reused_runs=125,total_runs=520,groups=26))
 print('ALL_COMPLETE_A20_MATCHED_520',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
