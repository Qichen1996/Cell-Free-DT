import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,subprocess,concurrent.futures,csv,importlib.util
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(R))
P=R/'experiments/replay_PPO_A_packet_diag.py'
sp=importlib.util.spec_from_file_location('diag',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
B=m.BASE;RES=B/'resume_20261024'
assert not RES.exists();(RES/'logs').mkdir(parents=True)
done=[]
for i,s in enumerate(m.specs()):
 d=m.OUT/s['tag']/'A/A/SEED101';v=json.loads((d/'verified.json').read_text());assert v['seed']==101 and v['steps']==10080
 record=B/'records'/s['tag']/'SEED101';drops=[json.loads(x) for x in (record/'drops.jsonl').read_text().splitlines()]
 counts=[0,0,0]
 for x in drops:counts[x['service']]+=1
 ue=list(csv.DictReader(open(d/'ue_stats.csv')))
 assert len(list(csv.DictReader(open(d/'reward.csv'))))==10080
 original=Path(s['output'])/'A/A/SEED101'
 matches={n:hashlib.sha256((d/f'{n}.csv').read_bytes()).hexdigest()==hashlib.sha256((original/f'{n}.csv').read_bytes()).hexdigest() for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']}
 m.save(record/'complete.json',dict(recovered_completed_run=True,seed=101,tag=s['tag'],dropped_packet_records=counts,original_csv_byte_match=matches))
 done.append([i,101])
print('REUSED_COMPLETED_DIAGNOSTICS',2,flush=True)
def task(pair):
 i,seed=pair
 with (RES/'logs'/f'{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',str(P),'worker',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert rc==0,f'Failed {pair}; no retry'
 return pair
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for pair in pool.map(task,[(i,s) for s in range(102,121) for i in range(2)]):
  done.append(pair);t=B/'state.tmp';t.write_text(json.dumps(dict(complete=len(done),total=40,jobs=done)));t.replace(B/'state.json');print('COMPLETE',pair,flush=True)
m.save(B/'complete.json',dict(runs=40,jobs=done));print('ALL_COMPLETE_PPO_A_PACKET_DIAG_40',flush=True)
