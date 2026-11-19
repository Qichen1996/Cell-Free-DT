"""Three lower actual RTG targets for the trained prompt-free DT."""
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,subprocess,concurrent.futures,hashlib
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/selected_seven_pc33_DT_lower_B'
STATS=ROOT/'analysis/sim_stats/selected_seven_pc33_DT_lower_B'
SOURCE=ROOT/'experiments/selected_seven_pc33_eval/eval_pc33.py'
HASH='239d7f9fbec9285dbfa2c8cf2aee0b0df09d94560e2a451405a9824a577f36f7'
TARGETS=[-516.0014544159304,-546.0014544159304,-576.0014544159304]
def worker(index,seed):
 assert index in range(3) and seed in range(101,106)
 source=SOURCE.read_text()
 replacements={
  "STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33'":'STATS=Path('+repr(str(STATS/f'test{index+1:02}'))+')',
  "target=json.loads((BASE/'rtg.json').read_text())['targets'][name]":'target='+repr(TARGETS[index]),
  'dt_results/selected_seven_pc33_v1/PDT/model.pt':'dt_results/selected_seven_pc33_v1/DT/model.pt',
  '738f006d0ff4e3c1ed665509aa0613341c00ef8aa7ca62a9f127e4ed3c5118b6':HASH,
  'model=training.Model(True)':'model=training.Model(False)',
  "'PROMPT',1.8/3.3,":"'PROMPT_ENABLED',False,",
  'pc_preference=1.8,prompt=1.8/3.3,':'prompt_enabled=False,'
 }
 for old,new in replacements.items():
  assert old in source,old
  source=source.replace(old,new)
 source=source.replace("assert checkpoint['prompt_normalization']", "assert checkpoint['prompt_enabled'] is False\n assert checkpoint['prompt_normalization']")
 scope={'__name__':'dt_lower_worker','__file__':str(SOURCE)}
 exec(compile(source,str(SOURCE),'exec'),scope)
 scope['evaluate']('B',seed)
def main():
 if len(sys.argv)==3:return worker(int(sys.argv[1]),int(sys.argv[2]))
 assert not (BASE/'plan.json').exists() and not STATS.exists()
 (BASE/'logs').mkdir(parents=True)
 tasks=[(i,s) for s in range(101,106) for i in range(3)]
 plan=dict(rtg_targets=TARGETS,seeds=list(range(101,106)),steps=10080,parallel=3,model=str(ROOT/'dt_results/selected_seven_pc33_v1/DT/model.pt'),model_sha256=HASH,prompt_enabled=False,environment=dict(scenario='B',area=1000,aps=16,spacing_m=250,neighbor_km=.25,cluster=.9,min_ap=1),reward=[40,.5,.5,3],source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),tasks=tasks)
 with (BASE/'plan.json').open('x') as f:json.dump(plan,f,indent=2)
 def job(task):
  i,s=task
  with (BASE/'logs'/f'test{i+1:02}-seed{s}.log').open('x') as f:
   rc=subprocess.call([sys.executable,'-u',str(Path(__file__)),str(i),str(s)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'FAILED {task} rc={rc}'
  print('CASE_COMPLETE',i,s,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(job,tasks))
 with (BASE/'complete.json').open('x') as f:json.dump(dict(runs=15),f)
 print('ALL_COMPLETE_DT_LOWER_RTG_B_15',flush=True)
if __name__=='__main__':main()
