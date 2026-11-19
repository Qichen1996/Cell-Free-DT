"""Remote launcher: fixed RTG, only prompt varies; no training or baseline writes."""
import os, sys, json, subprocess, concurrent.futures, hashlib
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/selected_seven_pc33_prompt_B'
STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33_prompt_B'
SOURCE=ROOT/'experiments/selected_seven_pc33_eval/eval_pc33.py'
PCS=[.6,.9,1.2,1.5,2.1,2.4,2.7,3.,3.3]
def main():
 if len(sys.argv)==3:
  pc=float(sys.argv[1]);seed=int(sys.argv[2]);assert pc in PCS and seed in range(101,106)
  source=SOURCE.read_text().replace("STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33'", "STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33_prompt_B'/"+repr('pc_'+str(pc)))
  source=source.replace('1.8/3.3',repr(pc)+'/3.3').replace('pc_preference=1.8,','pc_preference='+repr(pc)+',')
  scope={'__name__':'prompt_sweep_worker','__file__':str(SOURCE)}
  exec(compile(source,str(SOURCE),'exec'),scope)
  scope['evaluate']('B',seed)
  return
 assert not (BASE/'plan.json').exists() and not STATS.exists()
 BASE.mkdir(parents=True,exist_ok=True);(BASE/'logs').mkdir()
 plan=dict(pc_preferences=PCS,prompts=[p/3.3 for p in PCS],seeds=list(range(101,106)),steps=10080,parallel=3,rtg=-426.00145441593037,baseline_pc=1.8,baseline=str(ROOT/'analysis/sim_stats/PDT_selected_seven_pc33/B/B'),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
 (BASE/'plan.json').write_text(json.dumps(plan,indent=2))
 def job(task):
  pc,seed=task
  with (BASE/'logs'/f'pc{pc}-seed{seed}.log').open('x') as f:
   rc=subprocess.call([sys.executable,'-u',str(Path(__file__)),str(pc),str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'FAILED pc={pc} seed={seed} rc={rc}'
  print('CASE_COMPLETE',pc,seed,flush=True)
 tasks=[(pc,seed) for seed in range(101,106) for pc in PCS]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(job,tasks))
 (BASE/'complete.json').write_text(json.dumps(dict(runs=len(tasks))))
 print('ALL_COMPLETE_PC33_B_PROMPT_45',flush=True)
if __name__=='__main__':main()
