from pathlib import Path
import sys,json,subprocess,concurrent.futures,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_pc24_pc27_pc30_checkpoint_eval'
scope={'__name__':'extra_helpers','__file__':str(B/'runner.py')}
exec(compile((B/'runner.py').read_text(),str(B/'runner.py'),'exec'),scope)
rows=scope['specs']();state=json.loads((B/'state.json').read_text());tasks=[]
# Reverse order avoids all original workers, which traverse the queue forward.
# Four original children already running before task locks were introduced stay excluded.
for i in reversed(range(1,len(rows))):
 for seed in range(5,0,-1):
  if f'{i}-{seed}' not in state['paths']:tasks.append((i,seed))
logs=B/'extra_parallel8_logs';logs.mkdir(exist_ok=False)
def run(task):
 i,seed=task;s=rows[i];dest=scope['O']/s['tag']/f'B/B/SEED{seed}'
 if (dest/'verified.json').exists():scope['audit'](dest,s,seed);return
 assert shutil.disk_usage(R).free>4*1024**3
 with (logs/f'{i}-{seed}.log').open('x') as f:
  rc=subprocess.call([sys.executable,'-u',str(B/'runner.py'),'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert rc==0,f'Failed extra worker {task}; no retry'
 print('EXTRA_COMPLETE',i,seed,flush=True)
print('EXTRA_PARALLEL_START',len(tasks),8,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(run,t) for t in tasks]
 for f in concurrent.futures.as_completed(futures):f.result()
print('ALL_EXTRA_WORKERS_COMPLETE',flush=True)
