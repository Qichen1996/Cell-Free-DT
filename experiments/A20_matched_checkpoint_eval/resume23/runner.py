"""Explicit user-authorized recovery of exactly 23 interrupted/pending runs."""
import importlib.util,json,sys,subprocess,shutil,concurrent.futures
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/A20_matched_checkpoint_eval'
REC=BASE/'resume23'
spec=importlib.util.spec_from_file_location('original',BASE/'runner.py')
orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
def inventory():
 missing=[];complete=0;partials=[]
 for i,s in enumerate(orig.jobs()):
  for seed in orig.SEEDS:
   d=orig.dest(s,seed)
   if (d/'verified.json').is_file():orig.check_existing(s,seed);complete+=1
   else:
    missing.append([i,seed])
    if d.exists():partials.append([i,seed,str(d)])
 assert complete==497 and missing==[[i,120] for i in range(2,25)],(complete,missing)
 return dict(complete=complete,missing=missing,partials=partials)
def main():
 inv=inventory();print('INVENTORY_OK',json.dumps(inv),flush=True)
 if len(sys.argv)>1 and sys.argv[1]=='inspect':return
 assert not (REC/'state.json').exists(),'No duplicate recovery'
 assert not (BASE/'complete.json').exists()
 assert shutil.disk_usage(ROOT).free>4*1024**3
 REC.mkdir(exist_ok=True);(REC/'logs').mkdir(exist_ok=True)
 orig.save(REC/'plan.json',dict(**inv,parallel=3,no_automatic_retry=True,original_runner_sha256=orig.sha(BASE/'runner.py')))
 shutil.copy2(BASE/'state.json',REC/'original_state.json')
 for i,seed,path in inv['partials']:
  d=Path(path);s=orig.jobs()[i]
  assert d.resolve()==orig.dest(s,seed).resolve() and d.resolve().is_relative_to((ROOT/'analysis/sim_stats/AB_new_training_checkpoint_eval').resolve())
  target=REC/'interrupted'/f'index{i}-seed{seed}';target.parent.mkdir(exist_ok=True)
  assert not target.exists();d.rename(target)
 state=dict(status='simulating',complete=0,total=23,jobs={})
 def update():
  t=REC/'state.tmp';t.write_text(json.dumps(state,indent=2));t.replace(REC/'state.json')
 def task(i,seed):
  assert shutil.disk_usage(ROOT).free>4*1024**3
  with (REC/'logs'/f'eval-{i}-seed{seed}.log').open('x') as f:
   rc=subprocess.call([sys.executable,'-u',str(BASE/'runner.py'),'job',str(i),str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Failed {i} {seed}; no automatic retry'
  orig.check_existing(orig.jobs()[i],seed)
  return i,seed
 update()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  it=iter(inv['missing']);pending={}
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
    state['jobs'][f'{i}-{seed}']='complete';state['complete']+=1;update();print('RESUME_CASE_COMPLETE',i,seed,flush=True);submit()
 # Reuse the unchanged original full-520 verification and aggregation implementation.
 source=(BASE/'runner.py').read_text()
 body=source[source.index(' summary=[]\n'):source.index("if __name__=='__main__':")]
 body='\n'.join(line[1:] if line.startswith(' ') else line for line in body.splitlines())
 original_state=json.loads((BASE/'state.json').read_text());original_state['new_complete']=395
 original_state['jobs'].update(state['jobs'])
 def update_original():
  t=BASE/'state.tmp';t.write_text(json.dumps(original_state,indent=2));t.replace(BASE/'state.json')
 scope=dict(vars(orig),specs=orig.jobs(),state=original_state,update=update_original)
 exec(compile(body,str(REC/'finalize_in_memory.py'),'exec'),scope)
 state['status']='complete';update();orig.save(REC/'complete.json',dict(resumed=23,total_verified=520))
 print('ALL_COMPLETE_A20_RESUME23_AND_520',flush=True)
if __name__=='__main__':main()
