from pathlib import Path
import os,json,sys,subprocess,concurrent.futures,importlib.util,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/PDT_refresh150_single200_A_seed20_v1'
def main():
 plan=json.loads((B/'resume_plan.json').read_text());state=json.loads((B/'state.json').read_text());state.update(phase='simulating',complete=plan['preserved']);state['jobs']=[]
 def save():
  p=B/'state.tmp';p.write_text(json.dumps(state));p.replace(B/'state.json')
 def job(pair):
  i,s=pair;assert shutil.disk_usage(R).free>3*1024**3
  with (B/'logs'/f'resume-eval-{i}-{s}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',str(B/'runner.py'),'job',str(i),str(s)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,(i,s,rc);return pair
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  it=iter(plan['jobs']);pending={}
  def submit():
   try:p=next(it)
   except StopIteration:return
   pending[pool.submit(job,p)]=p
  for _ in range(6):submit()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    pair=pending.pop(f)
    try:f.result()
    except Exception:state.update(phase='failed',failed=pair);save();raise
    state['complete']+=1;state['jobs'].append(pair);save();print('COMPLETE',state['complete'],flush=True);submit()
 state['phase']='complete';save();assert state['complete']==300
 (B/'complete.json').write_text(json.dumps(dict(new_runs=300,reused_runs=100,total=400)));print('ALL_COMPLETE_PDT_TWO_MODELS_A_SEED20_400',flush=True)
if __name__=='__main__':main()
