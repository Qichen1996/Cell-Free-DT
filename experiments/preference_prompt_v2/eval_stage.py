from common import *
import workflow as w
import subprocess,concurrent.futures,threading,time
CENTRE=-435.27698835441845
TARGETS=[-535.2769883544184,-485.27698835441845,-385.27698835441845,-335.27698835441845]
LOCK=threading.RLock()
def validate_models():
 for kind in ['PDT','DT']:
  d=MODELS/kind;c=json.loads((d/'complete.json').read_text());assert c['updates']==270000 and c['rounds']==1800 and sha(d/'model.pt')==c['sha256']
  from preference_model import PreferenceTransformer
  m=PreferenceTransformer(kind=='PDT');ck=torch.load(d/'model.pt',map_location='cpu',weights_only=False)
  assert ck['updates']==270000 and ck['preference_enabled']==(kind=='PDT');m.load_state_dict(ck['model_state_dict'],strict=True)
 return {k:sha(MODELS/k/'model.pt') for k in ['PDT','DT']}
def job(kind,pc,seed,case):
 validate_models();space()
 if case>=0:
  w.BASE=BASE/'dt_rtg_scan'/f'test{case+1:02d}';w.STATS=STATS/'dt_rtg_scan'/f'test{case+1:02d}'
 w.evaluate(kind,pc,seed)
def main():
 hashes=validate_models();space();assert (BASE/'training-complete.json').exists()
 assert not (STATS/'evaluation').exists() and not (STATS/'dt_rtg_scan').exists()
 for i,target in enumerate(TARGETS):save_new(BASE/'dt_rtg_scan'/f'test{i+1:02d}'/'rtg.json',dict(target=target,scale=1))
 tasks=[('PDT',pc,s,-1) for pc in PCS for s in range(6,11)]+[('DT',0.,s,-1) for s in range(6,11)]+[('DT',0.,s,i) for i in range(4) for s in range(6,11)]
 assert len(tasks)==60
 state=dict(status='running',jobs={},hashes=hashes,started=time.time())
 path=BASE/'evaluation-state.json';assert not path.exists()
 def save():
  with LOCK:
   tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
 save();logs=BASE/'eval_logs';logs.mkdir(exist_ok=False)
 def launch(t):
  kind,pc,seed,case=t;key=f'{kind}-{pc:.1f}-{seed}-{case}';space()
  with (logs/(key+'.log')).open('x') as f:
   p=subprocess.Popen([sys.executable,__file__,'job',kind,str(pc),str(seed),str(case)],stdout=f,stderr=subprocess.STDOUT)
   with LOCK:state['jobs'][key]=dict(pid=p.pid,status='running');save()
   code=p.wait()
  with LOCK:state['jobs'][key].update(status='complete' if code==0 else 'failed',exit_code=code);save()
  if code:raise RuntimeError(f'{key} failed {code}; no retry')
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(launch,t) for t in tasks]
  for f in concurrent.futures.as_completed(futures):f.result()
 state['status']='complete';save()
if __name__=='__main__':
 if len(sys.argv)>1:job(sys.argv[2],float(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
 else:main()
