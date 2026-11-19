from pathlib import Path
import subprocess,sys,json,hashlib,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_pc12_pc15_seeds23';E=R/'experiments/B_pc12_pc15_seeds23_checkpoint_eval'
procs=[]
for pc in [1.2,1.5]:
 f=(B/f'pc_{pc}/workflow.log').open('a')
 procs.append((pc,subprocess.Popen([sys.executable,'-u',str(B/f'pc_{pc}/runner.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT),f))
for pc,p,f in procs:
 rc=p.wait();f.close();assert rc==0,f'Training failed weight{pc}; no retry'
import yaml
keys=list(json.loads((R/'results/weights/B1000_pc_sweep/pc_1.8/architecture.json').read_text()))
for pc in [1.2,1.5]:
 for seed in [2,3]:
  tr=R/f'results/weights/B_pc12_pc15_seeds23/pc_{pc}/seed{seed}/pc_{pc}'
  c=json.loads((tr/'complete.json').read_text());src=Path(c['actor_path']).parent
  cfg={k:v['value'] for k,v in yaml.safe_load((src/'config.yaml').read_text()).items() if isinstance(v,dict) and 'value' in v}
  assert cfg['seed']==seed and cfg['w_pc']==pc and [cfg['w_str'],cfg['w_sen'],cfg['w_tol']]==[80,.2,.2]
  for cp in [180,200,220,230,240]:
   dest=E/f'models/B_pc{pc}_trainseed{seed}/eps{cp}';dest.mkdir(parents=True);pair={}
   for kind in ['actor','critic']:
    source=src/f'{kind}_eps{cp}.pt';target=dest/f'{kind}.pt';shutil.copy2(source,target)
    h=hashlib.sha256(target.read_bytes()).hexdigest();assert h==hashlib.sha256(source.read_bytes()).hexdigest();pair[target.name]=h
   (dest/'manifest.json').write_text(json.dumps(pair,indent=2))
   (dest/'architecture.json').write_text(json.dumps({k:cfg[k] for k in keys},indent=2))
   (dest/'provenance.json').write_text(json.dumps(dict(training_seed=seed,pc=pc,checkpoint=cp,training_dir=str(tr),original_model_dir=str(src),manifest=pair),indent=2))
(B/'complete.json').write_text(json.dumps(dict(models=4,rounds_each=250,steps_each=2520000)))
rc=subprocess.call([sys.executable,'-u',str(E/'runner.py')],cwd=R)
assert rc==0,'Evaluation failed; no retry'
print('ALL_COMPLETE_B_PC12_PC15_TRAIN_EVAL',flush=True)
