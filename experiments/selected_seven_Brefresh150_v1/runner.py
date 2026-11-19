import os
os.environ.update(OMP_NUM_THREADS='5',MKL_NUM_THREADS='5',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,sys,subprocess,hashlib
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/selected_seven_Brefresh150_v1';M=R/'dt_results/selected_seven_Brefresh150_v1';sys.path.insert(0,str(R))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scope_build():
 t=R/'experiments/selected_seven_pc33_v1/train_mixed.py';s=t.read_text().replace("BASE=ROOT/'experiments/selected_seven_pc33_v1';MODELS=ROOT/'dt_results/selected_seven_pc33_v1'",f'BASE=Path({str(B)!r});MODELS=Path({str(M)!r})').replace('382','517').replace('4772','7012')
 scope={'__name__':'refresh_training','__file__':str(B/'runner.py')};exec(compile(s,str(t),'exec'),scope)
 original=scope['entries'];old=original();rows=[x for x in old if x['env']!='B']
 baseline=json.loads((R/'experiments/selected_seven_baseline_aug_v1/baseline-dataset-manifest.json').read_text())
 for x in baseline['files']:
  p=x['environment'];rows.append(dict(env=x['env'],path=x['path'],traffic={'A':0,'B':1,'C':2}[p['scenario']],aps=p['aps'],area=p['area'],spacing=p['spacing_m'],cluster=p['cluster'],episodes=5,preference=None,expected_sha=x['sha256'],baseline_policy=x['policy']))
 man=json.loads((R/'experiments/tradeoff_selected123_collect_cross/dataset-manifest.json').read_text())
 for f in man['files']:
  m=f['model'];rows.append(dict(env='B',path=f['path'],traffic=1,aps=16,area=1000,spacing=250,cluster=.9,episodes=1,preference=m['pc'],expected_sha=f['sha256'],training_seed=m['training_seed'],checkpoint=m['checkpoint'],collection_seed=f['seed']))
 assert sum(x['episodes'] for x in rows)==517 and sum(x['episodes']*x['aps'] for x in rows)==7012
 assert len([x for x in rows if x['env']=='B' and x.get('baseline_policy') is None])==150
 scope['entries']=lambda:rows
 return scope,rows,[x for x in old if x['env']=='B']
if len(sys.argv)>1:
 scope,rows,removed=scope_build();scope['run'](sys.argv[1])
else:
 assert not M.exists() and not (B/'plan.json').exists()
 scope,rows,removed=scope_build()
 for x in rows:
  assert Path(x['path']).is_file()
  if 'expected_sha' in x:assert sha(x['path'])==x['expected_sha']
 (B/'plan.json').write_text(json.dumps(dict(models=['PDT','DT'],from_scratch=True,seed=1,rounds=1800,updates_each=270000,batch=256,context=8,prompt='PC/3.3 for PDT; no prompt for DT',episodes=517,ap_trajectories=7012,replacement='B PPO85 replaced by new150; retain all rule data and other environments',removed_from_training=removed,files=rows),indent=2))
 for kind in ['PDT','DT']:
  (B/'state.json').write_text(json.dumps(dict(phase='training',model=kind)))
  with (B/(kind+'.log')).open('x') as log:rc=subprocess.call([sys.executable,'-u',__file__,kind],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  assert rc==0,f'{kind} failed; no retry'
  d=json.loads((M/kind/'complete.json').read_text());assert d['updates']==270000 and sha(M/kind/'model.pt')==d['sha256']
 (B/'complete.json').write_text(json.dumps(dict(models=['PDT','DT'],updates_each=270000,episodes=517)))
 (B/'state.json').write_text(json.dumps(dict(phase='complete')))
 print('ALL_COMPLETE_BREFRESH150_PDT_DT',flush=True)
