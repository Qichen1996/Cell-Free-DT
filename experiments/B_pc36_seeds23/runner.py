"""From-scratch B PC3.6 repeats using the original sweep hyperparameters."""
import sys,json,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/B_pc36_seeds23'
OUT=ROOT/'results/weights/B_pc36_seeds23'
def worker(seed,preflight=False):
 source=(ROOT/'experiments/B1000_pc_next/worker.py').read_text()
 source=source.replace("OUTPUT=PROJECT/'results/weights/B1000_pc_sweep_next'",f"OUTPUT=PROJECT/'results/weights/B_pc36_seeds23/seed{seed}'")
 source=source.replace("scenario='B',area_size=nc.areaSize", "scenario='B',area_size=nc.areaSize")
 source=source.replace("seed=1,scenario='B'",f"seed={seed},scenario='B'")
 source=source.replace("use_wandb=True,model_dir=None,use_eval=False,seed=1,",f"use_wandb=True,model_dir=None,use_eval=False,seed={seed},")
 source=source.replace("'seed':1,'rounds'",f"'seed':{seed},'rounds'")
 anchor='parser_factories=[train.get_config,get_mappo_config]'
 assert anchor in source
 source=source.replace(anchor,anchor+"\n    reference=json.loads((PROJECT/'results/weights/B1000_pc_sweep_next/pc_3.6/experiment.json').read_text())\n    aa,remaining=train.get_config().parse_known_args(reference['argv'])\n    bb,_=get_mappo_config().parse_known_args(remaining)\n    cfg.update(vars(aa));cfg.update(vars(bb))")
 source=source.replace("f'B1000_pc{pc:.1f}'",f"f'B1000_pc{{pc:.1f}}_seed{seed}'")
 source=source.replace("'Two additional weights3.3/3.6 trained from scratch with matched250round budget.'", "'User-authorized PC3.6 repeat from scratch, training seeds2/3, same250round2520000step budget as original seed1.'")
 scope={'__name__':'b_pc18_seed_repeat','__file__':str(BASE/'runner.py')}
 exec(compile(source,str(BASE/'worker_in_memory.py'),'exec'),scope)
 scope['run'](3.6,False,preflight)
def main():
 assert not OUT.exists();OUT.mkdir()
 for seed in [2,3]:
  (OUT/f'seed{seed}').mkdir()
  r=subprocess.run([sys.executable,'-u',__file__,'preflight',str(seed)],cwd=ROOT,capture_output=True,text=True)
  with (BASE/'logs'/f'preflight-seed{seed}.log').open('x') as f:f.write(r.stdout+r.stderr)
  assert r.returncode==0,r.stdout+r.stderr
  print('PREFLIGHT_OK',seed,flush=True)
 with (BASE/'plan.json').open('x') as f:json.dump(dict(scenario='B',weights=[80,.2,.2,3.6],seeds=[2,3],steps=2520000,rounds=250,parallel=2,from_scratch=True,cluster=.9,spacing_m=250,neighbor_km=.25,reference=str(ROOT/'results/weights/B1000_pc_sweep_next/pc_3.6/experiment.json')),f,indent=2)
 def job(seed):
  with (BASE/'logs'/f'train-seed{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'train',str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Training seed{seed} failed; no automatic retry'
  print('TRAIN_COMPLETE',seed,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,[2,3]))
 with (BASE/'complete.json').open('x') as f:json.dump(dict(seeds=[2,3],steps_each=2520000),f)
 print('ALL_COMPLETE_B_PC36_SEEDS23',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[2]),sys.argv[1]=='preflight')
 else:main()
