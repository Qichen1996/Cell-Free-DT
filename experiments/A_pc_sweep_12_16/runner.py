"""Isolated A IPPO sweep based on the existing verified training worker."""
import os,sys,json,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
BASE=ROOT/'experiments/A_pc_sweep_12_16'
OUT=ROOT/'results/weights/A_pc_sweep_12_16'
PCS=[1.2,1.3,1.4,1.5,1.6]
def worker(pc,preflight=False):
 source=(ROOT/'experiments/B1000_pc_next/worker.py').read_text()
 replacements={
  "OUTPUT=PROJECT/'results/weights/B1000_pc_sweep_next'":"OUTPUT=PROJECT/'results/weights/A_pc_sweep_12_16'",
  "SOURCE=PROJECT/'results/DT/B/mappo/check/wandb/run-20261003_065202-mfxf31yk/files/config.yaml'":"SOURCE=PROJECT/'results/DT/A/mappo/check/wandb/run-20261006_160354-dcoph43i/files/config.yaml'",
  'nc.coverageGainFraction=.9;nc.minCandidateBS=1':'nc.coverageGainFraction=.98;nc.minCandidateBS=1',
  'UserEquipment.coverage_gain_fraction=.9;UserEquipment.min_candidate_bs=1':'UserEquipment.coverage_gain_fraction=.98;UserEquipment.min_candidate_bs=1',
  "scenario='B'":"scenario='A'",
  'w_str=80,w_sen=.2,w_tol=.2':'w_str=400,w_sen=1,w_tol=1',
  'ENV_VERIFIED B1000 16AP SPACING_M=250 NEIGHBOR_KM=.25 CLUSTER=.90':'ENV_VERIFIED A1000 16AP SPACING_M=250 NEIGHBOR_KM=.25 CLUSTER=.98',
  "experiment_name='B1000_weights'":"experiment_name='A_weights'",
  'else 2520000':'else 5040000',
  "['--scenario','B','--accelerate','3000','--w_str','80','--w_sen','0.2','--w_tol','0.2','--w_pc',str(pc)]":"['--scenario','A','--accelerate','3000','--w_str','400','--w_sen','1','--w_tol','1','--w_pc',str(pc)]",
  "'weights':[80,.2,.2,pc]":"'weights':[400,1,1,pc]",
  "'cluster':.9":"'cluster':.98",
  'else 250':'else 500',
  "f'B1000_pc{pc:.1f}'":"f'A1000_pc{pc:.1f}'",
  "'Two additional weights3.3/3.6 trained from scratch with matched250round budget.'":"'A-only PC sweep1.2-1.6 from scratch; 5040000 steps,500 rounds. Original A weights400/1/1 retained.'",
  "'source_rollout_workers':21":"'source_rollout_workers':28",
  "'source_rollout_length':480":"'source_rollout_length':360",
 }
 for old,new in replacements.items():
  assert old in source,old
  source=source.replace(old,new)
 source=source.replace("'source_config':str(SOURCE)","'scenario':'A','train_cluster':.98,'source_config':str(SOURCE)")
 scope={'__name__':'isolated_a_training','__file__':str(BASE/'runner.py')}
 exec(compile(source,str(BASE/'worker_generated_in_memory.py'),'exec'),scope)
 scope['run'](pc,False,preflight)
def main():
 assert not OUT.exists();OUT.mkdir()
 for pc in PCS:
  result=subprocess.run([sys.executable,'-u',__file__,'preflight',str(pc)],capture_output=True,text=True,cwd=ROOT)
  with (BASE/'logs'/f'preflight-{pc:.1f}.log').open('x') as f:f.write(result.stdout+result.stderr)
  assert result.returncode==0,result.stdout+result.stderr
  print('PREFLIGHT_OK',pc,flush=True)
 with (BASE/'plan.json').open('x') as f:json.dump(dict(scenario='A',pc_weights=PCS,other_weights=[400,1,1],cluster=.98,area=1000,aps=16,spacing_m=250,neighbor_km=.25,seed=1,steps=5040000,rounds=500,parallel=2,from_scratch=True,wandb=True),f,indent=2)
 def job(pc):
  with (BASE/'logs'/f'train-{pc:.1f}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'train',str(pc)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  assert rc==0,f'Train failed pc{pc}; no automatic retry'
  print('TRAIN_COMPLETE',pc,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,PCS))
 with (BASE/'complete.json').open('x') as f:json.dump(dict(pc_weights=PCS,steps_each=5040000),f)
 print('ALL_COMPLETE_A_PC_SWEEP_5',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(float(sys.argv[2]),sys.argv[1]=='preflight')
 else:main()
