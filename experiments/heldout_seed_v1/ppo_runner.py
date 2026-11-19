# [heldout_seed_v1 step 1] PPO experts of the 7 PDT scenes on HELD-OUT env seeds 201-210 (never in PDT/DT training data:
# training collections used seeds 1-35, B4 1-140, B36 1-17). Seeds 201-205 -> prompt source for PDT; 206-210 -> PPO reference on PDT test seeds.
# (user: 用ppo在新的seed下仿真的结果中截取traj来作为prompt给pdt test用; 7场景全做)
# Same experts + clusters as the b2000cl90_pdt_v1 training data: A pc1.5 cp440 (.98), B pc2.1 s5 final (.9), C run1 (.9), B4 k164wiqz (.9),
# B36 jywudswq (.9), B500 bbcyap5g final (.98), B2000 zdovv6uu cp170 (.9). Worker = tradeoff_selected_single_collect200_v1 (collect mode, pkl).
# trajectory.csv deleted per run with record (user rule). No training, no retries, no other deletions.
import os,sys,json,hashlib,subprocess,concurrent.futures,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');B=Path(__file__).parent
SRC=R/'experiments/tradeoff_selected_single_collect200_v1/runner.py'
O=R/'analysis/sim_stats/heldout_seed_v1/ppo';D=R/'analysis/trajectory/heldout_seed_v1'
SEEDS=list(range(201,211))
WANT=['A_pc1.5_trainseed1_cp440','pc2.1_trainseed5_cpfinal','C_run1','B4_k164wiqz_final','B36_jywudswq_final','B2000_zdovv6uu_pc1.2_cp170']
CL={'A':.98,'B':.9,'C':.9,'B4':.9,'B36':.9,'B500':.98,'B2000':.9}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make_specs():
 a=json.loads((R/'experiments/rewardparts_collect_v1/specs.json').read_text())
 b=json.loads((R/'experiments/B500_evalw_final_cluster98_collect_v1/specs.json').read_text())
 sp=[s for t in WANT for s in a if s['tag']==t]+[s for s in b if s['tag']=='B500evalw_bbcyap5g_cpfinal_cl98']
 assert [s['env'] for s in sp]==['A','B','C','B4','B36','B2000','B500'],[s['env'] for s in sp]
 for s in sp:
  assert all(sha(Path(s['source'])/n)==h for n,h in s['manifest'].items()),s['tag']
  s['seeds']=[SEEDS[0],SEEDS[-1]]
 (B/'specs.json').write_text(json.dumps(sp,indent=1))
def load():
 code=SRC.read_text()
 rep={"B=R/'experiments/tradeoff_selected_single_collect200_v1'":f'B=Path({str(B)!r})',
      "O=R/'analysis/sim_stats/tradeoff_selected_single_collect200_v1'":f'O=Path({str(O)!r})',
      "D=R/'analysis/trajectory/tradeoff_selected_single_seed1to20_v1'":f'D=Path({str(D)!r})',
      "assert name=='B' and n==16":"assert name==s['env'] and n==expected(name)['aps']",
      "prompt=np.array([1,16,1000],np.float32)":"prompt=np.array([{'A':0,'B':1,'C':2}[scenario],n,expected(name)['area']],np.float32)",
      "training_weights=[80,.2,.2,s['pc']]":"training_weights=s['training_weights']",
      "'B500':('B',500,16,125.,.125,.9)":"'B500':('B',500,16,125.,.125,.98)",
      "'B2000':('B',2000,16,500.,.5,.95)":"'B2000':('B',2000,16,500.,.5,.9)",
      "exec(compile(f.read_text(),str(f),'exec'),scope)\n":"exec(compile(f.read_text(),str(f),'exec'),scope)\n _e=scope['training'].entries;scope['training'].entries=lambda:[dict(e,cluster=%r[e['env']]) if e['env'] in ('B500','B2000') else e for e in _e()]\n"%CL}
 for a,b in rep.items():assert code.count(a)==1,a;code=code.replace(a,b)
 assert code.count('4*1024**3')==2;code=code.replace('4*1024**3','1.5*1024**3')
 sc={'__name__':'collect_heldout','__file__':str(SRC)};exec(compile(code,str(B/'collect_in_memory.py'),'exec'),sc)
 for e,c in CL.items():assert sc['expected'](e)['cluster']==c,(e,sc['expected'](e))
 return sc
def cases():
 sp=json.loads((B/'specs.json').read_text());return [(i,s['tag'],s['env'],seed) for i,s in enumerate(sp) for seed in SEEDS]
def worker(k):
 i,tag,env,seed=cases()[k];sc=load();sc['worker']('collect',i,env,seed)
 d=O/'collection'/tag/env/sc['expected'](env)['scenario']/f'SEED{seed}'
 t=d/'trajectory.csv';(d/'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv',bytes=t.stat().st_size,sha256=sha(t),reason='disk; user rule')));t.unlink()
 r=json.loads((d/'result.json').read_text());r.update(tag=tag,env=env,seed=seed)
 (B/'done').mkdir(exist_ok=True);(B/'done'/f'{tag}__{env}__{seed}.json').write_text(json.dumps(r))
 print('CASE_DONE',tag,env,seed,r['power_w'],r['dropped'],flush=True)
def main(only=None):
 if not (B/'specs.json').exists():make_specs()
 C=cases();assert len(C)==70;print('cases',len(C),flush=True)
 if only=='check':load();print('CHECK_OK',flush=True);return
 assert not (B/'plan.json').exists();(B/'plan.json').write_text(json.dumps(dict(cases=C,runs=len(C),parallel=70,dataset_dir=str(D),clusters=CL),indent=1))
 (B/'logs').mkdir(exist_ok=True)
 def job(k):
  assert shutil.disk_usage(R).free>1.5*1024**3,'disk'
  with (B/'logs'/f'case-{k}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(k)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  print('JOB',k,C[k][1:],rc,flush=True);return rc
 with concurrent.futures.ThreadPoolExecutor(max_workers=70) as p:rcs=list(p.map(job,range(len(C))))
 (B/'complete.json').write_text(json.dumps(dict(runs=len(C),rc=rcs,failed=[C[k] for k,r in enumerate(rcs) if r])));print('ALL_COMPLETE',sum(r!=0 for r in rcs),'failed',flush=True)
if __name__=='__main__':
 if len(sys.argv)>2 and sys.argv[1]=='job':worker(int(sys.argv[2]))
 else:main(sys.argv[1] if len(sys.argv)>1 else None)
