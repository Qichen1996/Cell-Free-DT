from pathlib import Path
E=Path('/workspace/Cell-Free-DT-new/experiments');s=(E/'B500_evalw_final_cluster98_collect_v1/runner.py').read_text()
NAME='B500_cl98_cross_B1000_B2000_v1'
i=s.index('def make_specs():');j=s.index('def load():')
MS='''def make_specs():
 M=[('B1000pc21_trainseed5_cpfinal_onB500cl98',R/'experiments/tradeoff_selected_single_collect200_v1/models/pc2.1_trainseed5_cpfinal',2.1,5,'final',[80,.2,.2,2.1]),
    ('B2000_zdovv6uu_pc1.2_cp170_onB500cl98',R/'experiments/newmodels_traj_collect_v1/models/B2000_zdovv6uu_pc1.2_cp170',1.2,1,170,[30,.1,.1,1.2])]
 sp=[]
 for tag,src,pc,ts,cp,tw in M:
  m=B/'models'/tag;m.mkdir(parents=True)
  for n in ['actor.pt','critic.pt','architecture.json']:shutil.copy(src/n,m/n)
  sp.append(dict(pc=pc,training_seed=ts,checkpoint=cp,tag=tag,env='B500',training_weights=tw,source=str(m),original_source=str(src),manifest={n:sha(m/n) for n in ['actor.pt','critic.pt','architecture.json']}))
 (B/'specs.json').write_text(json.dumps(sp,indent=1))
'''
s=s[:i]+MS+s[j:]
rep=[("# [B500_evalw_final_cluster98_collect_v1] Collect trajectories for PDT: B500 PPO bbcyap5g cpfinal (trained w/ eval weights) under CLUSTER .98, seeds 1-35.",
      "# [B500_cl98_cross_B1000_B2000_v1] B1000 PPO pc2.1_trainseed5_cpfinal and B2000 PPO zdovv6uu cp170 run in B500 area, CLUSTER .98, seeds 1-35, collect traj. (user: 1000和2000的模型都在500 98%跑一下1-35seed看看)\n# Based on B500_evalw_final_cluster98_collect_v1/runner.py. Previous header:"),
 ("O=R/'analysis/sim_stats/B500_evalw_final_cluster98_collect_v1';D=R/'analysis/trajectory/B500_evalw_final_cluster98_collect_v1'",f"O=R/'analysis/sim_stats/{NAME}';D=R/'analysis/trajectory/{NAME}'"),
 ("C=cases();assert len(C)==35,len(C)","C=cases();assert len(C)==70,len(C)"),
 ("parallel=35,","parallel=70,"),("max_workers=35","max_workers=70")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
compile(s,'r','exec')
(E/NAME).mkdir(exist_ok=False);(E/NAME/'runner.py').write_text(s);print('ok')
