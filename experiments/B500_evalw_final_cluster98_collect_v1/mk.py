from pathlib import Path
E=Path('/workspace/Cell-Free-DT-new/experiments');s=(E/'B500_evalw_ckpt_eval_v1/runner.py').read_text()
NAME='B500_evalw_final_cluster98_collect_v1'
rep=[("# Eval + collect (seeds 1-5) checkpoints of B500 PPO trained with evaluation weights (B500_evalw_40_05_05_pc3_seed1_ep300, run bbcyap5g): cp200/250/290/final.",
      "# [B500_evalw_final_cluster98_collect_v1] Collect trajectories for PDT: B500 PPO bbcyap5g cpfinal (trained w/ eval weights) under CLUSTER .98, seeds 1-35. (user: 把他多跑一些seed 跑到35 这次要收集traj以备pdt来运行)\n# Based on B500_evalw_ckpt_eval_v1/runner.py (which was:) Eval + collect (seeds 1-5) checkpoints of B500 PPO trained with evaluation weights: cp200/250/290/final."),
 ("O=R/'analysis/sim_stats/B500_evalw_ckpt_eval_v1';D=R/'analysis/trajectory/B500_evalw_ckpt_eval_v1'",f"O=R/'analysis/sim_stats/{NAME}';D=R/'analysis/trajectory/{NAME}'"),
 ("CPS=[200,250,290,'final'];SEEDS=range(1,6)","CPS=['final'];SEEDS=range(1,36)"),
 ("tag=f'B500evalw_bbcyap5g_cp{cp}'","tag=f'B500evalw_bbcyap5g_cp{cp}_cl98'"),
 ("      \"training_weights=[80,.2,.2,s['pc']]\":\"training_weights=s['training_weights']\"}",
  "      \"training_weights=[80,.2,.2,s['pc']]\":\"training_weights=s['training_weights']\",\n      \"'B500':('B',500,16,125.,.125,.9)\":\"'B500':('B',500,16,125.,.125,.98)\",\n      \"exec(compile(f.read_text(),str(f),'exec'),scope)\n\":\"exec(compile(f.read_text(),str(f),'exec'),scope)\n _e=scope['training'].entries;scope['training'].entries=lambda:[dict(e,cluster=.98) if e['env']=='B500' else e for e in _e()]\n\"}"),
 ("C=cases();assert len(C)==20,len(C)","C=cases();assert len(C)==35,len(C)"),
 ("parallel=40,","parallel=35,cluster=.98,"),
 ("max_workers=40","max_workers=35")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
(E/NAME).mkdir(exist_ok=False);(E/NAME/'runner.py').write_text(s);print('ok')
