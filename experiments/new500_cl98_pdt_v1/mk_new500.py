from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');s=(R/'experiments/bigmodel_pdt_v1/train_big.py').read_text()
NAME='new500_cl98_pdt_v1'
rep=[('"""[bigmodel_pdt_v1] Ablation:',f'"""[{NAME}] Same as bigmodel_pdt_v1 except B500 data replaced by B500 PPO bbcyap5g cpfinal under cluster .98, seeds 1-35\n(analysis/trajectory/B500_evalw_final_cluster98_collect_v1; 189.7 W, str .517%). (user: 就用这个500的final作为新的500来收集traj再训练dt和pdt)\nBase was bigmodel_pdt_v1:'),
 ("BASE=ROOT/'experiments/bigmodel_pdt_v1';MODELS=ROOT/'dt_results/bigmodel_pdt_v1'",f"BASE=ROOT/'experiments/{NAME}';MODELS=ROOT/'dt_results/{NAME}'"),
 ("('B500',seeds(NEW/'B500_5m820ao4_cp290',1,35),1,16,500,.9,35),","('B500',seeds(TR/'B500_evalw_final_cluster98_collect_v1/B500evalw_bbcyap5g_cpfinal_cl98',1,35),1,16,500,.98,35),")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
d=R/'experiments'/NAME;d.mkdir(exist_ok=False);(d/'train_new500.py').write_text(s);print('ok')
