from pathlib import Path
src=Path('/workspace/Cell-Free-DT-new/experiments/selfhist_pdt_v1/train_selfhist.py').read_text()
rep=[
 ('"""Self-history prompt DT','"""[mixedq_pdt_v1] Step 2 of beat-PPO: same as selfhist_pdt_v1 but B500/B1000 use mixed-quality data\n(B500: cp100/150/200/250 seeds1-10 + cp290 seeds1-35; B1000: 10 trade-off policies x seeds1-20 + pc2.1 seeds21-35),\nbigger model (256 dim, 6 blocks, 4 heads), checkpoints every 600 rounds.\nOriginal doc: Self-history prompt DT'),
 ("BASE=ROOT/'experiments/selfhist_pdt_v1';MODELS=ROOT/'dt_results/selfhist_pdt_v1'","BASE=ROOT/'experiments/mixedq_pdt_v1';MODELS=ROOT/'dt_results/mixedq_pdt_v1'"),
 ("OLD21=TR/'tradeoff_selected_single_seed1to20_v1/pc2.1_trainseed5_cpfinal'","OLD21=TR/'tradeoff_selected_single_seed1to20_v1/pc2.1_trainseed5_cpfinal';TO=TR/'tradeoff_selected_single_seed1to20_v1';MQ=TR/'mixedq_collect_v1'\nTO_POL=['pc0.6_trainseed1_cp240','pc0.9_trainseed1_cp230','pc1.2_trainseed3_cpfinal','pc1.5_trainseed3_cp240','pc1.8_trainseed5_cpfinal','pc2.1_trainseed5_cpfinal','pc2.4_trainseed3_cp240','pc2.7_trainseed1_cp220','pc3.0_trainseed1_cp200','pc3.3_trainseed1_cp200']"),
 (" ('B',seeds(OLD21,1,20)+seeds(NEW/'pc2.1_trainseed5_cpfinal',21,35),1,16,1000,.9,35),"," ('B',sum([seeds(TO/p,1,20) for p in TO_POL],[])+seeds(NEW/'pc2.1_trainseed5_cpfinal',21,35),1,16,1000,.9,215),"),
 (" ('B500',seeds(NEW/'B500_5m820ao4_cp290',1,35),1,16,500,.9,35),"," ('B500',sum([seeds(MQ/f'B500_5m820ao4_cp{c}',1,10) for c in (100,150,200,250)],[])+seeds(NEW/'B500_5m820ao4_cp290',1,35),1,16,500,.9,75),"),
 ("super().__init__(110,2,3,128,CTX,1,.1,","super().__init__(110,2,6,256,CTX,4,.1,"),
 ("rn%300==0","rn%600==0"),
]
for a,b in rep:
 assert src.count(a)==1,a;src=src.replace(a,b)
n=src.count('128');src=src.replace('128','256');print('128->256',n)
Path('/workspace/Cell-Free-DT-new/experiments/mixedq_pdt_v1/train_mixedq.py').write_text(src)
