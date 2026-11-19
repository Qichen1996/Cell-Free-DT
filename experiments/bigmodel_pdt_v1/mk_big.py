from pathlib import Path
src=Path('/workspace/Cell-Free-DT-new/experiments/selfhist_pdt_v1/train_selfhist.py').read_text()
rep=[
 ('"""Self-history prompt DT','"""[bigmodel_pdt_v1] Ablation: identical data/settings to selfhist_pdt_v1 (expert data), only bigger model\n(256 dim, 6 blocks, 4 heads), checkpoints every 600 rounds.\nOriginal doc: Self-history prompt DT'),
 ("BASE=ROOT/'experiments/selfhist_pdt_v1';MODELS=ROOT/'dt_results/selfhist_pdt_v1'","BASE=ROOT/'experiments/bigmodel_pdt_v1';MODELS=ROOT/'dt_results/bigmodel_pdt_v1'"),
 ("super().__init__(110,2,3,128,CTX,1,.1,","super().__init__(110,2,6,256,CTX,4,.1,"),
 ("rn%300==0","rn%600==0"),
]
for a,b in rep:
 assert src.count(a)==1,a;src=src.replace(a,b)
src=src.replace('128','256')
Path('/workspace/Cell-Free-DT-new/experiments/bigmodel_pdt_v1/train_big.py').write_text(src)
print('ok')
