from pathlib import Path
R=Path("/workspace/Cell-Free-DT-new");s=(R/"experiments/new500_cl98_pdt_v1/train_new500.py").read_text()
NAME="b2000cl90_pdt_v1"
rep=[("\"\"\"[new500_cl98_pdt_v1] Same as bigmodel_pdt_v1",f"\"\"\"[{NAME}] Same as new500_cl98_pdt_v1 except B2000 data replaced by zdovv6uu cp170 recollected at cluster .90, seeds 1-35\n(analysis/trajectory/newmodels_traj_collect_B2000cl90_v1; 795.9 W, str .460%). (user: 要重新训练根据更新traj的pdt和dt)\n[new500_cl98_pdt_v1] Same as bigmodel_pdt_v1"),
 ("BASE=ROOT/\x27experiments/new500_cl98_pdt_v1\x27;MODELS=ROOT/\x27dt_results/new500_cl98_pdt_v1\x27",f"BASE=ROOT/\x27experiments/{NAME}\x27;MODELS=ROOT/\x27dt_results/{NAME}\x27"),
 ("(\x27B2000\x27,seeds(NEW/\x27B2000_zdovv6uu_pc1.2_cp170\x27,1,35),1,16,2000,.95,35)","(\x27B2000\x27,seeds(TR/\x27newmodels_traj_collect_B2000cl90_v1/B2000_zdovv6uu_pc1.2_cp170\x27,1,35),1,16,2000,.9,35)")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
d=R/"experiments"/NAME;d.mkdir(exist_ok=False);(d/"train_b2000cl90.py").write_text(s);print("ok")
