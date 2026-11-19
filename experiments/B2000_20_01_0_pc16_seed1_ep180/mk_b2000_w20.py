from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new/experiments');s=(R/'B2000_30_01_01_pc12_seed1_ep180/runner.py').read_text()
NAME='B2000_20_01_0_pc16_seed1_ep180'
rep=[('"""Train the authorized B2000 seed1 reward-weight experiment from scratch."""','"""B2000 PPO, training weights str 20 / sen .1 / tol 0 / pc 1.6, cluster .95, seed 1, 180 ep from scratch (user: 用20 0.1 0 1.6 来训练一下). Based on B2000_30_01_01_pc12_seed1_ep180/runner.py (zdovv6uu); only weights changed, wandb offline (no login on new server)."""'),
 ("NAME = 'B2000_30_01_01_pc12_seed1_ep180'",f"NAME = {NAME!r}"),
 ("w_str=30., w_sen=.1, w_tol=.1, w_pc=1.2, seed=1)","w_str=20., w_sen=.1, w_tol=0., w_pc=1.6, seed=1)"),
 ("assert (args.w_str, args.w_sen, args.w_tol, args.w_pc) == (30., .1, .1, 1.2)","assert (args.w_str, args.w_sen, args.w_tol, args.w_pc) == (20., .1, 0., 1.6)"),
 ("dict(weights=[30, .1, .1, 1.2],","dict(weights=[20, .1, 0, 1.6],"),
 ("wandb='online'))","wandb='offline'))"),
 ("WANDB_MODE='online',","WANDB_MODE='offline',"),
 ("'PREFLIGHT_PASS B2000 30/0.1/0.1/1.2 SEED1 EP180 STEPS1814400'","'PREFLIGHT_PASS B2000 20/0.1/0/1.6 SEED1 EP180'"),
 ("'--w_str', '30', '--w_sen', '.1', '--w_tol', '.1',\n        '--w_pc', '1.2',","'--w_str', '20', '--w_sen', '.1', '--w_tol', '0',\n        '--w_pc', '1.6',"),
 ("'ALL_COMPLETE_B2000_30_01_01_PC12_EP180'","'ALL_COMPLETE'")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
compile(s,'r','exec')
d=R/NAME;d.mkdir(exist_ok=False);(d/'runner.py').write_text(s);print('ok')
