from pathlib import Path
E=Path('/workspace/Cell-Free-DT-new/experiments');s=(E/'B2000_30_01_01_pc12_cp170_cluster90/runner.py').read_text()
NAME='B1000_pc21_cluster_sensitivity_v1'
rep=[('"""Isolated, deterministic cross-area evaluation of B500 run 5m820ao4 CP290."""','"""B1000 PPO pc2.1_trainseed5_cpfinal (DT teacher) under different UE clustering: (cluster,min_bs) in (.9,1) baseline, (.98,1), (.9,2); area 1000, seeds 1-5, eval weights. No training. (user: 你测试一下)\nBased on B2000_30_01_01_pc12_cp170_cluster90/runner.py."""'),
 ("NAME = 'B2000_30_01_01_pc12_cp170_cluster90'",f"NAME = {NAME!r}"),
 ("SOURCE = Path('/workspace/Cell-Free-DT-new/experiments/B2000_30_01_01_pc12_checkpoint_eval_v1/models/170')","SOURCE = Path('/workspace/Cell-Free-DT-new/experiments/tradeoff_selected_single_collect200_v1/models/pc2.1_trainseed5_cpfinal')"),
 ("CP = '170'","CP = 'final'"),
 ("def evaluate(area, seed, preflight=False):","CFG = {'c90m1': (.90, 1), 'c98m1': (.98, 1), 'c90m2': (.90, 2)}\n\n\ndef evaluate(cfg, seed, preflight=False):\n    area = 1000"),
 ("    cluster = 0.90\n","    cluster, min_bs = CFG[cfg]\n"),
 ("    config.coverageGainFraction = cluster\n","    config.coverageGainFraction = cluster\n    config.minCandidateBS = min_bs\n"),
 ("    UserEquipment.min_candidate_bs = 1\n","    UserEquipment.min_candidate_bs = min_bs\n    sizes = []\n    _orig_set = UserEquipment.channel_gains.fset\n    def _set(self, g):\n        r = _orig_set(self, g); sizes.append(self.cluster_size); return r\n    UserEquipment.channel_gains = UserEquipment.channel_gains.setter(_set)\n"),
 ("'--stats_dir', str(OUTPUT / str(area)),","'--stats_dir', str(OUTPUT / cfg),"),
 ("str(BASE / 'logs' / f'{area}_{seed}_detail.log')","str(BASE / 'logs' / f'{cfg}_{seed}_detail.log')"),
 ("str(BASE / 'logs' / f'{area}_{seed}_performance.csv')","str(BASE / 'logs' / f'{cfg}_{seed}_performance.csv')"),
 ("    record = dict(area=area, seed=seed, training_seed=1, checkpoint=CP, source_run='zdovv6uu',","    sz = np.array(sizes)\n    record = dict(cfg=cfg, min_candidate_bs=min_bs, cluster_size_hist={str(k): float((sz == k).mean()) for k in range(1, 5)}, cluster_size_ge5=float((sz >= 5).mean()), cluster_size_mean=float(sz.mean()), area=area, seed=seed, training_seed=5, checkpoint=CP, source_run='pc2.1_trainseed5_cpfinal',"),
 ("        areas=[2000], seeds=list(range(1, 6)), steps=10080, parallel=4,\n        cluster={'2000': .90},","        configs=CFG, area=1000, seeds=list(range(1, 6)), steps=10080, parallel=15,"),
 ("    for area in [2000]:\n        with (BASE / 'logs' / f'preflight_{area}.log').open('x') as log:\n            subprocess.run([sys.executable, '-u', __file__, 'preflight', str(area), '1'],","    for area in CFG:\n        with (BASE / 'logs' / f'preflight_{area}.log').open('x') as log:\n            subprocess.run([sys.executable, '-u', __file__, 'preflight', area, '1'],"),
 ("            subprocess.run([sys.executable, '-u', __file__, 'worker', str(area), str(seed)],","            subprocess.run([sys.executable, '-u', __file__, 'worker', area, str(seed)],"),
 ("    jobs = [(area, seed) for area in [2000] for seed in range(1, 6)]\n    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:","    jobs = [(area, seed) for area in CFG for seed in range(1, 6)]\n    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as pool:"),
 ("    assert len(records) == 5\n","    assert len(records) == 15\n"),
 ("    for area in [2000]:\n        rows = [row for row in records if row['area'] == area]","    for area in CFG:\n        rows = [row for row in records if row['cfg'] == area]"),
 ("        summary.append(dict(area=area, power_w=","        summary.append(dict(cfg=area, cluster_size_mean=sum(r['cluster_size_mean'] for r in rows) / 5, single_ap=sum(r['cluster_size_hist']['1'] for r in rows) / 5, str_dropped=dropped[0], power_w="),
 ("dict(runs=5, hashes=hashes)","dict(runs=15, hashes=hashes)"),
 ("'ALL_COMPLETE_B2000_CP170_CLUSTER90_5'","'ALL_COMPLETE'"),
 ("        evaluate(int(sys.argv[2]), int(sys.argv[3]), preflight=sys.argv[1] == 'preflight')","        evaluate(sys.argv[2], int(sys.argv[3]), preflight=sys.argv[1] == 'preflight')")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
(E/NAME).mkdir(exist_ok=False);(E/NAME/'runner.py').write_text(s);print('ok')
