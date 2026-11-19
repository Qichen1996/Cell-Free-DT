from pathlib import Path
E=Path('/workspace/Cell-Free-DT-new/experiments');s=(E/'B500_cp290_cluster_v2/runner.py').read_text()
NAME='B500_evalw_bbcyap5g_cluster98_v1'
i=s.index('"""');j=s.index('"""',3)
s='"""B500 PPO bbcyap5g (trained with eval weights 40/.5/.5/3) cp200/cp290/cpfinal under cluster .98 (min_bs 1), area 500, seeds 1-5, no training. (user: 试一下200和290以及final再98%) Based on B500_cp290_cluster_v2/runner.py. trajectory.csv deleted after verify."""'+s[j+3:]
rep=[("NAME = 'B500_cp290_cluster_v2'",f"NAME = {NAME!r}"),
 ("SOURCE = Path('/workspace/Cell-Free-DT-new/experiments/B500_5m820ao4_cp290_cross_1000_2000_v1/models')","SOURCE = Path('/workspace/Cell-Free-DT-new/experiments/B500_evalw_ckpt_eval_v1/models')"),
 ("CP = '290'\nVERSION = '' if CP == 'final' else '_eps'+CP","VERSION = ''"),
 ("CFG = {'c90': .90, 'c95': .95, 'c98': .98}","CFG = {'cp200': .98, 'cp290': .98, 'cpfinal': .98}"),
 ("'--model_dir', str(MODEL),","'--model_dir', str(MODEL / cfg),"),
 ("checkpoint=CP, source_run='5m820ao4',","checkpoint=cfg, source_run='bbcyap5g',"),
 ("model_sha256=sha(MODEL / ('actor'+VERSION+'.pt'))","model_sha256=sha(MODEL / cfg / 'actor.pt')"),
 ("    for filename in [('actor'+VERSION+'.pt'), ('critic'+VERSION+'.pt'), 'config.yaml']:\n        shutil.copy2(SOURCE / filename, MODEL / filename)\n        hashes[filename] = sha(MODEL / filename)",
  "    for cfg in CFG:\n        (MODEL / cfg).mkdir()\n        for filename in ['actor.pt', 'critic.pt', 'architecture.json']:\n            shutil.copy2(SOURCE / f'B500evalw_bbcyap5g_{cfg}' / filename, MODEL / cfg / filename)\n            hashes[cfg + '/' + filename] = sha(MODEL / cfg / filename)")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
assert 'CP' not in s.replace('CPU','')
(E/NAME).mkdir(exist_ok=False);(E/NAME/'runner.py').write_text(s);print('ok')
