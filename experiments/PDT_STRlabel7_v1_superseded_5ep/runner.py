# PDT with prompt = actual per-trajectory stringent drop (%) in every scene.
#  B: tradeoff_selected_single_collect200_v1 (10 PPO policies x seeds 1-20), prompt = that episode's own STR.
#  A, C, B4, B36, B2000: original seven-scene pkls, only episodes seed 1-5 (pkl index 0-4), prompt = STR from local E: stats
#     (sites/area/deployment, seeds 1-5; user decision 2026-11-05: label with local values). B500 dropped (no local stats for e450).
# Training code/hyper-parameters identical to selected_seven_pc33_v1 (train_mixed.py, patched in-memory). PDT only.
import os, json, sys, subprocess, hashlib
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); sys.path.insert(0, str(R))
B = Path(__file__).parent; M = R / 'dt_results/PDT_STRlabel7_v1'
COL = R / 'experiments/tradeoff_selected_single_collect200_v1'
LOCAL = json.loads((B / 'local_seed1to5_STR.json').read_text())
T = R / 'analysis/trajectory'
NONB = [('A', 'uniform/A.pkl.zst', 0, 16, 1000, 250, .98, 35, 'A'), ('C', 'uniform/C.pkl.zst', 2, 16, 1000, 250, .9, 35, 'C'),
        ('B4', 'B40-05-05-3/B4/B.pkl.zst', 1, 4, 1000, 500, .9, 140, 'B4'), ('B36', 'B40-05-05-3/B36/B.pkl.zst', 1, 36, 1000, 1000 / 6, .9, 17, 'B36'),
        ('B2000', 'B2000_e180_uniform/B.pkl.zst', 1, 16, 2000, 500, .95, 35, 'B2000')]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows_build():
    rows = []
    for env, rel, tf, aps, area, sp, cl, n, key in NONB:
        loc = {r['seed']: r for r in LOCAL[key]['rows']}
        for s in range(1, 6):
            r = loc[s]
            rows.append(dict(env=env, path=str(T / rel), traffic=tf, aps=aps, area=area, spacing=sp, cluster=cl, episodes=1, preference=r['str'] * 3.3,
                             prompt=r['str'], check_seed=None if env == 'B2000' else s, label_source=LOCAL[key]['source'] + f'/SEED{s}', label_pc_w=r['pc_w']))
        rows.append(dict(env=env, path=str(T / rel), traffic=tf, aps=aps, area=area, spacing=sp, cluster=cl, episodes=n - 5, preference=None, skip=True))
    for f in json.loads((COL / 'dataset-manifest.json').read_text())['files']:
        st = 100 * f['dropped'][0] / (f['dropped'][0] + f['completed'][0]); m = f['model']
        rows.append(dict(env='B', path=f['path'], traffic=1, aps=16, area=1000, spacing=250, cluster=.9, episodes=1, preference=st * 3.3, prompt=st,
                         check_seed=f['seed'], expected_sha=f['sha256'], tag=m['tag'], label_pc_w=f['power_w']))
    return rows
def scope_build():
    t = R / 'experiments/selected_seven_pc33_v1/train_mixed.py'; rows = rows_build()
    loaded = sum(x['episodes'] for x in rows); used = [x for x in rows if not x.get('skip')]
    ep, ap = sum(x['episodes'] for x in used), sum(x['episodes'] * x['aps'] for x in used)
    s = t.read_text()
    for a, b in [("BASE=ROOT/'experiments/selected_seven_pc33_v1';MODELS=ROOT/'dt_results/selected_seven_pc33_v1'", f'BASE=Path({str(B)!r});MODELS=Path({str(M)!r})'),
                 ("assert len(self.base.trajectories)==382", f"assert len(self.base.trajectories)=={loaded}"),
                 ("self.groups={e[0]:[] for e in SPECS}", "self.groups={e:[] for e in ['A','B','C','B4','B36','B2000']}"),
                 ("if pref is not None:assert tr['preference_pc']==pref", "if spec.get('check_seed') is not None:assert tr['seed']==spec['check_seed'],(spec['env'],tr.get('seed'))"),
                 ("self.ranges.append((start,ti));self.groups[spec['env']].append(len(self.ranges)-1)", "self.ranges.append((start,ti))\n   if not spec.get('skip'):self.groups[spec['env']].append(len(self.ranges)-1)"),
                 ("env=self.envs[int(torch.randint(7,()))]", "env=self.envs[int(torch.randint(len(self.envs),()))]"),
                 ("'actual source PPO PC weight'", "'actual per-trajectory stringent drop %'")]:
        assert s.count(a) == 1, a; s = s.replace(a, b)
    s = s.replace('episodes=382', f'episodes={ep}').replace('ap_trajectories=4772', f'ap_trajectories={ap}')
    scope = {'__name__': 'strlabel_training', '__file__': str(B / 'runner.py')}; exec(compile(s, str(t), 'exec'), scope)
    scope['entries'] = lambda: rows
    return scope, rows, ep, ap
if len(sys.argv) > 1:
    scope, *_ = scope_build(); scope['run'](sys.argv[1])
else:
    assert not M.exists() and not (B / 'plan.json').exists()
    scope, rows, ep, ap = scope_build()
    for x in rows:
        assert Path(x['path']).is_file()
        if 'expected_sha' in x: assert sha(x['path']) == x['expected_sha']
    (B / 'plan.json').write_text(json.dumps(dict(models=['PDT'], from_scratch=True, seed=1, rounds=1800, updates_each=270000, batch=256, context=8,
        prompt='actual per-trajectory stringent drop % (prompt vector = STR)', episodes=ep, ap_trajectories=ap, dropped_scene='B500', files=rows), indent=2))
    (B / 'state.json').write_text(json.dumps(dict(phase='training', model='PDT')))
    with (B / 'PDT.log').open('x') as log: rc = subprocess.call([sys.executable, '-u', __file__, 'PDT'], cwd=R, stdout=log, stderr=subprocess.STDOUT)
    assert rc == 0, 'PDT failed; no retry'
    d = json.loads((M / 'PDT' / 'complete.json').read_text()); assert d['updates'] == 270000 and sha(M / 'PDT' / 'model.pt') == d['sha256']
    (B / 'complete.json').write_text(json.dumps(dict(model_sha256=d['sha256'], episodes=ep, ap_trajectories=ap)))
    (B / 'state.json').write_text(json.dumps(dict(phase='complete')))
    print('ALL_COMPLETE_PDT_STRLABEL7', flush=True)
