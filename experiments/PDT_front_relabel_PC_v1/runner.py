# PDT trained with hindsight-relabelled prompts on the B Pareto front (PPO_front_B_collect_v1, seeds 1-12).
# Identical to selected_seven_Bsingle200_v1 except the 200 B episodes are replaced by front-policy episodes and their prompts:
#   MODE=PC : prompt = (PCmax - PC) / (PCmax - PCmin) over the front (PC = 12-seed mean of that policy)
#   MODE=STR: prompt = pooled 12-seed stringent drop ratio of that policy, in %
# Non-B data (A, C, B4, B36, B500, B2000 uniform at prompt 1.8/3.3) unchanged. PDT only (no DT).
import os
from pathlib import Path
import json, sys, subprocess, hashlib
MODE = os.environ['RELABEL_MODE'] if 'RELABEL_MODE' in os.environ else sys.argv[1]
R = Path('/workspace/Cell-Free-DT-new'); sys.path.insert(0, str(R))
B = R / f'experiments/PDT_front_relabel_{MODE}_v1'; M = R / f'dt_results/PDT_front_relabel_{MODE}_v1'
C = R / 'experiments/PPO_front_B_collect_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def front():
    man = json.loads((C / 'dataset-manifest.json').read_text())['files']; pol = {}
    for f in man:
        t = f['model']['tag']; p = pol.setdefault(t, dict(model=f['model'], files=[], pc=[], d=0, c=0))
        p['files'].append(f); p['pc'].append(f['power_w']); p['d'] += f['dropped'][0]; p['c'] += f['completed'][0]
    for p in pol.values():
        assert sorted(x['seed'] for x in p['files']) == list(range(1, 13))
        p['pc_w'] = sum(p['pc']) / 12; p['str'] = 100 * p['d'] / (p['d'] + p['c'])
    P = list(pol.values())
    F = [p for p in P if not any((q['pc_w'] <= p['pc_w'] and q['str'] <= p['str']) and (q['pc_w'] < p['pc_w'] or q['str'] < p['str']) for q in P)]
    hi, lo = max(p['pc_w'] for p in F), min(p['pc_w'] for p in F)
    for p in F: p['prompt'] = (hi - p['pc_w']) / (hi - lo) if MODE == 'PC' else p['str']
    return sorted(F, key=lambda p: p['prompt']), P
def scope_build():
    t = R / 'experiments/selected_seven_pc33_v1/train_mixed.py'; F, P = front(); nB = 12 * len(F)
    ep, ap = 297 + nB, 3412 + 16 * nB
    s = t.read_text()
    for a, b in [("BASE=ROOT/'experiments/selected_seven_pc33_v1';MODELS=ROOT/'dt_results/selected_seven_pc33_v1'", f'BASE=Path({str(B)!r});MODELS=Path({str(M)!r})'),
                 ("if pref is not None:assert tr['preference_pc']==pref", "if pref is not None:assert tr['preference_pc']==spec['source_pc'] and tr['seed']==spec['collection_seed']"),
                 ("'actual source PPO PC weight'", f"'hindsight relabel ({MODE}) of B Pareto-front policy'")]:
        assert s.count(a) == 1, a; s = s.replace(a, b)
    s = s.replace('382', str(ep)).replace('4772', str(ap))
    scope = {'__name__': 'relabel_training', '__file__': str(B / 'runner.py')}; exec(compile(s, str(t), 'exec'), scope)
    rows = [x for x in scope['entries']() if x['env'] != 'B']
    assert sum(x['episodes'] for x in rows) == 297 and sum(x['episodes'] * x['aps'] for x in rows) == 3412
    for p in F:
        for f in sorted(p['files'], key=lambda f: f['seed']):
            m = p['model']
            rows.append(dict(env='B', path=f['path'], traffic=1, aps=16, area=1000, spacing=250, cluster=.9, episodes=1, preference=p['prompt'] * 3.3,
                             prompt=p['prompt'], source_pc=m['pc'], expected_sha=f['sha256'], training_seed=m['training_seed'], checkpoint=m['checkpoint'],
                             collection_seed=f['seed'], policy_pc_w=p['pc_w'], policy_str=p['str']))
    assert sum(x['episodes'] for x in rows) == ep
    scope['entries'] = lambda: rows
    return scope, rows, F, P
if len(sys.argv) > 2:
    scope, rows, F, P = scope_build(); scope['run'](sys.argv[2])
else:
    assert not M.exists() and not (B / 'plan.json').exists(); B.mkdir(parents=True, exist_ok=True)
    scope, rows, F, P = scope_build()
    for x in rows:
        assert Path(x['path']).is_file()
        if 'expected_sha' in x: assert sha(x['path']) == x['expected_sha']
    (B / 'plan.json').write_text(json.dumps(dict(mode=MODE, models=['PDT'], from_scratch=True, seed=1, rounds=1800, updates_each=270000, batch=256, context=8,
        prompt=('(PCmax-PC)/(PCmax-PCmin) on B front' if MODE == 'PC' else 'pooled stringent drop % on B front'),
        front=[dict(tag=p['model']['tag'], pc_w=p['pc_w'], str=p['str'], prompt=p['prompt']) for p in F],
        pool=[dict(tag=p['model']['tag'], pc_w=p['pc_w'], str=p['str']) for p in P], files=rows), indent=2))
    (B / 'state.json').write_text(json.dumps(dict(phase='training', model='PDT')))
    with (B / 'PDT.log').open('x') as log: rc = subprocess.call([sys.executable, '-u', __file__, MODE, 'PDT'], cwd=R, stdout=log, stderr=subprocess.STDOUT)
    assert rc == 0, 'PDT failed; no retry'
    d = json.loads((M / 'PDT' / 'complete.json').read_text()); assert d['updates'] == 270000 and sha(M / 'PDT' / 'model.pt') == d['sha256']
    (B / 'complete.json').write_text(json.dumps(dict(mode=MODE, model_sha256=d['sha256'], front=len(F), episodes=sum(x['episodes'] for x in rows))))
    (B / 'state.json').write_text(json.dumps(dict(phase='complete')))
    print(f'ALL_COMPLETE_PDT_FRONT_RELABEL_{MODE}', flush=True)
