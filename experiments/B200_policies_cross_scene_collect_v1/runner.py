# Collect trajectories: existing B200 PPO tradeoff policies (all 10 weights) run unchanged in scenes A, C, B36, B500, env seeds 1-5 (200 episodes).
# Uses the collection runner's own setup/worker in 'collect' mode (pkl saved + round-trip check), with: output dirs redirected,
# B-only/16-AP assert lifted (B36 has 36 APs), disk guard 4G -> 0.5G. trajectory.csv deleted after each run with a record (user rule).
# pkls are moved to the local PC by a separate mover (sha-verified before server copy is removed, user-approved).
import os, sys, json, hashlib, subprocess, concurrent.futures, shutil
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); B = Path(__file__).parent
SRC = R / 'experiments/tradeoff_selected_single_collect200_v1/runner.py'
O = R / 'analysis/sim_stats/B200_policies_cross_scene_collect_v1'; D = R / 'analysis/trajectory/B200_policies_cross_scene_v1'
ENVS = ['A', 'C', 'B36', 'B500']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load():
    code = SRC.read_text()
    rep = {"O=R/'analysis/sim_stats/tradeoff_selected_single_collect200_v1'": f'O=Path({str(O)!r})',
           "D=R/'analysis/trajectory/tradeoff_selected_single_seed1to20_v1'": f'D=Path({str(D)!r})',
           "assert name=='B' and n==16": "assert name in %r" % ENVS}
    for a, b in rep.items(): assert code.count(a) == 1, a; code = code.replace(a, b)
    assert code.count('4*1024**3') == 2; code = code.replace('4*1024**3', '0.5*1024**3')
    scope = {'__name__': 'collect_cross', '__file__': str(SRC)}; exec(compile(code, str(B / 'collect_in_memory.py'), 'exec'), scope); return scope
def cases():
    sp = load()['specs'](); return [(i, s['tag'], e, seed) for e in ENVS for i, s in enumerate(sp) for seed in range(1, 6)]
def worker(k):
    sc = load(); i, tag, env, seed = cases()[k]
    sc['worker']('collect', i, env, seed)
    d = O / 'collection' / tag / env / sc['expected'](env)['scenario'] / f'SEED{seed}'
    t = d / 'trajectory.csv'; (d / 'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv', bytes=t.stat().st_size, sha256=sha(t), reason='disk; user rule'))); t.unlink()
    r = json.loads((d / 'result.json').read_text()); r.update(tag=tag, env=env, seed=seed)
    (B / 'done').mkdir(exist_ok=True); (B / 'done' / f'{tag}__{env}__{seed}.json').write_text(json.dumps(r))
    print('CASE_DONE', tag, env, seed, r['power_w'], r['dropped'], flush=True)
def main():
    assert not O.exists() and not D.exists() and not (B / 'plan.json').exists()
    C = cases(); assert len(C) == 200
    (B / 'logs').mkdir(exist_ok=True); (B / 'plan.json').write_text(json.dumps(dict(cases=C, runs=len(C), parallel=12, envs=ENVS, seeds=[1, 2, 3, 4, 5], dataset_dir=str(D)), indent=1))
    def job(k):
        assert shutil.disk_usage(R).free > 0.5 * 1024**3, 'disk'
        with (B / 'logs' / f'case-{k}.log').open('x') as f: rc = subprocess.call([sys.executable, '-u', __file__, 'job', str(k)], cwd=R, stdout=f, stderr=subprocess.STDOUT)
        print('JOB', k, C[k][1:], rc, flush=True); return rc
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p: rcs = list(p.map(job, range(len(C))))
    (B / 'complete.json').write_text(json.dumps(dict(runs=len(C), rc=rcs, failed=[k for k, r in enumerate(rcs) if r]))); print('ALL_COMPLETE', sum(r != 0 for r in rcs), 'failed', flush=True)
if __name__ == '__main__':
    if len(sys.argv) > 1: worker(int(sys.argv[2]))
    else: main()
