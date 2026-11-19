# Pre-check: run existing B200 PPO tradeoff policies (pc0.6, pc1.8, pc3.3) unchanged in the other six scenes, env seeds 1-2.
# Uses the collection runner's own setup/worker ('cross' mode = no pkl saved), output dir redirected, disk guard 4G -> 0.8G.
# After each run trajectory.csv is deleted with a trajectory_deleted.json record (user rule). No training, nothing else deleted.
import os, sys, json, hashlib, subprocess, concurrent.futures, shutil
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); B = Path(__file__).parent
SRC = R / 'experiments/tradeoff_selected_single_collect200_v1/runner.py'
O = R / 'analysis/sim_stats/B200_policies_cross_scene_precheck_v1'
TAGS = ['pc0.6_trainseed1_cp240', 'pc1.8_trainseed5_cpfinal', 'pc3.3_trainseed1_cp200']
CASES = [(t, e, s) for e in ['A', 'C', 'B4', 'B36', 'B500', 'B2000'] for t in TAGS for s in [1, 2]]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(k):
    tag, env, seed = CASES[k]; code = SRC.read_text()
    a = "O=R/'analysis/sim_stats/tradeoff_selected_single_collect200_v1'"; assert code.count(a) == 1; code = code.replace(a, f'O=Path({str(O)!r})')
    assert code.count('4*1024**3') == 2; code = code.replace('4*1024**3', '0.8*1024**3')
    scope = {'__name__': 'precheck', '__file__': str(SRC)}; exec(compile(code, str(B / 'collect_in_memory.py'), 'exec'), scope)
    i = next(j for j, s in enumerate(scope['specs']()) if s['tag'] == tag)
    scope['worker']('cross', i, env, seed)
    d = O / 'cross' / tag / env / scope['expected'](env)['scenario'] / f'SEED{seed}'
    t = d / 'trajectory.csv'; (d / 'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv', bytes=t.stat().st_size, sha256=sha(t), reason='disk; user rule'))); t.unlink()
    print('CASE_DONE', tag, env, seed, (d / 'result.json').read_text(), flush=True)
def main():
    assert not O.exists() and not (B / 'plan.json').exists()
    (B / 'logs').mkdir(exist_ok=True); (B / 'plan.json').write_text(json.dumps(dict(cases=CASES, runs=len(CASES), parallel=12, mode='cross, no pkl'), indent=1))
    def job(k):
        assert shutil.disk_usage(R).free > 0.8 * 1024**3
        with (B / 'logs' / f'case-{k}.log').open('x') as f: rc = subprocess.call([sys.executable, '-u', __file__, 'job', str(k)], cwd=R, stdout=f, stderr=subprocess.STDOUT)
        print('JOB', k, CASES[k], rc, flush=True); return rc
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p: rcs = list(p.map(job, range(len(CASES))))
    (B / 'complete.json').write_text(json.dumps(dict(runs=len(CASES), rc=rcs))); print('ALL_COMPLETE', rcs, flush=True)
if __name__ == '__main__':
    if len(sys.argv) > 1: worker(int(sys.argv[2]))
    else: main()
