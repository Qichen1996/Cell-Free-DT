# Weight-normalized PDT (selected_seven_Bsingle200_v1, prompt = PC weight / 3.3; old scenes trained with 1.8/3.3).
# Every AP gets the SAME prompt = the prompt that scene/policy had in training. Two RTG variants:
#  M  : mean RTG (mean per-AP return over all training trajectories of that policy / scene), same value for every AP
#  AP : each AP its own return from the training trajectory with the same env seed
# Scenes: B (10 tradeoff PPO policies, prompt pc/3.3), A and C (prompt 1.8/3.3); env seeds 1-5.
import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys, json, subprocess, concurrent.futures, hashlib, shutil
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); B = Path(__file__).parent; O = R / 'analysis/sim_stats/PDT_single200_uniformprompt_RTG_v1'
M = R / 'dt_results/selected_seven_Bsingle200_v1/PDT/model.pt'; H = '00c227342a337f9239a3572e387b1cd7696e084cfede9acc443d04bd6a190514'
SRC = R / 'experiments/PDT_APSTR_eval_v2/eval_pc33_diskguard07.py'
MAN = json.loads((R / 'experiments/tradeoff_selected_single_collect200_v1/dataset-manifest.json').read_text())['files']
TAGS = sorted({f['model']['tag'] for f in MAN}, key=lambda t: float(t[2:5])); PC = {f['model']['tag']: f['model']['pc'] for f in MAN}
CASES = [dict(env='B', tag=t, pc=PC[t], seed=s, var=v) for v in ['M', 'AP'] for t in TAGS for s in range(1, 6)] + \
        [dict(env=e, tag=e, pc=1.8, seed=s, var=v) for v in ['M', 'AP'] for e in ['A', 'C'] for s in range(1, 6)]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    sys.path.insert(0, str(R)); import numpy as np
    from trajectory_io import load_trajectory_pickle
    lab = {}
    for t in TAGS:
        rt = {f['seed']: load_trajectory_pickle(f['path'])[0]['rewards'][:, :, 0].sum(0) for f in MAN if f['model']['tag'] == t}
        assert sorted(rt) == list(range(1, 21))
        lab[t] = dict(mean=float(np.mean([v.mean() for v in rt.values()])), per_seed={s: rt[s].tolist() for s in range(1, 6)}, n_traj=20)
    for e in ['A', 'C']:
        d = load_trajectory_pickle(R / f'analysis/trajectory/uniform/{e}.pkl.zst'); rt = [tr['rewards'][:, :, 0].sum(0) for tr in d]
        for k in range(5): assert d[k].get('seed') in (None, k + 1)
        lab[e] = dict(mean=float(np.mean([v.mean() for v in rt])), per_seed={s: rt[s - 1].tolist() for s in range(1, 6)}, n_traj=len(rt)); del d
    (B / 'labels.json').write_text(json.dumps(lab))
def worker(i):
    c = CASES[i]; L = json.loads((B / 'labels.json').read_text())[c['tag']]; assert sha(M) == H
    p = c['pc'] / 3.3; rtg = [L['mean']] * 16 if c['var'] == 'M' else L['per_seed'][str(c['seed'])]; assert len(rtg) == 16
    code = SRC.read_text(); dest = O / c['var'] / c['tag']
    rep = {"STATS=ROOT/'analysis/sim_stats/PDT_selected_seven_pc33'": f'STATS=Path({str(dest)!r})', "MODEL=ROOT/'dt_results/selected_seven_pc33_v1/PDT/model.pt'": f'MODEL=Path({str(M)!r})',
           '738f006d0ff4e3c1ed665509aa0613341c00ef8aa7ca62a9f127e4ed3c5118b6': H, "target=json.loads((BASE/'rtg.json').read_text())['targets'][name]": f"target={L['mean']!r}",
           "running=torch.full((n,1),target);prompt=torch.full((n,1),1.8/3.3)": f"prompt=torch.full((n,1),{p!r});running=torch.tensor({rtg!r},dtype=torch.float32).reshape(n,1)",
           "'PROMPT',1.8/3.3,": f"'PROMPT',{p!r},",
           "pc_preference=1.8,prompt=1.8/3.3,rtg_target=target,": f"pc_preference={c['pc']!r},prompt={p!r},rtg_target={rtg!r},variant={c['var']!r},policy={c['tag']!r},"}
    for a, b in rep.items(): assert code.count(a) == 1, a; code = code.replace(a, b)
    code = code.replace(" model.load_state_dict(", " assert checkpoint['prompt_enabled'] is True and checkpoint['context']==8 and checkpoint['seed']==1\n model.load_state_dict(")
    scope = {'__name__': 'worker', '__file__': str(SRC)}; exec(compile(code, str(B / 'worker_in_memory.py'), 'exec'), scope); scope['evaluate'](c['env'], c['seed'])
    d = dest / c['env'] / c['env'] / f"SEED{c['seed']}"
    t = d / 'trajectory.csv'; (d / 'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv', bytes=t.stat().st_size, sha256=sha(t), reason='disk; user rule'))); t.unlink()
def main():
    assert not O.exists() and not (B / 'plan.json').exists() and sha(M) == H
    (B / 'logs').mkdir(exist_ok=True); prepare()
    (B / 'plan.json').write_text(json.dumps(dict(model=str(M), sha256=H, cases=CASES, runs=len(CASES), parallel=12), indent=1))
    def job(i):
        assert shutil.disk_usage(R).free > 0.8 * 1024**3
        with (B / 'logs' / f'case-{i}.log').open('x') as f: rc = subprocess.call([sys.executable, '-u', __file__, 'job', str(i)], cwd=R, stdout=f, stderr=subprocess.STDOUT)
        assert rc == 0, i; return i
    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p:
        for i in p.map(job, range(len(CASES))):
            done.append(i); (B / 'state.json').write_text(json.dumps(dict(complete=len(done), total=len(CASES)))); print('COMPLETE', i, len(done), flush=True)
    (B / 'complete.json').write_text(json.dumps(dict(runs=len(CASES), model_sha256=H))); print('ALL_COMPLETE', flush=True)
if __name__ == '__main__':
    if len(sys.argv) > 1: worker(int(sys.argv[2]))
    else: main()
