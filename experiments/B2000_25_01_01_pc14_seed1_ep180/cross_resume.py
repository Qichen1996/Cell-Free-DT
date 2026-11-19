"""Isolated, deterministic cross-area evaluation of B500 run 5m820ao4 CP290."""
from pathlib import Path
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = Path('/workspace/Cell-Free-DT-new')
NAME = 'B2000_25_01_01_pc14_seed1_ep180_cross'
BASE = ROOT / 'experiments' / NAME
OUTPUT = ROOT / 'analysis/sim_stats' / NAME
SOURCE = Path(json.loads((ROOT/'experiments'/'B2000_25_01_01_pc14_seed1_ep180'/'complete.json').read_text())['model_dir'])
MODEL = BASE / 'models'
CP = os.environ['WINNER_CP']
VERSION = '' if CP == 'final' else '_eps'+CP


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(area, seed, preflight=False):
    import numpy as np
    import torch
    import network.config as config

    spacing = area / 4
    cluster = 0.90
    config.areaSize = np.array([float(area), float(area)])
    config.interBSDist = spacing
    config.bsPositions = config.square_grid_centers(4, 4, float(area), float(area))
    config.coverageGainFraction = cluster
    from network.base_station import BaseStation
    from network.user_equipment import UserEquipment
    BaseStation.inter_dist = spacing / 1000
    UserEquipment.coverage_gain_fraction = cluster
    UserEquipment.min_candidate_bs = 1
    from env import MultiCellNetEnv
    MultiCellNetEnv.bs_poses = config.bsPositions
    original_init = MultiCellNetEnv.__init__

    def checked_init(self, *args, **kwargs):
        kwargs['area_size'] = config.areaSize
        original_init(self, *args, **kwargs)
        assert np.array_equal(self.net.area, config.areaSize)
        assert self.num_agents == 16
        assert self.observation_space[0].shape == (110,)
        assert np.allclose(self.bs_poses, config.bsPositions)
        assert (self.w_str, self.w_sen, self.w_tol, self.w_pc) == (40, .5, .5, 3)

    MultiCellNetEnv.__init__ = checked_init
    torch.set_num_threads(1)
    sys.argv = ['simulate.py', '--algorithm_name', 'mappo', '--experiment_name', 'test',
                '--scenario', 'B', '--days', '7', '--model_dir', str(MODEL),
                '--model_version', VERSION, '--hidden_size', '64', '--layer_N', '1',
                '--seed', str(seed), '--w_str', '40', '--w_sen', '.5', '--w_tol', '.5',
                '--w_pc', '3', '--w_qos', '40', '--w_xqos', '0', '--include_bs_info',
                '--no_traj_pkl', '--cuda', '--n_training_threads', '1',
                '--stats_dir', str(OUTPUT / str(area)),
                '--sim_log_path', str(BASE / 'logs' / f'{area}_{seed}_detail.log'),
                '--perf_save_path', str(BASE / 'logs' / f'{area}_{seed}_performance.csv')]
    script = (ROOT / 'simulate.py').read_text()
    scope = {'__name__': '__main__', '__file__': str(ROOT / 'simulate.py')}
    if preflight:
        original = '\nsimulate()\nenv.close()'
        assert script.count(original) == 1
        script = script.replace(original, '\nobs, _, _ = env.reset(False)\nassert np.asarray(obs).shape == (16,110)\nactions = agent.act(obs, deterministic=True)\nassert np.isfinite(actions).all()\nprint("PREFLIGHT_PASS", flush=True)')
    exec(compile(script, str(ROOT / 'simulate.py'), 'exec'), scope)
    if preflight:
        return
    env = scope['env']
    folder = Path(env.full_stats_dir)
    expected = ['reward.csv', 'ue_stats.csv', 'bs_stats.csv', 'net_stats.csv', 'trajectory.csv']
    assert all((folder / filename).stat().st_size > 0 for filename in expected)
    with (folder / 'reward.csv').open() as handle:
        powers = [float(row['pc_kw']) for row in csv.DictReader(handle)]
    assert len(powers) == 10080 and all(math.isfinite(value) for value in powers)
    dropped = [0, 0, 0]
    completed = [0, 0, 0]
    with (folder / 'ue_stats.csv').open() as handle:
        for row in csv.DictReader(handle):
            for index, budget in enumerate([.05, .15, .3]):
                if math.isclose(float(row['delay_budget']), budget, abs_tol=1e-8):
                    dropped[index] += int(float(row['dropped_packets']))
                    completed[index] += int(float(row['completed_packets']))
    record = dict(area=area, seed=seed, training_seed=1, checkpoint=CP, source_run=SOURCE.parent.name.rsplit('-',1)[-1],
                  model_sha256=sha(MODEL / ('actor'+VERSION+'.pt')), steps=10080, cluster=cluster,
                  aps=16, spacing_m=spacing, weights=[40, .5, .5, 3], deterministic=True,
                  power_w=sum(powers) / len(powers) * 1000, dropped=dropped, completed=completed,
                  source=str(folder))
    (folder / 'verified.json').write_text(json.dumps(record, indent=2))


def main():
    assert not (BASE / 'started.json').exists(), 'This experiment was already started'
    assert shutil.disk_usage(ROOT).free > 550 * 1024 ** 2
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / 'logs').mkdir(exist_ok=True)
    MODEL.mkdir(exist_ok=True)
    hashes = {}
    for filename in [('actor'+VERSION+'.pt'), ('critic'+VERSION+'.pt'), 'config.yaml']:
        shutil.copy2(SOURCE / filename, MODEL / filename)
        hashes[filename] = sha(MODEL / filename)
    (BASE / 'plan.json').write_text(json.dumps(dict(source=str(SOURCE), hashes=hashes,
        areas=[500, 1000], seeds=list(range(1, 6)), steps=10080, parallel=4,
        cluster={'500': .90, '1000': .90}, evaluation_weights=[40, .5, .5, 3]), indent=2))
    (BASE / 'started.json').write_text(json.dumps(dict(pid=os.getpid())))
    for area in [500, 1000]:
        with (BASE / 'logs' / f'preflight_{area}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', __file__, 'preflight', str(area), '1'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('PREFLIGHT_PASS', area, flush=True)

    def run(job):
        area, seed = job
        assert shutil.disk_usage(ROOT).free > 500 * 1024 ** 2
        print('SEED_START', area, seed, flush=True)
        with (BASE / 'logs' / f'{area}_{seed}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', __file__, 'worker', str(area), str(seed)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('SEED_COMPLETE', area, seed, flush=True)

    jobs = [(area, seed) for area in [500, 1000] for seed in range(1, 6)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, jobs))
    records = [json.loads(path.read_text()) for path in OUTPUT.rglob('verified.json')]
    assert len(records) == 10
    summary = []
    for area in [500, 1000]:
        rows = [row for row in records if row['area'] == area]
        dropped = [sum(row['dropped'][i] for row in rows) for i in range(3)]
        completed = [sum(row['completed'][i] for row in rows) for i in range(3)]
        summary.append(dict(area=area, power_w=sum(row['power_w'] for row in rows) / 5,
            drop_pct=[100 * d / (d + c) for d, c in zip(dropped, completed)], seeds=rows))
    (BASE / 'summary.json').write_text(json.dumps(summary, indent=2))
    (BASE / 'complete.json').write_text(json.dumps(dict(runs=10, hashes=hashes)))
    print('ALL_COMPLETE_B2000_WINNER_CROSS_10', flush=True)


if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      DEBUG='0', TRAIN='0', EVAL='1', WANDB_MODE='disabled')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if len(sys.argv) > 1:
        evaluate(int(sys.argv[2]), int(sys.argv[3]), preflight=sys.argv[1] == 'preflight')
    else:
        main()
