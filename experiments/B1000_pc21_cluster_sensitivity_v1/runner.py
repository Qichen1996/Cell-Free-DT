"""B1000 PPO pc2.1_trainseed5_cpfinal (DT teacher) under different UE clustering: (cluster,min_bs) in (.9,1) baseline, (.98,1), (.9,2); area 1000, seeds 1-5, eval weights. No training. (user: 你测试一下)
Based on B2000_30_01_01_pc12_cp170_cluster90/runner.py."""
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
NAME = 'B1000_pc21_cluster_sensitivity_v1'
BASE = ROOT / 'experiments' / NAME
OUTPUT = ROOT / 'analysis/sim_stats' / NAME
SOURCE = Path('/workspace/Cell-Free-DT-new/experiments/tradeoff_selected_single_collect200_v1/models/pc2.1_trainseed5_cpfinal')
MODEL = BASE / 'models'
CP = 'final'
VERSION = '' if CP == 'final' else '_eps'+CP


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


CFG = {'c90m1': (.90, 1), 'c98m1': (.98, 1), 'c90m2': (.90, 2)}


def evaluate(cfg, seed, preflight=False):
    area = 1000
    import numpy as np
    import torch
    import network.config as config

    spacing = area / 4
    cluster, min_bs = CFG[cfg]
    config.areaSize = np.array([float(area), float(area)])
    config.interBSDist = spacing
    config.bsPositions = config.square_grid_centers(4, 4, float(area), float(area))
    config.coverageGainFraction = cluster
    config.minCandidateBS = min_bs
    from network.base_station import BaseStation
    from network.user_equipment import UserEquipment
    BaseStation.inter_dist = spacing / 1000
    UserEquipment.coverage_gain_fraction = cluster
    UserEquipment.min_candidate_bs = min_bs
    sizes = []
    _orig_set = UserEquipment.channel_gains.fset
    def _set(self, g):
        r = _orig_set(self, g); sizes.append(self.cluster_size); return r
    UserEquipment.channel_gains = UserEquipment.channel_gains.setter(_set)
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
                '--stats_dir', str(OUTPUT / cfg),
                '--sim_log_path', str(BASE / 'logs' / f'{cfg}_{seed}_detail.log'),
                '--perf_save_path', str(BASE / 'logs' / f'{cfg}_{seed}_performance.csv')]
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
    sz = np.array(sizes)
    record = dict(cfg=cfg, min_candidate_bs=min_bs, cluster_size_hist={str(k): float((sz == k).mean()) for k in range(1, 5)}, cluster_size_ge5=float((sz >= 5).mean()), cluster_size_mean=float(sz.mean()), area=area, seed=seed, training_seed=5, checkpoint=CP, source_run='pc2.1_trainseed5_cpfinal',
                  model_sha256=sha(MODEL / ('actor'+VERSION+'.pt')), steps=10080, cluster=cluster,
                  aps=16, spacing_m=spacing, weights=[40, .5, .5, 3], deterministic=True,
                  power_w=sum(powers) / len(powers) * 1000, dropped=dropped, completed=completed,
                  source=str(folder))
    (folder / 'verified.json').write_text(json.dumps(record, indent=2))


def main():
    assert not (BASE / 'started.json').exists(), 'This experiment was already started'
    assert shutil.disk_usage(ROOT).free > 520 * 1024 ** 2
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / 'logs').mkdir(exist_ok=True)
    MODEL.mkdir(exist_ok=True)
    hashes = {}
    for filename in [('actor'+VERSION+'.pt'), ('critic'+VERSION+'.pt'), 'architecture.json']:
        shutil.copy2(SOURCE / filename, MODEL / filename)
        hashes[filename] = sha(MODEL / filename)
    (BASE / 'plan.json').write_text(json.dumps(dict(source=str(SOURCE), hashes=hashes,
        configs=CFG, area=1000, seeds=list(range(1, 6)), steps=10080, parallel=15, evaluation_weights=[40, .5, .5, 3]), indent=2))
    (BASE / 'started.json').write_text(json.dumps(dict(pid=os.getpid())))
    for area in CFG:
        with (BASE / 'logs' / f'preflight_{area}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', __file__, 'preflight', area, '1'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('PREFLIGHT_PASS', area, flush=True)

    def run(job):
        area, seed = job
        assert shutil.disk_usage(ROOT).free > 500 * 1024 ** 2
        print('SEED_START', area, seed, flush=True)
        with (BASE / 'logs' / f'{area}_{seed}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', __file__, 'worker', area, str(seed)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('SEED_COMPLETE', area, seed, flush=True)

    jobs = [(area, seed) for area in CFG for seed in range(1, 6)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as pool:
        list(pool.map(run, jobs))
    records = [json.loads(path.read_text()) for path in OUTPUT.rglob('verified.json')]
    assert len(records) == 15
    summary = []
    for area in CFG:
        rows = [row for row in records if row['cfg'] == area]
        dropped = [sum(row['dropped'][i] for row in rows) for i in range(3)]
        completed = [sum(row['completed'][i] for row in rows) for i in range(3)]
        summary.append(dict(cfg=area, cluster_size_mean=sum(r['cluster_size_mean'] for r in rows) / 5, single_ap=sum(r['cluster_size_hist']['1'] for r in rows) / 5, str_dropped=dropped[0], power_w=sum(row['power_w'] for row in rows) / 5,
            drop_pct=[100 * d / (d + c) for d, c in zip(dropped, completed)], seeds=rows))
    (BASE / 'summary.json').write_text(json.dumps(summary, indent=2))
    (BASE / 'complete.json').write_text(json.dumps(dict(runs=15, hashes=hashes)))
    print('ALL_COMPLETE', flush=True)


if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      DEBUG='0', TRAIN='0', EVAL='1', WANDB_MODE='disabled')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if len(sys.argv) > 1:
        evaluate(sys.argv[2], int(sys.argv[3]), preflight=sys.argv[1] == 'preflight')
    else:
        main()