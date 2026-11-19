import os
os.environ.update(DEBUG='0', TRAIN='0', EVAL='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

PROJECT = Path('/workspace/Cell-Free-DT-new')
sys.path.insert(0, str(PROJECT))
BASE = PROJECT / 'experiments/preference_prompt_v2'
DATA = PROJECT / 'analysis/trajectory/B1000_preference_v2'
STATS = PROJECT / 'analysis/sim_stats/B1000_preference_v2'
MODELS = PROJECT / 'dt_results/B1000_preference_v2'
TEACHERS = PROJECT / 'results/weights/B1000_pc_sweep_extension'
PCS = [.6, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
STEPS = 10080
COMMON_WEIGHTS = [40, .5, .5, 3]
EXPECTED = {2.4: ('3293eac73793e7f81fc0cb5da981ef25c9c69b67bf7adde228d4b3fe358f9d5a', '1631acbe08fa9d65334713bfdacc2987228652a2acb804058fa5f585dde6d2e0'), 2.7: ('3dbad535ee3f6bf239ad83f98d8c15aac47007fbee162e6f33ab80aacd874a98', '170eee46c23d8637edb98d8324fbf26981bd8fcb4ad1cb7b7fc57031a20f7a72'), 3.0: ('74d97ffd7ea4c41d3afaeded5420c54103512d135f193ab174bfb342b7e1c9c5', '0d1641ea51c837b855efc3193ba58a450d9d8892ce3410b2542fbb191568c424')}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)

def space():
    free = shutil.disk_usage(PROJECT).free
    if free < 2 * 1024**3:
        raise RuntimeError(f'Free disk below2GiB: {free}; no deletion authorized')
    return free

def validate_sources():
    for pc, pair in EXPECTED.items():
        folder = TEACHERS / f'pc_{pc:.1f}'
        for name, expected in zip(['actor.pt', 'critic.pt'], pair):
            assert sha(folder / name) == expected, (pc, name)
        for name, expected in json.loads((folder / 'manifest.json').read_text()).items():
            assert sha(folder / name) == expected, (pc, name)

def environment(seed, stats):
    from network import config as nc
    nc.areaSize = np.array([1000., 1000.])
    nc.numBS = 16
    nc.interBSDist = 250.
    nc.bsPositions = nc.square_grid_centers(4, 4, 1000., 1000.)
    nc.coverageGainFraction = .9
    nc.minCandidateBS = 1
    from network.base_station import BaseStation
    from network.network import MultiCellNetwork
    from network.user_equipment import UserEquipment
    from env import MultiCellNetEnv
    BaseStation.inter_dist = .25
    MultiCellNetwork.inter_bs_dist = 250.
    MultiCellNetwork.default_area = nc.areaSize
    MultiCellNetwork.default_bs_poses = nc.bsPositions
    MultiCellNetEnv.bs_poses = nc.bsPositions
    MultiCellNetEnv.num_agents = 16
    UserEquipment.coverage_gain_fraction = .9
    UserEquipment.min_candidate_bs = 1
    from utils import set_log_level
    set_log_level('ERROR')
    env = MultiCellNetEnv(seed=seed, scenario='B', area_size=nc.areaSize,
        accelerate=3000, episode_len=STEPS, w_str=40, w_sen=.5, w_tol=.5,
        w_qos=40, w_xqos=0, w_pc=3, save_trajectory=True, stats_dir=str(stats))
    obs, _, _ = env.reset()
    assert np.asarray(obs).shape == (16, 110)
    assert np.array_equal(env.net.area, nc.areaSize)
    assert np.array_equal(np.array([b.pos[:2] for b in env.net.bss.values()]), nc.bsPositions)
    assert BaseStation.inter_dist == .25 and UserEquipment.coverage_gain_fraction == .9
    assert UserEquipment.min_candidate_bs == 1
    return env, obs

def teacher(pc, env):
    from arguments import get_config
    from agents.mappo.policy import MappoPolicy
    folder = TEACHERS / f'pc_{pc:.1f}'
    config = get_config().parse_args([])
    for key, value in json.loads((folder / 'architecture.json').read_text()).items():
        setattr(config, key, value)
    config.model_version = ''
    config.use_wandb = False
    policy = MappoPolicy(config, env.observation_space[0], env.cent_observation_space[0], env.action_space[0])
    policy.actor.load_state_dict(torch.load(folder / 'actor.pt', map_location='cpu', weights_only=False), strict=True)
    policy.critic.load_state_dict(torch.load(folder / 'critic.pt', map_location='cpu', weights_only=False), strict=True)
    policy.prep_rollout()
    return policy

def check_trajectory(tr):
    assert np.array_equal(tr['prompt'], [1,16,1000])
    for key, shape in [('observations',(STEPS,16,110)),('actions',(STEPS,16,2)),('rewards',(STEPS,16,1))]:
        arr = np.asarray(tr[key])
        assert arr.shape == shape and np.isfinite(arr).all(), (key, arr.shape)
    assert np.equal(tr['actions'], np.floor(tr['actions'])).all()
    assert ((tr['actions'][...,0] >= 0) & (tr['actions'][...,0] < 3)).all()
    assert ((tr['actions'][...,1] >= 0) & (tr['actions'][...,1] < 4)).all()
    assert tr['preference_pc'] in PCS and tr['weights'] == COMMON_WEIGHTS

def validate_csv(folder):
    import csv
    for name in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
        path = Path(folder) / (name+'.csv')
        assert path.is_file() and path.stat().st_size > 0, str(path)
    with (Path(folder)/'reward.csv').open() as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == STEPS
