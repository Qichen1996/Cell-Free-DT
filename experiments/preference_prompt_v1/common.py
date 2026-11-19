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
BASE = PROJECT / 'experiments/preference_prompt_v1'
DATA = PROJECT / 'analysis/trajectory/B1000_preference_v1'
STATS = PROJECT / 'analysis/sim_stats/B1000_preference_v1'
MODELS = PROJECT / 'dt_results/B1000_preference_v1'
TEACHERS = PROJECT / 'results/weights/B1000_pc_sweep'
PCS = [.6, 1.5, 1.8, 2.1]
STEPS = 10080
COMMON_WEIGHTS = [40, .5, .5, 3]
EXPECTED = {
 .6: ('37d6011481b3f4718f8e1631350ca0477b8c22eb6fb69d32c4b054201f2eac3b','edba00fd14de93ce64b46f10bef3ee83e2898b6dd7ed2922e8b8983c73954b0b'),
 .9: ('80d16912e47aa8577147b4dba8ade9ae7cae4d8d38e578aadb6e314b6a02d519','0662845f4c3c6a6cb56e21c1e29133f3ec057a32b4dc0c566f6ebec14f3e34be'),
 1.2: ('6751da872775e90ef6d198987a3b3134e74e3cbea6fe21e46c5c94d072300ce7','bbd89672c34210aee166dc47d3fa05f40a89c85631ad9ac89aef7838ff767dcb'),
 1.5: ('9fb1662bc4ac1c34c925233834672fb50811db11d38558898ff4eb94e4a50601','72e9d0f10ada3c47df8841791f9906a14cb49324efe2557591c7da90d2f436e4'),
 1.8: ('25614a2e8302063cc6b724af45ad2775952245c2fc111e68bf0f920724a8dc71','cc33a54efc05b4fe52a1a09b30f611a84240388c8437f5c44d348e3094165dbf'),
 2.1: ('82c23f6fecb8d132ba42d8a20e5afffbc53066c2755341fd98ca9cc4835de8e1','c9e2af43d8f2df358c1a66ca0e9864dcfe233bfe2dccaaf853460689bcff5e73'),
}

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
