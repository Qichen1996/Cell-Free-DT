# Cross-scene eval: B2000 [30,.1,.1] pc1.2 seed1 cp170 -> B500x2, cluster .9, seeds 1-5.
# Generic checkpoint sweep (actor_eps10..240 + final), seeds 1-5. Env: SWEEP_SCENE (A3x|B500x2), SWEEP_W (weights dir), SWEEP_NAME (output name).
# Scene/env patching identical to eval_A3x_B500x2_c90_v1 (cluster .9, traffic scale). Eval only, no training.
# Deterministic evaluation of the two from-scratch PPO models (train_A3x_B500x2_c90_v1) in their own training scenes:
# A3x: A traffic x3, area 1000, 16 AP, cluster .9;  B500x2: B traffic x2, area 500, 16 AP, cluster .9. Env seeds 1-5.
# Eval reward weights [40,.5,.5,3] (standard eval; only affects reward column). Final actor.pt (sha from complete.json).
# trajectory.csv deleted after each run with a record (user rule). No training, no other deletions.
import os
os.environ.update(DEBUG='0', TRAIN='0', EVAL='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import sys, json, csv, math, hashlib, statistics, subprocess, concurrent.futures, shutil
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); B = Path(__file__).parent
WD = R / 'results/weights'
O = R / 'analysis/sim_stats/cross_B2000_b500_v1'
MD = R / 'results/DT/B/mappo/B2000_30_01_01_pc12_seed1_ep180/wandb/run-20261105_143047-zdovv6uu/files'
TGT = {'B2000pc12': ['B500x2']}
ARCH = R / 'results/weights/B1000_checkpoint_test'  # architecture.json of same training recipe (searched below)
SCENES = {'A3x': dict(scenario='A', area=1000., spacing=250., nbr=.25, scale=3.), 'B500x2': dict(scenario='B', area=500., spacing=125., nbr=.125, scale=2.)}
SEEDS = [1, 2, 3, 4, 5]
sys.path.insert(0, str(R))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(model, scene, seed):
    import numpy as np, torch
    SC = SCENES[scene]; cp = '170'
    actor = MD / 'actor_eps170.pt'; critic = MD / 'critic_eps170.pt'
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    from network import config as nc
    nc.areaSize = np.array([SC['area'], SC['area']]); nc.numBS = 16; nc.interBSDist = SC['spacing']
    nc.bsPositions = nc.square_grid_centers(4, 4, SC['area'], SC['area'])
    nc.coverageGainFraction = .9; nc.minCandidateBS = 1; nc.coverageGainFractionA = .9; nc.minCandidateBSA = 1
    from traffic.model import TrafficModel
    _gar = TrafficModel.get_arrival_rates
    TrafficModel.get_arrival_rates = lambda self, time, dt: _gar(self, time, dt) * SC['scale']
    from network.base_station import BaseStation
    from network.network import MultiCellNetwork
    from network.user_equipment import UserEquipment
    from env import MultiCellNetEnv
    BaseStation.inter_dist = SC['nbr']; MultiCellNetwork.inter_bs_dist = SC['spacing']
    MultiCellNetwork.default_area = nc.areaSize; MultiCellNetwork.default_bs_poses = nc.bsPositions
    MultiCellNetEnv.bs_poses = nc.bsPositions; MultiCellNetEnv.num_agents = 16
    UserEquipment.coverage_gain_fraction = .9; UserEquipment.min_candidate_bs = 1
    UserEquipment.coverage_gain_fraction_A = .9; UserEquipment.min_candidate_bs_A = 1
    from utils import set_log_level; set_log_level('ERROR')
    stats = O / f'{model}_cp{cp}' / scene; dest = stats / SC['scenario'] / f'SEED{seed}'; assert not dest.exists()
    env = MultiCellNetEnv(seed=seed, scenario=SC['scenario'], area_size=nc.areaSize, accelerate=3000, episode_len=10080,
                          w_str=40, w_sen=.5, w_tol=.5, w_qos=40, w_xqos=0, w_pc=3, save_trajectory=True, stats_dir=str(stats))
    obs, _, _ = env.reset(); assert np.asarray(obs).shape == (16, 110)
    tm = env.net.traffic_model; r1 = _gar(tm, 0., 60.); r2 = tm.get_arrival_rates(0., 60.); assert np.allclose(r2, r1 * SC['scale'])
    assert env.net.scenario == SC['scenario'] and np.array_equal(env.net.area, nc.areaSize)
    from arguments import get_config
    from agents.mappo.policy import MappoPolicy
    args = get_config().parse_args([])
    for k, v in json.loads(Path(os.environ['ARCH_JSON']).read_text()).items(): setattr(args, k, v)
    args.model_version = ''; args.use_wandb = False
    policy = MappoPolicy(args, env.observation_space[0], env.cent_observation_space[0], env.action_space[0])
    policy.actor.load_state_dict(torch.load(actor, map_location='cpu', weights_only=False), strict=True)
    policy.critic.load_state_dict(torch.load(critic, map_location='cpu', weights_only=False), strict=True)
    policy.prep_rollout()
    ev = dict(model=model, scene=scene, cp=cp, seed=seed, steps=10080, actor=str(actor), actor_sha256=sha(actor), cluster=.9, rate_scale=SC['scale'],
              scenario=SC['scenario'], area=SC['area'], aps=16, eval_weights=[40, .5, .5, 3], deterministic=True)
    print('ENV_OK', json.dumps(ev), flush=True)
    with torch.no_grad():
        for step in range(10080):
            obs, _, _, done, _, _ = env.step(policy.act(obs, deterministic=True))
            assert bool(done) == (step == 10079)
            if (step + 1) % 1440 == 0:
                assert shutil.disk_usage(R).free > 0.5 * 1024**3; print('DAY', scene, seed, (step + 1) // 1440, flush=True)
    env.close()
    with (dest / 'reward.csv').open() as f: rr = list(csv.DictReader(f))
    assert len(rr) == 10080 and all(math.isfinite(float(v['pc_kw'])) for v in rr)
    ds = [0] * 3; cs = [0] * 3
    with (dest / 'ue_stats.csv').open() as f:
        for v in csv.DictReader(f):
            j = [.05, .15, .3].index(float(v['delay_budget'])); ds[j] += int(float(v['dropped_packets'])); cs[j] += int(float(v['completed_packets']))
    t = dest / 'trajectory.csv'
    if t.exists():
        (dest / 'trajectory_deleted.json').write_text(json.dumps(dict(file='trajectory.csv', bytes=t.stat().st_size, sha256=sha(t), reason='disk; user rule'))); t.unlink()
    res = dict(ev, power_w=statistics.mean(float(v['pc_kw']) for v in rr) * 1000, rtg0=sum(float(v['reward']) for v in rr), dropped=ds, completed=cs)
    (dest / 'result.json').write_text(json.dumps(res, indent=1))
    (B / 'done').mkdir(exist_ok=True); (B / 'done' / f'{model}__cp{cp}__{scene}__{seed}.json').write_text(json.dumps(res))
    print('CASE_DONE', scene, seed, res['power_w'], ds, cs, flush=True)

def main():
    assert not O.exists() and not (B / 'plan.json').exists()
    C = [(m, sc, str(seed)) for m in TGT for sc in TGT[m] for seed in SEEDS]
    (B / 'logs').mkdir(exist_ok=True); (B / 'plan.json').write_text(json.dumps(dict(cases=C, arch=os.environ['ARCH_JSON']), indent=1))
    def job(c):
        with (B / 'logs' / f'{c[0]}-{c[1]}-{c[2]}.log').open('x') as f:
            rc = subprocess.call([sys.executable, '-u', __file__, c[0], c[1], c[2]], cwd=R, stdout=f, stderr=subprocess.STDOUT)
        print('JOB', c, rc, flush=True); return rc
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as p: rcs = list(p.map(job, C))
    (B / 'complete.json').write_text(json.dumps(dict(runs=len(C), rc=rcs, failed=[C[k] for k, r in enumerate(rcs) if r])))
    print('ALL_COMPLETE failed', sum(r != 0 for r in rcs), flush=True)

if __name__ == '__main__':
    if len(sys.argv) > 1: worker(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else: main()
