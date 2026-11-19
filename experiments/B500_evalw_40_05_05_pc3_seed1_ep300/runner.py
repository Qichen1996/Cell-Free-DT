"""Train B500 PPO with training weights = evaluation weights (str40 sen.5 tol.5 pc3 qos40), seed1, 300 episodes; same B500 setup/hyperparams as B500_150_cluster90_mb16_p5_r28_pc07_ep400 (28 workers x 360, 16 minibatches). Purpose: consistent objective for DT data (user: 在500试一下)."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import sys
import time

ROOT = Path('/workspace/Cell-Free-DT-new')
NAME = 'B500_evalw_40_05_05_pc3_seed1_ep300'
BASE = ROOT / 'experiments' / NAME
STEPS = 300 * 10080


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    os.environ.update(TRAIN='1', EVAL='0', DEBUG='0', WANDB_MODE='online',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    assert not (BASE / 'started.json').exists(), 'Experiment already started'
    assert shutil.disk_usage(ROOT).free > 1024 ** 3
    import numpy as np
    import torch
    import network.config as config
    config.areaSize = np.array([500., 500.])
    config.numBS = 16
    config.interBSDist = 125.
    config.bsPositions = config.square_grid_centers(4, 4, *config.areaSize)
    config.coverageGainFraction = .9
    config.minCandidateBS = 1
    from network.base_station import BaseStation
    from network.user_equipment import UserEquipment
    BaseStation.inter_dist = .125
    UserEquipment.coverage_gain_fraction = .9
    UserEquipment.min_candidate_bs = 1
    from env import MultiCellNetEnv
    MultiCellNetEnv.bs_poses = config.bsPositions
    MultiCellNetEnv.num_agents = 16
    environment = MultiCellNetEnv(area_size=config.areaSize, scenario='B', accelerate=3000,
                                  w_str=40., w_sen=.5, w_tol=.5, w_pc=3., seed=1)
    assert environment.num_agents == 16
    assert environment.observation_space[0].shape == (110,)
    assert np.array_equal(environment.net.area, config.areaSize)
    assert environment.episode_len // 28 == 360
    assert torch.cuda.is_available()
    torch.ones(1, device='cuda').sum().item()

    import train
    original_kwargs = train.get_env_kwargs
    train.get_env_kwargs = lambda args: dict(original_kwargs(args), area_size=config.areaSize)
    original_run_dir = train.get_run_dir

    def checked_run_dir(args, env_args):
        args.opti_eps = float(args.opti_eps)
        assert args.num_env_steps == STEPS
        assert args.episode_length == 360 and args.n_rollout_threads == 28
        assert not args.use_linear_lr_decay and args.model_dir is None
        assert (args.w_str, args.w_sen, args.w_tol, args.w_pc) == (40., .5, .5, 3.)
        assert args.lr == args.critic_lr == .0005
        values = {key: value for key, value in vars(args).items()
                  if isinstance(value, (str, int, float, bool, type(None)))
                  and not any(word in key.lower() for word in ['key', 'token', 'password'])}
        atomic_json(BASE / 'actual_config.json', values)
        path = original_run_dir(args, env_args)
        assert not path.exists(), 'Existing model directory'
        return path

    train.get_run_dir = checked_run_dir
    from trainers.mappo_trainer import MappoTrainer
    original_init = MappoTrainer.__init__
    original_log = MappoTrainer.log_train
    state = {}

    def capture_init(self, configuration):
        original_init(self, configuration)
        state['trainer'] = self
        atomic_json(BASE / 'model_location.json', dict(path=str(self.save_dir)))

    def capture_log(self, values, steps):
        clean = {}
        for key, value in values.items():
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            clean[key] = float(np.asarray(value).mean())
            assert math.isfinite(clean[key]), key
        original_log(self, clean, steps)
        atomic_json(BASE / 'progress.json', dict(steps=int(steps), episode=int(steps) // 10080,
                                               target_steps=STEPS, metrics=clean))
        with (BASE / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(steps=int(steps), **clean)) + '\n')
        if shutil.disk_usage(ROOT).free < 500 * 1024 ** 2:
            raise RuntimeError('Training stopped because free storage is below 500 MiB')

    MappoTrainer.__init__ = capture_init
    MappoTrainer.log_train = capture_log
    atomic_json(BASE / 'plan.json', dict(weights=[40, .5, .5, 3], seed=1, episodes=300,
        steps=STEPS, area_m=500, aps=16, spacing_m=125, cluster=.9, min_candidate_bs=1,
        rollout_workers=28, rollout_steps=360, training_threads=5, ppo_epochs=5,
        minibatches=16, lr=.0005, critic_lr=.0005, linear_lr_decay=False,
        from_scratch=True, checkpoint_interval=10, wandb='online'))
    atomic_json(BASE / 'preflight.json', dict(status='PASS', obs=[16, 110], cuda=True,
                                             area=[500, 500], cluster=.9))
    atomic_json(BASE / 'started.json', dict(pid=os.getpid(), time=time.time()))
    print('PREFLIGHT_PASS B500 40/0.5/0.5/3 SEED1 EP300', flush=True)
    train.main(['--algorithm_name', 'mappo', '--experiment_name', NAME, '--scenario', 'B',
        '--accelerate', '3000', '--seed', '1', '--n_training_threads', '5',
        '--n_rollout_threads', '28', '--num_mini_batch', '16', '--num_env_steps', str(STEPS),
        '--ppo_epoch', '5', '--gain', '.01', '--gamma', '.99', '--lr', '.0005',
        '--critic_lr', '.0005', '--value_loss_coef', '1', '--save_interval', '10',
        '--log_interval', '1', '--w_str', '40', '--w_sen', '.5', '--w_tol', '.5',
        '--w_pc', '3', '--w_qos', '40', '--w_xqos', '0', '--user_name', 'qichenw',
        '--log_level', 'NOTICE'])
    trainer = state['trainer']
    progress = json.loads((BASE / 'progress.json').read_text())
    assert progress['steps'] == STEPS
    hashes = {}
    for filename in ['actor.pt', 'critic.pt']:
        path = Path(trainer.save_dir) / filename
        assert path.is_file()
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(BASE / 'complete.json', dict(steps=STEPS, episodes=300, hashes=hashes,
                                             model_dir=str(trainer.save_dir)))
    print('ALL_COMPLETE_B500_EVALW_EP300', flush=True)


if __name__ == '__main__':
    main()

