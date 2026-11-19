#!/usr/bin/env python
# %%
import yaml
import torch
import pickle
from utils import *
from agents import *
from arguments import *
from env import MultiCellNetEnv
from decision_transformer.utils import TrajectoryDataset, evaluate_on_env
from decision_transformer.model import DecisionTransformer
from visualize.render import create_dash_app
# %reload_ext autoreload
# %autoreload 2

sim_days = 7
accelerate = 3000
render_interval = 4
model_params = dict(w_str=600, w_sen=200, w_tol=30, w_pc=10, no_interf=False, max_sleep=3, no_offload=False)

parser = get_config()
parser.add_argument("--perf_save_path", default="results/performance.csv",
                    help="path to save the performance of the simulation")
parser.add_argument("--render_interval", type=int, default=render_interval,
                    help="interval of rendering")
parser.add_argument("--days", type=int, default=sim_days,
                    help="number of days to simulate")
# parser.add_argument("--count-flops", action="store_true",
#                     help="count flops of the model")
parser.add_argument("--stochastic", action="store_true",
                    help="whether to use stochastic policy")
parser.add_argument("--model_index", type=int, default=0,
                    help=("index of the compatible saved run to load after sorting "
                          "by modification time (0 is the newest; default: 0)"))
parser.add_argument("--traj_save_path", default="analysis/trajectory",
                    help="path to save trajectory")
parser.add_argument("--traj_name", default=None,
                    help=("trajectory file name (without .pkl) under traj_save_path; "
                          "default: <scenario>_<numBS>bs_<area side>m, e.g. B_25bs_1000m"))

parser.add_argument('--dataset_dir', type=str, default='analysis/trajectory')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--rtg_scale', type=int, default=1)
parser.add_argument('--max_eval_ep_len', type=int)
parser.add_argument('--context_len', type=int, default=20)
parser.add_argument('--target_rtg', type=float)
parser.add_argument('--n_blocks', type=int, default=3)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--dropout_p', type=float, default=0.1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--no_traj_pkl', action='store_true', help='Do not collect or save trajectory PKL; keep simulation CSVs')


env_parser = get_env_config()
    
parser.set_defaults(log_level='NOTICE', group_name='C')
env_parser.set_defaults(accelerate=accelerate)

try:
    args, env_args = parser.parse_known_args()
    env_args = env_parser.parse_args(env_args)
except:
    args = parser.parse_args([])
    env_args = env_parser.parse_args([])
    
AGENT = args.algorithm_name

if args.experiment_name == 'test': args.use_wandb = False
args.num_env_steps = args.days * 24 * 3600 * 50 // env_args.accelerate
if env_args.stats_dir is None:
    env_args.stats_dir = f'analysis/sim_stats/{env_args.scenario}'
args.max_eval_ep_len = args.num_env_steps

# %%
set_log_level(args.log_level)

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# %% [markdown]
# ## Simulation Parameters

# %%
def make_env(args, seed=None):
    kwds = vars(args).copy()
    kwds['no_interf'] = False  # always has interference at test time
    env = MultiCellNetEnv(seed=seed, **{
        k: v for k, v in kwds.items() if v is not None})
    pars = inspect.signature(MultiCellNetEnv.__init__).parameters
    # [setattr(args, k, pars[k].default) for k, v in kwds.items() if v is None]
    for k, v in kwds.items():
        if v is None and k in pars:
            setattr(args, k, pars[k].default)
    return env

def get_model_dir(args, env_args, run_dir, version=''):
    assert run_dir.exists(), "Run directory does not exist: {}".format(run_dir)
    if args.model_dir is not None:
        
        return run_dir / args.model_dir
    if args.model_index < 0:
        raise ValueError("--model_index must be non-negative")
    p = 'wandb/run-*%s/files/' if args.use_wandb else '%s/models/'
    dirs = run_dir.glob(p % version)
    compatible_dirs = []
    for d in sorted(dirs, key=os.path.getmtime, reverse=True):
        config_path = d/'config.yaml'
        if not config_path.exists():
            warn("no config file in %s" % d)
            compatible_dirs.append(d)
            continue
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            if all(getattr(env_args, k) == cfg[k]['value']
                    for k in model_params if k in cfg):
                compatible_dirs.append(d)

    if not compatible_dirs:
        raise FileNotFoundError("no compatible model directory")
    if args.model_index >= len(compatible_dirs):
        raise IndexError(
            f"--model_index {args.model_index} is out of range; "
            f"found {len(compatible_dirs)} compatible run(s)")

    selected_dir = compatible_dirs[args.model_index]
    print(f"Loading model index {args.model_index}: {selected_dir}")
    return selected_dir

env = make_env(env_args, seed=args.seed)

obs_space = env.observation_space[0]
cent_obs_space = env.cent_observation_space[0]
action_space = env.action_space[0]

run_dir = get_run_dir(args, env_args)

if args.sim_log_path is None:
    # fn = '{}_{}_{}_acc-{}.log'.format(
    #     AGENT, env_args.scenario,
    #     re.sub('(, |:)', '-', env.net.world_time_repr),
    #     env.net.accelerate)
    fn = 'simulation.log'
    args.sim_log_path = 'logs/' + fn

if AGENT == 'mappo':
    model_dir = args.model_dir or get_model_dir(args, env_args, run_dir, version=args.run_version)
    agent = MappoPolicy(args, obs_space, cent_obs_space, action_space, model_dir=model_dir)
elif AGENT == 'dqn':
    model_dir = args.model_dir or get_model_dir(args, env_args, run_dir, version=args.run_version)
    agent = DQNPolicy(obs_space, action_space, model_dir=model_dir, model_version=args.model_version)
elif AGENT == 'dt':
    device = torch.device(args.device)
    eval_max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    use_norm = args.norm
    rtg_scale = args.rtg_scale

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    eval_model = DecisionTransformer(
        state_dim=obs_space.shape[0],
        act_dim=action_space.shape[0],
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        max_timestep=10080
    ).to(device)
    agent = eval_model
elif AGENT == 'fixed':
    agent = AlwaysOnPolicy(action_space, env.num_agents)
elif AGENT == 'random':
    agent = RandomPolicy(action_space, env.num_agents)
elif AGENT == 'simple':
    agent = SimplePolicy(action_space, env.num_agents)
elif AGENT == 'simple1':
    agent = SimplePolicySM1Only(action_space, env.num_agents)
elif AGENT == 'simple2':
    agent = SimplePolicyNoSM3(action_space, env.num_agents)
elif AGENT == 'sleepy':
    agent = SleepyPolicy(action_space, env.num_agents)
else:
    raise ValueError('invalid agent type')
print('Policy:', type(agent).__name__)

for par, defval in model_params.items():
    val = getattr(env_args, par)
    if val != defval:
        env.stats_dir += f'_{par}={val}'
if args.model_version:
    env.stats_dir += '_eps=%s' % args.model_version

print(env.full_stats_dir)
if os.path.exists(env.full_stats_dir):
    print('Simulation already done.')
    exit()

env.print_info()
env.net.traffic_model.print_info()

set_log_file(args.sim_log_path)

traffic_to_prompt = {
    "A": 0,
    "B": 1,
    "C": 2,
}
scenario = env.net.scenario

# %%
from datetime import datetime

render_mode = args.use_render and ('dash' if args.use_dash else 'frame')

# %%
def simulate():
    step_rewards = []
    obs, _, _ = env.reset(render_mode)

    # for i in trange(args.num_env_steps, file=sys.stdout):
    #     actions = agent.act(obs, deterministic=not args.stochastic)
    #     obs, _, rewards, done, _, _ = env.step(
    #         actions, render_mode=render_mode, render_interval=render_interval)
    # env._trajectory = env._trajectory[-1:]

    if AGENT == 'dt':

        if use_norm:
            eval_chk_pt_path = f'dt_results/dt_norm_{context_len}_{rtg_scale}.pt'
        else:
            eval_chk_pt_path = f'dt_results/dt_{context_len}_{rtg_scale}.pt'
        eval_model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device)['model_state_dict'])
        print("model loaded from: " + eval_chk_pt_path)

        result_dir = "dt_results"
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"eval_results.csv")

        if use_norm:
            dataset_path = f'{args.dataset_dir}/c.pkl'
            traj_dataset = TrajectoryDataset(dataset_path, context_len)
            state_mean, state_std = traj_dataset.get_state_stats()
        else:
            state_mean, state_std = None, None

        # evaluate on env
        rtg_targets = [args.target_rtg]
        for target in rtg_targets:
            results = evaluate_on_env(args, eval_model, device, context_len, env, target, rtg_scale, eval_max_eval_ep_len, state_mean, state_std)
    
            eval_avg_pc = results['eval/avg_pc']
            eval_avg_drop_ratio = results['eval/avg_drop_ratio']
    
            print("=" * 60 + '\n' +
                  "rtg target: " + str(target) + '\n' +
                  "eval avg pc: " + format(eval_avg_pc, ".4f") + '\n' +
                  "eval avg drop ratio: " + format(eval_avg_drop_ratio, ".4f")
            )
    
            env.stats_dir = f'analysis/sim_stats/dt'
            
    else:
        # collect trajectory
        obs_buffer = []
        act_buffer = []
        rew_buffer = []

        for i in trange(args.num_env_steps, file=sys.stdout):
            actions = agent.act(obs, deterministic=not args.stochastic)

            if not args.no_traj_pkl:
                obs_buffer.append(np.array(obs))
                act_buffer.append(actions)
            
            obs, _, rewards, done, infos, _ = env.step(
                actions, render_mode=render_mode, render_interval=render_interval)

            if not args.no_traj_pkl:
                rew_buffer.append(np.array(rewards))
            step_rewards.append(np.mean(rewards))

        if not args.no_traj_pkl:
            save_traj_pkl(obs_buffer, act_buffer, rew_buffer)

        rewards = pd.Series(np.squeeze(step_rewards), name='reward')

        info = rewards.describe()
        info.index = ['reward_' + str(i) for i in info.index]
        info['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info['agent'] = AGENT
        info['total_steps'] = args.num_env_steps
        info['accelerate'] = env_args.accelerate
        info['scenario'] = env.net.traffic_scenario
        info['traffic_density'] = env.net.traffic_model.density_mean
        info['w_pc'] = env.w_pc
        info['w_qos'] = env.w_qos
        info['w_xqos'] = env.w_xqos
        print(info)
        info.to_frame().T.set_index('time').to_csv(args.perf_save_path, mode='a', header=not os.path.exists(args.perf_save_path))
        if args.use_render and not args.use_dash:
            return env.animate()

def save_traj_pkl(obs_buffer, act_buffer, rew_buffer):
        trajectory = {
            # [traffic scenario id (A/B/C -> 0/1/2), number of BSs, area side length in m]
            "prompt": np.array([traffic_to_prompt[scenario], env.net.num_bs, env.net.area[0]], dtype=np.float32),
            'observations': np.array(obs_buffer, dtype=np.float32),   # (T, 123)
            'actions': np.array(act_buffer, dtype=np.float32),        # (T, 2)
            'rewards': np.array(rew_buffer, dtype=np.float32),        # (T, 1)
        }

        # if args.seed == 1:
        save_dir = args.traj_save_path
        os.makedirs(args.traj_save_path, exist_ok=True)
        # name by scenario, number of BSs and area side length (not by model)
        traj_name = args.traj_name or f"{scenario}_{env.net.num_bs}bs_{int(env.net.area[0])}m"
        save_path = os.path.join(save_dir, f"{traj_name}.pkl")
    
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = []
        
        data.append(trajectory)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Trajectory saved to: {save_path}")
    

simulate()
env.close()

# %%
if args.use_dash:
    # threading.Thread(target=simulate).start()
    app = create_dash_app(env, args)
    app.run_server(debug=True)
