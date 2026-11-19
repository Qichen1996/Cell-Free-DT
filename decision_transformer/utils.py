import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import *

traffic_to_prompt = {"A": 0, "B": 1, "C": 2}

def discount_cumsum(x, gamma):
    
    if x.ndim == 2:
        x = x[..., None]  # (T, N, 1)
    
    T, N, _ = x.shape
    disc_cumsum = np.zeros_like(x, dtype=np.float32)  # (T, N, 1)

    # loop over time only; all agents are updated at once
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(T - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
            
    return disc_cumsum

@torch.no_grad()
def evaluate_on_env(args, model, device, context_len, env, rtg_target, rtg_scale, max_test_ep_len, state_mean=None, state_std=None):

    eval_batch_size = env.num_agents  # required for forward pass

    prompts = torch.full(
        (eval_batch_size,),
        traffic_to_prompt[env.net.scenario],
        dtype=torch.long,
        device=device,
    )

    if getattr(model, 'deployment_prompt', False):
        prompts = torch.tensor([traffic_to_prompt[env.net.scenario], env.num_agents, env.net.area[0]], device=device).repeat(eval_batch_size, 1)

    results = {}

    state_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    # import wandb
    # import socket
    # wandb.init(
    #             config=args,
    #             project=args.env_name,
    #             entity=args.user_name,
    #             notes=socket.gethostname(),
    #             name=f"{args.algorithm_name}_{args.experiment_name}_seed{args.seed}",
    #             group=env_args.scenario,
    #             job_type="training",
    #             reinit=True,
    #         )
    
    # zeros place holders
    actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                        dtype=torch.float32, device=device)
    states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                        dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                        dtype=torch.float32, device=device)

    # init episode
    obs, _, _ = env.reset()
    running_reward = 0
    running_rtg = torch.ones((eval_batch_size, 1), device=device) * (rtg_target / rtg_scale)

    total_pc = 0
    total_drop_ratio = np.zeros(3, dtype=np.float64)
    steps = 0

    for t in trange(max_test_ep_len, file=sys.stdout):

        # add state in placeholder
        states[:, t] = torch.from_numpy(np.stack(obs)).to(device)
        states[:, t] = (states[:, t] - state_mean) / state_std

        # calcualate running rtg and add it in placeholder
        running_rtg = running_rtg - running_reward / rtg_scale
        rewards_to_go[:, t] = running_rtg

        if t < context_len:
            logits = model.forward(timesteps[:,:context_len],
                                   states[:,:context_len],
                                   actions[:,:context_len],
                                   rewards_to_go[:,:context_len],
                                   prompts=prompts,)
            idx = t
        else:
            logits = model.forward(timesteps[:,t-context_len+1:t+1],
                                   states[:,t-context_len+1:t+1],
                                   actions[:,t-context_len+1:t+1],
                                   rewards_to_go[:,t-context_len+1:t+1],
                                   prompts=prompts,)
            idx = -1
        act = torch.stack([l[:, idx].argmax(dim=-1) for l in logits], dim=-1).unsqueeze(1)

        obs, _, rewards, done, infos, _ = env.step(act.view(eval_batch_size, -1).cpu().numpy())
        running_reward = torch.tensor(rewards, dtype=torch.float32, device=device)
        # print(f"time: {infos['time']}, pc: {infos['pc']}, dp: {infos['drop_ratio']}, rwd: {infos['reward']}")
        # if t > 20:
        #     sys.exit()

        # add action in placeholder
        actions[:, t] = act.view(eval_batch_size, -1)

        steps += 1
        # Reward statistics are the same records exported to reward.csv.
        # EVAL info_dict does not expose the legacy drop_ratio field.
        reward_stats = env._reward_stats[-1]
        total_pc += reward_stats['pc_kw']
        total_drop_ratio += np.array([
            reward_stats['str_drop'], reward_stats['sen_drop'],
            reward_stats['tor_drop'],
        ], dtype=np.float64)
        
        if done:
            break

    results['eval/avg_pc_kw'] = total_pc / steps
    for category, ratio in zip(('str', 'sen', 'tor'), total_drop_ratio / steps):
        results[f'eval/avg_{category}_drop_ratio'] = float(ratio)

    return results


class TrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale=1.0, seed_fraction=1.0, single_ap=False):

        self.context_len = context_len

        # load dataset
        paths = dataset_path if isinstance(dataset_path, (list, tuple)) else [dataset_path]
        if not 0 < seed_fraction <= 1:
            raise ValueError('seed_fraction must be in (0, 1]')
        self.trajectories = []
        # Older local exports may use NumPy 2 module paths on NumPy 1 runtimes.
        import sys
        import numpy.core.numeric
        import numpy.core.multiarray
        sys.modules.setdefault('numpy._core.numeric', numpy.core.numeric)
        sys.modules.setdefault('numpy._core.multiarray', numpy.core.multiarray)
        for path in paths:
            from trajectory_io import load_trajectory_pickle
            items = load_trajectory_pickle(path)
            keep = max(1, int(len(items) * seed_fraction))
            print('DATA_SELECTION', path, 'selected', keep, 'of', len(items), flush=True)
            self.trajectories.extend(items[:keep])
        self.single_ap = single_ap
        self.window_ends = []
        self.window_counts = []

        self.windows = []
        for ti, traj in enumerate(self.trajectories):
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale
            T = traj['observations'].shape[0]
            assert T >= self.context_len, "trajectory too short"
            count = T - self.context_len + 1
            self.window_counts.append(count)
            self.window_ends.append((self.window_ends[-1] if self.window_ends else 0) + count * (traj['observations'].shape[1] if single_ap else 1))

    def get_state_stats(self):
        # state_means = []
        # state_stds = []
        # for traj in self.trajectories:
        #     # used for input normalization
        #     mean = np.mean(traj['observations'], axis=0)
        #     std = np.std(traj['observations'], axis=0) + 1e-6
        #     state_means.append(mean)
        #     state_stds.append(std)
        #     traj['observations'] = (traj['observations'] - mean) / std

        # state_means = np.vstack(state_means)
        # state_stds = np.vstack(state_stds)
    
        # self.state_mean = np.mean(state_means, axis=0, keepdims=True)  # (1, state_dim)
        # self.state_std = np.mean(state_stds, axis=0, keepdims=True) + 1e-6  # (1, state_dim)

        traj = self.trajectories[0]
        self.state_mean = np.mean(traj['observations'], axis=0)
        self.state_std = np.std(traj['observations'], axis=0) + 1e-6

        for traj in self.trajectories:
            # used for input normalization
            # mean = np.mean(traj['observations'], axis=0)
            # std = np.std(traj['observations'], axis=0) + 1e-6
            mean = self.state_mean
            std = self.state_std
            traj['observations'] = (traj['observations'] - mean) / std

        return self.state_mean, self.state_std

    def __len__(self):
        return self.window_ends[-1]

    def __getitem__(self, idx):
        import bisect
        traj_idx = bisect.bisect_right(self.window_ends, idx)
        offset = idx - (self.window_ends[traj_idx-1] if traj_idx else 0)
        ap, si = divmod(offset, self.window_counts[traj_idx])
        traj = self.trajectories[traj_idx]
        ei = si + self.context_len

        obs = traj['observations']           # shape: (T, N, obs_dim)
        acts = traj['actions']               # shape: (T, N, act_dim)
        rtgs = traj['returns_to_go']         # shape: (T, N) or (T, N, 1)

        states = torch.from_numpy(obs[si:ei])
        actions = torch.from_numpy(acts[si:ei])
        returns_to_go = torch.from_numpy(rtgs[si:ei])
        if self.single_ap:
            states = states[:, ap:ap+1]
            actions = actions[:, ap:ap+1]
            returns_to_go = returns_to_go[:, ap:ap+1]
        prompt = torch.tensor(traj["prompt"], dtype=torch.long)

        timesteps = torch.arange(start=si, end=ei, step=1)

        return prompt, timesteps, states, actions, returns_to_go
