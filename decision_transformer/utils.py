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
    total_drop_ratio = 0
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
        total_pc += infos['pc'] 
        total_drop_ratio += infos['drop_ratio']
        
        if done:
            break

    results['eval/avg_pc'] = total_pc / steps
    results['eval/avg_drop_ratio'] = total_drop_ratio / steps

    return results


class TrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale=1.0):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.windows = []
        for ti, traj in enumerate(self.trajectories):
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale
            T = traj['observations'].shape[0]
            assert T >= self.context_len, "trajectory too short"
            for si in range(0, T - self.context_len + 1, 1):
                self.windows.append((ti, si))

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
        return len(self.windows)

    def __getitem__(self, idx):
        traj_idx, si = self.windows[idx]
        traj = self.trajectories[traj_idx]
        ei = si + self.context_len

        obs = traj['observations']           # shape: (T, N, obs_dim)
        acts = traj['actions']               # shape: (T, N, act_dim)
        rtgs = traj['returns_to_go']         # shape: (T, N) or (T, N, 1)

        states = torch.from_numpy(obs[si:ei])
        actions = torch.from_numpy(acts[si:ei])
        returns_to_go = torch.from_numpy(rtgs[si:ei])
        prompt = torch.tensor(traj["prompt"], dtype=torch.long)

        timesteps = torch.arange(start=si, end=ei, step=1)

        return prompt, timesteps, states, actions, returns_to_go
