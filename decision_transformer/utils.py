import random
import time
import pickle
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import *



def discount_cumsum(x, gamma):
    
    if x.ndim == 2:
        x = x[..., None]  # (T, N, 1)
    
    T, N, _ = x.shape
    disc_cumsum = np.zeros_like(x, dtype=np.float32)  # (T, N, 1)

    for j in range(N):
        disc_cumsum[-1, j, 0] = x[-1, j, 0]
        for t in reversed(range(T - 1)):
            disc_cumsum[t, j, 0] = x[t, j, 0] + gamma * disc_cumsum[t + 1, j, 0]
            
    return disc_cumsum

def plot_attention_map(attn_map, context_len=10, save_path="attn_map.png", show=False):
    """
    可视化 Decision Transformer 的注意力矩阵（仅包含 s, a 两类 token）
    每个时间步包含 s, a 两种 token，因此序列长度 T = 2 * context_len
    """
    # ---- 转 numpy ----
    if not isinstance(attn_map, np.ndarray):
        attn_map = attn_map.detach().cpu().numpy()

    T = attn_map.shape[0]
    expected_T = 3 * context_len
    assert (
        T == expected_T
    ), f"attn_map shape {T} 与 3*context_len={expected_T} 不匹配"

    token_labels = ["r", "s", "a"]  # 只有 state 和 action

    # ---- 画图 ----
    plt.figure(figsize=(7, 6))
    sns.heatmap(attn_map, cmap="viridis", square=True, cbar=True)
    plt.title(f"Attention Map (context_len={context_len})", fontsize=12)
    plt.xlabel("Key tokens (past)")
    plt.ylabel("Query tokens (current)")

    # ---- 添加分割线：蓝=s, 绿=a ----
    colors = ["red", "blue", "green"]
    for i in range(context_len):
        for j, color in enumerate(colors):
            idx = 3 * i + j
            plt.axvline(idx, color=color, linestyle="--", linewidth=0.6)
            plt.axhline(idx, color=color, linestyle="--", linewidth=0.6)

    # ---- 添加 tick label ----
    tick_pos = np.arange(1.5, 3 * context_len, 3)
    tick_labels = [f"{label}{i}" for i in range(context_len) for label in token_labels]
    plt.xticks(range(T), tick_labels, rotation=90, fontsize=6)
    plt.yticks(range(T), tick_labels, rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) 
    if show:
        plt.show()
    plt.close()

    print(f"✅ Attention map saved at: {save_path}")

def evaluate_on_env(args, model, device, context_len, env, rtg_target, rtg_scale, max_test_ep_len, state_mean=None, state_std=None):

    eval_batch_size = 4  # required for forward pass

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

    obs_buffer = []
    act_buffer = []
    rew_buffer = []
    saliency_list = []
    attn_sum = None
    count = 0

    for t in trange(max_test_ep_len, file=sys.stdout):

        # add state in placeholder
        # state = torch.from_numpy(np.stack(obs)).to(device)
        # state = (state - state_mean) / state_std
        # state.requires_grad_(True)
        # states[:, t] = state
        
        states[:, t] = torch.from_numpy(np.stack(obs)).to(device)
        states[:, t] = (states[:, t] - state_mean) / state_std

        # calcualate running rtg and add it in placeholder
        running_rtg = running_rtg - running_reward / rtg_scale
        rewards_to_go[:, t] = running_rtg

        if t < context_len:
            act_preds, logits, attn_map = model.forward(timesteps[:,:context_len],
                                                        states[:,:context_len],
                                                        actions[:,:context_len],
                                                        rewards_to_go[:,:context_len])
            act = act_preds[:, t].unsqueeze(1).detach()
        else:
            act_preds, logits, attn_map = model.forward(timesteps[:,t-context_len+1:t+1],
                                                        states[:,t-context_len+1:t+1],
                                                        actions[:,t-context_len+1:t+1],
                                                        rewards_to_go[:,t-context_len+1:t+1])
            act = act_preds[:, -1].unsqueeze(1).detach()

        # attn_cur = attn_map.mean(dim=1).detach().cpu()  # 平均多头，形状 [B, T, T]
        # attn_sum = attn_cur if attn_sum is None else attn_sum + attn_cur
        # count += 1
        
        # act_idx = torch.argmax(logits[0], dim=-1) 
        # model.zero_grad()
        # target_logit = torch.gather(logits[0], dim=2, index=act_idx.unsqueeze(-1))
        # target_logit = target_logit.mean()
        # target_logit.backward(retain_graph=True)
        # saliency = (state * state.grad).abs().mean(dim=0).detach().cpu().numpy()
        # # print(f"[Step {t}] Top-5 important state dims: {np.argsort(-saliency)[:5]} {np.sort(-saliency)[:5]}")
        # saliency_list.append(saliency)

        obs, _, rewards, done, infos, _ = env.step(act.view(eval_batch_size, -1).cpu().numpy())
        running_reward = torch.tensor(rewards, dtype=torch.float32, device=device)
        # print(f"time: {infos['time']}, pc: {infos['pc']}, dp: {infos['drop_ratio']}, rwd: {infos['reward']}")
        # if t > 20:
        #     sys.exit()

        obs_buffer.append(np.array(obs))
        act_buffer.append(act.cpu().numpy())
        rew_buffer.append(np.array(rewards))

        # add action in placeholder
        actions[:, t] = act.view(eval_batch_size, -1)

        steps += 1
        total_pc += infos['pc'] 
        total_drop_ratio += infos['drop_ratio']
        
        if done:
            break

    # attn_mean = (attn_sum / count)[0]
    # plot_attention_map(attn_mean, context_len=context_len, save_path="attn_mean.png")
    # plot_and_save_saliency(saliency_list, save_dir=f"saliency_target")
    trajectory = {
            'observations': np.array(obs_buffer, dtype=np.float32),   # (T, 123)
            'actions': np.array(act_buffer, dtype=np.float32),        # (T, 2)
            'rewards': np.array(rew_buffer, dtype=np.float32),        # (T, 1)
    }

    save_dir = args.traj_save_path
    os.makedirs(args.traj_save_path, exist_ok=True)
    save_path = os.path.join(save_dir, f"1.pkl")

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = []
    
    data.append(trajectory)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(data, f)
    # print(f"Trajectory saved to: {save_path}")

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

        timesteps = torch.arange(start=si, end=ei, step=1)

        return timesteps, states, actions, returns_to_go
