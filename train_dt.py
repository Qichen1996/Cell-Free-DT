import csv
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils import *
from arguments import *
from decision_transformer.utils import TrajectoryDataset, evaluate_on_env
from decision_transformer.model import DecisionTransformer



def train(args, env_args):

    rtg_scale = args.rtg_scale      # normalize returns to go
    use_norm = args.norm

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    # load data from this file
    dataset_path = f'{args.dataset_dir}/ab.pkl'

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M")

    if use_norm:
        save_model_name = f"dt_norm_{context_len}_{rtg_scale}.pt"
        log_csv_name = f"log_norm_{context_len}_{rtg_scale}.csv"
    else:
        log_csv_name = f"log_{context_len}_{rtg_scale}.csv"
        save_model_name = f"dt_{context_len}_{rtg_scale}.pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    # save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_path = os.path.join(log_dir, log_csv_name)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    traj_dataset = TrajectoryDataset(dataset_path, context_len, rtg_scale)
    # subset = Subset(traj_dataset, [0])
    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True
                        )

    data_iter = iter(traj_data_loader)

    # dims from dataset: observations (T, N, state_dim), actions (T, N, act_dim)
    state_dim = traj_dataset.trajectories[0]['observations'].shape[-1]
    act_dim = traj_dataset.trajectories[0]['actions'].shape[-1]

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        max_timestep=10080
    ).to(device)

    if use_norm:
        traj_dataset.get_state_stats()

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    if use_norm:
        eval_chk_pt_path = f'dt_results/dt_norm_{context_len}_{rtg_scale}.pt'
    else:
        eval_chk_pt_path = f'dt_results/dt_{context_len}_{rtg_scale}.pt'
    # checkpoint = torch.load(eval_chk_pt_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

    scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lr_lambda=lambda step: 1 - step / max_train_iters / num_updates_per_iter / 4
                        )

    log_records = []
    total_updates = 0

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                # timesteps, states, actions, returns_to_go = next(data_iter)
                prompts, timesteps, states, actions, returns_to_go = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                # timesteps, states, actions, returns_to_go = next(data_iter)
                prompts, timesteps, states, actions, returns_to_go = next(data_iter)

            timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x N x state_dim
            actions = actions.to(device)        # B x T x N x act_dim
            returns_to_go = returns_to_go.to(device)  # B x T x N x 1
            prompts = prompts.to(device)

            B, T, N, state_dim = states.shape
            act_dim = actions.shape[-1]

            states_flat = states.permute(0, 2, 1, 3).reshape(B * N, T, state_dim)
            actions_flat = actions.permute(0, 2, 1, 3).reshape(B * N, T, act_dim)
            returns_to_go_flat = returns_to_go.permute(0, 2, 1, 3).reshape(B * N, T, 1)
            timesteps_flat = timesteps.unsqueeze(1).expand(B, N, T).reshape(B * N, T)
            prompts_flat = prompts[:, None].expand(B, N).reshape(B * N)

            action_logits = model.forward(
                                                    timesteps=timesteps_flat,
                                                    states=states_flat,
                                                    actions=actions_flat,
                                                    returns_to_go=returns_to_go_flat,
                                                    prompts=prompts_flat,
                                                )

            target_1 = actions_flat[..., 0].long()
            target_2 = actions_flat[..., 1].long()

            loss1 = F.cross_entropy(
                action_logits[0].reshape(-1, action_logits[0].size(-1)),
                target_1.reshape(-1),
                reduction='mean'
            )
            loss2 = F.cross_entropy(
                action_logits[1].reshape(-1, action_logits[1].size(-1)),
                target_2.reshape(-1),
                reduction='mean'
            )

            action_loss = (loss1 + loss2) / 2
            
            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())

        # # evaluate action accuracy
        # results = evaluate_on_env(model, device, context_len, env, rtg_target, max_eval_ep_len)

        # eval_avg_pc = results['eval/avg_pc']
        # eval_avg_drop_ratio = results['eval/avg_drop_ratio']

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n'
                # "eval avg pc: " + format(eval_avg_pc, ".4f") + '\n' +
                # "eval avg drop ratio: " + format(eval_avg_drop_ratio, ".4f")
            )

        print(log_str)

        log_records.append([time_elapsed, total_updates, mean_action_loss])

        # save model
        # print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        # if eval_d4rl_score >= max_d4rl_score:
        #     print("saving max d4rl score model at: " + save_best_model_path)
        #     torch.save(model.state_dict(), save_best_model_path)
        #     max_d4rl_score = eval_d4rl_score

        # torch.save(model.state_dict(), save_model_path)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_model_path)

    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["duration", "num_updates", "action_loss"])
        writer.writerows(log_records)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M")
    print("total training time: " + time_elapsed)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--rtg_scale', type=int, default=1)
    parser.add_argument('--norm', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='analysis/trajectory')
    parser.add_argument('--log_dir', type=str, default='dt_results/')

    parser.add_argument('--context_len', type=int, default=10)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)

    parser.add_argument('--max_train_iters', type=int, default=1500)
    parser.add_argument('--num_updates_per_iter', type=int, default=150)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    env_parser = get_env_config()

    try:
        args, env_args = parser.parse_known_args()
        env_args = env_parser.parse_args(env_args)
    except:
        args = parser.parse_args([])
        env_args = env_parser.parse_args([])

    train(args, env_args)