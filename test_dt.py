import torch
import csv
import numpy as np
from utils import *
from arguments import *
from env import MultiCellNetEnv
from datetime import datetime
from decision_transformer.utils import evaluate_on_env
from decision_transformer.model import DecisionTransformer




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


def test(args, env_args):

    eval_max_eval_ep_len = args.max_eval_ep_len  # max len of one episode

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    device = torch.device(args.device)
    print("device set to: ", device)

    eval_env = make_env(env_args, seed=args.seed)

    state_dim = eval_env.observation_space[0].shape[0]
    act_dim = eval_env.action_space[0].shape[0]

    eval_model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        max_timestep=25200
    ).to(device)

    eval_chk_pt_path = f'dt_results/model_{context_len}.pt'
    eval_model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))
    print("model loaded from: " + eval_chk_pt_path)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = "dt_results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"eval_results_{timestamp}.csv")

    # evaluate on env
    rtg_targets = [-200]
    for target in rtg_targets:
        results = evaluate_on_env(eval_model, context_len, eval_env, target, eval_max_eval_ep_len)

        eval_avg_pc = results['eval/avg_pc']
        eval_avg_drop_ratio = results['eval/avg_drop_ratio']

        print("=" * 60 + '\n' +
              "rtg target: " + str(target) + '\n' +
              "eval avg pc: " + format(eval_avg_pc, ".4f") + '\n' +
              "eval avg drop ratio: " + format(eval_avg_drop_ratio, ".4f")
        )

        eval_env.stats_dir = f'analysis/sim_stats/dt_{target}'
        eval_env.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_eval_ep_len', type=int, default=10080)
    parser.add_argument('--context_len', type=int, default=5)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')

    env_parser = get_env_config()

    try:
        args, env_args = parser.parse_known_args()
        env_args = env_parser.parse_args(env_args)
    except:
        args = parser.parse_args([])
        env_args = env_parser.parse_args([])

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env_args.accelerate = 3000

    test(args, env_args)