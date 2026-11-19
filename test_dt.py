import argparse
import json
from pathlib import Path
import numpy as np
import torch
from arguments import get_env_config
from env import MultiCellNetEnv
from decision_transformer.model import DecisionTransformer
from decision_transformer.utils import evaluate_on_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--target_rtg', type=float, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,2,3,4,5])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--context_len', type=int, default=8)
    parser.add_argument('--rtg_scale', type=float, default=1)
    parser.add_argument('--max_eval_ep_len', type=int, default=10080)
    parser.add_argument('--eval_area', type=float)
    parser.add_argument('--eval_grid', type=int)
    parser.add_argument('--eval_cluster', type=float)
    parser.add_argument('--validate_env_only', action='store_true')
    args, remaining = parser.parse_known_args()
    env_args = get_env_config().parse_args(remaining)
    env_args.accelerate = 3000
    if args.eval_area is not None:
        if args.eval_grid is None or args.eval_cluster is None:
            parser.error('--eval_area requires --eval_grid and --eval_cluster')
        from network import config as nc
        from network.base_station import BaseStation
        from network.network import MultiCellNetwork
        from network.user_equipment import UserEquipment
        nc.areaSize = np.array([args.eval_area, args.eval_area])
        nc.numBS = args.eval_grid ** 2
        nc.interBSDist = args.eval_area / args.eval_grid
        nc.bsPositions = nc.square_grid_centers(args.eval_grid, args.eval_grid, args.eval_area, args.eval_area)
        BaseStation.inter_dist = nc.interBSDist / 1000
        MultiCellNetwork.inter_bs_dist = nc.interBSDist
        MultiCellNetwork.default_area = nc.areaSize
        MultiCellNetwork.default_bs_poses = nc.bsPositions
        MultiCellNetEnv.bs_poses = nc.bsPositions
        MultiCellNetEnv.num_agents = nc.numBS
        env_args.area_size = nc.areaSize
        nc.coverageGainFraction = args.eval_cluster
        nc.coverageGainFractionA = .98
        UserEquipment.coverage_gain_fraction = args.eval_cluster
        UserEquipment.coverage_gain_fraction_A = .98
        UserEquipment.min_candidate_bs = 1
        UserEquipment.min_candidate_bs_A = 1
        print('EVAL_ENV', args.eval_area, nc.numBS, 'INTER_DIST_M', nc.interBSDist,
              'NEIGHBOR_DIST_KM', BaseStation.inter_dist, 'CLUSTER', args.eval_cluster, flush=True)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=True)
    state = checkpoint['model_state_dict']
    model = DecisionTransformer(state_dim=state['embed_state.weight'].shape[1], act_dim=2,
        n_blocks=3,h_dim=128,context_len=args.context_len,n_heads=1,drop_p=.1,
        max_timestep=state['embed_timestep.weight'].shape[0],deployment_prompt='prompt_scale' in state,
        no_prompt=checkpoint.get('args', {}).get('no_prompt', False)).to(args.device)
    model.load_state_dict(state)
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = MultiCellNetEnv(seed=seed, **{k:v for k,v in vars(env_args).items() if v is not None})
        obs = [bs.get_observation() for bs in env.net.bss.values()]
        assert np.asarray(obs).shape == (env.num_agents, state['embed_state.weight'].shape[1]), np.asarray(obs).shape
        print('OBS_SHAPE_OK', np.asarray(obs).shape, flush=True)
        if args.validate_env_only:
            return
        output = Path(env.full_stats_dir)
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
        print('SEED_START', seed, 'RTG', args.target_rtg, flush=True)
        try:
            result = evaluate_on_env(args,model,torch.device(args.device),args.context_len,env,args.target_rtg,args.rtg_scale,args.max_eval_ep_len)
        finally:
            env.close()
        (output/'evaluation.json').write_text(json.dumps({'seed':seed,'rtg':args.target_rtg,'model':args.model_path,**result}))
        print('SEED_COMPLETE', seed, flush=True)
    print('ALL_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
