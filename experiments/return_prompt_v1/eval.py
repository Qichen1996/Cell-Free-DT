"""Isolated evaluation of trained raw-RTG + return-prompt model."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
os.environ.update(TRAIN='0', EVAL='1', DEBUG='0')
import numpy as np
import torch
from model import ReturnPromptDT

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(PROJECT))


def make_env(case, seed, stats_dir):
    from network import config as nc
    from network.base_station import BaseStation
    from network.network import MultiCellNetwork
    from network.user_equipment import UserEquipment
    from env import MultiCellNetEnv
    from utils import set_log_level
    set_log_level('ERROR')
    area, grid, cluster = case['area'], case['grid'], case['cluster']
    nc.areaSize = np.array([float(area), float(area)])
    nc.numBS = grid**2
    nc.interBSDist = area/grid
    nc.bsPositions = nc.square_grid_centers(grid, grid, area, area)
    nc.coverageGainFraction = cluster
    nc.coverageGainFractionA = .98
    nc.minCandidateBS = nc.minCandidateBSA = 1
    BaseStation.inter_dist = nc.interBSDist/1000
    MultiCellNetwork.inter_bs_dist = nc.interBSDist
    MultiCellNetwork.default_area = nc.areaSize
    MultiCellNetwork.default_bs_poses = nc.bsPositions
    MultiCellNetEnv.bs_poses = nc.bsPositions
    MultiCellNetEnv.num_agents = nc.numBS
    UserEquipment.coverage_gain_fraction = cluster
    UserEquipment.coverage_gain_fraction_A = .98
    UserEquipment.min_candidate_bs = UserEquipment.min_candidate_bs_A = 1
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    e = MultiCellNetEnv(seed=seed, area_size=nc.areaSize,
        scenario=case['scenario'], accelerate=3000, episode_len=10080,
        w_str=40., w_sen=.5, w_tol=.5, w_pc=3.,
        save_trajectory=True, include_bs_info=True, stats_dir=stats_dir)
    obs = np.asarray([b.get_observation() for b in e.net.bss.values()])
    assert obs.shape == (grid**2, 110)
    assert np.array_equal(np.asarray(e.net.area), nc.areaSize)
    assert np.array_equal(np.asarray([b.pos[:2] for b in e.net.bss.values()]), nc.bsPositions)
    assert all(b.pos[2] == BaseStation.bs_height for b in e.net.bss.values())
    assert BaseStation.inter_dist == area/grid/1000
    print('EVAL_ENV', case['name'], 'AREA', area, 'AP', grid**2,
          'INTER_DIST_M', area/grid, 'NEIGHBOR_KM', BaseStation.inter_dist,
          'CLUSTER', cluster, 'MIN_AP', 1, 'OBS_SHAPE_OK', obs.shape, flush=True)
    return e


def validate_outputs(output):
    for name in ('reward','ue_stats','trajectory','net_stats','bs_stats'):
        assert (output/f'{name}.csv').stat().st_size > 0
    with (output/'reward.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==10080
    assert all(np.isfinite(float(r['pc_kw'])) for r in rows)


@torch.no_grad()
def rollout(model, env, target, prompt_value):
    obs, _, _ = env.reset()
    n = env.num_agents
    states=torch.zeros(n,10080,110)
    actions=torch.zeros(n,10080,2)
    rtgs=torch.zeros(n,10080,1)
    timesteps=torch.arange(10080).repeat(n,1)
    running=torch.full((n,1),float(target))
    start=time.monotonic()
    for t in range(10080):
        states[:,t]=torch.as_tensor(np.asarray(obs),dtype=torch.float32)
        rtgs[:,t]=running
        lo=max(0,t-7)
        p=torch.full((n,t-lo+1,1),float(prompt_value))
        logits=model(timesteps[:,lo:t+1],states[:,lo:t+1],actions[:,lo:t+1],rtgs[:,lo:t+1],p)
        act=torch.stack([v[:,-1].argmax(-1) for v in logits],dim=-1)
        obs,_,reward,done,_,_=env.step(act.numpy())
        actions[:,t]=act
        # Original reward units, no subtraction from normalized RTG or prompt.
        running-=torch.as_tensor(reward,dtype=torch.float32).reshape(n,1)
        assert torch.isfinite(running).all()
        if (t+1)%1440==0:
            print('DAY_COMPLETE', (t+1)//1440, 'STEPS',t+1,
                  'SECONDS',round(time.monotonic()-start,1),flush=True)
        if done:
            assert t==10079, f'Early episode termination at {t+1}'
    return {'steps':10080,'duration_seconds':time.monotonic()-start,
            'final_raw_rtg_mean':float(running.mean())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True)
    p.add_argument('--seeds',nargs='+',type=int,help='Explicit supplemental seeds; no overwrite')
    p.add_argument('--validate_only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    plan=json.loads((Path(__file__).parent/'evaluation_plan.json').read_text())
    case=next(c for c in plan['cases'] if c['name']==args.case)
    model_path=PROJECT/plan['model']
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=False)
    assert checkpoint['round']==1800 and checkpoint['updates']==270000
    model=ReturnPromptDT().eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    prompt=float(checkpoint['inference_prompt'])
    assert 0<prompt<1
    sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    seeds=args.seeds if args.seeds is not None else plan['seeds']
    if len(set(seeds))!=len(seeds) or not all(1<=s<=5 for s in seeds):
        p.error('Seeds must be unique values in 1..5')
    for seed in seeds:
        output_root=PROJECT/case['output']
        output=output_root/case['scenario']/f'SEED{seed}'
        if not args.validate_only and output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
        env=make_env(case,seed,str(output_root))
        assert Path(env.full_stats_dir)==output
        if args.validate_only:
            obs=np.asarray([b.get_observation() for b in env.net.bss.values()])
            logits=model(torch.zeros(env.num_agents,1,dtype=torch.long),
                torch.tensor(obs[:,None],dtype=torch.float32),torch.zeros(env.num_agents,1,2),
                torch.full((env.num_agents,1,1),case['rtg']),torch.full((env.num_agents,1,1),prompt))
            assert all(torch.isfinite(x).all() for x in logits)
            print('MODEL_FORWARD_OK',case['name'],'RTG',case['rtg'],'PROMPT',prompt,flush=True)
            return
        print('SEED_START',seed,'RTG',case['rtg'],'PROMPT',prompt,flush=True)
        try:
            result=rollout(model,env,case['rtg'],prompt)
        finally:
            env.close()
        validate_outputs(output)
        metadata={'seed':seed,'rtg':case['rtg'],'return_prompt':prompt,
          'model':str(model_path),'model_sha256':sha,'environment':case,
          'weights':[40,.5,.5,3],'prompt_mode':'constant historical training maximum',**result}
        (output/'evaluation.json').write_text(json.dumps(metadata,indent=2))
        print('SEED_COMPLETE',seed,flush=True)
    print('ALL_COMPLETE',args.case,flush=True)


if __name__=='__main__':
    main()
