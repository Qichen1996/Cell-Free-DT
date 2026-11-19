"""CSV-only checkpoint screening with disk guards and exact model provenance."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'experiments/return_prompt_v1'))
sys.path.insert(1,str(PROJECT))
import numpy as np
import torch
import yaml
from eval import make_env,validate_outputs
from arguments import get_config
from agents.mappo.policy import MappoPolicy

SOURCE=PROJECT/'results/DT/B/mappo/check/wandb/run-20261003_065202-mfxf31yk/files'
CASE=dict(name='B',scenario='B',area=1000,grid=4,cluster=.90)
ROOT=PROJECT/'analysis/sim_stats/B1000_checkpoint_screen'


def sha(p):
    return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()


@torch.no_grad()
def main(cp,validate=False):
    torch.set_num_threads(1)
    model_path=SOURCE/f'actor_eps{cp}.pt'
    assert model_path.is_file()
    config=get_config().parse_args([])
    cfg=yaml.safe_load((SOURCE/'config.yaml').read_text())
    for k,v in cfg.items():
        if hasattr(config,k) and isinstance(v,dict) and 'value' in v:setattr(config,k,v['value'])
    config.model_version=f'_eps{cp}'
    config.use_wandb=False
    actor_sha=sha(model_path)
    for seed in ([999] if validate else [1,2,3]):
        free=shutil.disk_usage(PROJECT).free
        if free<2*1024**3:raise RuntimeError(f'Disk reserve under2GiB: {free}; stopping without deletion')
        base=ROOT/f'cp{cp:03d}';out=base/'B'/f'SEED{seed}'
        if not validate and out.exists():raise FileExistsError(out)
        env=make_env(CASE,seed,str(base))
        policy=MappoPolicy(config,env.observation_space[0],env.cent_observation_space[0],
            env.action_space[0],model_dir=str(SOURCE))
        obs,_,_=env.reset()
        first=policy.act(obs,deterministic=True)
        assert first.shape==(16,2) and np.isfinite(first).all()
        if validate:
            print('MODEL_ENV_VALIDATED',cp,'SHA256',actor_sha,flush=True)
            return
        print('SEED_START',cp,seed,'FREE_BYTES',free,flush=True)
        start=time.monotonic()
        for t in range(10080):
            act=first if t==0 else policy.act(obs,deterministic=True)
            obs,_,reward,done,_,_=env.step(act)
            if done:assert t==10079
            if (t+1)%1440==0:
                print('DAY_COMPLETE',cp,seed,(t+1)//1440,round(time.monotonic()-start,1),flush=True)
                if shutil.disk_usage(PROJECT).free<2*1024**3:
                    raise RuntimeError('Disk reserve below2GiB during rollout; no deletion or automatic retry')
        env.close();validate_outputs(out)
        assert sha(model_path)==actor_sha
        meta={'model_source':'mfxf31yk','checkpoint_suffix':f'_eps{cp}',
            'actor':str(model_path),'actor_sha256':actor_sha,'seed':seed,'steps':10080,
            'environment':CASE,'training_weights':[80,.2,.2,1.8],
            'evaluation_weights':[40,.5,.5,3],'traj_pkl':False,'seconds':time.monotonic()-start}
        with (out/'evaluation.json').open('x') as f:json.dump(meta,f,indent=2)
        print('SEED_COMPLETE',cp,seed,flush=True)
    print('ALL_COMPLETE',cp,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=int,choices=[40,120,220],required=True)
    p.add_argument('--validate_only',action='store_true');a=p.parse_args();main(a.checkpoint,a.validate_only)
