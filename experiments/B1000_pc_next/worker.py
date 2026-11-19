"""Isolated Vast PC3.3/3.6 training; source code and old model outputs untouched."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.update(DEBUG='0', TRAIN='1', EVAL='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
PROJECT=Path('/workspace/Cell-Free-DT-new')
OUTPUT=PROJECT/'results/weights/B1000_pc_sweep_next'
SOURCE=PROJECT/'results/DT/B/mappo/check/wandb/run-20261003_065202-mfxf31yk/files/config.yaml'
sys.path.insert(0,str(PROJECT))

# Apply deployment settings in process, including any spawned rollout workers.
import numpy as np
from network import config as nc
nc.areaSize=np.array([1000.,1000.]);nc.numBS=16;nc.interBSDist=250.
nc.bsPositions=nc.square_grid_centers(4,4,1000.,1000.)
nc.coverageGainFraction=.9;nc.minCandidateBS=1
from network.base_station import BaseStation
from network.network import MultiCellNetwork
from network.user_equipment import UserEquipment
from env import MultiCellNetEnv
BaseStation.inter_dist=.25
MultiCellNetwork.inter_bs_dist=250.
MultiCellNetwork.default_area=nc.areaSize
MultiCellNetwork.default_bs_poses=nc.bsPositions
MultiCellNetEnv.bs_poses=nc.bsPositions;MultiCellNetEnv.num_agents=16
UserEquipment.coverage_gain_fraction=.9;UserEquipment.min_candidate_bs=1


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(pc,smoke=False,preflight=False):
    import torch
    import train
    from utils import set_log_level
    set_log_level('ERROR')
    assert torch.cuda.is_available(),'CUDA required for this Vast sweep'
    assert float(torch.ones(1,device='cuda').sum())==1.
    env=MultiCellNetEnv(seed=1,scenario='B',area_size=nc.areaSize,
        accelerate=3000,episode_len=10080,w_str=80,w_sen=.2,w_tol=.2,w_pc=pc)
    obs=np.array([bs.get_observation() for bs in env.net.bss.values()])
    assert obs.shape==(16,110),obs.shape
    assert np.array_equal(env.net.area,nc.areaSize)
    assert np.array_equal(np.array([bs.pos[:2] for bs in env.net.bss.values()]),nc.bsPositions)
    print('ENV_VERIFIED B1000 16AP SPACING_M=250 NEIGHBOR_KM=.25 CLUSTER=.90 OBS',obs.shape,flush=True)
    if preflight:return
    dest=OUTPUT/(f'_smoke_pc_{pc:.1f}' if smoke else f'pc_{pc:.1f}')
    if dest.exists():raise FileExistsError(f'No overwrite: {dest}')
    dest.mkdir(parents=True)
    train.get_run_dir=lambda args,env_args:dest
    original=train.get_env_config
    def env_parser():
        p=original();p.set_defaults(area_size=nc.areaSize);return p
    train.get_env_config=env_parser
    # Preserve learner settings from the actual source config where parsers expose them.
    import yaml
    cfg={k:v['value'] for k,v in yaml.safe_load(SOURCE.read_text()).items()
         if isinstance(v,dict) and 'value' in v}
    from trainers.mappo_trainer import get_mappo_config,MappoTrainer
    parser_factories=[train.get_config,get_mappo_config]
    runtime=dict(algorithm_name='mappo',experiment_name='B1000_weights',n_rollout_threads=7,
        n_training_threads=2,num_env_steps=10080 if smoke else 2520000,
        use_wandb=True,model_dir=None,use_eval=False,seed=1,log_level='ERROR',
        sim_log_path=str(dest/'environment.log'),log_interval=1,save_interval=1)
    argv=[];seen=set()
    for factory in parser_factories:
        parser=factory()
        for action in parser._actions:
            name=action.dest
            if name in seen or name in ['help','model_dir'] or not action.option_strings:continue
            seen.add(name);value=runtime.get(name,cfg.get(name,action.default))
            if value is None:continue
            if isinstance(action,argparse._StoreTrueAction):
                if value:argv.append(action.option_strings[0])
            elif isinstance(action,argparse._StoreFalseAction):
                if not value:argv.append(action.option_strings[0])
            elif isinstance(action,argparse._StoreAction) and action.nargs in [None,'?']:
                if isinstance(value,(str,int,float)):argv.extend([action.option_strings[0],str(value)])
    argv+=['--scenario','B','--accelerate','3000','--w_str','80','--w_sen','0.2','--w_tol','0.2','--w_pc',str(pc)]
    metadata={'pc':pc,'weights':[80,.2,.2,pc],'area':1000,'AP':16,'spacing_m':250,'cluster':.9,
        'seed':1,'rounds':1 if smoke else 250,'total_steps':10080 if smoke else 2520000,
        'sampling_workers':7,'logical_rollout_length':1440,'source_rollout_workers':21,
        'source_rollout_length':480,'runtime_difference':'Vast training; match the previous local sweep with 7 workers x1440, same 10080 steps per update. Different hardware/backend may affect numerical results.',
        'baseline_note':'Two additional weights3.3/3.6 trained from scratch with matched250round budget.',
        'source_config':str(SOURCE),'source_config_sha256':hash_file(SOURCE),
        'argv':argv,'wandb':True,'algorithm_unchanged':'mappo, use_centralized_V as source config',
        'source_sha256':{str(p.relative_to(PROJECT)):hash_file(p) for p in
            [PROJECT/'train.py',PROJECT/'utils.py',PROJECT/'network/config.py',PROJECT/'trainers/mappo_trainer.py']}}
    (dest/'experiment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    old_log=MappoTrainer.log_train
    def record(self,info,steps):
        old_log(self,info,steps)
        row={'round':steps//10080,'steps':steps,**{k:float(v) for k,v in info.items() if np.isscalar(v)}}
        with (dest/'training_metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print('ROUND_COMPLETE',row['round'],'STEPS',steps,'PC',row.get('pc_kw_mean'),'REWARD',row.get('reward_mean'),flush=True)
    MappoTrainer.log_train=record
    print('TRAIN_START',pc,'SMOKE',smoke,'OUTPUT',dest,flush=True)
    started=time.time();os.chdir(PROJECT)
    import wandb
    init_original=wandb.init
    saved={}
    def named_init(*args,**kwargs):
        kwargs['name']=f'B1000_pc{pc:.1f}'+('_smoke' if smoke else '')
        run=init_original(*args,**kwargs)
        saved.update(run_id=run.id,model_dir=run.dir,url=run.url)
        (dest/'wandb-run.json').write_text(json.dumps(saved,indent=2))
        return run
    wandb.init=named_init
    train.main(argv)
    model=Path(saved['model_dir'])/'actor.pt'
    assert model.is_file() and model.stat().st_size>0
    with (dest/'complete.json').open('x',encoding='utf-8') as f:
        json.dump({'pc':pc,'seconds':time.time()-started,'actor_sha256':hash_file(model),
                   'actor_path':str(model),'steps':metadata['total_steps']},f,indent=2)
    print('ALL_TRAIN_COMPLETE',pc,flush=True)


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    p=argparse.ArgumentParser();p.add_argument('--pc',type=float,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--preflight',action='store_true')
    a=p.parse_args();assert a.pc in [3.3,3.6]
    run(a.pc,a.smoke,a.preflight)


