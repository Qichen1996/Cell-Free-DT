"""A3x retrain with w_str=80 (user: [80,.1,.1,1]); otherwise identical. From-scratch PPO training (same recipe as B1000_pc_next/worker.py, 300 rounds) in new scenes, cluster .90:
A3x: A traffic x3, area 1000, 16 AP, reward [50,.1,.1,pc] (user). B500x2: B traffic x2, area 500, 16 AP, reward [80,.2,.2,pc]."""
import argparse
import os as _o
SCENES={'A3x':dict(scenario='A',area=1000.,spacing=250.,nbr=.25,scale=3.,wq=(100,.2,.1)),'B500x2':dict(scenario='B',area=500.,spacing=125.,nbr=.125,scale=2.,wq=(80,.2,.2))}
SC=SCENES[_o.environ['FT_SCENE']]
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.update(DEBUG='0', TRAIN='1', EVAL='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
PROJECT=Path('/workspace/Cell-Free-DT-new')
OUTPUT=PROJECT/'results/weights/train_A3x_s100_n02_e300_c90_v1'
SOURCE=PROJECT/'results/DT/B/mappo/check/wandb/run-20261003_065202-mfxf31yk/files/config.yaml'
sys.path.insert(0,str(PROJECT))

# Apply deployment settings in process, including any spawned rollout workers.
import numpy as np
from network import config as nc
nc.areaSize=np.array([SC['area'],SC['area']]);nc.numBS=16;nc.interBSDist=SC['spacing']
nc.bsPositions=nc.square_grid_centers(4,4,SC['area'],SC['area'])
nc.coverageGainFraction=.9;nc.minCandidateBS=1;nc.coverageGainFractionA=.9;nc.minCandidateBSA=1
from traffic.model import TrafficModel
_gar=TrafficModel.get_arrival_rates
TrafficModel.get_arrival_rates=lambda self,time,dt:_gar(self,time,dt)*SC['scale']
from network.base_station import BaseStation
from network.network import MultiCellNetwork
from network.user_equipment import UserEquipment
from env import MultiCellNetEnv
BaseStation.inter_dist=SC['nbr']
MultiCellNetwork.inter_bs_dist=SC['spacing']
MultiCellNetwork.default_area=nc.areaSize
MultiCellNetwork.default_bs_poses=nc.bsPositions
MultiCellNetEnv.bs_poses=nc.bsPositions;MultiCellNetEnv.num_agents=16
UserEquipment.coverage_gain_fraction=.9;UserEquipment.min_candidate_bs=1
UserEquipment.coverage_gain_fraction_A=.9;UserEquipment.min_candidate_bs_A=1


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
    env=MultiCellNetEnv(seed=1,scenario=SC['scenario'],area_size=nc.areaSize,
        accelerate=3000,episode_len=10080,w_str=SC['wq'][0],w_sen=SC['wq'][1],w_tol=SC['wq'][2],w_pc=pc)
    obs=np.array([bs.get_observation() for bs in env.net.bss.values()])
    assert obs.shape==(16,110),obs.shape
    assert np.array_equal(env.net.area,nc.areaSize)
    assert np.array_equal(np.array([bs.pos[:2] for bs in env.net.bss.values()]),nc.bsPositions)
    tm=env.net.traffic_model; r1=_gar(tm,0.,60.); r2=tm.get_arrival_rates(0.,60.); assert np.allclose(r2,r1*SC['scale'])
    assert UserEquipment.coverage_gain_fraction_A==.9 and UserEquipment.coverage_gain_fraction==.9
    assert (env.net.w_str,env.net.w_sen,env.net.w_tol,env.net.w_pc)==(*SC['wq'],pc)
    print('ENV_VERIFIED',_o.environ['FT_SCENE'],SC,'scene',env.net.scenario,'rate_ratio',float(r2.sum()/max(r1.sum(),1e-12)),'REWARD_W',env.net.w_str,env.net.w_sen,env.net.w_tol,env.net.w_pc,'OBS',obs.shape,flush=True)
    if preflight:return
    dest=OUTPUT/_o.environ['FT_SCENE']/(f'_smoke_pc_{pc:.1f}' if smoke else f'pc_{pc:.1f}')
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
    runtime=dict(algorithm_name='mappo',experiment_name='train_'+_o.environ['FT_SCENE'],n_rollout_threads=7,
        n_training_threads=2,num_env_steps=10080 if smoke else 3024000,
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
    argv+=['--scenario',SC['scenario'],'--accelerate','3000','--w_str',str(SC['wq'][0]),'--w_sen',str(SC['wq'][1]),'--w_tol',str(SC['wq'][2]),'--w_pc',str(pc)]
    metadata={'pc':pc,'weights':[*SC['wq'],pc],'scene':_o.environ['FT_SCENE'],**SC,'AP':16,'cluster':.9,'init':'from scratch',
        'seed':1,'rounds':1 if smoke else 300,'total_steps':10080 if smoke else 3024000,
        'sampling_workers':7,'logical_rollout_length':1440,'source_rollout_workers':21,
        'source_rollout_length':480,'runtime_difference':'Vast training; match the previous local sweep with 7 workers x1440, same 10080 steps per update. Different hardware/backend may affect numerical results.',
        'baseline_note':'From scratch, 300 rounds, same budget as B1000 sweep.',
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
        kwargs['name']=f"{_o.environ['FT_SCENE']}_pc{pc:.1f}"+('_smoke' if smoke else '')
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
    a=p.parse_args();assert a.pc in [0.6,0.9,1.0,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3]
    run(a.pc,a.smoke,a.preflight)


