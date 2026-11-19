"""Matched original PPO ABC runs, stop after first7200 action steps."""
import argparse,csv,json,time,pickle
from pathlib import Path
import screen as s

ROOT=s.PROJECT/'analysis/sim_stats/PPO_ABC_first5'

@s.torch.no_grad()
def main(scenario,validate=False):
    s.torch.set_num_threads(1)
    plan=json.loads((Path(__file__).parent/'abc_first5_plan.json').read_text())
    case=plan[scenario];source=s.PROJECT/case['source'];actor=source/f"actor{case['suffix']}.pt"
    assert actor.is_file()
    cfg=s.get_config().parse_args([])
    if (source/'config.yaml').exists():
        for k,v in s.yaml.safe_load((source/'config.yaml').read_text()).items():
            if hasattr(cfg,k) and isinstance(v,dict) and 'value' in v:setattr(cfg,k,v['value'])
    cfg.model_version=case['suffix'];cfg.use_wandb=False
    actor_sha=s.sha(actor)
    if scenario=='C':assert actor_sha=='646b1b61e38879c3deb885b2ef7e6162f4c997f7e61323d838915064a2e0bc3f'
    for seed in ([1] if validate else range(1,6)):
        assert s.shutil.disk_usage(s.PROJECT).free>2*1024**3,'Disk reserve below2GiB'
        base=ROOT/scenario;out=base/scenario/f'SEED{seed}'
        assert not out.exists(),f'Refusing overwrite: {out}'
        env=s.make_env(dict(name=scenario,scenario=scenario,area=1000,grid=4,cluster=.98 if scenario=='A' else .9),seed,str(base))
        policy=s.MappoPolicy(cfg,env.observation_space[0],env.cent_observation_space[0],env.action_space[0],model_dir=str(source))
        obs,_,_=env.reset();policy.prep_rollout()
        if validate:
            for t,expected in enumerate(case['prefix_pc'][str(seed)]):
                obs,_,_,_,_,_=env.step(policy.act(obs,deterministic=True))
                actual=float(env.net.power_consumption)*.001
                assert abs(actual-expected)<1e-8,(scenario,seed,t,actual,expected)
            print('PREFLIGHT_MATCHED_10_STEPS',scenario,'ACTOR_SHA256',actor_sha,flush=True)
            return
        print('SEED_START',scenario,seed,flush=True);start=time.monotonic()
        for t in range(7200):
            obs,_,_,done,_,_=env.step(policy.act(obs,deterministic=True))
            assert not done,'Unexpected early end'
            if t<10:
                expected=case['prefix_pc'][str(seed)][t]
                assert abs(float(env.net.power_consumption)*.001-expected)<1e-8,(scenario,seed,t,'original7day prefix mismatch')
            if (t+1)%1440==0:
                print('DAY_COMPLETE',scenario,seed,(t+1)//1440,flush=True)
                assert s.shutil.disk_usage(s.PROJECT).free>2*1024**3
        env.close()
        for name in ['reward','trajectory','ue_stats','net_stats','bs_stats']:assert (out/f'{name}.csv').stat().st_size>0
        with (out/'reward.csv').open() as f:assert sum(1 for _ in f)==7201
        with (out/'trajectory.csv').open() as f:assert sum(1 for _ in f)==7202
        assert s.sha(actor)==actor_sha
        meta={'scenario':scenario,'seed':seed,'steps':7200,'environment_horizon':10080,'area':1000,'ap_count':16,
              'inter_dist_m':250,'cluster':.98 if scenario=='A' else .9,'min_ap':1,'weights':[40,.5,.5,3],
              'actor':str(actor),'actor_sha256':actor_sha,'original_7day_reference':case['reference'],
              'original_prefix_verified_steps':10,'packet_metric':'ue.ipynb pooled dropped/(completed+dropped)',
              'seconds':time.monotonic()-start,'traj_pkl':False}
        with (out/'evaluation.json').open('x') as f:json.dump(meta,f,indent=2)
        print('SEED_COMPLETE',scenario,seed,flush=True)
    print('ALL_COMPLETE_FIRST5',scenario,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',choices=list('ABC'),required=True)
    parser.add_argument('--validate_only',action='store_true');args=parser.parse_args()
    main(args.scenario,args.validate_only)
