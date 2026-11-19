"""Matched CSV-only evaluation of completed pc3.3/3.6 PPO policies."""
import sys
from pathlib import Path
PROJECT=Path('/workspace/Cell-Free-DT-new')
sys.path.insert(0,str(PROJECT/'experiments/preference_prompt_v1'))
from common import environment,validate_csv,save_new,sha,space,np,torch,os,json
import concurrent.futures
import subprocess
import csv

BASE=PROJECT/'results/weights/B1000_pc_sweep_next'
OUT=PROJECT/'analysis/sim_stats/B1000_pc_sweep_next'
EXPECTED={3.3:'c0aaac3cedf7a740855ba5644ed4b49033fca640094abeecedc8ff7dc5c016c8',
          3.6:'5cd5343c5fe520695dcb1cbcf0d1660a6609f0f3a39be9dac89fcf63f4dbcf8e'}

def source(pc):
    folder=BASE/f'pc_{pc:.1f}'
    completed=json.loads((folder/'complete.json').read_text())
    experiment=json.loads((folder/'experiment.json').read_text())
    actor=Path(completed['actor_path'])
    assert actor.resolve().is_relative_to(folder.resolve()) and actor.name=='actor.pt'
    assert completed['steps']==2520000 and completed['actor_sha256']==EXPECTED[pc]
    assert sha(actor)==EXPECTED[pc]
    assert experiment['weights']==[80,.2,.2,pc] and experiment['rounds']==250
    reference=PROJECT/'results/weights/B1000_pc_sweep/pc_0.6'
    previous=json.loads((reference/'experiment.json').read_text())
    assert experiment['source_config_sha256']==previous['source_config_sha256']
    architecture=json.loads((reference/'architecture.json').read_text())
    return actor,actor.parent/'critic.pt',architecture

@torch.no_grad()
def job(pc,seed,preflight=False):
    from arguments import get_config
    from agents.mappo.policy import MappoPolicy
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor,critic,architecture=source(pc)
    critic_hash=sha(critic)
    stats=OUT/f'pc_{pc:.1f}'
    dest=stats/'B'/f'SEED{seed}'
    assert not dest.exists(),dest
    env,obs=environment(seed,stats)
    args=get_config().parse_args([])
    for key,value in architecture.items():
        setattr(args,key,value)
    args.model_version=''
    args.use_wandb=False
    policy=MappoPolicy(args,env.observation_space[0],env.cent_observation_space[0],env.action_space[0])
    policy.actor.load_state_dict(torch.load(actor,map_location='cpu',weights_only=False),strict=True)
    policy.critic.load_state_dict(torch.load(critic,map_location='cpu',weights_only=False),strict=True)
    policy.prep_rollout()
    act=policy.act(obs,deterministic=True)
    assert act.shape==(16,2) and np.isfinite(act).all()
    if preflight:
        print('PREFLIGHT_OK',pc,'DIST250m NEIGHBOR.25 CLUSTER.90 OBS16x110 SHA',sha(actor),flush=True)
        return
    print('SEED_START',pc,seed,flush=True)
    for step in range(10080):
        act=policy.act(obs,deterministic=True)
        obs,_,_,done,_,_=env.step(act)
        assert bool(done)==(step==10079)
        if (step+1)%1440==0:
            space()
            print('DAY_COMPLETE',pc,seed,(step+1)//1440,flush=True)
    env.close()
    validate_csv(dest)
    assert sha(actor)==EXPECTED[pc] and sha(critic)==critic_hash
    save_new(dest/'verified.json',dict(pc_weight=pc,seed=seed,steps=10080,complete=True,
        actor=str(actor),actor_sha256=EXPECTED[pc],critic_sha256=critic_hash,architecture=architecture,
        evaluation_weights=[40,.5,.5,3],training_weights=[80,.2,.2,pc],
        spacing_m=250,neighbor_km=.25,cluster=.9,min_ap=1,deterministic=True))
    print('SEED_COMPLETE',pc,seed,flush=True)

def summarize(root,pc):
    counts=np.zeros((3,2),dtype=float)
    power=[]
    for seed in range(1,6):
        folder=root/f'pc_{pc:.1f}'/'B'/f'SEED{seed}'
        validate_csv(folder)
        meta=json.loads((folder/'verified.json').read_text())
        assert meta['complete'] and meta['seed']==seed and meta['evaluation_weights']==[40,.5,.5,3]
        with (folder/'ue_stats.csv').open() as stream:
            for row in csv.DictReader(stream):
                indices=[i for i,v in enumerate([.05,.15,.3]) if np.isclose(float(row['delay_budget']),v)]
                assert len(indices)==1
                values=np.array([float(row['dropped_packets']),float(row['completed_packets'])])
                assert np.isfinite(values).all() and (values>=0).all()
                counts[indices[0]]+=values
        with (folder/'reward.csv').open() as stream:
            values=np.array([float(row['pc_kw']) for row in csv.DictReader(stream)])
            assert len(values)==10080 and np.isfinite(values).all()
            power.append(values.mean()*1000)
    assert (counts.sum(axis=1)>0).all()
    return dict(pc_weight=pc,power_w=float(np.mean(power)),power_seed_sd_w=float(np.std(power,ddof=1)),
        drop_pct=(100*counts[:,0]/counts.sum(axis=1)).tolist(),packet_counts=counts.tolist())

def main():
    assert not (BASE/'simulation-plan.json').exists() and not OUT.exists(),'No duplicate or overwrite'
    for pc in EXPECTED:
        source(pc)
    save_new(BASE/'simulation-plan.json',dict(weights=list(EXPECTED),seeds=list(range(1,6)),
        steps=10080,concurrency=3,evaluation_weights=[40,.5,.5,3],no_traj_pkl=True))
    def launch(pc,seed):
        space()
        with (BASE/f'sim-{pc:.1f}-{seed}.log').open('x') as stream:
            process=subprocess.Popen([sys.executable,__file__,'job',str(pc),str(seed)],stdout=stream,stderr=subprocess.STDOUT)
            print('JOB_START',pc,seed,process.pid,flush=True)
            code=process.wait()
            print('JOB_EXIT',pc,seed,code,flush=True)
            if code:
                raise RuntimeError(f'pc{pc}seed{seed} failed; no automatic retries')
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=3)
    futures=[pool.submit(launch,pc,seed) for seed in range(1,6) for pc in EXPECTED]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    rows=[]
    for namespace,pcs in [('B1000_pc_sweep',[.6,.9,1.2,1.5,1.8,2.1]),('B1000_pc_sweep_extension',[2.4,2.7,3.]),('B1000_pc_sweep_next',[3.3,3.6])]:
        for pc in pcs:
            rows.append(summarize(PROJECT/'analysis/sim_stats'/namespace,pc))
    save_new(BASE/'simulation-comparison.json',dict(results=rows,seeds=list(range(1,6)),steps=10080,
        caveat='One PPO training seed per weight. First9 trained locally; last2 onVast, matched budget but different backend. No statistical significance claim.'))
    save_new(BASE/'simulation-complete.json',dict(new_seed_runs=10,compared_models=11))
    print('ALL_COMPLETE_PC33_PC36',json.dumps(rows),flush=True)

if __name__=='__main__':
    if len(sys.argv)==1:
        main()
    else:
        job(float(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
