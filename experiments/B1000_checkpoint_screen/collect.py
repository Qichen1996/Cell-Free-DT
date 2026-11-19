"""Collect an isolated 35-trajectory multi-checkpoint dataset without overwrites."""
import pickle
import shutil
import time
import json
from pathlib import Path
import screen as s

ROOT=s.PROJECT/'analysis/trajectory/B1000_multicp_70'
SOURCE_DATA=s.PROJECT/'analysis/trajectory/uniform/B.pkl'
RESERVE=2*1024**3

def check(tr):
    assert s.np.array_equal(tr['prompt'],[1,16,1000])
    for key,shape in [('observations',(10080,16,110)),('actions',(10080,16,2)),('rewards',(10080,16,1))]:
        arr=s.np.asarray(tr[key]);assert arr.shape==shape,(key,arr.shape)
        assert s.np.isfinite(arr).all(),key

def space(required=0):
    free=shutil.disk_usage(s.PROJECT).free
    if free<RESERVE+required:raise RuntimeError(f'Insufficient disk: free={free}, required={required}+reserve={RESERVE}; no automatic deletion')

@s.torch.no_grad()
def main():
    s.torch.set_num_threads(1)
    assert not ROOT.exists(),f'Refusing existing output: {ROOT}'
    space(3800000000)
    source_sha=s.sha(SOURCE_DATA)
    with SOURCE_DATA.open('rb') as f:old=pickle.load(f)
    assert len(old)==35
    selected=[]
    for seed in range(1,22):
        tr=next(x for x in old if x.get('seed')==seed)
        assert tr.get('model_source')=='mfxf31yk_eps220',tr.get('model_source')
        check(tr);selected.append(tr)
    del old
    ROOT.mkdir();parts=ROOT/'_parts';parts.mkdir()
    manifest={'source':str(SOURCE_DATA),'source_sha256':source_sha,'evaluation_weights':[40,.5,.5,3],
              'environment':s.CASE,'steps':10080,'reuse_checkpoint220_seeds':list(range(1,22)),
              'new_checkpoint40_seeds':list(range(1,8)),'new_checkpoint120_seeds':list(range(1,8))}
    with (ROOT/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    for cp in [40,120]:
        config=s.get_config().parse_args([])
        cfg=s.yaml.safe_load((s.SOURCE/'config.yaml').read_text())
        for k,v in cfg.items():
            if hasattr(config,k) and isinstance(v,dict) and 'value' in v:setattr(config,k,v['value'])
        config.model_version=f'_eps{cp}';config.use_wandb=False
        actor_sha=s.sha(s.SOURCE/f'actor_eps{cp}.pt')
        for seed in range(1,8):
            space(2700000000)
            base=s.PROJECT/'analysis/sim_stats/B1000_multicp_70'/f'cp{cp:03d}'
            out=base/'B'/f'SEED{seed}';assert not out.exists()
            env=s.make_env(s.CASE,seed,str(base))
            policy=s.MappoPolicy(config,env.observation_space[0],env.cent_observation_space[0],env.action_space[0],model_dir=str(s.SOURCE))
            obs,_,_=env.reset();ob=[];ac=[];rw=[];start=time.monotonic()
            print('SEED_START',cp,seed,flush=True)
            for t in range(10080):
                act=policy.act(obs,deterministic=True)
                ob.append(s.np.array(obs,copy=True));ac.append(s.np.array(act,copy=True))
                obs,_,reward,done,_,_=env.step(act);rw.append(s.np.array(reward,copy=True))
                if done:assert t==10079
                if (t+1)%1440==0:
                    space();print('DAY_COMPLETE',cp,seed,(t+1)//1440,flush=True)
            env.close();s.validate_outputs(out)
            assert s.sha(s.SOURCE/f'actor_eps{cp}.pt')==actor_sha
            tr={'prompt':s.np.array([1,16,1000],dtype=s.np.float32),'observations':s.np.asarray(ob,dtype=s.np.float32),
                'actions':s.np.asarray(ac,dtype=s.np.float32),'rewards':s.np.asarray(rw,dtype=s.np.float32),
                'seed':seed,'weights':[40,.5,.5,3],'model_source':f'mfxf31yk_eps{cp}', 'actor_sha256':actor_sha}
            check(tr)
            with (parts/f'cp{cp:03d}_seed{seed}.pkl').open('xb') as f:pickle.dump([tr],f,pickle.HIGHEST_PROTOCOL)
            selected.append(tr)
            print('SEED_COMPLETE',cp,seed,'SECONDS',time.monotonic()-start,flush=True)
            del ob,ac,rw,env,policy
    assert len(selected)==35
    for tr in selected:check(tr)
    space(2700000000)
    assert s.sha(SOURCE_DATA)==source_sha
    target=ROOT/'B.pkl'
    with target.open('xb') as f:pickle.dump(selected,f,pickle.HIGHEST_PROTOCOL)
    del selected
    with target.open('rb') as f:verified=pickle.load(f)
    assert len(verified)==35
    for tr in verified:check(tr)
    print('ALL_COMPLETE_35',str(target),'BYTES',target.stat().st_size,'SHA256',s.sha(target),flush=True)

if __name__=='__main__':main()
