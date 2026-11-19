"""Collect balanced compressed episodes, train PDT/DT, run held-out fixed-RTG tests."""
from common import *
import argparse
import bisect
import concurrent.futures
import csv
import pickle
import subprocess
import time
import zstandard

def dataset_path(pc):
    return DATA / f'pc_{pc:.1f}' / 'B.pkl.zst'

@torch.no_grad()
def collect(pc):
    torch.set_num_threads(1)
    validate_sources()
    target = dataset_path(pc)
    assert not target.exists(), target
    trajectories = []
    for seed in range(1,6):
        space()
        torch.manual_seed(seed)
        stats = STATS / 'collection' / f'pc_{pc:.1f}'
        dest = stats / 'B' / f'SEED{seed}'
        assert not dest.exists(), dest
        env, obs = environment(seed, stats)
        policy = teacher(pc, env)
        observations = np.empty((STEPS,16,110),dtype=np.float32)
        actions = np.empty((STEPS,16,2),dtype=np.float32)
        rewards = np.empty((STEPS,16,1),dtype=np.float32)
        print('SEED_START', pc, seed, flush=True)
        for step in range(STEPS):
            act = policy.act(obs,deterministic=True)
            observations[step] = np.asarray(obs)
            actions[step] = act
            obs, _, reward, done, _, _ = env.step(act)
            rewards[step] = np.asarray(reward).reshape(16,1)
            assert bool(done) == (step == STEPS-1)
            if (step+1) % 1440 == 0:
                space()
                print('DAY_COMPLETE',pc,seed,(step+1)//1440,flush=True)
        env.close()
        validate_csv(dest)
        tr = dict(prompt=np.array([1,16,1000],dtype=np.float32), observations=observations,
                  actions=actions,rewards=rewards,seed=seed,preference_pc=pc,
                  training_weights=[80,.2,.2,pc],weights=COMMON_WEIGHTS,
                  actor_sha256=EXPECTED[pc][0],model_source=f'B1000_pc_sweep/pc_{pc:.1f}')
        check_trajectory(tr)
        save_new(dest/'evaluation.json',dict(seed=seed,preference_pc=pc,steps=STEPS,
            actor_sha256=EXPECTED[pc][0],evaluation_weights=COMMON_WEIGHTS))
        trajectories.append(tr)
        print('SEED_COMPLETE',pc,seed,flush=True)
        del env,policy
    space()
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(str(target)+'.partial')
    with temporary.open('xb') as stream:
        with zstandard.ZstdCompressor(level=3,write_checksum=True).stream_writer(stream,closefd=False) as writer:
            pickle.dump(trajectories,writer,protocol=pickle.HIGHEST_PROTOCOL)
    # Partial suffix is intentionally not treated as plain pickle by the loader.
    with temporary.open('rb') as stream,zstandard.ZstdDecompressor().stream_reader(stream) as reader:
        loaded = pickle.load(reader)
    assert len(loaded)==5
    for expected, actual in zip(trajectories,loaded):
        check_trajectory(actual)
        for key in ['observations','actions','rewards','prompt']:
            assert np.array_equal(expected[key],actual[key])
        assert expected['seed']==actual['seed'] and expected['preference_pc']==actual['preference_pc']
    assert not target.exists()
    os.link(temporary,target)
    temporary.unlink()
    save_new(target.parent/'complete.json',dict(episodes=5,ap_trajectories=80,steps=STEPS,
        compressed_bytes=target.stat().st_size,sha256=sha(target),preference_pc=pc,
        actor_sha256=EXPECTED[pc][0],evaluation_weights=COMMON_WEIGHTS))
    print('COLLECTION_COMPLETE',pc,str(target),flush=True)

def prepare_data():
    from trajectory_io import load_trajectory_pickle
    returns = []
    groups = []
    for pc in PCS:
        data = load_trajectory_pickle(dataset_path(pc))
        assert len(data)==5
        values = []
        for tr in data:
            check_trajectory(tr)
            assert tr['preference_pc']==pc
            values.extend(tr['rewards'].sum(axis=0,dtype=np.float64).reshape(-1).tolist())
        returns.extend(values)
        groups.append(dict(preference_pc=pc,mean=float(np.mean(values)),min=float(np.min(values)),max=float(np.max(values))))
    save_new(BASE/'rtg.json',dict(target=float(np.mean(returns)),scale=1,
        method='Equal20episodes and equal16AP mean initial common-reward undiscounted return',groups=groups))

class PreferenceDataset:
    def __init__(self):
        from decision_transformer.utils import TrajectoryDataset
        self.base = TrajectoryDataset([str(dataset_path(pc)) for pc in PCS],8,1,seed_fraction=1,single_ap=True)
        assert len(self.base.trajectories)==20
    def __len__(self):
        return len(self.base)
    def __getitem__(self,index):
        _,timesteps,states,actions,rtg = self.base[index]
        ti = bisect.bisect_right(self.base.window_ends,index)
        pc = self.base.trajectories[ti]['preference_pc']
        return torch.tensor(pc,dtype=torch.float32),timesteps,states,actions,rtg

def train(kind, smoke=False):
    from preference_model import PreferenceTransformer
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,Subset
    torch.manual_seed(1)
    np.random.seed(1)
    torch.set_num_threads(5)
    folder = BASE/'smoke'/kind if smoke else MODELS/kind
    folder.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    if smoke:
        # Cheap synthetic check before long collection; no claim about learned performance.
        data = [(torch.tensor(.6),torch.arange(8),torch.zeros(8,1,110),torch.zeros(8,1,2),torch.zeros(8,1,1))]*256
        workers, rounds, updates = 0,1,2
    else:
        data = PreferenceDataset()
        workers,rounds,updates = 4,1800,150
    loader = DataLoader(data,batch_size=256,shuffle=True,num_workers=workers,pin_memory=True,
                        generator=torch.Generator().manual_seed(1))
    iterator = iter(loader)
    model = PreferenceTransformer(kind=='PDT').to('cuda:0')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1-step/(rounds*updates)/4)
    count = 0
    with (folder/'loss.jsonl').open('x') as log:
        for iteration in range(rounds):
            losses = []
            for _ in range(updates):
                try:
                    batch=next(iterator)
                except StopIteration:
                    iterator=iter(loader)
                    batch=next(iterator)
                pref,ts,states,actions,rtg=[x.to('cuda:0') for x in batch]
                states,actions,rtg=states.squeeze(2),actions.squeeze(2),rtg.squeeze(2)
                logits=model(ts,states,actions,rtg,pref)
                loss=sum(F.cross_entropy(head.reshape(-1,head.shape[-1]),actions[...,i].long().reshape(-1))
                         for i,head in enumerate(logits))/2
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),.25)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                count+=1
            row=dict(round=iteration+1,updates=count,loss=float(np.mean(losses)),seconds=time.monotonic()-start)
            log.write(json.dumps(row)+'\n')
            log.flush()
            if iteration==0 or (iteration+1)%20==0:
                space()
                print('TRAIN_PROGRESS',kind,json.dumps(row),flush=True)
    checkpoint = folder/'model.pt'
    with checkpoint.open('xb') as stream:
        torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            preference_enabled=kind=='PDT',updates=count,seed=1,context=8,rtg_scale=1,
            data=[str(dataset_path(pc)) for pc in PCS]),stream)
    save_new(folder/'complete.json',dict(kind=kind,updates=count,rounds=rounds,
        seconds=time.monotonic()-start,sha256=sha(checkpoint),parameters=sum(x.numel() for x in model.parameters())))
    print('TRAIN_COMPLETE',kind,count,flush=True)

@torch.no_grad()
def evaluate(kind,pc,seed):
    from preference_model import PreferenceTransformer
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    label = f'PDT_pc_{pc:.1f}' if kind=='PDT' else 'DT'
    stats = STATS/'evaluation'/label
    dest = stats/'B'/f'SEED{seed}'
    assert not dest.exists(), dest
    checkpoint = MODELS/kind/'model.pt'
    model = PreferenceTransformer(kind=='PDT')
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    model.eval()
    target = json.loads((BASE/'rtg.json').read_text())['target']
    env,obs = environment(seed,stats)
    ts=torch.arange(STEPS).repeat(16,1)
    states=torch.zeros(16,STEPS,110)
    actions=torch.zeros(16,STEPS,2)
    rtg=torch.zeros(16,STEPS,1)
    running=torch.full((16,1),target)
    pref=torch.full((16,),pc)
    for step in range(STEPS):
        states[:,step]=torch.as_tensor(np.asarray(obs),dtype=torch.float32)
        rtg[:,step]=running
        left=max(0,step-7)
        right=max(8,step+1)
        logits=model(ts[:,left:right],states[:,left:right],actions[:,left:right],rtg[:,left:right],pref)
        index=step if step<8 else -1
        act=torch.stack([head[:,index].argmax(-1) for head in logits],dim=-1)
        actions[:,step]=act
        obs,_,reward,done,_,_=env.step(act.numpy())
        running-=torch.as_tensor(np.asarray(reward).reshape(16,1),dtype=torch.float32)
        assert bool(done)==(step==STEPS-1)
        if (step+1)%1440==0:
            space()
            print('EVAL_DAY',label,seed,(step+1)//1440,flush=True)
    env.close()
    validate_csv(dest)
    save_new(dest/'evaluation.json',dict(model_kind=kind,preference_pc=pc if kind=='PDT' else None,
        seed=seed,steps=STEPS,rtg_target=target,model_sha256=sha(checkpoint),
        evaluation_weights=COMMON_WEIGHTS,heldout_seed=True))
    print('EVAL_COMPLETE',label,seed,flush=True)

@torch.no_grad()
def sensitivity():
    from preference_model import PreferenceTransformer
    torch.set_num_threads(1)
    data=PreferenceDataset()
    model=PreferenceTransformer(True)
    model.load_state_dict(torch.load(MODELS/'PDT/model.pt',map_location='cpu',weights_only=False)['model_state_dict'])
    model.eval()
    indices=np.random.default_rng(123).integers(len(data),size=512)
    items=[data[int(index)] for index in indices]
    ts=torch.stack([x[1] for x in items])
    states=torch.stack([x[2] for x in items]).squeeze(2)
    actions=torch.stack([x[3] for x in items]).squeeze(2)
    rtg=torch.stack([x[4] for x in items]).squeeze(2)
    predictions=[]
    probabilities=[]
    for pc in PCS:
        heads=model(ts,states,actions,rtg,torch.full((len(items),),pc))
        predictions.append(torch.stack([h[:,-1].argmax(-1) for h in heads],dim=-1))
        probabilities.append([h[:,-1].softmax(-1) for h in heads])
    pairs=[]
    for i in range(len(PCS)):
        for j in range(i+1,len(PCS)):
            pairs.append(dict(pc_a=PCS[i],pc_b=PCS[j],action_disagreement=float((predictions[i]!=predictions[j]).any(-1).float().mean()),
                mean_absolute_probability_change=float(torch.cat([abs(a-b).flatten() for a,b in zip(probabilities[i],probabilities[j])]).mean())))
    save_new(BASE/'prompt_sensitivity.json',dict(contexts=512,pairs=pairs,
        scope='Fixed TRAINING contexts and identical raw RTG; mechanistic diagnostic, not heldout accuracy'))

def report():
    import pandas as pd
    rows=[]
    for label in [f'PDT_pc_{pc:.1f}' for pc in PCS]+['DT']:
        counts=np.zeros((3,2),dtype=np.float64)
        powers=[]
        for seed in range(6,11):
            dest=STATS/'evaluation'/label/'B'/f'SEED{seed}'
            validate_csv(dest)
            meta=json.loads((dest/'evaluation.json').read_text())
            assert meta['heldout_seed'] and meta['seed']==seed
            ue=pd.read_csv(dest/'ue_stats.csv')
            for i,budget in enumerate([.05,.15,.3]):
                part=ue[np.isclose(ue.delay_budget,budget)]
                values=part[['dropped_packets','completed_packets']].to_numpy(dtype=float)
                assert np.isfinite(values).all() and (values>=0).all()
                counts[i]+=values.sum(axis=0)
            power=pd.read_csv(dest/'reward.csv')['pc_kw'].to_numpy(dtype=float)
            assert np.isfinite(power).all()
            powers.append(float(power.mean()))
        assert (counts.sum(axis=1)>0).all()
        rows.append(dict(label=label,rtg_target=json.loads((BASE/'rtg.json').read_text())['target'],
            power_kw=float(np.mean(powers)),power_seed_sd_kw=float(np.std(powers,ddof=1)),
            drop_pct=(100*counts[:,0]/counts.sum(axis=1)).tolist(),packet_counts=counts.tolist(),
            seeds=list(range(6,11))))
    save_new(BASE/'comparison.json',dict(results=rows,interpretation='Pilot: one training seed. Preference-conditioned imitation, not proof of superiority or monotonic control. Teacher seed1..5 not matched comparison.'))
    print('ALL_COMPLETE_PREFERENCE_PILOT',json.dumps(rows),flush=True)

def launch(command,label):
    space()
    folder=BASE/'logs'
    folder.mkdir(exist_ok=True)
    with (folder/(label+'.log')).open('x') as log:
        process=subprocess.Popen([sys.executable,__file__]+command,stdout=log,stderr=subprocess.STDOUT)
        print('JOB_START',label,process.pid,flush=True)
        code=process.wait()
        print('JOB_EXIT',label,code,flush=True)
        if code:
            raise RuntimeError(f'{label} failed exit{code}; see log; no automatic retry')

def parallel_jobs(tasks,workers):
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures=[pool.submit(launch,command,label) for command,label in tasks]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    pool.shutdown(wait=True)

def pipeline():
    validate_sources()
    assert not (BASE/'plan.json').exists(), 'Existing workflow: do not duplicate'
    assert not DATA.exists() and not MODELS.exists(), 'Existing output: do not overwrite'
    save_new(BASE/'plan.json',dict(preferences=PCS,collection_seeds=list(range(1,6)),evaluation_seeds=list(range(6,11)),
        steps=STEPS,common_reward=COMMON_WEIGHTS,train_rounds=1800,updates_per_round=150,training_seed=1,
        model_provenance=EXPECTED,environment=dict(scenario='B',area=1000,aps=16,spacing_m=250,neighbor_km=.25,cluster=.9,min_ap=1)))
    def state(phase):
        path=BASE/'state.json'
        temp=BASE/'state.tmp'
        temp.write_text(json.dumps(dict(phase=phase,updated=time.time()),indent=2))
        temp.replace(path)
        print('PHASE',phase,flush=True)
    state('collecting')
    parallel_jobs([(['collect',str(pc)],f'collect_pc_{pc:.1f}') for pc in PCS],4)
    prepare_data()
    state('training')
    for kind in ['PDT','DT']:
        launch(['train',kind],'train_'+kind)
    state('sensitivity')
    sensitivity()
    state('evaluating')
    tasks=[('PDT',pc,seed) for pc in PCS for seed in range(6,11)]+[('DT',0.,seed) for seed in range(6,11)]
    parallel_jobs([(['evaluate',kind,str(pc),str(seed)],f'eval_{kind}_pc{pc:.1f}_seed{seed}') for kind,pc,seed in tasks],3)
    report()
    state('complete')

def smoke():
    validate_sources()
    torch.set_num_threads(1)
    for pc in PCS:
        env,obs=environment(999,BASE/'smoke_stats'/f'pc_{pc:.1f}')
        policy=teacher(pc,env)
        with torch.no_grad():
            act=policy.act(obs,deterministic=True)
            assert act.shape==(16,2)
            obs,_,reward,_,_,_=env.step(act)
            assert np.asarray(reward).reshape(16,1).shape==(16,1)
        print('PREFLIGHT_OK',pc,'OBS16x110 DIST250m CLUSTER.90',flush=True)
        # Do not close/save incomplete preflight rollout as a formal result.
    for kind in ['PDT','DT']:
        train(kind,smoke=True)
    from preference_model import PreferenceTransformer
    from decision_transformer.model import DecisionTransformer
    torch.manual_seed(1)
    baseline=PreferenceTransformer(False).eval()
    reference=DecisionTransformer(110,2,3,128,8,1,.1,max_timestep=10080,no_prompt=True).eval()
    reference.load_state_dict(baseline.state_dict(),strict=True)
    inputs=(torch.arange(8).repeat(2,1),torch.randn(2,8,110),torch.zeros(2,8,2),torch.randn(2,8,1))
    with torch.no_grad():
        a=baseline(*inputs,torch.ones(2))
        b=reference(*inputs,torch.ones(2))
        assert all(torch.equal(x,y) for x,y in zip(a,b)), 'Baseline backbone changed'
    print('SMOKE_COMPLETE',flush=True)

if __name__=='__main__':
    if len(sys.argv)==1:
        pipeline()
    elif sys.argv[1]=='collect':
        collect(float(sys.argv[2]))
    elif sys.argv[1]=='train':
        train(sys.argv[2])
    elif sys.argv[1]=='evaluate':
        evaluate(sys.argv[2],float(sys.argv[3]),int(sys.argv[4]))
    elif sys.argv[1]=='smoke':
        smoke()
    else:
        raise ValueError(sys.argv)
