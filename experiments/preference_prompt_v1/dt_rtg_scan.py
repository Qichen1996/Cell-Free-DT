"""Fixed pure-DT model, four additional RTGs; preserve/reuse the baseline five seeds."""
from common import *
import concurrent.futures
import subprocess
import time
import workflow as original

SCAN = BASE / 'dt_rtg_scan'
OUT = STATS / 'dt_rtg_scan'
MODEL_SHA = '62e926bb101cbfbc106ef5426843462a60260a23fa5a9b0d709bb4b93df16df8'
CENTRE = -435.27698835441845
CASES = {'test01':CENTRE-100, 'test02':CENTRE-50, 'test03':CENTRE+50, 'test04':CENTRE+100}

def baseline():
    assert sha(MODELS/'DT/model.pt') == MODEL_SHA
    checkpoint=torch.load(MODELS/'DT/model.pt',map_location='cpu',weights_only=False)
    assert checkpoint['preference_enabled'] is False and checkpoint['updates']==270000
    for seed in range(6,11):
        folder=STATS/'evaluation/DT/B'/f'SEED{seed}'
        validate_csv(folder)
        meta=json.loads((folder/'evaluation.json').read_text())
        assert meta['rtg_target']==CENTRE and meta['model_sha256']==MODEL_SHA
        assert meta['seed']==seed and meta['evaluation_weights']==COMMON_WEIGHTS

def job(case,seed):
    assert case in CASES and seed in range(6,11)
    baseline()
    config=SCAN/case
    assert json.loads((config/'rtg.json').read_text())['target']==CASES[case]
    # Evaluate with the SAME tested function: only its target-file and result root differ.
    # Overrides are process-local; no project source, model or old result is modified.
    original.BASE=config
    original.STATS=OUT/case
    original.evaluate('DT',0.,seed)
    meta=json.loads((OUT/case/'evaluation/DT/B'/f'SEED{seed}'/'evaluation.json').read_text())
    assert meta['rtg_target']==CASES[case] and meta['model_sha256']==MODEL_SHA

def metrics(folder):
    import csv
    counts=np.zeros((3,2),dtype=np.float64)
    powers=[]
    for seed in range(6,11):
        dest=folder/'B'/f'SEED{seed}'
        validate_csv(dest)
        with (dest/'ue_stats.csv').open() as stream:
            for row in csv.DictReader(stream):
                budget=float(row['delay_budget'])
                matches=[i for i,value in enumerate([.05,.15,.3]) if np.isclose(budget,value)]
                assert len(matches)==1,budget
                values=np.array([float(row['dropped_packets']),float(row['completed_packets'])])
                assert np.isfinite(values).all() and (values>=0).all()
                counts[matches[0]]+=values
        with (dest/'reward.csv').open() as stream:
            values=np.array([float(row['pc_kw']) for row in csv.DictReader(stream)])
            assert len(values)==STEPS and np.isfinite(values).all()
            powers.append(float(values.mean()))
    assert (counts.sum(axis=1)>0).all()
    return dict(power_w=float(np.mean(powers)*1000),power_seed_sd_w=float(np.std(powers,ddof=1)*1000),
        drop_pct=(100*counts[:,0]/counts.sum(axis=1)).tolist(),packet_counts=counts.tolist(),seeds=list(range(6,11)))

def report():
    dt=[]
    for case,target in [('baseline',CENTRE)]+list(CASES.items()):
        folder=STATS/'evaluation/DT' if case=='baseline' else OUT/case/'evaluation/DT'
        for seed in range(6,11):
            meta=json.loads((folder/'B'/f'SEED{seed}'/'evaluation.json').read_text())
            assert meta['rtg_target']==target and meta['model_sha256']==MODEL_SHA
        dt.append(dict(case=case,rtg_target=target,**metrics(folder)))
    dt.sort(key=lambda row:row['rtg_target'])
    pdt=[]
    for pc in PCS:
        pdt.append(dict(preference_pc=pc,rtg_target=CENTRE,**metrics(STATS/'evaluation'/f'PDT_pc_{pc:.1f}')))
    comparisons=[]
    for item in pdt:
        vector=np.array([item['power_w']]+item['drop_pct'])
        dominators=[]
        for other in dt:
            candidate=np.array([other['power_w']]+other['drop_pct'])
            if (candidate<=vector+1e-10).all() and (candidate<vector-1e-10).any():
                dominators.append(other['rtg_target'])
        comparisons.append(dict(preference_pc=item['preference_pc'],dominated_by_tested_dt_rtgs=dominators))
    save_new(SCAN/'comparison.json',dict(dt=dt,pdt=pdt,dominance=comparisons,
        scope='Only five tested DT RTGs and four PDT preferences at one common RTG; one training seed; no claim about all possible RTGs or statistical significance. drop pooled by packet count, power mean reward.pc_kw.'))
    lines=['# Pure DT RTG sweep versus preference PDT','',
           'Same held-out seeds6–10,7days,model fixed,common evaluation rewards40/.5/.5/3. Baseline reused,20new simulations.','',
           '| Model | Actual RTG | Preference pc | Power W | Stringent drop % | Sensitive drop % | Tolerant drop % |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for kind,rows in [('DT',dt),('PDT',pdt)]:
        for row in rows:
            lines.append('| '+ ' | '.join([kind,f"{row['rtg_target']:.9f}",str(row.get('preference_pc','—')),
                f"{row['power_w']:.4f}",*[f'{value:.6f}' for value in row['drop_pct']]])+' |')
    lines+=['','Finite sweep: lack of matching point is not proof that RTG cannot substitute for prompt. No extra automatic sweep.']
    with (SCAN/'comparison.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print('ALL_COMPLETE_DT_RTG_SCAN',json.dumps(dict(dt=dt,pdt=pdt)),flush=True)

def preflight():
    from preference_model import PreferenceTransformer
    baseline()
    torch.set_num_threads(1)
    torch.manual_seed(999)
    env,obs=environment(999,OUT/'preflight')
    model=PreferenceTransformer(False)
    model.load_state_dict(torch.load(MODELS/'DT/model.pt',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    model.eval()
    assert not model.preference_enabled and not any('preference_projection' in key for key in model.state_dict())
    with torch.no_grad():
        ts=torch.arange(8).repeat(16,1)
        states=torch.zeros(16,8,110)
        states[:,0]=torch.as_tensor(np.asarray(obs),dtype=torch.float32)
        actions=torch.zeros(16,8,2)
        for target in CASES.values():
            rtg=torch.zeros(16,8,1)
            rtg[:,0]=target
            heads=model(ts,states,actions,rtg,torch.zeros(16))
            act=torch.stack([h[:,0].argmax(-1) for h in heads],dim=-1)
            assert act.shape==(16,2) and torch.isfinite(act).all()
            print('PREFLIGHT_OK','RTG',target,'PURE_DT OBS(16,110) DIST250m CLUSTER.90',flush=True)

def main():
    baseline()
    assert not SCAN.exists() and not OUT.exists(), 'Existing scan outputs: do not duplicate or overwrite'
    SCAN.mkdir()
    save_new(SCAN/'plan.json',dict(model_sha256=MODEL_SHA,targets=CASES,reused_target=CENTRE,
        reused_seeds=list(range(6,11)),new_simulations=20,steps=STEPS,concurrency=3,
        unchanged_environment=dict(area=1000,aps=16,spacing_m=250,neighbor_km=.25,cluster=.9,min_ap=1),
        evaluation_weights=COMMON_WEIGHTS))
    for case,target in CASES.items():
        save_new(SCAN/case/'rtg.json',dict(target=target,scale=1))
    (SCAN/'logs').mkdir()
    def launch(case,seed):
        space()
        name=f'{case}_seed{seed}'
        with (SCAN/'logs'/(name+'.log')).open('x') as stream:
            process=subprocess.Popen([sys.executable,__file__,'job',case,str(seed)],stdout=stream,stderr=subprocess.STDOUT)
            print('JOB_START',name,process.pid,flush=True)
            code=process.wait()
            print('JOB_EXIT',name,code,flush=True)
            if code:
                raise RuntimeError(f'{name} failed exit{code}; no auto retry')
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=3)
    # Interleave RTG cases, so partial progress covers every target.
    futures=[pool.submit(launch,case,seed) for seed in range(6,11) for case in CASES]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    report()
    save_new(SCAN/'complete.json',dict(new_seed_runs=20,reused_seed_runs=5,finished=time.time()))

if __name__=='__main__':
    if len(sys.argv)==1:
        main()
    elif sys.argv[1]=='preflight':
        preflight()
    elif sys.argv[1]=='job':
        job(sys.argv[2],int(sys.argv[3]))
    else:
        raise ValueError(sys.argv)
