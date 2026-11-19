import csv, json, hashlib
from pathlib import Path
import numpy as np
P=Path('/workspace/Cell-Free-DT-new')
C=-435.27698835441845
PCS=[.6,1.5,1.8,2.1,2.4,2.7,3.0]
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def metrics(v,label,target,relative):
    root=P/'analysis/sim_stats'/f'B1000_preference_v{v}'/relative
    ck=P/'dt_results'/f'B1000_preference_v{v}'/('PDT' if label.startswith('PDT') else 'DT')/'model.pt'
    expected=hashlib.sha256(ck.read_bytes()).hexdigest()
    counts=np.zeros((3,2)); seeds=[]
    for seed in range(6,11):
        d=root/'B'/f'SEED{seed}'
        meta=json.loads((d/'evaluation.json').read_text())
        assert meta['seed']==seed and meta['steps']==10080 and meta['rtg_target']==target
        assert meta['evaluation_weights']==[40,.5,.5,3] and meta['model_sha256']==expected
        for name in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
            assert (d/(name+'.csv')).stat().st_size>0
        reward=read(d/'reward.csv'); assert len(reward)==10080
        power=np.array([float(r['pc_kw']) for r in reward]); assert np.isfinite(power).all()
        tr=read(d/'trajectory.csv'); assert len(tr)==10081
        assert all(f'bs_{i}_sleep_mode' in tr[0] for i in range(16))
        bs=read(d/'bs_stats.csv'); assert len(bs)==16 and len({r['id'] for r in bs})==16
        sleep=np.array([np.fromstring(r['sleep_time'].strip('[]'),sep=' ')/float(r['time']) for r in bs])
        assert sleep.shape==(16,4) and np.isfinite(sleep).all() and (sleep>=0).all()
        # Legacy NumPy CSV strings use limited significant digits; retain raw ratios.
        residual=float(np.max(np.abs(sleep.sum(1)-1)))
        with (d/'net_stats.csv').open() as f: keys=[r[0] for r in csv.reader(f)]
        assert len(keys)==len(set(keys)) and {'energy','arrived'}<=set(keys)
        local=np.zeros((3,2))
        for r in read(d/'ue_stats.csv'):
            idx=np.flatnonzero(np.isclose(float(r['delay_budget']),[.05,.15,.3])); assert len(idx)==1
            vals=np.array([float(r['dropped_packets']),float(r['completed_packets'])]); assert np.isfinite(vals).all() and (vals>=0).all()
            local[idx[0]]+=vals
        assert (local.sum(1)>0).all(); counts+=local
        seeds.append(dict(seed=seed,power_w=float(power.mean()*1000),drop_pct=(100*local[:,0]/local.sum(1)).tolist(),packet_counts=local.tolist(),sleep_ratio=sleep.mean(0).tolist(),sleep_max_sum_residual=residual))
    powers=[r['power_w'] for r in seeds]; sleeps=np.array([r['sleep_ratio'] for r in seeds])
    return dict(version=v,label=label,rtg_target=target,model_sha256=expected,power_w=float(np.mean(powers)),power_sd_w=float(np.std(powers,ddof=1)),drop_pct=(100*counts[:,0]/counts.sum(1)).tolist(),packet_counts=counts.tolist(),sleep_ratio=sleeps.mean(0).tolist(),sleep_sd=sleeps.std(0,ddof=1).tolist(),sleep_hours_168=(sleeps.mean(0)*168).tolist(),seeds=seeds)
def main():
    state=json.loads((P/'experiments/preference_prompt_v2/evaluation-state.json').read_text())
    assert len(state['jobs'])==60 and all(j['status']=='complete' and j['exit_code']==0 for j in state['jobs'].values())
    rows=[]
    for v in [1,2]:
        for pc in (PCS[:4] if v==1 else PCS):
            label=f'PDT_pc_{pc:.1f}'; rows.append(metrics(v,label,C,Path('evaluation')/label))
        rows.append(metrics(v,'DT',C,Path('evaluation/DT')))
        for i,delta in enumerate([-100,-50,50,100],1):
            rows.append(metrics(v,'DT',C+delta,Path(f'dt_rtg_scan/test{i:02d}/evaluation/DT')))
    dt=[r for r in rows if r['version']==2 and r['label']=='DT']
    dominance=[]
    for r in rows:
        if r['version']!=2 or r['label']=='DT': continue
        vec=np.array([r['power_w']]+r['drop_pct']); dominators=[]
        for other in dt:
            alt=np.array([other['power_w']]+other['drop_pct'])
            if (alt<=vec+1e-10).all() and (alt<vec-1e-10).any(): dominators.append(other['rtg_target'])
        dominance.append(dict(label=r['label'],dominated_by_tested_dt_rtgs=dominators))
    out=dict(verified_new_runs=60,rows=rows,dominance=dominance,method='Drops pool all five seeds by packet counts; power mean reward.pc_kw per seed then equal mean times1000. Sleep mean16BS sleep_time/time then mean5, SM0 active, SM1..3 sleep; 168hour equivalents. One training seed; seeds6..10 already inspected in v1; finite five-RTG control only.')
    dest=P/'experiments/preference_prompt_v2/verified-comparison.json'
    with dest.open('x') as f: json.dump(out,f,indent=2)
    print(json.dumps([dict(version=r['version'],label=r['label'],rtg=r['rtg_target'],power=r['power_w'],drop=r['drop_pct']) for r in rows]))
if __name__=='__main__': main()
