"""Validate completed diagnostic rollouts and report packet-weighted metrics."""
import csv
import hashlib
import json
import math
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'analysis/prompt_diagnostic_v1'


def measure(base):
    counts=[[0.,0.] for _ in range(3)];power=[];per_seed=[];sources=[]
    for seed in [1,2,3]:
        p=base/'B'/f'SEED{seed}'
        metadata=json.loads((p/'evaluation.json').read_text())
        assert metadata['rtg']==-523.07568359375 and metadata['steps']==10080
        assert metadata['weights']==[40,.5,.5,3]
        e=metadata['environment'];assert (e['area'],e['grid'],e['cluster'])==(1000,4,.9)
        assert metadata['model_sha256']==hashlib.file_digest(Path(metadata['model']).open('rb'),'sha256').hexdigest()
        seed_counts=[[0.,0.] for _ in range(3)]
        for name in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
            assert (p/f'{name}.csv').stat().st_size>0
        with (p/'ue_stats.csv').open() as f:
            for row in csv.DictReader(f):
                d=float(row['delay_budget']);i=min(range(3),key=lambda j:abs(d-[.05,.15,.3][j]))
                assert abs(d-[.05,.15,.3][i])<1e-8
                for j,k in enumerate(['dropped_packets','completed_packets']):
                    x=float(row[k]);assert math.isfinite(x) and x>=0
                    counts[i][j]+=x;seed_counts[i][j]+=x
        with (p/'reward.csv').open() as f: pc=[float(r['pc_kw']) for r in csv.DictReader(f)]
        assert len(pc)==10080 and all(math.isfinite(x) for x in pc)
        power.extend(pc)
        per_seed.append({'seed':seed,'drop_percent':[100*d/(d+c) for d,c in seed_counts],'pc_kw':sum(pc)/len(pc)})
        sources.append(str(p))
    return {'drop_percent':[100*d/(d+c) for d,c in counts],'packet_counts':counts,
      'pc_kw':sum(power)/len(power),'per_seed':per_seed,'sources':sources}


if __name__=='__main__':
    results={};pending=[]
    for v in ['dt','deployment','q25','q50','q75','mapped','max']:
        base=PROJECT/'analysis/sim_stats'/('PDTreturn7_B' if v=='max' else 'PDTdiag_B_'+v)
        if not all((base/'B'/f'SEED{s}'/'evaluation.json').exists() for s in [1,2,3]):pending.append(v);continue
        results[v]=measure(base)
    output={'common_rtg':-523.07568359375,'seeds':[1,2,3],'pending':pending,'results':results,
      'scope':'Same evaluator context8/perAP; trained checkpoints differ, so this is not a controlled retraining ablation.'}
    print(json.dumps(output,indent=2))
    if not pending:
        p=ROOT/'simulation_summary.json'
        with p.open('x') as f:json.dump(output,f,indent=2)
