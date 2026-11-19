# Read-only analysis of PDT_pc18_dual_v1. Cold start = first stringent packet of a UE finishing at step k; window W = steps k-3..k on its covering APs.
import json, numpy as np
from pathlib import Path
O = Path('/workspace/Cell-Free-DT-new/analysis/sim_stats/PDT_pc18_dual_v1'); out = {}
for tag in ['pc1.5_trainseed3_cp240', 'pc1.8_trainseed5_cpfinal', 'pc2.1_trainseed5_cpfinal']:
    for drv in ['PPO', 'PDT']:
        agg = dict(all_agree=[], all_deeper=[], cs_agree=[], cs_deeper=[], cs_shallower=[], drop_agree=[], drop_deeper=[], ok_deeper=[], sleep_hist=np.zeros(4), cs_minsleep=np.zeros(4), drops=0, pkts=0, cold=0, cold_drops=0)
        for s in [1, 2, 3]:
            b = O / drv / tag / f'SEED{s}'; z = np.load(b / 'dual.npz'); E = json.loads((b / 'events.json').read_text())['events']
            pp, pd, st = z['a_ppo'][..., 1].astype(int), z['a_pdt'][..., 1].astype(int), z['state'][..., 0].astype(int)
            agg['all_agree'].append((pp == pd).mean()); agg['all_deeper'].append((pd > pp).mean())
            agg['sleep_hist'] += np.bincount(st.ravel(), minlength=4)
            agg['pkts'] += len(E); agg['drops'] += sum(not e['success'] for e in E)
            for e in E:
                if e['pkt_index'] > 1 or not e['cover']: continue
                k = e['step']; ks = list(range(max(0, k - 3), k + 1)); c = e['cover']
                a, d = pp[np.ix_(ks, c)], pd[np.ix_(ks, c)]
                agg['cold'] += 1; agg['cold_drops'] += not e['success']
                agg['cs_agree'].append((a == d).mean()); agg['cs_deeper'].append((d > a).mean()); agg['cs_shallower'].append((d < a).mean())
                (agg['ok_deeper'] if e['success'] else agg['drop_deeper']).append((d > a).mean())
                if not e['success']: agg['drop_agree'].append((a == d).mean())
                agg['cs_minsleep'][st[max(0, k - 3), c].min()] += 1
        r = {k: (float(np.mean(v)) if isinstance(v, list) and v else v) for k, v in agg.items()}
        r['sleep_hist'] = (agg['sleep_hist'] / agg['sleep_hist'].sum()).round(3).tolist(); r['cs_minsleep'] = (agg['cs_minsleep'] / max(1, agg['cs_minsleep'].sum())).round(3).tolist()
        r['drop_rate_pct'] = 100 * agg['drops'] / agg['pkts']; out[f'{tag}|{drv}'] = r
        print(tag, drv, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
Path('/workspace/Cell-Free-DT-new/experiments/PDT_pc18_seed20_events_v1/dual_analysis.json').write_text(json.dumps(out, indent=1))
