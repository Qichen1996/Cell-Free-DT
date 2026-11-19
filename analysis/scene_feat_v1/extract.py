# read-only: windowed scene statistics from existing sim_stats reward.csv
import csv, glob, json, numpy as np
from pathlib import Path
S = Path('/workspace/Cell-Free-DT-new/analysis/sim_stats')
SRC = [('A3x','A3x_cp270', S/'sweep_A3x_s100_n02_e300/cp270/A'),
       ('B500x2','B500x2_cp250', S/'sweep_B500x2_s120_n02_t02_e300/cp250/B'),
       ('B1000','A3x_cp270', S/'cross_scene_v2/A3x_cp270/B1000/B'),
       ('B1000','B500x2_cp250', S/'cross_scene_v2/B500x2_cp250/B1000/B'),
       ('B1000','B500x2_cp220', S/'cross_scene_v2/B500x2b_cp220/B1000/B'),
       ('C','A3x_cp270', S/'cross_scene_v2/A3x_cp270/C/C'),
       ('B2000','B500x2_cp250', S/'cross_scene_v2/B500x2_cp250/B2000/B'),
       ('B2000','B500x2_cp220', S/'cross_scene_v2/B500x2b_cp220/B2000/B')]
K = 60
rows = []
for scene, model, d in SRC:
    for sd in sorted(d.glob('SEED*')):
        r = list(csv.DictReader(open(sd/'reward.csv')))
        a = np.array([float(v['arrival_rate']) for v in r]); se = np.array([float(v['avg_se']) for v in r]); pc = np.array([float(v['pc_kw']) for v in r])
        for w in range(len(r)//K):
            s = slice(w*K, (w+1)*K); A, E, P = a[s], se[s], pc[s]
            la = np.log1p(A); nz = E[E > 0]
            rows.append(dict(scene=scene, model=model, seed=sd.name, win=w,
                la_mean=la.mean(), la_std=la.std(), la_max=la.max(),
                se_mean=E.mean(), se_nzmean=nz.mean() if len(nz) else 0., se_p90=np.percentile(E,90), se_zero=(E==0).mean(),
                se_per_arr=E.mean()/max(A.mean(),1e-9), pc_mean=P.mean(), pc_std=P.std()))
json.dump(rows, open(Path(__file__).parent/'features.json','w'))
print('rows', len(rows), {k: sum(1 for x in rows if x['scene']==k) for k in sorted({x['scene'] for x in rows})})
