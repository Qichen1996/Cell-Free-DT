from pathlib import Path
D=Path('/workspace/Cell-Free-DT-new/experiments/new500_cl98_pdt_v1');s=(D/'eval_new500.py').read_text()
a=s.index(" name,files,traffic,aps,*_=next(");b=s.index(" print('PREP'",a)
s=s[:a]+" E.save(OUT/'prep.json',dict(rtg=dict(mean={e:m[e] for e in ENVS}),source='selfhist_pdt_v1/eval/rtg.json (same as bigmodel eval); demos selfhist_pdt_v1/eval_pdt'));(OUT/'logs').mkdir()\n"+s[b:]
rep=[("OUT=D/'eval';","OUT=D/'eval_other';"),
("ENVS=['B500','B'];","ENVS=['A','C','B4','B36','B2000'];"),
("STATS/f'{k}_{q}'/e/'B'/f'SEED{s}'","STATS/f'{k}_{q}'/e/['A','B','C'][E.SPEC[e][2]]/f'SEED{s}'"),
("OUT.mkdir(exist_ok=False);m=json.loads((ROOT/'experiments/selfhist_pdt_v1/eval/rtg.json').read_text())['targets'];from trajectory_io import load_trajectory_pickle\n","OUT.mkdir(exist_ok=False);m=json.loads((ROOT/'experiments/selfhist_pdt_v1/eval/rtg.json').read_text())['targets']\n"),
('"""','"""[new500 eval_other] Same as eval_new500.py but scenes A,C,B4,B36,B2000 (old rtg.json mean, selfhist demos, scene cluster from T.SCENES); user: 都测一下.\n',1)]
for r in rep:
 o,n=r[0],r[1]
 assert o in s,o;s=s.replace(o,n,1)
assert "OUT/f'demo_{name}_{seed}.npz' if name=='B500'" in s
(D/'eval_new500_other.py').write_text(s);print('ok')
