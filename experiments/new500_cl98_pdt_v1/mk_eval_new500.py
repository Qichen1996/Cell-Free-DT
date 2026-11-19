from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');s=(R/'experiments/bigmodel_pdt_v1/eval_big.py').read_text()
rep=[('"""[bigmodel_pdt_v1 eval]','"""[new500_cl98_pdt_v1 eval] DT/PDT retrained with new B500 data (bbcyap5g final, cluster .98). Envs B500 (cluster .98) and B (B1000, .9), seeds 1-5.\nB500: RTG = mean initial AP return of NEW B500 training data; PDT demos drawn from NEW B500 data with the rerun_b500_s1to5 rule (rng=seed).\nB: identical to bigmodel eval (old rtg.json mean, selfhist demos) for direct comparison. Base was eval_big.py:\n[bigmodel_pdt_v1 eval]'),
 ("D=ROOT/'experiments/bigmodel_pdt_v1'","D=ROOT/'experiments/new500_cl98_pdt_v1'"),
 ("import train_big as T\nimport eval_selfhist as E\n","import train_new500 as T\nimport eval_selfhist as E\nE.SPEC={s[0]:s for s in T.SCENES};assert E.SPEC['B500'][5]==.98 and E.SPEC['B'][5]==.9\n"),
 ("STATS=ROOT/'analysis/sim_stats/bigmodel_pdt_v1'","STATS=ROOT/'analysis/sim_stats/new500_cl98_pdt_v1'"),
 ("MP=lambda k:ROOT/f'dt_results/bigmodel_pdt_v1/{k}/model.pt'","MP=lambda k:ROOT/f'dt_results/new500_cl98_pdt_v1/{k}/model.pt'"),
 (" d=np.load(ROOT/f'experiments/selfhist_pdt_v1/eval_pdt/demo_{name}_{seed}.npz')"," d=np.load(OUT/f'demo_{name}_{seed}.npz' if name=='B500' else ROOT/f'experiments/selfhist_pdt_v1/eval_pdt/demo_{name}_{seed}.npz')"),
 (" for k in KINDS:assert (ROOT/f'dt_results/bigmodel_pdt_v1/{k}/complete.json').is_file()"," for k in KINDS:assert (ROOT/f'dt_results/new500_cl98_pdt_v1/{k}/complete.json').is_file()"),
 ("E.save(OUT/'prep.json',dict(rtg=dict(mean={e:m[e] for e in ENVS}),source='selfhist_pdt_v1/eval/rtg.json'));(OUT/'logs').mkdir()",
  "from trajectory_io import load_trajectory_pickle\n name,files,traffic,aps,*_=next(x for x in T.SCENES if x[0]=='B500');eps=[];demos={}\n for p in files:eps.append(float(load_trajectory_pickle(p)[0]['rewards'][...,0].astype(np.float64).sum(0).mean()))\n for seed in SEEDS:\n  rng=np.random.default_rng(seed);hs=[];ha=[];hr=[];src=[]\n  for ap in range(aps):\n   p=files[rng.integers(len(files))];a2=int(rng.integers(aps));end=int(rng.integers(T.STRIDE*(T.H-1)+1,10081))\n   tr=load_trajectory_pickle(p)[0];hi=end-1-T.STRIDE*np.arange(T.H)\n   hs.append(tr['observations'][hi,a2]);ha.append(tr['actions'][hi,a2]);hr.append(tr['rewards'][hi,a2]);src.append([p,0,a2,end])\n  np.savez(OUT/f'demo_B500_{seed}.npz',hs=np.array(hs,np.float32),ha=np.array(ha,np.float32),hr=np.array(hr,np.float32));demos[str(seed)]=src\n E.save(OUT/'prep.json',dict(rtg=dict(mean={'B500':float(np.mean(eps)),'B':m['B']}),source='B500: new data mean of %d eps (old was %s); B: selfhist_pdt_v1/eval/rtg.json'%(len(eps),m['B500']),b500_demos=demos));(OUT/'logs').mkdir()\n print('PREP',json.loads((OUT/'prep.json').read_text())['rtg'],flush=True)")]
for a,b in rep:
 assert s.count(a)==1,a;s=s.replace(a,b)
compile(s,'r','exec')
p=R/'experiments/new500_cl98_pdt_v1/eval_new500.py';assert not p.exists();p.write_text(s);print('ok')
