from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');src=(R/'experiments/bigmodel_pdt_v1/train_big.py').read_text()
SC='''RP=TR/'rewardparts_collect_v1'
BTAGS=['pc0.6_trainseed1_cp240','pc0.9_trainseed1_cp230','pc1.2_trainseed3_cpfinal','pc1.5_trainseed3_cp240','pc1.8_trainseed5_cpfinal','pc2.1_trainseed5_cpfinal','pc2.4_trainseed3_cp240','pc2.7_trainseed1_cp220','pc3.0_trainseed1_cp200','pc3.3_trainseed1_cp200']
C_OMEGA=[40,.5,.5,3]  # C_run1 training weights unknown -> eval weights (stated to user)
SCENES=[  # env, files, traffic, aps, area, cluster, episodes
 ('A',seeds(RP/'A_pc1.5_trainseed1_cp440',1,35),0,16,1000,.98,35),
 ('B',sum([seeds(RP/t,1,35 if t=='pc2.1_trainseed5_cpfinal' else 20) for t in BTAGS],[]),1,16,1000,.9,215),
 ('C',seeds(RP/'C_run1',1,35),2,16,1000,.9,35),
 ('B4',seeds(RP/'B4_k164wiqz_final',1,140),1,4,1000,.9,140),
 ('B36',seeds(RP/'B36_jywudswq_final',1,17),1,36,1000,.9,17),
 ('B500',seeds(RP/'B500_5m820ao4_cp290',1,35),1,16,500,.9,35),
 ('B2000',seeds(RP/'B2000_zdovv6uu_pc1.2_cp170',1,35),1,16,2000,.95,35)]
'''
i=src.index('SCENES=[');j=src.index('def sha(')
src=src[:i]+SC+src[j:]
rep=[
 ('"""[bigmodel_pdt_v1] Ablation:','"""[modt_pdt_v1] Multi-objective prompt DT (PEDA-style): vector RTG of 4 reward parts (str/sen/tol drop ratio, pc kW,\neach divided by dataset mean episode total) + preference omega=log10(training weights) feed the RTG token.\nData: rewardparts_collect_v1 (all 512 episodes, B has 10 trade-off policies). Base was bigmodel_pdt_v1:'),
 ("BASE=ROOT/'experiments/bigmodel_pdt_v1';MODELS=ROOT/'dt_results/bigmodel_pdt_v1'","BASE=ROOT/'experiments/modt_pdt_v1';MODELS=ROOT/'dt_results/modt_pdt_v1'"),
 ("     r=tr['rewards'].astype(np.float32)\n     rtg=np.flip(np.cumsum(np.flip(r,0),0,dtype=np.float64),0).astype(np.float32)\n     self.trajs.append(dict(obs=tr['observations'].astype(np.float32),act=a.astype(np.float32),rew=r,rtg=rtg))",
  "     r=tr['rewards'].astype(np.float32);pr=tr['reward_parts'];assert pr.shape==(10080,aps,4) and np.isfinite(pr).all()\n     rtg=np.flip(np.cumsum(np.flip(pr,0),0,dtype=np.float64),0).astype(np.float32)\n     tw=tr['training_weights'] if tr['training_weights'] is not None else C_OMEGA;assert len(tw)==4 and min(tw)>0\n     self.trajs.append(dict(obs=tr['observations'].astype(np.float32),act=a.astype(np.float32),rew=r,rtg=rtg,w=torch.tensor(np.log10(tw),dtype=torch.float32)))"),
 ("  self.envs=list(self.groups)\n","  self.envs=list(self.groups)\n  self.scale=np.mean([t['rtg'][0].mean(0) for t in self.trajs],0).astype(np.float32);assert (self.scale>0).all()\n  for t in self.trajs:t['rtg']/=self.scale\n"),
 ("torch.from_numpy(tr['rtg'][sl,ap].copy()))","torch.from_numpy(tr['rtg'][sl,ap].copy()),tr['w'])"),
 ("  self.h_rew=nn.Linear(1,256)","  self.embed_rtgv=nn.Linear(8,256)\n  self.h_rew=nn.Linear(1,256)"),
 (" def forward(self,hs,ha,hr,hv,ts,s,a,r):\n  B,L,_=s.shape;t=self.embed_timestep(ts)\n"," def forward(self,hs,ha,hr,hv,ts,s,a,r,w):\n  B,L,_=s.shape;t=self.embed_timestep(ts);rt=self.embed_rtgv(torch.cat((r,w.unsqueeze(1).expand(B,L,4)),-1))\n"),
 (" hs,ha,hr,hv,t,s,a,r=batch;heads=model(hs,ha,hr,hv,t,s,a,r)"," hs,ha,hr,hv,t,s,a,r,w=batch;heads=model(hs,ha,hr,hv,t,s,a,r,w)"),
 ("(35 if e in('C',) else 140 if e=='B4' else 17 if e=='B36' else 1)","1"),
 ("reward=[40,.5,.5,3],","reward=[40,.5,.5,3],rtg_parts_scale=data.scale.tolist(),omega='log10(training_weights); C_run1 unknown -> [40,.5,.5,3]',"),
 ("rtg_scale=1,","rtg_parts_scale=data.scale.tolist(),"),
 ("kind='DT' if 'DT' in a else 'PDT' if 'PDT' in a else 'SHPDT'","kind='PDT'"),
]
for a,b in rep:
 assert src.count(a)==1,a;src=src.replace(a,b)
assert src.count('self.embed_rtg(r)+t')==2;src=src.replace('self.embed_rtg(r)+t','rt+t')
(R/'experiments/modt_pdt_v1').mkdir(exist_ok=False)
(R/'experiments/modt_pdt_v1/train_modt.py').write_text(src);print('ok')
