from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new/experiments');src=(R/'B2000_30_01_01_pc12_seed1_ep180/runner.py').read_text()
CP='/workspace/Cell-Free-DT-new/results/DT/B/mappo/B2000_30_01_01_pc12_seed1_ep180/wandb/run-20261105_143047-zdovv6uu/files'
def mk(name,seed,eps,lr,model_dir=None,save=10,doc=''):
 s=src
 rep=[('"""Train the authorized B2000 seed1 reward-weight experiment from scratch."""',f'"""{doc} Based on B2000_30_01_01_pc12_seed1_ep180/runner.py (same env/weights 30/.1/.1/1.2/cluster .95)."""'),
  ("NAME = 'B2000_30_01_01_pc12_seed1_ep180'",f"NAME = {name!r}"),
  ("STEPS = 180 * 10080",f"STEPS = {eps} * 10080"),
  ("w_str=30., w_sen=.1, w_tol=.1, w_pc=1.2, seed=1)",f"w_str=30., w_sen=.1, w_tol=.1, w_pc=1.2, seed={seed})"),
  ("assert not args.use_linear_lr_decay and args.model_dir is None",f"assert not args.use_linear_lr_decay and args.model_dir == {model_dir!r}"),
  ("assert args.lr == args.critic_lr == .0005",f"assert args.lr == args.critic_lr == {lr}"),
  ("dict(weights=[30, .1, .1, 1.2], seed=1, episodes=180,",f"dict(weights=[30, .1, .1, 1.2], seed={seed}, episodes={eps},"),
  ("minibatches=2, lr=.0005, critic_lr=.0005, linear_lr_decay=False,",f"minibatches=2, lr={lr}, critic_lr={lr}, linear_lr_decay=False,"),
  ("from_scratch=True, checkpoint_interval=10,",f"from_scratch={model_dir is None}, init_model={(model_dir+'/actor_eps170.pt') if model_dir else None!r}, checkpoint_interval={save},"),
  ("'PREFLIGHT_PASS B2000 30/0.1/0.1/1.2 SEED1 EP180 STEPS1814400'",f"'PREFLIGHT_PASS {name}'"),
  ("'--seed', '1',",f"'--seed', '{seed}',"),
  ("'--lr', '.0005',",f"'--lr', '{lr}',"),("'--critic_lr', '.0005',",f"'--critic_lr', '{lr}',"),
  ("'--save_interval', '10',",f"'--save_interval', '{save}',"+(f" '--model_dir', {model_dir!r}, '--model_version', '_eps170'," if model_dir else '')),
  ("dict(steps=STEPS, episodes=180, hashes",f"dict(steps=STEPS, episodes={eps}, hashes"),
  ("'ALL_COMPLETE_B2000_30_01_01_PC12_EP180'","'ALL_COMPLETE'")]
 for a,b in rep:
  assert s.count(a)==1,a;s=s.replace(a,b)
 d=R/name;d.mkdir(exist_ok=False);(d/'runner.py').write_text(s);print(name)
mk('B2000_30_01_01_pc12_ft_cp170_lr1e-4_ep40',1,40,.0001,CP,5,'Fine-tune zdovv6uu cp170 (actor+critic) with lr 1e-4 (1/5) for 40 episodes, ckpt every 5 (user: 试一下1和2).')
for sd in (2,3):mk(f'B2000_30_01_01_pc12_seed{sd}_ep180',sd,180,.0005,None,10,f'Same config as zdovv6uu, training seed {sd}, from scratch 180 episodes (user: 试一下1和2).')
