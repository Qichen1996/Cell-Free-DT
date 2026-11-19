from pathlib import Path
import sys,json,subprocess,concurrent.futures
R=Path('/workspace/Cell-Free-DT-new');NAME='B_pc18_pc21_seeds45';B=R/'experiments'/NAME;O=R/'results/weights'/NAME
JOBS=[(pc,seed) for pc in [1.8,2.1] for seed in [4,5]]
def worker(pc,seed,pre=False):
 code='18' if pc==1.8 else '21';old=f'B_pc{code}_seeds23'
 s=(R/'experiments'/old/'runner.py').read_text().replace(old,NAME).replace('training seeds2/3','training seeds4/5')
 injection=r"""
 source=source.replace('use_wandb=True,model_dir=None,use_eval=False,seed=', 'use_linear_lr_decay=False,use_wandb=True,model_dir=None,use_eval=False,seed=')
 patch='''
    original_init=MappoTrainer.__init__
    def checked_init(self,config):
        assert not config['all_args'].use_linear_lr_decay
        original_init(self,config)
        assert not self.use_linear_lr_decay and self.lr==.0005 and self.critic_lr==.0005
        print('ACTUAL_FIXED_LR_VERIFIED',self.lr,self.critic_lr,flush=True)
    MappoTrainer.__init__=checked_init
'''
 source=source.replace('    parser_factories=[train.get_config,get_mappo_config]',patch+'\n    parser_factories=[train.get_config,get_mappo_config]')
 source=source.replace('        old_log(self,info,steps)',"        info=dict(info,actor_lr=self.actor_optimizer.param_groups[0]['lr'],critic_lr=self.critic_optimizer.param_groups[0]['lr'])\n        assert abs(info['actor_lr']-.0005)<1e-12 and abs(info['critic_lr']-.0005)<1e-12\n        old_log(self,info,steps)")
 source=source.replace("    (dest/'experiment.json').write_text", "    metadata['lr_schedule']={'type':'constant','actor_lr':.0005,'critic_lr':.0005}\n    (dest/'experiment.json').write_text")
 """
 marker=" scope={'__name__'";assert marker in s;s=s.replace(marker,injection+'\n'+marker)
 scope={'__name__':'seed45_template','__file__':str(B/'runner.py')}
 exec(compile(s,str(B/f'worker_pc{pc}_memory.py'),'exec'),scope)
 scope['worker'](seed,pre)
def main():
 assert not O.exists();O.mkdir()
 plan=dict(weights=[[80,.2,.2,pc] for pc in [1.8,2.1]],training_seeds=[4,5],rounds=250,steps_per_model=2520000,parallel=4,from_scratch=True,lr_schedule='constant 0.0005 actor and critic',environment=dict(scenario='B',area=1000,aps=16,spacing_m=250,neighbor_km=.25,cluster=.9,min_ap=1),checkpoint_eval=[180,200,220,230,240],evaluation_seeds=[1,2,3,4,5],auto_retry=False)
 (B/'plan.json').write_text(json.dumps(plan,indent=2))
 def job(x):
  pc,seed=x
  with (B/'logs'/f'train-pc{pc}-seed{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'train',str(pc),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert rc==0,f'Training failed: {x}; no auto retry'
  p=O/f'seed{seed}/pc_{pc}/complete.json';assert json.loads(p.read_text())['steps']==2520000
  print('TRAIN_COMPLETE',pc,seed,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(job,JOBS))
 (B/'complete.json').write_text(json.dumps(dict(models=4,steps_each=2520000)))
 print('ALL_COMPLETE_B_PC18_PC21_SEEDS45_TRAIN',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(float(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
 else:main()
