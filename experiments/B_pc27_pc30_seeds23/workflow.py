from pathlib import Path
import subprocess,sys,json,concurrent.futures,shutil
R=Path('/workspace/Cell-Free-DT-new');B=R/'experiments/B_pc27_pc30_seeds23'
def group(code,pc):
    out=R/f'results/weights/B_pc{code}_seeds23'
    assert not out.exists();out.mkdir()
    for seed in [2,3]:
        assert shutil.disk_usage(R).free>4*1024**3
        (out/f'seed{seed}').mkdir()
        log=R/f'experiments/B_pc{code}_seeds23/logs/train-seed{seed}.log'
        with log.open('x') as f:
            rc=subprocess.call([sys.executable,'-u',str(R/f'experiments/B_pc{code}_seeds23/runner.py'),'train',str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
        assert rc==0,f'PC{pc} seed{seed} failed; no automatic retry'
        done=json.loads((out/f'seed{seed}/pc_{pc}/complete.json').read_text())
        assert done['steps']==2520000
        print('TRAIN_COMPLETE',pc,seed,flush=True)
    (R/f'experiments/B_pc{code}_seeds23/complete.json').write_text(json.dumps(dict(pc=pc,seeds=[2,3],steps_each=2520000)))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(group,code,pc) for code,pc in [('27',2.7),('30',3.0)]]
    for f in futures:f.result()
with (B/'complete.json').open('x') as f:json.dump(dict(models=4,steps_each=2520000),f)
print('ALL_COMPLETE_B_PC27_PC30_SEEDS23',flush=True)
