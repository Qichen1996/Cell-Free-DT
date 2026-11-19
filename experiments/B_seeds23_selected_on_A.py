import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,subprocess,concurrent.futures,hashlib,shutil
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new');sys.path.insert(0,str(R))
BASE=R/'experiments/B_seeds23_selected_on_A'
OUT=R/'analysis/sim_stats/B_seeds23_selected_on_A'
SPECS=[dict(training_seed=2,checkpoint=180),dict(training_seed=3,checkpoint=200)]
def spec(i):
    s=dict(SPECS[i]);s['tag']=f"B_pc1.8_trainseed{s['training_seed']}_cp{s['checkpoint']}"
    s['source']=str(R/'experiments/AB_new_training_checkpoint_eval/models'/f"B_pc1.8_trainseed{s['training_seed']}"/f"eps{s['checkpoint']}")
    return s
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2)
def worker(i,seed,preflight=False):
    s=spec(i);src=Path(s['source']);m=json.loads((src/'manifest.json').read_text())
    for n,h in m.items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h
    provenance=json.loads((src/'provenance.json').read_text())
    assert provenance['training_seed']==s['training_seed'] and provenance['checkpoint']==s['checkpoint']
    code=(R/'experiments/IPPO_pc21_cp200_cross/runner.py').read_text().replace('pc_weight=2.1,checkpoint=200',f"pc_weight=1.8,checkpoint={s['checkpoint']}")
    scope={'__name__':'selected_cross_worker','__file__':__file__};exec(compile(code,str(BASE/'in_memory_worker.py'),'exec'),scope)
    scope.update(SOURCE=src,AH=m['actor.pt'],CH=m['critic.pt'],OUT=(BASE/'preflight' if preflight else OUT)/s['tag'])
    scope['worker']('A',seed,preflight)
    if not preflight:save(OUT/s['tag']/'A/A'/f'SEED{seed}'/'training_provenance.json',dict(**s,manifest=m,source_provenance=provenance))
def main():
    assert not BASE.exists() and not OUT.exists(),'No duplicate launch'
    assert shutil.disk_usage(R).free>6*1024**3
    (BASE/'logs').mkdir(parents=True)
    save(BASE/'plan.json',dict(models=[spec(0),spec(1)],evaluation_seeds=list(range(101,121)),runs=40,parallel=2,steps=10080,evaluation_weights=[40,.5,.5,3],environment=dict(scenario='A',area=1000,aps=16,spacing_m=250,neighbor_km=.25,cluster=.98,min_ap=1),auto_retry=False,save_traj_pkl=False))
    for i in range(2):
        with (BASE/'logs'/f'preflight-{i}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'preflight',str(i),'101'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
        assert rc==0,f'Preflight failed {i}'
        print('PREFLIGHT_OK',i,flush=True)
    def task(x):
        i,seed=x;assert shutil.disk_usage(R).free>4*1024**3
        with (BASE/'logs'/f'{i}-{seed}.log').open('x') as f:rc=subprocess.call([sys.executable,'-u',__file__,'job',str(i),str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
        assert rc==0,f'Failed {i} {seed}; no retry'
        return x
    done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for x in pool.map(task,[(i,s) for s in range(101,121) for i in range(2)]):
            done.append(x);p=BASE/'state.tmp';p.write_text(json.dumps(dict(complete=len(done),total=40,jobs=done)));p.replace(BASE/'state.json')
            print('COMPLETE',x,flush=True)
    save(BASE/'complete.json',dict(runs=40,jobs=done));print('ALL_COMPLETE_B_SEEDS23_SELECTED_ON_A_40',flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]),sys.argv[1]=='preflight')
    else:main()
