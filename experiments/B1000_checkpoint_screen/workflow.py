"""Wait for verified collection; train matched deployment-PDT/DT; evaluate B1000."""
import configparser
import csv
import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

PROJECT=Path('/workspace/Cell-Free-DT-new')
os.chdir(PROJECT)
ROOT=PROJECT/'experiments/B1000_multicp_comparison'
DATA=PROJECT/'analysis/trajectory/B1000_multicp_70/B.pkl'
RTG=-523.07568359375

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def reserve():
    assert shutil.disk_usage(PROJECT).free>2*1024**3,'Disk reserve below2GiB; stopped without deletion'

def run(cmd,log):
    reserve()
    with Path(log).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    if ROOT.exists():
        assert ROOT.is_dir() and not ROOT.is_symlink() and not any(ROOT.iterdir()),f'Refusing nonempty workflow output: {ROOT}'
    else:ROOT.mkdir()
    print('WAIT_COLLECTION',flush=True)
    while True:
        log=Path('/tmp/new_B1000_multicp_traj35.log').read_text()
        if 'ALL_COMPLETE_35' in log and DATA.exists():break
        status=subprocess.run(['supervisorctl','status','new_B1000_multicp_traj35'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False).stdout
        if 'RUNNING' not in status and 'STARTING' not in status:raise RuntimeError('Collection not running or verified complete: '+status)
        time.sleep(30)
    reserve()
    with DATA.open('rb') as f:data=pickle.load(f)
    assert len(data)==35
    counts={}
    for tr in data:
        assert np.array_equal(tr['prompt'],[1,16,1000])
        for key,shape in [('observations',(10080,16,110)),('actions',(10080,16,2)),('rewards',(10080,16,1))]:
            assert tr[key].shape==shape and np.isfinite(tr[key]).all()
        cp=tr['model_source'];counts[cp]=counts.get(cp,0)+1
    assert counts=={'mfxf31yk_eps220':21,'mfxf31yk_eps40':7,'mfxf31yk_eps120':7},counts
    del data
    commands={};datasets=None
    for kind in ['PDT','DT']:
        config=configparser.ConfigParser(interpolation=None)
        config.read(f'/etc/supervisor/conf.d/new_{kind}_seven_70_ctx8.conf')
        cmd=shlex.split(config[f'program:new_{kind}_seven_70_ctx8']['command'])
        idx=cmd.index('analysis/trajectory/uniform/B.pkl');cmd[idx]=str(DATA)
        cmd[cmd.index('--log_dir')+1]=f'dt_results/{kind}_seven_multicp70_ctx8'
        assert not Path(cmd[cmd.index('--log_dir')+1]).exists()
        commands[kind]=cmd
        files=cmd[cmd.index('--dataset_files')+1:cmd.index('--seed_fraction')]
        if datasets is None:datasets=files
        else:assert datasets==files
    provenance=[]
    expected=[35,35,35,140,17,35,35]
    for file,n in zip(datasets,expected):
        with open(file,'rb') as f:entries=pickle.load(f)
        assert len(entries)==n,(file,len(entries))
        for tr in entries:
            for key in ['observations','actions','rewards']:
                assert len(tr[key])==10080 and np.isfinite(tr[key]).all()
        provenance.append({'path':file,'count':n,'sha256':sha(file)})
        del entries
    record={'datasets':provenance,'total_trajectories':332,'commands':commands,
            'rtg':RTG,'seeds':[1,2,3,4,5],'weights':[40,.5,.5,3],
            'environment':{'area':1000,'grid':4,'inter_dist_m':250,'neighbor_km':.25,'cluster':.9,'min_ap':1},
            'note':'deployment-prompt PDT versus no-prompt DT; only B1000 training dataset replaced; no RTG sweep'}
    with (ROOT/'plan.json').open('x') as f:json.dump(record,f,indent=2)
    print('DATA_VERIFIED_332',flush=True)
    for kind,cmd in commands.items():
        print('TRAIN_START',kind,flush=True);start=time.monotonic()
        run(cmd,f'/tmp/new_{kind}_seven_multicp70_train.log')
        directory=Path(cmd[cmd.index('--log_dir')+1])
        with (directory/'log_8_1.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==1800 and int(rows[-1]['num_updates'])==270000
        assert all(np.isfinite(float(row['action_loss'])) for row in rows)
        assert (directory/'dt_8_1.pt').stat().st_size>0
        print('TRAIN_COMPLETE',kind,'SECONDS',time.monotonic()-start,'LOSS',rows[0]['action_loss'],rows[-1]['action_loss'],flush=True)
    for label,model in [('PDT_multicp','PDT_seven_multicp70_ctx8'),('DT_multicp','DT_seven_multicp70_ctx8'),
                         ('PDT_original','PDT_seven_70_ctx8'),('DT_original','DT_seven_70_ctx8')]:
        stats=f'analysis/sim_stats/B1000_multicp_comparison/{label}'
        assert not Path(stats).exists()
        cmd=['/venv/main/bin/python','-u','test_dt.py','--model_path',f'dt_results/{model}/dt_8_1.pt',
             '--target_rtg',str(RTG),'--eval_area','1000','--eval_grid','4','--eval_cluster','.9',
             '--scenario','B','--stats_dir',stats,'--w_str','40','--w_sen','.5','--w_tol','.5','--w_pc','3',
             '--include_bs_info','--device','cpu','--seeds','1','2','3','4','5']
        print('SIM_START',label,flush=True)
        run(cmd,f'/tmp/new_B1000_multicp_{label}_sim5.log')
        for seed in range(1,6):
            folder=Path(stats)/'B'/f'SEED{seed}'
            for name in ['reward','ue_stats','trajectory','net_stats','bs_stats']:
                assert (folder/f'{name}.csv').stat().st_size>0
            with (folder/'reward.csv').open() as f:assert sum(1 for _ in f)==10081
        print('SIM_COMPLETE',label,flush=True)
    print('ALL_TRAIN_AND_B1000_SIM_COMPLETE',flush=True)

if __name__=='__main__':main()
