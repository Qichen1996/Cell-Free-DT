import os
os.environ.update(DEBUG='0',TRAIN='0',EVAL='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,subprocess,concurrent.futures,hashlib,shutil,csv
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
sys.path.insert(0,str(ROOT))
BASE=ROOT/'experiments/PPO_A_packet_diag_v2'
OUT=ROOT/'analysis/sim_stats/PPO_A_packet_diag_v2'
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2)
def specs():
    p=json.loads((ROOT/'experiments/A20_matched_checkpoint_eval/plan.json').read_text())
    return [p['models'][25],p['models'][17]]
def worker(i,seed):
    s=specs()[i]; assert i==0 or (s['pc']==1.5 and s['checkpoint']==440)
    dest=OUT/s['tag']/'A/A'/f'SEED{seed}'
    diag=BASE/'records'/s['tag']/f'SEED{seed}'
    diag.mkdir(parents=True,exist_ok=False)
    from network.user_equipment import UserEquipment as U
    counter=[0]; totals=[[0,0] for _ in range(3)]
    arrivals=open(diag/'arrivals.jsonl','x'); losses=open(diag/'drops.jsonl','x')
    def emit(f,x):f.write(json.dumps(x,separators=(',',':'))+'\n');f.flush()
    def bs(b):return dict(id=int(b.id),sm=int(b.sleep),next_sm=int(b._next_sleep),wake_remaining=float(b.wakeup_time),ants=int(b.num_ant),serving=len(b.ues),queue=len(b.queue))
    def snap(u):return dict(t=float(u.net._time),world=float(u.net.world_time),status=int(u.status),associated=[bs(b) for b in u.bss],coverage=[bs(b) for b in sorted(u._cover_bss,key=lambda b:b.id)])
    prop=U.channel_gains
    def gains(u,x):
        ret=prop.fset(u,x)
        if not hasattr(u,'_pd'):
            counter[0]+=1
            u._pd=dict(uid=counter[0],service=int(u.service),budget=float(u.delay_budget),pos=u.pos.tolist(),arrival=snap(u))
            u._ph=[];u._requests=[];u._packet_start=float(u.net._time)
            emit(arrivals,u._pd)
        return ret
    U.channel_gains=property(prop.fget,gains)
    original_request=U.request_connection
    def request(u):
        before=snap(u);ret=original_request(u)
        u._requests.append(dict(before=before,after=snap(u)))
        return ret
    U.request_connection=request
    original_step=U.step
    def step(u,dt):
        # Read only raw state. Do not call data_rate/SINR or any RNG from instrumentation.
        u._ph.append(dict(state=snap(u),delay=float(u.delay),remaining=float(u.demand),dt=float(dt)))
        return original_step(u,dt)
    U.step=step
    original_finish=U.finish_packet
    def finish(u,success):
        k=int(u.service);totals[k][0 if success else 1]+=1
        if not success:
            emit(losses,dict(**u._pd,packet=int(u.completed_packets+u.dropped_packets+1),packet_start=u._packet_start,end=snap(u),remaining_bits=float(u.demand),packet_size=float(u.packet_size),delay=float(u.delay),requests=u._requests,history=u._ph))
        ret=original_finish(u,success)
        u._ph=[];u._requests=[];u._packet_start=float(u.net._time)
        return ret
    U.finish_packet=finish
    m=json.loads((Path(s['source'])/'manifest.json').read_text())
    code=(ROOT/'experiments/IPPO_pc21_cp200_cross/runner.py').read_text().replace('pc_weight=2.1,checkpoint=200',f"pc_weight={s['pc']},checkpoint={s['checkpoint']}")
    scope={'__name__':'diagnostic_worker','__file__':__file__}
    exec(compile(code,str(BASE/'original_worker_in_memory.py'),'exec'),scope)
    scope.update(SOURCE=Path(s['source']),AH=m['actor.pt'],CH=m['critic.pt'],OUT=OUT/s['tag'])
    scope['worker']('A',seed,False)
    arrivals.close();losses.close()
    old=Path(s['output'])/'A/A'/f'SEED{seed}'
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    comparison={n:sha(dest/(n+'.csv'))==sha(old/(n+'.csv')) for n in ['reward','ue_stats','trajectory','net_stats','bs_stats']}
    save(diag/'complete.json',dict(seed=seed,tag=s['tag'],packet_totals_completed_dropped=totals,ue_arrivals=counter[0],original_csv_byte_match=comparison,source=s,model_manifest=m))
    print('DIAGNOSTIC_COMPLETE',i,seed,comparison,flush=True)
def main():
    assert not BASE.exists() and not OUT.exists(),'No duplicate launch'
    assert shutil.disk_usage(ROOT).free>8*1024**3
    BASE.mkdir();(BASE/'logs').mkdir()
    save(BASE/'plan.json',dict(models=specs(),seeds=list(range(101,121)),runs=40,parallel=2,steps=10080,auto_retry=False,instrumentation='process-local read-only hooks; no source/model modifications; arrivals and dropped-packet histories'))
    def task(pair):
        i,seed=pair
        assert shutil.disk_usage(ROOT).free>4*1024**3
        with (BASE/'logs'/f'{i}-{seed}.log').open('x') as f:
            rc=subprocess.call([sys.executable,'-u',__file__,'worker',str(i),str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        assert rc==0,f'Failed {i} {seed}; no automatic retry'
        return [i,seed]
    done=[]
    # Two smoke runs must complete before dispatching the other 38.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for pair in pool.map(task,[(0,101),(1,101)]):done.append(pair)
    print('SMOKE_COMPLETE',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for pair in pool.map(task,[(i,s) for s in range(102,121) for i in range(2)]):
            done.append(pair)
            temp=BASE/'state.tmp';temp.write_text(json.dumps(dict(complete=len(done),total=40,jobs=done)));temp.replace(BASE/'state.json')
    save(BASE/'complete.json',dict(runs=40,jobs=done))
    print('ALL_COMPLETE_PPO_A_PACKET_DIAG_40',flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:worker(int(sys.argv[2]),int(sys.argv[3]))
    else:main()
