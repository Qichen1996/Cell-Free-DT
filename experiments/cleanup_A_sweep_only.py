import json,hashlib,shutil,re,sys
from pathlib import Path
R=Path('/workspace/Cell-Free-DT-new')
assert R.resolve()==R
train=R/'results/weights/A_pc_sweep_12_16'
ab=R/'experiments/AB_new_training_checkpoint_eval'
a20=R/'experiments/A20_matched_checkpoint_eval'
plans=[json.loads((x/'plan.json').read_text()) for x in (ab,a20)]
models=plans[1]['models']
assert len(models)==26
keep=models[17];assert keep['tag']=='A_pc1.5_trainseed1' and keep['checkpoint']==440
targets=[]
for pc in ['1.2','1.3','1.4','1.6']:targets.append(train/f'pc_{pc}')
files=train/'pc_1.5/wandb/run-20261021_123616-b2lvzbig/files'
for p in files.glob('*.pt'):
    if p.name not in ['actor_eps440.pt','critic_eps440.pt']:targets.append(p)
for i,s in enumerate(models[:25]):
    assert s['tag']==f"A_pc{s['pc']:.1f}_trainseed1" and s['checkpoint'] in [400,420,440,460,480]
    if i==17:continue
    targets.extend([Path(s['source']),Path(s['output']),a20/'models'/s['tag']/f"eps{s['checkpoint']}"])
    for base in (ab,a20):
        for p in (base/'logs').iterdir():
            if re.match(rf'^(?:preflight|eval)-{i}(?:[-.]|$)',p.name):targets.append(p)
        p=base/'preflight'/str(i)
        if p.exists():targets.append(p)
for pc in ['1.2','1.3','1.4','1.6']:
    for name in [f'preflight-{pc}.log',f'train-{pc}.log']:
        targets.append(R/'experiments/A_pc_sweep_12_16/logs'/name)
targets=sorted(set(p for p in targets if p.exists()),key=str)
allowed=[train,ab/'models',R/'analysis/sim_stats/AB_new_training_checkpoint_eval',a20/'models',ab/'logs',a20/'logs',ab/'preflight',a20/'preflight',R/'experiments/A_pc_sweep_12_16/logs']
for p in targets:
    assert p.resolve()==p and any(a in p.parents for a in allowed),str(p)
    assert not any(q.is_symlink() for q in [p,*p.parents]),str(p)
    assert not any(q in targets for q in p.parents)
protected=[Path(keep['source']),Path(keep['output']),a20/'models'/keep['tag']/'eps440',files/'actor_eps440.pt',files/'critic_eps440.pt']
for base in (ab,a20):
    protected.extend(p for p in (base/'models').iterdir() if p.name.startswith('B'))
def hashes():
    out={}
    for p in protected:
        assert p.exists(),str(p)
        for f in ([p] if p.is_file() else p.rglob('*')):
            if f.is_file():out[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    return out
before=hashes()
for p in targets:assert not any(p==q or p in q.parents or q in p.parents for q in protected)
for name in ['actor','critic']:
    assert hashlib.sha256((files/f'{name}_eps440.pt').read_bytes()).hexdigest()==json.loads((Path(keep['source'])/'manifest.json').read_text())[f'{name}.pt']
sizes={str(p):sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink()) if p.is_dir() else p.stat().st_size for p in targets}
report=dict(scope='ONLY A_pc_sweep_12_16 and its 24 unselected checkpoint groups',keep=keep,targets=sizes,bytes=sum(sizes.values()),protected_hashes=before,apply='--apply' in sys.argv)
audit=R/'experiments/A_pc_sweep_12_16/cleanup_keep_pc15_cp440.json'
if report['apply']:
    assert not audit.exists(),'Already applied'
    audit.write_text(json.dumps(report,indent=2))
    for p in targets:
        assert p.resolve()==p and any(a in p.parents for a in allowed)
        if p.is_dir():shutil.rmtree(p)
        else:p.unlink()
    assert hashes()==before
    assert all(not p.exists() for p in targets)
    report['verified']=True
    audit.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(apply=report['apply'],targets=len(targets),bytes=report['bytes'],verified=report.get('verified'),paths=[str(p) for p in targets],audit=str(audit)),indent=2))
