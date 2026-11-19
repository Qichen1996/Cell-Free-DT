"""B1000 diagnosis only: no training and no writes to original results."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
os.environ.update(TRAIN='0', EVAL='1', DEBUG='0')
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'experiments/return_prompt_v1'))
sys.path.insert(1, str(PROJECT))
from model import ReturnPromptDT
from eval import make_env, validate_outputs
from decision_transformer.model import DecisionTransformer

ROOT = PROJECT/'analysis/prompt_diagnostic_v1'
TARGET = -523.07568359375
NEW_PATH = PROJECT/'dt_results/PDT_return_prompt_seven_ctx8/model.pt'
STATS = PROJECT/'dt_results/PDT_return_prompt_seven_ctx8/return_prompt_stats.npz'
CASE = dict(name='B', scenario='B', area=1000, grid=4, cluster=.90)


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def load_model(kind, device='cpu'):
    path = NEW_PATH if kind=='return' else PROJECT/f'dt_results/{"DT" if kind=="dt" else "PDT"}_seven_70_ctx8/dt_8_1.pt'
    c=torch.load(path,map_location='cpu',weights_only=False)
    state=c['model_state_dict']
    if kind=='return': m=ReturnPromptDT()
    else:
        m=DecisionTransformer(110,2,3,128,8,1,.1,
            max_timestep=state['embed_timestep.weight'].shape[0],
            deployment_prompt=kind=='deployment',no_prompt=kind=='dt')
    m.load_state_dict(state)
    return m.to(device).eval(),path


def mapping(r,t,mu,sigma):
    return torch.sigmoid((r-mu[t][...,None])/sigma[t][...,None])


def metrics(pred,true,previous):
    p=pred.cpu().numpy(); y=true.cpu().numpy(); prev=previous.cpu().numpy()
    out={'samples':len(p),'antenna_accuracy':float((p[:,0]==y[:,0]).mean()),
         'sleep_accuracy':float((p[:,1]==y[:,1]).mean()),
         'joint_accuracy':float((p==y).all(1).mean())}
    cm=np.zeros((4,4),dtype=int)
    np.add.at(cm,(y[:,1],p[:,1]),1)
    out['sleep_confusion_true_rows_pred_cols']=cm.tolist()
    out['sleep_class_recall']=[float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None for i in range(4)]
    active=y[:,1]==0
    out['expert_active_pred_sleep_rate']=float((p[active,1]>0).mean()) if active.any() else None
    transition=(prev[:,1]==0)&(y[:,1]>0)
    out['expert_active_to_sleep_samples']=int(transition.sum())
    out['expert_active_to_sleep_recall']=float((p[transition,1]>0).mean()) if transition.any() else None
    return out


@torch.no_grad()
def offline():
    ROOT.mkdir(parents=True,exist_ok=True)
    dest=ROOT/'offline.json'
    if dest.exists(): raise FileExistsError(dest)
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with (PROJECT/'analysis/trajectory/uniform/B.pkl').open('rb') as f: data=pickle.load(f)
    assert len(data)==35 and [d['seed'] for d in data]==list(range(1,36))
    assert all(d['model_source']=='mfxf31yk_eps220' for d in data)
    z=np.load(STATS); mu_np=z['mu'];sigma_np=z['sigma']
    prompt_samples=[]; batch_s=[];batch_a=[];batch_r=[];batch_deploy=[];batch_t=[];previous=[]
    for tr in data:
        rew=np.asarray(tr['rewards'],dtype=np.float32)
        rtg=np.cumsum(rew[::-1],axis=0,dtype=np.float32)[::-1].copy()
        # Uniform hourly samples across all 35 retained training seeds for quantiles.
        times=np.arange(30,10080,60)
        zz=(rtg[times]-mu_np[times,None,None])/sigma_np[times,None,None]
        prompt_samples.append((1/(1+np.exp(-np.clip(zz,-60,60)))).ravel())
        if tr['seed']>5: continue
        cumulative=np.concatenate([np.zeros_like(rew[:1]),np.cumsum(rew[:-1],axis=0,dtype=np.float32)])
        deploy=TARGET-cumulative
        for t in times:
            sl=slice(t-7,t+1)
            batch_s.append(np.asarray(tr['observations'][sl]).transpose(1,0,2))
            batch_a.append(np.asarray(tr['actions'][sl]).transpose(1,0,2))
            batch_r.append(rtg[sl].transpose(1,0,2))
            batch_deploy.append(deploy[sl].transpose(1,0,2))
            batch_t.append(np.tile(np.arange(t-7,t+1),(16,1)))
            previous.append(np.asarray(tr['actions'][t-1]))
    ps=np.concatenate(prompt_samples)
    q=np.quantile(ps,[0,.01,.25,.5,.75,.99,1])
    variants={'q25':float(q[2]),'q50':float(q[3]),'q75':float(q[4]),'max':float(z['prompt_max'])}
    ss,aa,rr,dd,tt=[torch.from_numpy(np.concatenate(v)).to(device) for v in [batch_s,batch_a,batch_r,batch_deploy,batch_t]]
    ss=ss.float();aa=aa.float();rr=rr.float();dd=dd.float();tt=tt.long()
    truth=aa[:,-1].long(); prev=torch.from_numpy(np.concatenate(previous)).to(device)
    mu=torch.tensor(mu_np,device=device);sigma=torch.tensor(sigma_np,device=device)
    report={'scope':'B1000; hourly t=30+60k, all16AP, seeds1..5, teacher-forced action histories',
      'not_holdout':'All selected seeds were in training; this is reproduction, not generalization accuracy.',
      'target_rtg':TARGET,'context':8,'prompt_quantiles':dict(zip(['min','p01','p25','p50','p75','p99','max'],map(float,q))),
      'variants':variants,'sample_count':len(ss),'evaluations':{},'models':{}}
    reference=None
    for kind in ['dt','deployment','return']:
        model,path=load_model(kind,device);report['models'][kind]={'path':str(path),'sha256':sha(path)}
        conditions=['true_rtg','deployment_rtg'] if kind!='return' else ['true_rtg_mapped','true_rtg_max','deployment_rtg_mapped']+['deployment_rtg_'+k for k in variants]
        for condition in conditions:
            r=rr if condition.startswith('true') else dd
            preds=[];probs=[]
            for start in range(0,len(ss),256):
                sl=slice(start,start+256);t=tt[sl]
                if kind=='return':
                    prompt=mapping(r[sl],t,mu,sigma) if condition.endswith('mapped') else torch.full_like(r[sl],variants[condition.rsplit('_',1)[-1]])
                else: prompt=torch.tensor([1,16,1000],device=device).repeat(len(t),1) if kind=='deployment' else torch.ones(len(t),device=device,dtype=torch.long)
                logits=model(t,ss[sl],aa[sl],r[sl],prompt)
                preds.append(torch.stack([x[:,-1].argmax(-1) for x in logits],dim=-1))
                probs.append(torch.cat([x[:,-1].softmax(-1) for x in logits],dim=-1))
            pred=torch.cat(preds);prob=torch.cat(probs)
            key=kind+'/'+condition
            report['evaluations'][key]=metrics(pred,truth,prev)
            if kind=='return':
                if condition=='deployment_rtg_mapped': reference=(pred,prob)
                elif condition.startswith('deployment'):
                    rp,rprob=reference
                    report['evaluations'][key]['joint_flip_vs_mapped']=float((pred!=rp).any(-1).float().mean())
                    report['evaluations'][key]['mean_head_total_variation_vs_mapped']=float((prob-rprob).abs().sum(-1).mean()/4)
            print('OFFLINE',key,report['evaluations'][key]['joint_accuracy'],flush=True)
        # Verify present action target and future cannot affect current prediction.
        sl=slice(0,16);a=aa[sl].clone();a[:,-1]=torch.tensor([2,3],device=device)
        p=mapping(rr[sl],tt[sl],mu,sigma) if kind=='return' else (torch.tensor([1,16,1000],device=device).repeat(16,1) if kind=='deployment' else torch.ones(16,device=device,dtype=torch.long))
        before=model(tt[sl],ss[sl],aa[sl],rr[sl],p);after=model(tt[sl],ss[sl],a,rr[sl],p)
        assert all(torch.allclose(x[:,-1],y[:,-1],atol=2e-6) for x,y in zip(before,after))
        del model
    # Source expert identity and exact action reproduction check.
    from arguments import get_config
    from agents.mappo.nn.actor_critic import Actor
    import yaml
    expert=PROJECT/'results/DT/B/mappo/check/wandb/run-20261003_065202-mfxf31yk/files'
    args=get_config().parse_args([])
    cfg=yaml.safe_load((expert/'config.yaml').read_text())
    for k,v in cfg.items():
        if hasattr(args,k) and isinstance(v,dict) and 'value' in v:setattr(args,k,v['value'])
    env=make_env(CASE,999,str(ROOT/'preflight'))
    actor=Actor(args,env.observation_space[0],env.action_space[0],torch.device(device)).eval()
    actor.load_state_dict(torch.load(expert/'actor_eps220.pt',map_location=device,weights_only=True))
    pred=[]
    for start in range(0,len(ss),1024):pred.append(actor(ss[start:start+1024,-1],deterministic=True)[0].long())
    report['expert']=metrics(torch.cat(pred),truth,prev)|{'sha256':sha(expert/'actor_eps220.pt')}
    assert report['expert']['joint_accuracy']==1.,'Source expert does not exactly reproduce labels'
    with dest.open('x') as f:json.dump(report,f,indent=2)
    print('ALL_OFFLINE_COMPLETE',dest,flush=True)


@torch.no_grad()
def rollout(variant,validate_only=False):
    torch.set_num_threads(1)
    report=json.loads((ROOT/'offline.json').read_text())
    kind=variant if variant in ['dt','deployment'] else 'return'
    model,path=load_model(kind)
    z=np.load(STATS);mu=torch.tensor(z['mu']);sigma=torch.tensor(z['sigma'])
    for seed in ([999] if validate_only else [1,2,3]):
        output_root=PROJECT/'analysis/sim_stats'/f'PDTdiag_B_{variant}'
        out=output_root/'B'/f'SEED{seed}'
        if not validate_only and out.exists():raise FileExistsError(out)
        env=make_env(CASE,seed,str(output_root))
        obs,_,_=env.reset();n=16
        s=torch.zeros(n,10080,110);a=torch.zeros(n,10080,2);r=torch.zeros(n,10080,1)
        t=torch.arange(10080).repeat(n,1);running=torch.full((n,1),TARGET)
        count=1 if validate_only else 10080
        start=time.monotonic()
        for step in range(count):
            s[:,step]=torch.tensor(np.asarray(obs),dtype=torch.float32);r[:,step]=running
            lo=max(0,step-7);sl=slice(lo,step+1)
            if kind=='return':
                p=mapping(r[:,sl],t[:,sl],mu,sigma) if variant=='mapped' else torch.full_like(r[:,sl],report['variants'][variant])
            else:p=torch.tensor([1,16,1000]).repeat(n,1) if kind=='deployment' else torch.ones(n,dtype=torch.long)
            logits=model(t[:,sl],s[:,sl],a[:,sl],r[:,sl],p)
            assert all(torch.isfinite(x).all() for x in logits)
            act=torch.stack([x[:,-1].argmax(-1) for x in logits],dim=-1)
            if validate_only:
                print('VALIDATED',variant,'TARGET',TARGET,'PROMPT',None if kind!='return' else ('mapped' if variant=='mapped' else report['variants'][variant]),flush=True)
                return
            obs,_,reward,done,_,_=env.step(act.numpy());a[:,step]=act
            running-=torch.as_tensor(reward,dtype=torch.float32).reshape(n,1)
            if done:assert step==10079
            if (step+1)%1440==0:print('DAY_COMPLETE',variant,seed,(step+1)//1440,round(time.monotonic()-start,1),flush=True)
        env.close();validate_outputs(out)
        metadata={'variant':variant,'model':str(path),'model_sha256':sha(path),'seed':seed,'rtg':TARGET,
          'prompt':('mapped sigmoid((rawRTG-mu_t)/sigma_t)' if variant=='mapped' else report['variants'].get(variant)),
          'environment':CASE,'weights':[40,.5,.5,3],'steps':10080,'seconds':time.monotonic()-start}
        with (out/'evaluation.json').open('x') as f:json.dump(metadata,f,indent=2)
        print('SEED_COMPLETE',variant,seed,flush=True)
    print('ALL_COMPLETE',variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');p.add_argument('--variant');p.add_argument('--validate_only',action='store_true')
    args=p.parse_args()
    if args.offline:offline()
    else:rollout(args.variant,args.validate_only)
