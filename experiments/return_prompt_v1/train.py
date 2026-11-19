"""Isolated original-RTG + return-prompt experiment. Does not edit old DT code."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import ReturnPromptDT
from data import ReturnPromptDataset


def smoke_test():
    torch.manual_seed(123)
    m=ReturnPromptDT(drop_p=0.).eval()
    t=torch.arange(8).repeat(2,1)
    s=torch.randn(2,8,110)
    a=torch.zeros(2,8,2)
    r=torch.randn(2,8,1)
    p=torch.rand(2,8,1)
    y=m(t,s,a,r,p)
    assert y[0].shape==(2,8,3) and y[1].shape==(2,8,4)
    # State prediction at t cannot see its own target action or any future token.
    aa=a.clone();aa[:,3:,0]=2;aa[:,3:,1]=3
    ss=s.clone();ss[:,4:]+=10
    rr=r.clone();rr[:,4:]+=10
    pp=p.clone();pp[:,4:]=1
    z=m(t,ss,aa,rr,pp)
    assert all(torch.allclose(x[:,:4],q[:,:4],atol=2e-6) for x,q in zip(y,z))
    loss=sum(v.square().mean() for v in y)
    loss.backward()
    assert m.embed_prompt.weight.grad.abs().sum()>0
    assert m.embed_rtg.weight.grad.abs().sum()>0
    print('SMOKE_OK shapes, causal mask, action-target nonleakage, RTG/prompt gradients',flush=True)


def main(args):
    torch.set_num_threads(2)
    smoke_test()
    if args.smoke_only:
        return
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    dest=Path(args.log_dir)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    dest.mkdir(parents=True)
    start=time.monotonic()
    ds=ReturnPromptDataset(args.dataset_files,8,args.bootstrap,args.seed)
    assert len(ds.trajectories)==332,'Unexpected seven-dataset trajectory count'
    manifest=ds.metadata|{'args':vars(args),'token_order':['raw_rtg','return_prompt','state','action'],
      'architecture_scope':'independent per-AP model as old baseline; no joint-BS attention-mask change',
      'paper_limit':'bootstrap details are an explicit implementation choice, not an exact full-paper reproduction',
      'inference':'raw RTG -= raw reward each step; historical maximum prompt fixed at every step for paper-style new-task evaluation; no evaluation statistics are fitted',
      'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (dest/'experiment.json').write_text(json.dumps(manifest,indent=2))
    np.savez(dest/'return_prompt_stats.npz',mu=ds.mu,sigma=ds.sigma,prompt_max=ds.prompt_max)
    # Validate real examples and a finite backward pass before the full loop.
    loader=DataLoader(ds,batch_size=256,shuffle=True,num_workers=args.workers,
                      pin_memory=True,persistent_workers=args.workers>0)
    it=iter(loader)
    model=ReturnPromptDT().to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    total=args.rounds*args.updates
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1-step/total/4)
    with (dest/'loss.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['round','updates','elapsed_seconds','action_loss'])
        for round_idx in range(args.rounds):
            losses=[];model.train()
            for _ in range(args.updates):
                try:batch=next(it)
                except StopIteration:
                    it=iter(loader);batch=next(it)
                t,s,a,r,p=(x.to(args.device,non_blocking=True) for x in batch)
                logits=model(t,s,a,r,p)
                loss=sum(F.cross_entropy(v.reshape(-1,v.shape[-1]),a[...,k].long().reshape(-1))
                         for k,v in enumerate(logits))/2
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),.25,error_if_nonfinite=True)
                opt.step();scheduler.step()
                losses.append(float(loss.detach()))
            row=[round_idx+1,(round_idx+1)*args.updates,round(time.monotonic()-start,2),float(np.mean(losses))]
            writer.writerow(row);f.flush()
            ckpt={'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),
                  'scheduler_state_dict':scheduler.state_dict(),'round':round_idx+1,
                  'updates':row[1],'args':vars(args),'prompt_metadata':manifest,
                  'mu':torch.from_numpy(ds.mu),'sigma':torch.from_numpy(ds.sigma),
                  'inference_prompt':ds.prompt_max}
            tmp=dest/'model.pt.tmp';torch.save(ckpt,tmp);os.replace(tmp,dest/'model.pt')
            print('TRAIN_PROGRESS',*row,flush=True)
    print('ALL_TRAIN_COMPLETE',total,'updates',time.monotonic()-start,'seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset_files',nargs='+')
    p.add_argument('--log_dir',default='dt_results/PDT_return_prompt_seven_ctx8')
    p.add_argument('--rounds',type=int,default=1800)
    p.add_argument('--updates',type=int,default=150)
    p.add_argument('--bootstrap',type=int,default=256)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke_only',action='store_true')
    main(p.parse_args())
