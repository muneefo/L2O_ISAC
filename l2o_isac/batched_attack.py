"""Same independent candidate protocol as attacks.attack, starts folded into batch.

This changes execution batching, not search budgets/domains/schedules. Reference
attacker stays immutable. Tests compare full search outputs including sensing pairs.
"""
from dataclasses import fields,replace
import math,time
import torch
from .attacks import ERRORS,error_gradients,keep_best,project_errors
from .data import ChannelBatch,random_ball
from .solver import SolverState,NAMES
from .physics import metrics


def repeat_batch(batch,n):
    return ChannelBatch(**{f.name:getattr(batch,f.name).repeat((n,)+(1,)*(getattr(batch,f.name).ndim-1)) for f in fields(batch)})


def reduce_starts(batch,state,cfg,count):
    b=len(batch);z=metrics(repeat_batch(batch,count),state,cfg)
    ic=z['comm'].reshape(count,b).argmin(0);iss=z['sense'].reshape(count,b).argmin(0);idx=torch.arange(b,device=batch.h.device)
    vals={}
    for name in NAMES:
        x=getattr(state,name).reshape((count,b)+getattr(state,name).shape[1:])
        vals[name]=x[ic if name=='dh' else iss if name in ('dt','dr') else torch.zeros_like(ic),idx]
    return SolverState(**vals)


@torch.no_grad()
def batched_attack(batch,design,cfg,budgets=(200,1000),restarts=5,seed=9001,step_size=.2,adam_steps=50,adam_lr=.01):
    budgets=tuple(sorted(set(budgets)))
    if not budgets or min(budgets)<1 or restarts<0:raise ValueError('Invalid attack budgets')
    started=time.perf_counter();design=design.detach();zero=replace(design,**{n:torch.zeros_like(getattr(design,n)) for n in ERRORS})
    terminal=project_errors(design,batch,cfg);starts=[zero,terminal];g=torch.Generator().manual_seed(seed)
    for k in range(restarts):
        errs={}
        for i,name in enumerate(ERRORS):
            x=getattr(design,name);radius=batch.eps[:,i,None,None]*math.sqrt(x.shape[-1]);v=random_ball(x.shape,radius,g,x.dtype,x.device)
            if k%2==1:v=project_errors(replace(zero,**{name:v/v.norm(dim=-1,keepdim=True).clamp_min(torch.finfo(v.real.dtype).tiny)*radius}),batch,cfg).__dict__[name]
            errs[name]=v
        starts.append(replace(design,**errs))
    best=terminal
    for s in starts:best=keep_best(batch,best,s,cfg)
    if adam_steps:
        cur=zero;m={n:torch.zeros_like(getattr(cur,n)) for n in ERRORS};v={n:torch.zeros_like(getattr(cur,n).real) for n in ERRORS}
        for t in range(1,adam_steps+1):
            grad=error_gradients(batch,cur,cfg);update={}
            for n in ERRORS:
                m[n]=.9*m[n]+.1*grad[n];v[n]=.999*v[n]+.001*grad[n].abs().square()
                update[n]=getattr(cur,n)-adam_lr*(m[n]/(1-.9**t))/((v[n]/(1-.999**t)).sqrt()+1e-8)
            cur=project_errors(replace(cur,**update),batch,cfg);best=keep_best(batch,best,cur,cfg)
    count=len(starts);wide=repeat_batch(batch,count)
    cur=SolverState(**{n:torch.cat([getattr(s,n) for s in starts]) for n in NAMES})
    lr_c=torch.full((len(wide),1,1),step_size,dtype=batch.eps.dtype,device=batch.h.device);lr_s=lr_c.clone();snapshots={}
    for t in range(1,max(budgets)+1):
        grad=error_gradients(wide,cur,cfg)
        prop=project_errors(replace(cur,**{n:getattr(cur,n)-(lr_c if n=='dh' else lr_s)*grad[n] for n in ERRORS}),wide,cfg)
        before,after=metrics(wide,cur,cfg),metrics(wide,prop,cfg)
        ac,ass=after['comm']<=before['comm'],after['sense']<=before['sense'];cur=keep_best(wide,cur,prop,cfg)
        lr_c=torch.where(ac[:,None,None],lr_c*1.05,lr_c*.5).clamp(max=1.)
        lr_s=torch.where(ass[:,None,None],lr_s*1.05,lr_s*.5).clamp(max=1.)
        if t in budgets:
            best=keep_best(batch,best,reduce_starts(batch,cur,cfg,count),cfg);snapshots[t]=best.clone()
    return {'state':best,'snapshots':snapshots,'metrics':{b:metrics(batch,s,cfg) for b,s in snapshots.items()},
            'seconds':time.perf_counter()-started,'tensor_error_gradient_calls':max(budgets)+adam_steps,
            'logical_start_error_gradient_calls':count*max(budgets)+adam_steps,'total_pgd_starts':count}
