"""Independent frozen-design attacks. Feasible candidate search, not certification.

No dependency on solver.pgda_step or physics.gradients. PyTorch complex
metric gradients /2 use the reference Wirtinger convention. Sensing (dt,dr)
are accepted together across all targets. Communication may be retained apart.
"""
from dataclasses import replace
import math
import time
import torch
from .physics import metrics
from .solver import project_ball, NAMES
from .data import random_ball

ERRORS = NAMES[:3]


def project_errors(state, batch, cfg):
    return replace(state, **{n: project_ball(getattr(state, n), batch.eps[:, i, None, None]
                     * math.sqrt(cfg.nr if n == 'dr' else cfg.nt)) for i,n in enumerate(ERRORS)})


def keep_best(batch, best, candidate, cfg):
    """Separable minima: preserve jointly consistent t/r errors for sensing."""
    b, c = metrics(batch,best,cfg), metrics(batch,candidate,cfg)
    if not all(torch.isfinite(x).all() for x in c.values()):
        raise FloatingPointError('Nonfinite independent attack candidate')
    return replace(best, **{n:torch.where((c['comm' if n=='dh' else 'sense'] <
                 b['comm' if n=='dh' else 'sense'])[:,None,None],getattr(candidate,n),getattr(best,n)) for n in ERRORS})


def error_gradients(batch, state, cfg):
    """Independent differentiation; beamformers never require attack gradients."""
    with torch.enable_grad():
        x={n:getattr(state,n).detach().requires_grad_(True) for n in ERRORS}
        z=metrics(batch,replace(state,**x),cfg)
        # Unweighted independent components, including at rho=0 or 1.
        g=torch.autograd.grad((z['comm']+z['sense']).sum(),tuple(x.values()))
    return {n:v.detach()/2 for n,v in zip(ERRORS,g)}


@torch.no_grad()
def attack(batch, design, cfg, budgets=(50,200), restarts=5, seed=9100,
           step_size=.05, adam_steps=50, adam_lr=.01):
    """Nested PGD prefixes + zero/terminal + random interior/boundary + Adam.

    Five restarts means FIVE random starts in addition to zero and terminal.
    Backtracking retains separate component improvements per scenario. Raw full
    best-error tensors and per-scenario budget curves are returned for union use.
    """
    budgets=tuple(sorted(set(budgets)))
    if not budgets or min(budgets)<1 or restarts<0 or adam_steps<0 or step_size<=0:
        raise ValueError('Invalid attack budget')
    started=time.perf_counter();design=design.detach()
    zero=replace(design,**{n:torch.zeros_like(getattr(design,n)) for n in ERRORS})
    terminal=project_errors(design,batch,cfg)
    starts=[zero,terminal]
    g=torch.Generator().manual_seed(seed)
    for k in range(restarts):
        errs={}
        for i,n in enumerate(ERRORS):
            x=getattr(design,n);radius=batch.eps[:,i,None,None]*math.sqrt(x.shape[-1])
            value=random_ball(x.shape,radius,g,x.dtype,x.device)
            if k%2==1: value=project_ball(value/value.norm(dim=-1,keepdim=True).clamp_min(torch.finfo(x.real.dtype).tiny)*radius,radius)
            errs[n]=value
        starts.append(replace(design,**errs))
    # Seed ALL budget records identically, ensuring nested candidate sets.
    best=terminal
    for s in starts: best=keep_best(batch,best,s,cfg)
    snapshots={b:best.clone() for b in budgets}; evaluations=0
    # Adam is an additional strategy, independent of PGD momentum/schedules.
    if adam_steps:
        cur=zero;first={n:torch.zeros_like(getattr(cur,n)) for n in ERRORS}
        second={n:torch.zeros_like(getattr(cur,n).real) for n in ERRORS}
        for t in range(1,adam_steps+1):
            grad=error_gradients(batch,cur,cfg);evaluations+=1
            updates={}
            for n in ERRORS:
                first[n]=.9*first[n]+.1*grad[n]
                second[n]=.999*second[n]+.001*grad[n].abs().square()
                updates[n]=getattr(cur,n)-adam_lr*(first[n]/(1-.9**t))/((second[n]/(1-.999**t)).sqrt()+1e-8)
            cur=project_errors(replace(cur,**updates),batch,cfg)
            best=keep_best(batch,best,cur,cfg)
        snapshots={b:best.clone() for b in budgets}
    for start in starts:
        cur=start
        lr_c=torch.full((len(batch),1,1),step_size,dtype=batch.eps.dtype,device=batch.h.device)
        lr_s=lr_c.clone()
        for t in range(1,max(budgets)+1):
            grad=error_gradients(batch,cur,cfg);evaluations+=1
            prop=project_errors(replace(cur,**{n:getattr(cur,n)-(lr_c if n=='dh' else lr_s)*grad[n] for n in ERRORS}),batch,cfg)
            before,after=metrics(batch,cur,cfg),metrics(batch,prop,cfg)
            ac,ass=after['comm']<=before['comm'],after['sense']<=before['sense']
            cur=keep_best(batch,cur,prop,cfg)
            lr_c=torch.where(ac[:,None,None],lr_c*1.05,lr_c*.5).clamp(max=1.)
            lr_s=torch.where(ass[:,None,None],lr_s*1.05,lr_s*.5).clamp(max=1.)
            if t in snapshots: snapshots[t]=keep_best(batch,snapshots[t],cur,cfg)
    # Each larger record includes all smaller-budget candidates, even roundoff ties.
    last=terminal
    for b in budgets:
        snapshots[b]=keep_best(batch,snapshots[b],last,cfg);last=snapshots[b]
    return {'state':last,'snapshots':snapshots,'metrics':{b:metrics(batch,s,cfg) for b,s in snapshots.items()},
            'seconds':time.perf_counter()-started,'gradient_evaluations':evaluations,
            'random_restarts':restarts,'total_pgd_starts':len(starts),
            'adam_steps':adam_steps,'adam_denominator_epsilon':1e-8}


@torch.no_grad()
def union_attack(batch, designs, candidates, cfg):
    """Every method receives the same feasible pool, retaining its own minima."""
    result={}
    for name,design in designs.items():
        best=project_errors(design.detach(),batch,cfg)
        for candidate in candidates:
            proposed=replace(design,**{n:getattr(candidate,n) for n in ERRORS})
            best=keep_best(batch,best,proposed,cfg)
        result[name]=best
    return result
