"""Original alternating PGDA plus an exact position partial and PAV projection."""
from dataclasses import replace
import time
import torch
from l2o_isac.physics import gradients,metrics
from l2o_isac.solver import pgda_step,constraint_errors
from .spatial import channels,channel_derivatives
from .geometry import project_positions,geometry_residuals


def position_gradient(scenario,x,state,cfg):
    """Fixed-error/fixed-beam partial; expression itself remains differentiable.

    Core returns unweighted comm h and unweighted sense transmit Wirtinger
    gradients. A real-coordinate derivative is 2Re(conj(g)*dchannel/dx).
    """
    batch=channels(scenario,x,cfg);g=gradients(batch,state,cfg);dh,dt=channel_derivatives(scenario,x,cfg)
    return 2*(cfg.rho*(g['dh'].conj()*dh).sum(1)+(1-cfg.rho)*(g['dt'].conj()*dt).sum(1)).real


def position_step(scenario,x,state,cfg,spec,step,safeguard=True):
    g=position_gradient(scenario,x,state,cfg)
    if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite geometry gradient')
    if not safeguard:
        return project_positions(x+step*g,spec['aperture_wavelengths'],spec['min_spacing_wavelengths']),{'position_gradients':1,'projections':1,'metric_evaluations':0}
    before=metrics(channels(scenario,x,cfg),state,cfg)['objective'];result=x.clone();remaining=torch.ones(len(x),dtype=torch.bool,device=x.device)
    trials=0
    for j in range(spec['position_backtracks']):
        proposal=project_positions(x+step*(.5**j)*g,spec['aperture_wavelengths'],spec['min_spacing_wavelengths'])
        value=metrics(channels(scenario,proposal,cfg),state,cfg)['objective']
        rhs=before+spec['position_armijo']*(g*(proposal-x)).sum(-1)
        accept=remaining&(value>=rhs-10*torch.finfo(x.dtype).eps)
        result=torch.where(accept[:,None],proposal,result);remaining=remaining&~accept;trials+=1
        if not remaining.any():break
    return result,{'position_gradients':1,'projections':trials,'metric_evaluations':trials+1}


def solve(scenario,initial_x,initial_state,cfg,spec,iterations,mu_inner,mu_outer,position_step_size=0.,safeguard=True,
          position_schedule=None,return_states=False):
    cur,prev=initial_state,initial_state;x=initial_x;trace=[];states=[]
    # Match the shared PGDA buffer initialization, including its float32-to-double convention.
    mu=torch.tensor([mu_inner]*3+[mu_outer]*3,dtype=torch.float32,device=x.device).to(x.dtype)
    eta=torch.full((6,),.01,dtype=torch.float32,device=x.device).to(x.dtype)
    counts={'full_analytic_calls':0,'position_projections':0,'position_metric_evaluations':0,'field_reconstructions':0}
    started=time.perf_counter()
    for i in range(iterations):
        batch=channels(scenario,x,cfg);nxt=pgda_step(batch,cur,prev,cfg,mu,eta);prev,cur=cur,nxt
        counts['full_analytic_calls']+=2;counts['field_reconstructions']+=1
        step=position_schedule[i] if position_schedule is not None else position_step_size
        if position_schedule is not None or position_step_size!=0:
            x,info=position_step(scenario,x,cur,cfg,spec,step,safeguard)
            counts['full_analytic_calls']+=1;counts['position_projections']+=info['projections'];counts['position_metric_evaluations']+=info['metric_evaluations'];counts['field_reconstructions']+=1+info['metric_evaluations']
        trace.append(metrics(channels(scenario,x,cfg),cur,cfg));counts['field_reconstructions']+=1
        if return_states:states.append((x,cur))
    return {'x':x,'state':cur,'trace':trace,'states':states,'counts':counts,'seconds':time.perf_counter()-started}


def all_residuals(x,state,scenario,cfg,spec):
    return {**constraint_errors(state,channels(scenario,x,cfg),cfg),**{k:v.max().item() for k,v in geometry_residuals(x,spec['aperture_wavelengths'],spec['min_spacing_wavelengths']).items()}}
