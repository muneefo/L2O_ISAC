from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from l2o_isac import SystemConfig,generate_channels,initialize_state
from l2o_isac.physics import metrics,components,gradients
from l2o_isac.solver import project_ball,project_beamformers,constraint_errors,NAMES,PGDA
from l2o_isac.attacks import attack,error_gradients,union_attack
from l2o_isac.execution_evidence import numpy_metrics


def tiny():
    torch.set_num_threads(1)
    c=SystemConfig(nt=5,nr=3,users=2,targets=2,clutter=1,rf_chains=2)
    b=generate_channels(c,3,seed=110,dtype=torch.complex128)
    return c,b,initialize_state(b,c,seed=111)


def test_numpy_components_and_sinrs():
    c,b,s=tiny();z=components(b,s,c);ref=numpy_metrics(b,s,c)
    for k,v in metrics(b,s,c).items():np.testing.assert_allclose(v.numpy(),ref[k],rtol=1e-11,atol=1e-11)
    for i in range(len(b)):
        p=(s.a[i]@s.d[i]).numpy();h=(b.h[i]+s.dh[i]).numpy();radar=((b.alpha[i,:,None,None])*(b.ar[i]+s.dr[i])[:,:,None]*(b.at[i]+s.dt[i])[:,None,:]).numpy()
        for m in range(c.users):
            powers=np.abs(h[m].conj()@p)**2
            refsinr=powers[m]/(c.noise_comm+sum(powers[k] for k in range(c.streams) if k!=m))
            assert (z['csignal']/z['cden'])[i,m].item()==pytest.approx(refsinr,rel=1e-11)
        for t in range(c.targets):
            w=s.w[i,t].numpy();powers=[np.sum(np.abs(w.conj()@r@p)**2) for r in radar]
            refsinr=powers[t]/(c.noise_sense*np.sum(np.abs(w)**2)+sum(powers[k] for k in range(c.paths) if k!=t))
            assert (z['ssignal']/z['sden'])[i,t].item()==pytest.approx(refsinr,rel=1e-11)


def test_projection_cases_and_idempotence():
    c,b,s=tiny()
    for r in [0.,.1,1.,10.]:
        x=project_ball(s.dh,r)
        assert (x.norm(dim=-1)<=r+1e-12).all()
        torch.testing.assert_close(project_ball(x,r),x,rtol=1e-12,atol=1e-12)
    a,d,w=project_beamformers(s.a,s.d,s.w,c.power)
    for x,y in zip((a,d,w),project_beamformers(a,d,w,c.power)):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
    small=1e-8;g=gradients(b,s,c)
    inner=replace(s,dh=project_ball(s.dh-small*g['dh'],b.eps[:,0,None,None]*math.sqrt(c.nt)))
    assert (metrics(b,inner,c)['comm']<=metrics(b,s,c)['comm']+1e-12).all()
    proposals=[getattr(s,n)+small*g[n] for n in NAMES[3:]]
    a,d,w=project_beamformers(*proposals,c.power)
    assert (metrics(b,replace(s,a=a,d=d,w=w),c)['objective']>=metrics(b,s,c)['objective']-1e-12).all()


def test_attack_frozen_nested_separable_repeatable_batch():
    c,b,s=tiny();before=s.clone()
    kwargs=dict(budgets=(2,6),restarts=1,seed=5,adam_steps=3)
    out=attack(b,s,c,**kwargs);again=attack(b,s,c,**kwargs)
    for n in NAMES:torch.testing.assert_close(getattr(s,n),getattr(before,n),atol=0,rtol=0)
    for n in NAMES[3:]:torch.testing.assert_close(getattr(out['state'],n),getattr(s,n),atol=0,rtol=0)
    for k in ('comm','sense','objective'):
        assert (out['metrics'][6][k]<=out['metrics'][2][k]+1e-12).all()
        assert (out['metrics'][6][k]<=metrics(b,s,c)[k]+1e-12).all()
        torch.testing.assert_close(out['metrics'][6][k],again['metrics'][6][k],atol=0,rtol=0)
    assert max(constraint_errors(out['state'],b,c).values())<1e-8
    g=error_gradients(b,s,c);analytic=gradients(b,s,c)
    for n in ('dh','dt','dr'):torch.testing.assert_close(g[n],analytic[n],rtol=1e-10,atol=1e-11)
    # Deterministic zero/terminal-only paths are invariant to batching.
    whole=attack(b,s,c,budgets=(5,),restarts=0,adam_steps=2)
    for i in range(len(b)):
        ss=type(s)(**{n:getattr(s,n)[i:i+1] for n in NAMES})
        one=attack(b[i],ss,c,budgets=(5,),restarts=0,adam_steps=2)
        torch.testing.assert_close(whole['metrics'][5]['objective'][i:i+1],one['metrics'][5]['objective'],rtol=1e-11,atol=1e-11)
    union=union_attack(b,{'a':s,'b':s},[out['state']],c)
    torch.testing.assert_close(metrics(b,union['a'],c)['objective'],out['metrics'][6]['objective'])


def test_roles_equivalence_freezing_sign_and_finite_difference():
    from l2o_isac.solver import UnrolledPGDA
    from l2o_isac.role_training import RoleSeparatedPGDA
    c,b,s=tiny();ref=UnrolledPGDA(c,layers=3).double();model=RoleSeparatedPGDA(ref)
    a=ref(b,s)[0];z=model(b,s)[0]
    for n in NAMES:torch.testing.assert_close(getattr(a,n),getattr(z,n),rtol=0,atol=0)
    model.set_phase('inner');frozen=[p.clone() for p in model.outer_parameters()]
    initial=metrics(b,model.inner_attack(b,a),c)['objective'].mean().item()
    opt=torch.optim.SGD(model.inner_parameters(),lr=.001)
    for _ in range(5):
        opt.zero_grad();loss=metrics(b,model.inner_attack(b,a),c)['objective'].mean();loss.backward();opt.step()
    assert metrics(b,model.inner_attack(b,a),c)['objective'].mean().item()<initial
    for p,q in zip(model.outer_parameters(),frozen):torch.testing.assert_close(p,q,rtol=0,atol=0)
    model.set_phase('outer');frozen=[p.clone() for p in model.inner_parameters()]
    # Hold an approximate adversary fixed, exactly the stated stop-gradient surrogate.
    adverse=attack(b,model(b,s)[0],c,budgets=(3,),restarts=0,adam_steps=0)['state']
    def loss():
        design=model(b,s)[0];design=replace(design,**{n:getattr(adverse,n).detach() for n in NAMES[:3]})
        return -metrics(b,design,c)['objective'].mean()
    value=loss();grad,=torch.autograd.grad(value,model.outer_mu)
    index=(1,1);old=model.outer_mu[index].item();delta=1e-6;vals=[]
    with torch.no_grad():
        for sign in [1,-1]:model.outer_mu[index]=old+sign*delta;vals.append(loss().item())
        model.outer_mu[index]=old
    assert grad[index].item()==pytest.approx((vals[0]-vals[1])/(2*delta),rel=1e-4,abs=1e-7)
    before=loss().item();opt=torch.optim.SGD(model.outer_parameters(),lr=1e-5)
    opt.zero_grad();loss().backward();opt.step();assert loss().item()<before
    for p,q in zip(model.inner_parameters(),frozen):torch.testing.assert_close(p,q,rtol=0,atol=0)


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('imaginary',[False,True])
def test_complex_coordinate_finite_difference(name,imaginary):
    c,b,s=tiny();x=getattr(s,name);direction=torch.zeros_like(x)
    direction.reshape(-1)[0]=1j if imaginary else 1.
    key='comm' if name=='dh' else 'sense' if name in ('dt','dr') else 'objective'
    delta=1e-6
    numeric=(metrics(b,replace(s,**{name:x+delta*direction}),c)[key].sum()-metrics(b,replace(s,**{name:x-delta*direction}),c)[key].sum())/(2*delta)
    exact=2*(gradients(b,s,c)[name].conj()*direction).sum().real
    torch.testing.assert_close(exact,numeric,rtol=1e-4,atol=1e-8)


def test_new_attacker_closed_form_zero_radius_and_adam_floor():
    c=SystemConfig(nt=4,nr=3,users=1,targets=1,clutter=0,rf_chains=1)
    b=generate_channels(c,3,seed=15,dtype=torch.complex128);s=initialize_state(b,c,seed=8,zero_errors=True)
    d=torch.zeros_like(s.d);d[:,0,0]=math.sqrt(c.power/c.nt);s=replace(s,d=d);p=(s.a@s.d)[...,0]
    amp=((b.h[:,0].conj()*p).sum(-1).abs()-math.sqrt(c.nt)*b.eps[:,0]*p.norm(dim=-1)).clamp_min(0)
    expected=torch.log1p(amp.square()/c.noise_comm)
    out=attack(b,s,c,budgets=(100,),restarts=1,adam_steps=20)
    torch.testing.assert_close(out['metrics'][100]['comm'],expected,rtol=1e-6,atol=1e-8)
    b0=replace(b,eps=b.eps*0);z=attack(b0,s,c,budgets=(3,),restarts=1,adam_steps=2)
    torch.testing.assert_close(z['metrics'][3]['objective'],metrics(b0,s,c)['objective'],rtol=0,atol=0)
    # Removing/changing Adam's optimizer epsilon must not change the feasible
    # closed-form agreement: a PGD-only search already reaches the same minimum.
    pgd=attack(b,s,c,budgets=(100,),restarts=1,adam_steps=0)
    torch.testing.assert_close(pgd['metrics'][100]['comm'],expected,rtol=1e-6,atol=1e-8)


def test_complete_role_layers_keep_hardware_and_unequal_array_balls(monkeypatch):
    import l2o_isac.role_training as role_module
    from l2o_isac.solver import UnrolledPGDA
    c,b,s=tiny();model=role_module.RoleSeparatedPGDA(UnrolledPGDA(c,layers=4).double())
    with torch.no_grad():
        model.inner_mu[1].mul_(-20);model.outer_mu.mul_(20);model.inner_eta.fill_(.7);model.outer_eta.fill_(.7)
    captured=[];original=role_module.pgda_step
    def checked(*args,**kwargs):
        result=original(*args,**kwargs);captured.append(result)
        return result
    monkeypatch.setattr(role_module,'pgda_step',checked)
    model(b,s)
    assert len(captured)==4
    for state in captured:
        assert max(constraint_errors(state,b,c).values())<1e-8
        assert all(torch.isfinite(getattr(state,n)).all() for n in NAMES)
