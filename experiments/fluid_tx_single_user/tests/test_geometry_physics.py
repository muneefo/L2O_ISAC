from dataclasses import replace
import itertools,math
import numpy as np
import pytest
from scipy.optimize import minimize,Bounds,LinearConstraint
import torch
from l2o_isac.physics import metrics
from l2o_isac.solver import PGDA,NAMES,SolverState
from l2o_isac.data import ChannelBatch
from l2o_isac.inference import load_checkpoint
from experiments.fluid_tx_single_user.geometry import project_positions,steering,steering_derivative,physical_steering,geometry_residuals
from experiments.fluid_tx_single_user.spatial import system,make_scenarios,channels,ula_positions,initialize,analytic_case_a
from experiments.fluid_tx_single_user.optimizer import position_gradient,position_step,solve,all_residuals


def setup(case='C',count=3):
    torch.set_num_threads(1)
    spec={'nt':4,'nr':3,'rf_chains':2,'aperture_wavelengths':1.8,'min_spacing_wavelengths':.2,'ula_spacing_wavelengths':.5,
          'rho':.6,'snr_db':5.,'epsilon':.05,'clutter_case_c':2,'communication_paths':3,'position_backtracks':6,'position_armijo':1e-4}
    cfg=system(spec,case);s=make_scenarios(spec,case,count,81,torch.complex128);x=ula_positions(spec,count,torch.float64)
    state=initialize(s,x,cfg,case,82)
    return spec,cfg,s,x,state


def test_ula_steering_field_repeatability_conjugation_and_units():
    spec,cfg,s,x,state=setup()
    expected=torch.exp(-1j*math.pi*s.sensing_angles.sin()[...,None]*torch.arange(cfg.nt,dtype=x.dtype))
    torch.testing.assert_close(steering(s.sensing_angles,x),expected,rtol=1e-13,atol=1e-13)
    a=channels(s,x,cfg);b=channels(s,x,cfg)
    for name in ['h','at','ar','alpha','eps']:torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    row=(s.beta[...,None]*steering(s.user_angles,x)).sum(1)
    torch.testing.assert_close(a.h[:,0].conj(),row)
    wavelength=.013;p=(x*wavelength).requires_grad_();v=physical_steering(s.sensing_angles,p,wavelength).real.sum()
    actual,=torch.autograd.grad(v,p)
    expected=steering_derivative(s.sensing_angles,x).real.sum(1)/wavelength
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


def numpy_reference(s,x,state,cfg):
    vals=[]
    for i in range(len(s)):
        xx=x[i].numpy();at=np.exp(-2j*np.pi*np.sin(s.sensing_angles[i].numpy())[:,None]*xx[None,:])+state.dt[i].numpy()
        ar=np.exp(-1j*np.pi*np.sin(s.sensing_angles[i].numpy())[:,None]*np.arange(cfg.nr)[None,:])+state.dr[i].numpy()
        hrow=np.sum(s.beta[i].numpy()[:,None]*np.exp(-2j*np.pi*np.sin(s.user_angles[i].numpy())[:,None]*xx[None,:]),axis=0)+state.dh[i,0].numpy().conj()
        p=state.a[i].numpy()@state.d[i].numpy();cp=np.abs(hrow@p)**2
        c=np.log1p(cp[0]/(cp[1:].sum()+cfg.noise_comm));w=state.w[i,0].numpy()
        radar=s.alpha[i].numpy()[:,None,None]*ar[:,:,None]*at[:,None,:]
        power=np.array([np.sum(np.abs(w.conj()@h@p)**2) for h in radar])
        sense=np.log1p(power[0]/(power[1:].sum()+cfg.noise_sense*np.sum(np.abs(w)**2)))
        vals.append([c,sense,cfg.rho*c+(1-cfg.rho)*sense])
    return np.array(vals)


def test_independent_spatial_signal_equations_including_incoherent_clutter():
    spec,cfg,s,x,state=setup()
    ref=numpy_reference(s,x,state,cfg);actual=metrics(channels(s,x,cfg),state,cfg)
    for i,k in enumerate(['comm','sense','objective']):np.testing.assert_allclose(actual[k].numpy(),ref[:,i],rtol=1e-11,atol=1e-11)
    radar=channels(s,x,cfg).radar;p=state.a@state.d
    waves=state.w[:,0,None,None,:].conj()@radar[:,1:]@p[:,None]
    incoherent=waves.abs().square().sum((-3,-2,-1));coherent=waves.sum(1).abs().square().sum((-2,-1))
    assert not torch.allclose(incoherent,coherent)


@pytest.mark.parametrize('case',['A','B','C'])
def test_fixed_error_position_partial_against_autograd_and_finite_difference(case):
    spec,cfg,s,x,state=setup(case);x=x+.05
    probe=x.clone().requires_grad_();value=metrics(channels(s,probe,cfg),state,cfg)['objective'].sum()
    expected,=torch.autograd.grad(value,probe);actual=position_gradient(s,x,state,cfg)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
    step=1e-6
    for index in range(cfg.nt):
        perturb=torch.zeros_like(x);perturb[:,index]=step
        plus=metrics(channels(s,x+perturb,cfg),state,cfg)['objective'];minus=metrics(channels(s,x-perturb,cfg),state,cfg)['objective']
        torch.testing.assert_close(actual[:,index],(plus-minus)/(2*step),rtol=1e-5,atol=1e-8)


def exhaustive_isotonic_qp(proposal,A,d):
    n=len(proposal);shifted=proposal-np.arange(n)*d;upper=A-(n-1)*d;best=None
    for mask in itertools.product([False,True],repeat=n-1):
        cuts=[0]+[i+1 for i,cut in enumerate(mask) if cut]+[n];v=np.empty(n)
        for left,right in zip(cuts[:-1],cuts[1:]):v[left:right]=np.clip(shifted[left:right].mean(),0,upper)
        if np.any(np.diff(v)<-1e-12):continue
        p=v+np.arange(n)*d;cost=np.sum((p-proposal)**2)
        if best is None or cost<best[0]:best=(cost,p)
    return best[1]


def test_exact_projection_against_two_independent_convex_qp_references():
    g=np.random.default_rng(100);n=4;A=1.8;d=.2
    diff=np.eye(n,k=1)[:-1]-np.eye(n)[:-1]
    for proposal in [np.zeros(n),np.arange(n)*.5,*g.normal(size=(24,n))*2]:
        got=project_positions(torch.tensor(proposal,dtype=torch.float64),A,d).numpy()
        result=minimize(lambda p:.5*np.sum((p-proposal)**2),np.arange(n)*.5,jac=lambda p:p-proposal,
                        bounds=Bounds(np.zeros(n),np.full(n,A)),constraints=LinearConstraint(diff,d,np.inf),method='SLSQP',options={'ftol':1e-12,'maxiter':200})
        assert result.success,result.message
        np.testing.assert_allclose(got,result.x,rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(got,exhaustive_isotonic_qp(proposal,A,d),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(project_positions(torch.tensor(got),A,d).numpy(),got,rtol=1e-12,atol=1e-12)
        assert got[0]>=-1e-12 and got[-1]<=A+1e-12 and np.min(np.diff(got))>=d-1e-12
    with pytest.raises(ValueError):project_positions(torch.zeros(4),.1,.2)


def test_projection_exact_jacobian_and_second_derivative_away_from_kinks():
    x=torch.tensor([[.4,.25,1.1,1.5]],dtype=torch.float64,requires_grad=True)
    f=lambda p:project_positions(p,1.8,.2)
    assert torch.autograd.gradcheck(f,(x,),eps=1e-6,atol=1e-6,rtol=1e-4)
    assert torch.autograd.gradgradcheck(f,(x,),eps=1e-6,atol=1e-6,rtol=1e-4)


def test_complete_iterations_feasibility_disabled_position_equivalence_and_freezing():
    spec,cfg,s,x,state=setup();initial=state.clone();batch=channels(s,x,cfg)
    fixed=solve(s,x,state,cfg,spec,4,.05,.005,return_states=True)
    reference=PGDA(cfg,iterations=4,mu=[.05]*3+[.005]*3).double()(batch,state)[0]
    for n in NAMES:torch.testing.assert_close(getattr(fixed['state'],n),getattr(reference,n),rtol=0,atol=0)
    torch.testing.assert_close(fixed['x'],x,rtol=0,atol=0)
    moved,info=position_step(s,x,state,cfg,spec,.02)
    for n in NAMES:torch.testing.assert_close(getattr(initial,n),getattr(state,n),rtol=0,atol=0)
    before=metrics(batch,state,cfg)['objective'];after=metrics(channels(s,moved,cfg),state,cfg)['objective']
    assert (after>=before-1e-12).all()
    fluid=solve(s,x,state,cfg,spec,6,.05,.005,.02,return_states=True)
    for px,ps in fluid['states']:
        assert max(all_residuals(px,ps,s,cfg,spec).values())<1e-10
    assert fluid['counts']['full_analytic_calls']==18


def test_short_position_schedule_full_unroll_derivative_and_zero_channel():
    spec,cfg,s,x,state=setup('B');x=x+.04
    schedule=torch.tensor([.0001,.0002],dtype=torch.float64,requires_grad=True)
    def f(v):
        r=solve(s,x,state,cfg,spec,2,.01,.001,safeguard=False,position_schedule=v)
        return metrics(channels(s,r['x'],cfg),r['state'],cfg)['objective'].mean()
    analytic,=torch.autograd.grad(f(schedule),schedule);step=1e-6
    for i in range(2):
        e=torch.zeros_like(schedule);e[i]=step
        torch.testing.assert_close(analytic[i],(f(schedule.detach()+e)-f(schedule.detach()-e))/(2*step),rtol=1e-4,atol=1e-7)
    zero=replace(s,beta=torch.zeros_like(s.beta));z=initialize(zero,x,cfg,'B',82)
    assert all(torch.isfinite(v).all() for v in metrics(channels(zero,x,cfg),z,cfg).values())


def test_aligned_case_a_closed_form_is_geometry_invariant_and_attack_reaches_it():
    from l2o_isac.batched_attack import batched_attack
    spec,cfg,s,x,state=setup('A');expected=analytic_case_a(s,cfg)
    for positions in [x,project_positions(x+.11*torch.randn_like(x),spec['aperture_wavelengths'],spec['min_spacing_wavelengths'])]:
        init=initialize(s,positions,cfg,'A',82);out=batched_attack(channels(s,positions,cfg),init,cfg,budgets=(200,),restarts=2,adam_steps=30)
        for k in ['comm','sense','objective']:torch.testing.assert_close(out['metrics'][200][k],expected[k],rtol=1e-7,atol=1e-9)


def test_read_only_existing_checkpoint_regression():
    from pathlib import Path
    root=Path('reports/corrected_per_channel_seed73129');cfg,_,model,_=load_checkpoint(root/'checkpoint.pt')
    raw=torch.load(root/'raw_beamformers_and_channels.pt',weights_only=True);batch=ChannelBatch(**raw['channel']);initial=SolverState(**raw['initial'])
    with torch.no_grad():state,trace,_=model(batch,initial,eps=raw['predicted_eps'])
    for n in NAMES:torch.testing.assert_close(getattr(state,n),raw['trained'][n],rtol=0,atol=0)
    torch.testing.assert_close(torch.stack([v['objective'] for v in trace]),raw['objective_trace'],rtol=0,atol=0)
