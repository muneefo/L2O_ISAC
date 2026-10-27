import json
from pathlib import Path
import torch
from dataclasses import replace
from l2o_isac.solver import UnrolledPGDA,NAMES
from experiments.fluid_tx_single_user.spatial import system,make_scenarios,ula_positions,initialize,channels
from experiments.fluid_tx_single_user.optimizer import solve,all_residuals
from experiments.fluid_tx_l2o.model import FluidUnroll

torch.set_num_threads(1)

def setup(case='B'):
    sp=json.loads(Path('experiments/fluid_tx_l2o/configs/study.json').read_text());sp.update(nt=4,nr=3,rf_chains=2,aperture_wavelengths=1.8)
    c=system(sp,case);s=make_scenarios(sp,case,2,95123,torch.complex128);x=ula_positions(sp,2,torch.float64);st=initialize(s,x,c,case,12)
    return sp,c,s,x,st


def test_disabled_positions_exactly_match_inherited_unroll():
    sp,c,s,x,st=setup();model=FluidUnroll(c,sp,3,fluid=False,dtype=torch.float64)
    ref=UnrolledPGDA(c,layers=3).double()
    with torch.no_grad():ref.raw_mu.copy_(model.mu);ref.raw_eta.copy_(model.eta)
    r=model(s,x,st);out,trace,_=ref(channels(s,x,c),st)
    for n in NAMES:torch.testing.assert_close(getattr(r['state'],n),getattr(out,n),atol=0,rtol=0)
    torch.testing.assert_close(model.loss(r),-torch.stack([v['objective'] for v in trace]).mean(),atol=0,rtol=0)


def test_zero_position_momentum_matches_prior_fluid_partial():
    sp,c,s,x,st=setup();m=FluidUnroll(c,sp,3,inner=.01,outer=.001,position=.02,position_momentum=False,dtype=torch.float64)
    r=m(s,x,st);ref=solve(s,x,st,c,sp,3,.01,.001,.02,safeguard=False,position_schedule=m.mu[:,6])
    torch.testing.assert_close(r['x'],ref['x'],atol=0,rtol=0)
    for n in NAMES:torch.testing.assert_close(getattr(r['state'],n),getattr(ref['state'],n),atol=0,rtol=0)


def test_coupled_schedules_finite_difference_and_no_input_mutation():
    sp,c,s,x,st=setup();x=x+.04;m=FluidUnroll(c,sp,2,inner=.01,outer=.001,position=.0002,dtype=torch.float64)
    before=st.clone();r=m(s,x,st,True);loss=m.loss(r);loss.backward()
    assert m.mu.grad is not None and m.eta.grad is not None
    for tensor in [m.mu.grad,m.eta.grad]:assert torch.isfinite(tensor).all()
    # Test representatives of every uncertainty, beamformer and position block.
    for p in [m.mu,m.eta]:
        grad=p.grad.clone()
        for layer,col in [(0,i) for i in range(7)]+[(1,6)]:
            original=p[layer,col].item();eps=1e-6
            with torch.no_grad():p[layer,col]=original+eps
            a=m.loss(m(s,x,st)).item()
            with torch.no_grad():p[layer,col]=original-eps
            b=m.loss(m(s,x,st)).item()
            with torch.no_grad():p[layer,col]=original
            torch.testing.assert_close(grad[layer,col],torch.tensor((a-b)/(2*eps),dtype=grad.dtype),atol=3e-6,rtol=3e-3)
    for n in NAMES:torch.testing.assert_close(getattr(st,n),getattr(before,n),atol=0,rtol=0)
    for px,state in r['states']:assert max(all_residuals(px,state,s,c,sp).values())<1e-10


def test_joint_training_changes_inherited_and_position_parameter_blocks():
    sp,c,s,x,st=setup();m=FluidUnroll(c,sp,4,inner=.01,outer=.001,dtype=torch.float64)
    oldmu=m.mu.detach().clone();oldeta=m.eta.detach().clone();opt=torch.optim.SGD(m.parameters(),lr=1e-4)
    opt.zero_grad();m.loss(m(s,x+.03,st)).backward();opt.step()
    assert (m.mu[:,:3]!=oldmu[:,:3]).any()  # MinNet
    assert (m.mu[:,3:6]!=oldmu[:,3:6]).any() # MaxNet
    assert (m.mu[:,6]!=oldmu[:,6]).any()    # positions
    assert (m.eta[:,6]!=oldeta[:,6]).any()  # position momentum
    nomom=FluidUnroll(c,sp,3,position_momentum=False,dtype=torch.float64)
    nomom.loss(nomom(s,x,st)).backward();assert not nomom.eta.grad[:,6].any()


def test_configurable_two_clutter_and_finite_zero_channels():
    sp,c,s,x,st=setup('C');sp['clutter_case_c']=2;c=system(sp,'C');s=make_scenarios(sp,'C',2,33,torch.complex128)
    s=replace(s,beta=s.beta*0);st=initialize(s,x,c,'C',12)
    m=FluidUnroll(c,sp,3,dtype=torch.float64);r=m(s,x,st,True)
    for v in r['trace']:
        assert all(torch.isfinite(z).all() for z in v.values())
    for px,state in r['states']:assert max(all_residuals(px,state,s,c,sp).values())<1e-10


def test_shared_candidates_scored_with_recipient_frozen_beamformers():
    from l2o_isac.attacks import union_attack,ERRORS
    from l2o_isac.batched_attack import batched_attack
    from l2o_isac.physics import metrics
    sp,c,s,x,st=setup('C');batch=channels(s,x,c)
    other=replace(st,d=st.d*.2,w=st.w*2)
    own=batched_attack(batch,st,c,budgets=(3,),restarts=1,adam_steps=0)['state']
    candidate=batched_attack(batch,other,c,budgets=(3,),restarts=1,adam_steps=0)['state']
    pool=[own,candidate];out=union_attack(batch,{'recipient':st},pool,c)['recipient']
    expected=[]
    for z in [st]+pool:expected.append(metrics(batch,replace(st,**{n:getattr(z,n) for n in ERRORS}),c))
    got=metrics(batch,out,c)
    for name in ['comm','sense']:
        torch.testing.assert_close(got[name],torch.stack([r[name] for r in expected]).min(0).values,rtol=0,atol=0)
    for n in NAMES[3:]:torch.testing.assert_close(getattr(out,n),getattr(st,n),rtol=0,atol=0)
