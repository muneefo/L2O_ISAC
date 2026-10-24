import torch
from dataclasses import replace
from l2o_isac import SystemConfig,generate_channels,initialize_state
from l2o_isac.solver import NAMES,constraint_errors
from l2o_isac.attacks import attack
from l2o_isac.batched_attack import batched_attack


def tiny():
    torch.set_num_threads(1);cfg=SystemConfig(nt=5,nr=3,users=2,targets=2,clutter=1,rf_chains=2)
    batch=generate_channels(cfg,3,seed=801,dtype=torch.complex128)
    return cfg,batch,initialize_state(batch,cfg,seed=802)


def test_batched_attack_matches_preserved_search_and_frozen_inputs():
    cfg,batch,state=tiny();old=state.clone();kwargs=dict(budgets=(5,15),restarts=3,adam_steps=5,seed=19,step_size=.2)
    serial=attack(batch,state,cfg,**kwargs);parallel=batched_attack(batch,state,cfg,**kwargs)
    for budget in kwargs['budgets']:
        for key in ['comm','sense','objective']:
            torch.testing.assert_close(serial['metrics'][budget][key],parallel['metrics'][budget][key],atol=1e-11,rtol=1e-10)
    for n in NAMES:torch.testing.assert_close(getattr(state,n),getattr(old,n),atol=0,rtol=0)
    assert max(constraint_errors(parallel['state'],batch,cfg).values())<1e-8
    assert (parallel['metrics'][15]['objective']<=parallel['metrics'][5]['objective']+1e-12).all()


def test_controlled_positive_mapping_and_exact_reference():
    from l2o_isac.solver import UnrolledPGDA
    from l2o_isac.controlled_training import ControlledSchedule
    c,b,s=tiny();ref=UnrolledPGDA(c,layers=3).double()
    with torch.no_grad():ref.raw_mu[0,0]=-.003;ref.raw_eta[1,0]=-.2
    original=ControlledSchedule(ref);mapped=ControlledSchedule(ref,True)
    for n in NAMES:torch.testing.assert_close(getattr(original(b,s)[0],n),getattr(ref(b,s)[0],n),rtol=0,atol=0)
    torch.testing.assert_close(mapped.inner_mu,ref.mu[:,:3].abs().clamp_min(1e-6))
    torch.testing.assert_close(mapped.inner_eta,original.inner_eta,rtol=0,atol=0)
    torch.testing.assert_close(mapped.outer_eta,original.outer_eta,rtol=0,atol=0)
    torch.testing.assert_close(mapped.outer_mu,original.outer_mu,rtol=0,atol=0)
    with torch.no_grad():mapped.inner_mu.fill_(-1);mapped.inner_eta.fill_(-.3)
    mapped.project_inner();assert mapped.inner_mu.min()>=1e-6;assert mapped.inner_eta.min()<0


def test_inner_full_search_lowers_numpy_utility_and_keeps_outer_frozen():
    from l2o_isac.solver import UnrolledPGDA
    from l2o_isac.controlled_training import ControlledSchedule
    from l2o_isac.execution_evidence import numpy_metrics
    import numpy as np
    c,b,s=tiny();model=ControlledSchedule(UnrolledPGDA(c,layers=3).double(),True)
    model.set_phase('inner');outer=[p.clone() for p in model.outer_parameters()]
    before=numpy_metrics(b,model.inner_attack(b,s),c)['objective'].mean();opt=torch.optim.SGD(model.inner_parameters(),lr=.001)
    for _ in range(8):
        opt.zero_grad();from l2o_isac.physics import metrics
        loss=metrics(b,model.inner_attack(b,s),c)['objective'].mean();loss.backward();opt.step();model.project_inner()
    after=numpy_metrics(b,model.inner_attack(b,s),c)['objective'].mean();assert after<before
    for p,q in zip(model.outer_parameters(),outer):torch.testing.assert_close(p,q,rtol=0,atol=0)
    model.set_phase('outer');inner=[p.clone() for p in model.inner_parameters()]
    adverse=batched_attack(b,model(b,s)[0],c,budgets=(3,),restarts=0,adam_steps=0)['state']
    def objective():
        state=model(b,s)[0]
        return -metrics(b,replace(state,**{n:getattr(adverse,n).detach() for n in NAMES[:3]}),c)['objective'].mean()
    initial=objective().item();opt=torch.optim.SGD(model.outer_parameters(),lr=1e-5);opt.zero_grad();objective().backward();opt.step()
    assert objective().item()<initial
    for p,q in zip(model.inner_parameters(),inner):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_controlled_joint_loss_and_through_update_gradient():
    from l2o_isac.solver import UnrolledPGDA,unrolled_loss
    from l2o_isac.controlled_training import ControlledSchedule,joint_update
    c,b,s=tiny();ref=UnrolledPGDA(c,layers=3).double();model=ControlledSchedule(ref,True)
    # Same positive forward parameterization used by C and D; no detach in joint unroll.
    loss=unrolled_loss(model(b,s)[1]);g,=torch.autograd.grad(loss,model.inner_mu)
    idx=(1,1);old=model.inner_mu[idx].item();delta=1e-6;values=[]
    with torch.no_grad():
        for sign in [1,-1]:model.inner_mu[idx]=old+sign*delta;values.append(unrolled_loss(model(b,s)[1]).item())
        model.inner_mu[idx]=old
    torch.testing.assert_close(g[idx],torch.tensor((values[0]-values[1])/(2*delta),dtype=torch.float64),rtol=1e-4,atol=1e-8)
    opt=torch.optim.SGD(model.parameters(),lr=.0001);r=joint_update(model,b,s,opt)
    assert r['full_analytic_calls']==6 and model.inner_mu.min()>=1e-6


def test_balanced_training_checkpoint_resume_uses_portable_metadata(tmp_path):
    from l2o_isac.controlled_study import ControlledStudy
    from l2o_isac.architecture_audit import write,state_slice
    study=ControlledStudy(output=tmp_path);study.spec['epochs']=1;study.spec['batch_size']=4
    study.data['train']=study.data['train'][:4];study.initial['train']=state_slice(study.initial['train'],slice(0,4))
    study.data['validation']=study.data['validation'][:2];study.initial['validation']=state_slice(study.initial['validation'],slice(0,2))
    write(tmp_path/'compute_matching.json',{'methods':{'B':{'median_seconds':1.},'C':{'median_seconds':1.},'D':{'median_seconds':2.4}}})
    path=study.train_case(0,'C',101,balanced=True)
    last=torch.load(path.parent/'latest.pt',weights_only=True)
    assert type(last['cycles_per_minibatch']) is float
    assert last['counts']['full_analytic_calls']==60
    params={k:v.clone() for k,v in last['model'].items()}
    study.train_case(0,'C',101,balanced=True)
    after=torch.load(path.parent/'latest.pt',weights_only=True)
    for k,v in params.items():torch.testing.assert_close(after['model'][k],v,atol=0,rtol=0)


def test_lossmatched_separation_uses_identical_outer_loss_to_joint_control():
    from l2o_isac.solver import UnrolledPGDA,unrolled_loss
    from l2o_isac.controlled_training import ControlledSchedule,separated_update
    c,b,s=tiny();model=ControlledSchedule(UnrolledPGDA(c,layers=3).double(),True)
    expected=unrolled_loss(model(b,s)[1]).item()
    # Zero optimizer LR isolates the choice of objective from changed parameters.
    io=torch.optim.SGD(model.inner_parameters(),lr=0.);oo=torch.optim.SGD(model.outer_parameters(),lr=0.)
    result=separated_update(model,b,s,io,oo,outer_loss_mode='joint')
    assert result['outer_loss']==expected
    assert result['error_gradient_calls']==0
    assert result['full_analytic_calls']==15
