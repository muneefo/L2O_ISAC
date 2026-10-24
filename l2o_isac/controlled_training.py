"""Controlled continuation; preserve physics and pretrained reference checkpoints."""
import copy,time
from dataclasses import replace
import torch
from .role_training import RoleSeparatedPGDA
from .solver import unrolled_loss
from .physics import metrics
from .attacks import attack,ERRORS


class ControlledSchedule(RoleSeparatedPGDA):
    def __init__(self,reference,positive_inner=False,floor=1e-6):
        super().__init__(reference);self.positive_inner=positive_inner;self.floor=floor
        if positive_inner:
            with torch.no_grad():self.inner_mu.copy_(self.inner_mu.abs().clamp_min(floor))

    def set_phase(self,phase):
        if phase=='joint':
            for p in self.parameters():p.requires_grad_(True)
        else:super().set_phase(phase)

    @torch.no_grad()
    def project_inner(self):
        if self.positive_inner:self.inner_mu.clamp_(min=self.floor)


def check_gradients(model):
    for p in model.parameters():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            raise FloatingPointError('Nonfinite/missing controlled-training gradient')


def joint_update(model,batch,initial,optimizer):
    model.set_phase('joint');optimizer.zero_grad(set_to_none=True)
    loss=unrolled_loss(model(batch,initial)[1]);loss.backward();check_gradients(model)
    optimizer.step();model.project_inner()
    return {'loss':loss.item(),'full_analytic_calls':2*model.layers,'error_gradient_calls':0,
            'differentiated_layers':model.layers,'differentiated_full_analytic_calls':2*model.layers,
            'backward_passes':1,'optimizer_step_calls':1,'inner_block_updates':1,'outer_block_updates':1}


def separated_update(model,batch,initial,inner_optimizer,outer_optimizer,attack_steps=15,step_size=.2,outer_loss_mode='terminal'):
    with torch.no_grad():design=model(batch,initial)[0]
    model.set_phase('inner');inner_optimizer.zero_grad(set_to_none=True)
    inner_loss=metrics(batch,model.inner_attack(batch,design),model.cfg)['objective'].mean()
    inner_loss.backward();check_gradients(model);inner_optimizer.step();model.project_inner()
    model.set_phase('outer');outer_optimizer.zero_grad(set_to_none=True)
    state,trace,_=model(batch,initial)
    if outer_loss_mode=='joint':
        outer_loss=unrolled_loss(trace)
    else:
        adverse=attack(batch,state,model.cfg,budgets=(attack_steps,),restarts=0,adam_steps=0,step_size=step_size)['state']
        surrogate=replace(state,**{n:getattr(adverse,n).detach() for n in ERRORS})
        outer_loss=-metrics(batch,surrogate,model.cfg)['objective'].mean()
    outer_loss.backward();check_gradients(model);outer_optimizer.step();model.project_inner()
    return {'inner_loss':inner_loss.item(),'outer_loss':outer_loss.item(),'full_analytic_calls':5*model.layers,
            'error_gradient_calls':0 if outer_loss_mode=='joint' else 2*attack_steps,'differentiated_layers':2*model.layers,
            'differentiated_full_analytic_calls':3*model.layers,'backward_passes':2,'optimizer_step_calls':2,
            'inner_block_updates':1,'outer_block_updates':1}
