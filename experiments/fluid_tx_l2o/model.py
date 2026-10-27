"""Inherited sequential MinNet/MaxNet plus an outer real-position update.

All six inherited step and momentum schedules are trainable. Position steps and
(optional) momentum add two schedules. Gradients are partials at each stage,
while training retains the full computational graph through all stages.
"""
import torch
from torch import nn
from l2o_isac.solver import pgda_step
from l2o_isac.physics import metrics
from experiments.fluid_tx_single_user.spatial import channels
from experiments.fluid_tx_single_user.geometry import project_positions
from experiments.fluid_tx_single_user.optimizer import position_gradient


class FluidUnroll(nn.Module):
    def __init__(self,cfg,spec,layers,inner=.0025,outer=.001,position=.02,
                 fluid=True,position_momentum=True,dtype=torch.float32):
        super().__init__();self.cfg=cfg;self.spec=spec;self.layers=layers;self.fluid=fluid;self.position_momentum=position_momentum
        self.mu=nn.Parameter(torch.tensor([inner]*3+[outer]*3+([position] if fluid else []),dtype=torch.float32).to(dtype).repeat(layers,1))
        self.eta=nn.Parameter(torch.tensor([.01]*6+([0.] if fluid else []),dtype=torch.float32).to(dtype).repeat(layers,1))

    def forward(self,scenario,x,initial,return_states=False):
        cur,prev=initial,initial;previous_x=x;trace=[];states=[]
        for k in range(self.layers):
            batch=channels(scenario,x,self.cfg)
            nxt=pgda_step(batch,cur,prev,self.cfg,self.mu[k,:6],self.eta[k,:6]);prev,cur=cur,nxt
            if self.fluid:
                gradient=position_gradient(scenario,x,cur,self.cfg)
                momentum=self.eta[k,6] if self.position_momentum else self.eta[k,6]*0
                proposal=x+self.mu[k,6]*gradient+momentum*(x-previous_x)
                previous_x,x=x,project_positions(proposal,self.spec['aperture_wavelengths'],self.spec['min_spacing_wavelengths'])
            trace.append(metrics(channels(scenario,x,self.cfg),cur,self.cfg))
            if return_states:states.append((x,cur))
        return {'x':x,'state':cur,'trace':trace,'states':states,
                'counts':{'full_analytic_calls':self.layers*(3 if self.fluid else 2),
                          'position_projections':self.layers if self.fluid else 0,
                          'field_reconstructions':self.layers*(3 if self.fluid else 2)}}

    def loss(self,result):
        # Paper Eq. (22): average negative utility over every layer and scenario.
        return -torch.stack([r['objective'] for r in result['trace']]).mean()

    def active_parameter_count(self):
        return self.mu.numel()+self.eta.numel()-(self.layers if self.fluid and not self.position_momentum else 0)
