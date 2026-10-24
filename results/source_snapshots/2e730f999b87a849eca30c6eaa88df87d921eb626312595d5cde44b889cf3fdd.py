"""Role-separated scalar schedules; deployment keeps reference projected layers."""
from dataclasses import replace
import torch
from torch import nn
from .solver import NAMES,pgda_step,project_ball
from .physics import gradients,metrics
import math


class RoleSeparatedPGDA(nn.Module):
    def __init__(self,reference):
        super().__init__();self.cfg=reference.cfg;self.layers=reference.layers
        self.inner_mu=nn.Parameter(reference.mu[:,:3].detach().clone())
        self.inner_eta=nn.Parameter(reference.eta[:,:3].detach().clone())
        self.outer_mu=nn.Parameter(reference.mu[:,3:].detach().clone())
        self.outer_eta=nn.Parameter(reference.eta[:,3:].detach().clone())

    def inner_parameters(self):return (self.inner_mu,self.inner_eta)
    def outer_parameters(self):return (self.outer_mu,self.outer_eta)

    def set_phase(self,phase):
        for p in self.inner_parameters():p.requires_grad_(phase=='inner')
        for p in self.outer_parameters():p.requires_grad_(phase=='outer')

    def forward(self,batch,initial):
        cur,prev=initial,initial;trace=[]
        for i in range(self.layers):
            nxt=pgda_step(batch,cur,prev,self.cfg,torch.cat((self.inner_mu[i],self.outer_mu[i])),
                           torch.cat((self.inner_eta[i],self.outer_eta[i])))
            prev,cur=cur,nxt;trace.append(metrics(batch,cur,self.cfg))
        return cur,trace,[]

    def inner_attack(self,batch,design):
        cur=replace(design.detach(),**{n:torch.zeros_like(getattr(design,n)) for n in NAMES[:3]});prev=cur
        for i in range(self.layers):
            g=gradients(batch,cur,self.cfg)
            updates={n:project_ball(getattr(cur,n)-self.inner_mu[i,j]*g[n]+
                       self.inner_eta[i,j]*(getattr(cur,n)-getattr(prev,n)),
                       batch.eps[:,j,None,None]*math.sqrt(self.cfg.nr if n=='dr' else self.cfg.nt))
                       for j,n in enumerate(NAMES[:3])}
            prev,cur=cur,replace(cur,**updates)
        return cur
