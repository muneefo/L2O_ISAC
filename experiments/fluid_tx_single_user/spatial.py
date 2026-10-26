"""Fixed field-response scenarios: geometry changes phases, never path draws."""
from dataclasses import dataclass,fields,replace
import math
import torch
from l2o_isac.config import SystemConfig
from l2o_isac.data import ChannelBatch,complex_normal,initialize_state
from l2o_isac.solver import project_beamformers
from .geometry import steering,steering_derivative,project_positions,check_geometry


@dataclass
class Scenario:
    beta:torch.Tensor
    user_angles:torch.Tensor
    sensing_angles:torch.Tensor
    alpha:torch.Tensor
    eps:torch.Tensor

    def __len__(self):return self.beta.shape[0]
    def __getitem__(self,index):
        if isinstance(index,int):index=slice(index,index+1)
        return Scenario(**{f.name:getattr(self,f.name)[index] for f in fields(self)})


def system(spec,case,rho=None):
    return SystemConfig(nt=spec['nt'],nr=spec['nr'],users=1,targets=1,clutter=spec['clutter_case_c'] if case=='C' else 0,
                        rf_chains=spec['rf_chains'],rho=spec['rho'] if rho is None else rho,snr_db=spec['snr_db'],
                        eps_h=spec['epsilon'],eps_t=spec['epsilon'],eps_r=spec['epsilon'])


def make_scenarios(spec,case,count,seed,dtype=torch.complex64):
    cfg=system(spec,case);g=torch.Generator().manual_seed(seed);real=torch.float64 if dtype==torch.complex128 else torch.float32
    target=(torch.rand(count,1,generator=g,dtype=real)*10+(20 if case=='C' else 25))*math.pi/180
    if case=='A':
        target=(torch.rand(count,1,generator=g,dtype=real)*100-50)*math.pi/180
        beta=torch.exp(2j*math.pi*torch.rand(count,1,generator=g,dtype=real));user=target.clone()
    else:
        k=spec['communication_paths'];user=(torch.rand(count,k,generator=g,dtype=real)*130-65)*math.pi/180
        user[:,0]=(-25+torch.rand(count,generator=g,dtype=real)*10-5)*math.pi/180
        powers=torch.tensor([.75**2]+[.4**2]*(k-1),dtype=real);powers=powers/powers.sum()
        beta=complex_normal((count,k),g,dtype)*powers.sqrt()
    angles=[target]
    for j in range(cfg.clutter):
        angle=target+(10+10*j+torch.rand(count,1,generator=g,dtype=real)*4-2)*math.pi/180
        angles.append(angle)
    sens=torch.cat(angles,1);alpha=complex_normal((count,cfg.paths),g,dtype)
    variance=torch.tensor([10**(cfg.target_variance_db/10)]+[10**(cfg.clutter_variance_db/10)]*cfg.clutter,dtype=real)
    alpha=alpha*variance.sqrt();eps=torch.full((count,3),spec['epsilon'],dtype=real)
    return Scenario(beta,user,sens,alpha,eps)


def channels(scenario,x,cfg):
    user_response=steering(scenario.user_angles,x)
    # h^H=sum beta*a^T; ChannelBatch stores h, deliberately conjugated.
    h=(scenario.beta[...,None]*user_response).sum(1).conj()[:,None,:]
    at=steering(scenario.sensing_angles,x)
    rx=torch.arange(cfg.nr,dtype=x.dtype,device=x.device).expand(len(scenario),-1)*.5
    ar=steering(scenario.sensing_angles,rx)
    return ChannelBatch(h,at,ar,scenario.alpha,scenario.eps)


def channel_derivatives(scenario,x,cfg):
    dh=(scenario.beta[...,None]*steering_derivative(scenario.user_angles,x)).sum(1).conj()[:,None,:]
    dt=steering_derivative(scenario.sensing_angles,x)
    return dh,dt


def ula_positions(spec,count,dtype=torch.float32):
    check_geometry(spec['nt'],spec['aperture_wavelengths'],spec['min_spacing_wavelengths'])
    x=torch.arange(spec['nt'],dtype=dtype)*spec['ula_spacing_wavelengths']
    if x[-1]>spec['aperture_wavelengths'] or spec['ula_spacing_wavelengths']<spec['min_spacing_wavelengths']:raise ValueError('ULA outside common geometry set')
    return x.expand(count,-1).clone()


def random_positions(spec,count,seed,dtype=torch.float32):
    g=torch.Generator().manual_seed(seed);slack=spec['aperture_wavelengths']-(spec['nt']-1)*spec['min_spacing_wavelengths']
    z=torch.rand(count,spec['nt'],generator=g,dtype=dtype).sort(-1).values*slack
    if spec['nt']>1:z[:,0]=0;z[:,-1]=slack
    return z+torch.arange(spec['nt'],dtype=dtype)*spec['min_spacing_wavelengths']


def initialize(scenario,x,cfg,case,seed):
    batch=channels(scenario,x,cfg);state=initialize_state(batch,cfg,seed=seed,zero_errors=True)
    a=state.a.clone();a[:,:,0]=batch.h[:,0]/batch.h[:,0].abs().clamp_min(torch.finfo(x.dtype).tiny)
    if cfg.rf_chains>1:a[:,:,1]=batch.at[:,0].conj()
    d=torch.zeros_like(state.d);share=1. if case=='A' else .9
    d[:,0,0]=math.sqrt(cfg.power*share/cfg.nt)
    if cfg.rf_chains>1:d[:,1,1]=math.sqrt(cfg.power*(1-share)/cfg.nt)
    else:d[:,0,1]=math.sqrt(cfg.power*(1-share)/cfg.nt)
    w=batch.ar[:,:1]/math.sqrt(cfg.nr)
    a,d,w=project_beamformers(a,d,w,cfg.power)
    return replace(state,a=a,d=d,w=w)


def analytic_case_a(scenario,cfg):
    c=torch.log1p(torch.full((len(scenario),),cfg.power*cfg.nt/cfg.noise_comm,dtype=scenario.eps.dtype)*(1-scenario.eps[:,0]).clamp_min(0).square())
    v=scenario.alpha[:,0].abs().square()*cfg.power*cfg.nt*cfg.nr/cfg.noise_sense
    s=torch.log1p(v*(1-scenario.eps[:,1]).clamp_min(0).square()*(1-scenario.eps[:,2]).clamp_min(0).square())
    return {'comm':c,'sense':s,'objective':cfg.rho*c+(1-cfg.rho)*s}


def single_stream_exact_errors(batch,state,cfg):
    """Closed-form feasible minima for case A's one active transmit stream.

    Adds a validation candidate; not used as a certificate outside this case.
    """
    p=(state.a@state.d)[...,0];pn=p.norm(dim=-1,keepdim=True)
    tiny=torch.finfo(p.real.dtype).tiny
    def phase(z):return z/z.abs().clamp_min(tiny)
    c=(batch.h[:,0].conj()*p).sum(-1,keepdim=True)
    radius=batch.eps[:,0,None]*math.sqrt(cfg.nt);amount=torch.minimum(radius,c.abs()/pn.clamp_min(tiny))
    dh=-amount*phase(c).conj()*p/pn.clamp_min(tiny)
    t=(batch.at[:,0]*p).sum(-1,keepdim=True)
    radius=batch.eps[:,1,None]*math.sqrt(cfg.nt);amount=torch.minimum(radius,t.abs()/pn.clamp_min(tiny))
    dt=-amount*phase(t)*p.conj()/pn.clamp_min(tiny)
    w=state.w[:,0];wn=w.norm(dim=-1,keepdim=True);r=(w.conj()*batch.ar[:,0]).sum(-1,keepdim=True)
    radius=batch.eps[:,2,None]*math.sqrt(cfg.nr);amount=torch.minimum(radius,r.abs()/wn.clamp_min(tiny))
    dr=-amount*phase(r)*w/wn.clamp_min(tiny)
    return replace(state,dh=dh[:,None,:],dt=dt[:,None,:],dr=dr[:,None,:])
