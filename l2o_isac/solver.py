from dataclasses import dataclass, fields, replace
import math
import torch
from torch import nn
from .physics import gradients, metrics


NAMES = ("dh", "dt", "dr", "a", "d", "w")


@dataclass
class SolverState:
    dh: torch.Tensor
    dt: torch.Tensor
    dr: torch.Tensor
    a: torch.Tensor
    d: torch.Tensor
    w: torch.Tensor

    def detach(self):
        return SolverState(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def clone(self):
        return SolverState(**{f.name: getattr(self, f.name).clone() for f in fields(self)})


def project_ball(x, radius):
    tiny = torch.finfo(x.real.dtype).tiny
    scale = (radius / x.norm(dim=-1, keepdim=True).clamp_min(tiny)).clamp(max=1)
    return x * scale


def unit_phase(a):
    # A zero complex value has no defined phase: select the feasible value +1.
    mag = a.abs()
    return torch.where(mag > 0, a / mag.clamp_min(torch.finfo(a.real.dtype).tiny), torch.ones_like(a))


def project_beamformers(a, d, w, power):
    a = unit_phase(a)
    norm = (a @ d).norm(dim=(-2, -1), keepdim=True)
    # Degenerate zero-power proposals use a deterministic feasible fallback.
    fallback_d = torch.zeros_like(d)
    fallback_d[:, 0, 0] = math.sqrt(power / a.shape[-2])
    d = torch.where(norm > 0, d * math.sqrt(power) / norm.clamp_min(torch.finfo(norm.dtype).tiny), fallback_d)
    wnorm = w.norm(dim=-1, keepdim=True)
    fallback_w = torch.zeros_like(w)
    fallback_w[..., 0] = 1
    w = torch.where(wnorm > 0, w / wnorm.clamp_min(torch.finfo(wnorm.dtype).tiny), fallback_w)
    return a, d, w


def pgda_step(batch, current, previous, cfg, mu, eta, eps=None):
    """One MinNet then one MaxNet; gradients within each stage are simultaneous."""
    eps = batch.eps if eps is None else eps
    g = gradients(batch, current, cfg)
    errors = {}
    for i, name in enumerate(NAMES[:3]):
        n = cfg.nr if name == 'dr' else cfg.nt
        x, old = getattr(current, name), getattr(previous, name)
        proposal = x - mu[i] * g[name] + eta[i] * (x - old)
        errors[name] = project_ball(proposal, eps[:, i, None, None] * math.sqrt(n))
    inner = replace(current, **errors)
    g = gradients(batch, inner, cfg)
    proposals = [getattr(current, name) + mu[i] * g[name]
                 + eta[i] * (getattr(current, name) - getattr(previous, name))
                 for i, name in enumerate(NAMES[3:], 3)]
    a, d, w = project_beamformers(*proposals, cfg.power)
    return replace(inner, a=a, d=d, w=w)


def constraint_errors(state, batch, cfg, eps=None):
    eps = batch.eps if eps is None else eps
    out = {"analog_modulus": (state.a.abs() - 1).abs().max().item(),
           "transmit_power": ((state.a @ state.d).abs().square().sum((-2, -1)) - cfg.power).abs().max().item(),
           "receive_norm": (state.w.norm(dim=-1) - 1).abs().max().item()}
    for i, name in enumerate(NAMES[:3]):
        n = cfg.nr if name == 'dr' else cfg.nt
        out[name + '_violation'] = (getattr(state, name).norm(dim=-1) - eps[:, i, None] * math.sqrt(n)).clamp_min(0).max().item()
    return out


class PGDA(nn.Module):
    def __init__(self, cfg, iterations=180, mu=None, eta=0.01):
        super().__init__()
        self.cfg, self.iterations = cfg, iterations
        self.register_buffer('mu', torch.tensor(mu if mu is not None else [0.0025] * 3 + [0.001] * 3))
        self.register_buffer('eta', torch.full((6,), float(eta)))

    def forward(self, batch, initial, eps=None, return_states=False):
        current = initial
        previous = initial  # first momentum is zero
        trace, states = [], []
        for _ in range(self.iterations):
            next_state = pgda_step(batch, current, previous, self.cfg, self.mu, self.eta, eps)
            previous, current = current, next_state
            trace.append(metrics(batch, current, self.cfg))
            if return_states:
                states.append(current)
        return current, trace, states


class UnrolledPGDA(nn.Module):
    def __init__(self, cfg, layers=15, learn_momentum=True, momentum=0.01,
                 parameterization='direct'):
        super().__init__()
        if layers < 1 or parameterization not in ('direct', 'positive', 'projected'):
            raise ValueError("Invalid layers or parameterization")
        self.cfg, self.layers, self.parameterization = cfg, layers, parameterization
        mu = torch.tensor([0.0025] * 3 + [0.001] * 3).repeat(layers, 1)
        eta = torch.full((layers, 6), float(momentum))
        # Direct unconstrained scalars match the paper. Positive is an explicit optional variant.
        if parameterization == 'positive':
            mu = torch.log(torch.expm1(mu))
            eta = torch.logit(eta.clamp(1e-7, 1 - 1e-7))
        self.raw_mu = nn.Parameter(mu)
        self.raw_eta = nn.Parameter(eta, requires_grad=learn_momentum)

    @property
    def mu(self):
        return torch.nn.functional.softplus(self.raw_mu) if self.parameterization == 'positive' else self.raw_mu

    @property
    def eta(self):
        return self.raw_eta.sigmoid() if self.parameterization == 'positive' else self.raw_eta

    @torch.no_grad()
    def project_hyperparameters(self):
        """Optional safeguard, not stated in the paper's learning algorithm."""
        if self.parameterization == 'projected':
            self.raw_mu.clamp_(min=0.)
            self.raw_eta.clamp_(min=0., max=0.999)

    def forward(self, batch, initial, eps=None, extra_layers=0, beta=1., phi=1., return_states=False):
        if extra_layers < 0:
            raise ValueError("extra_layers must be nonnegative")
        current, previous = initial, initial
        trace, states = [], []
        all_mu, all_eta = self.mu, self.eta
        for i in range(self.layers + extra_layers):
            if i < self.layers:
                mu, eta = all_mu[i], all_eta[i]
            else:
                depth = i - self.layers + 1
                mu, eta = all_mu[-1] * beta ** depth, all_eta[-1] * phi ** depth
            next_state = pgda_step(batch, current, previous, self.cfg, mu, eta, eps)
            previous, current = current, next_state
            trace.append(metrics(batch, current, self.cfg))
            if return_states:
                states.append(current)
        return current, trace, states


def unrolled_loss(trace):
    # Equation (22): average over samples AND every layer, without detached gradients.
    return -torch.stack([x['objective'] for x in trace]).mean()
