"""Equations (3)-(7), (32)-(70); gradients are Wirtinger d f / d z*.

PyTorch's complex autograd gradient for a real scalar is twice this convention.
Factorized calculations avoid forming dense radar covariance matrices.
"""
import torch


def components(batch, state, cfg):
    h, at, ar = batch.h + state.dh, batch.at + state.dt, batch.ar + state.dr
    p = state.a @ state.d
    c = h.conj() @ p
    cp = c.abs().square()
    cmask = torch.ones(cfg.users, cfg.streams, dtype=cp.dtype, device=cp.device)
    cmask[torch.arange(cfg.users), torch.arange(cfg.users)] = 0
    cden = (cp * cmask).sum(-1) + cfg.noise_comm
    csignal = cp[:, torch.arange(cfg.users), torch.arange(cfg.users)]
    ctotal = cden + csignal
    s = at @ p
    r = state.w.conj() @ ar.transpose(-1, -2)
    alpha2 = batch.alpha.abs().square()
    s2 = s.abs().square().sum(-1)
    energy = r.abs().square() * (alpha2 * s2)[:, None, :]
    smask = torch.ones(cfg.targets, cfg.paths, dtype=cp.dtype, device=cp.device)
    smask[torch.arange(cfg.targets), torch.arange(cfg.targets)] = 0
    sden = (energy * smask).sum(-1) + cfg.noise_sense * state.w.abs().square().sum(-1)
    ssignal = energy[:, torch.arange(cfg.targets), torch.arange(cfg.targets)]
    stotal = sden + ssignal
    return dict(h=h, at=at, ar=ar, p=p, c=c, cmask=cmask, cden=cden, ctotal=ctotal,
                csignal=csignal, s=s, r=r, alpha2=alpha2, s2=s2, smask=smask,
                sden=sden, stotal=stotal, ssignal=ssignal)


def metrics(batch, state, cfg):
    z = components(batch, state, cfg)
    comm = torch.log1p(z['csignal'] / z['cden']).sum(-1)
    sense = torch.log1p(z['ssignal'] / z['sden']).sum(-1)
    return {"comm": comm, "sense": sense, "objective": cfg.rho * comm + (1 - cfg.rho) * sense}


def gradients(batch, state, cfg):
    """All six exact partial gradients, with unweighted inner minimization as in (11)."""
    z = components(batch, state, cfg)
    cc = 1 / z['ctotal'][..., None] - z['cmask'] / z['cden'][..., None]
    sc = 1 / z['stotal'][..., None] - z['smask'] / z['sden'][..., None]
    gc_p = z['h'].transpose(-1, -2) @ (cc * z['c'])
    gh = (cc * z['c'].conj()) @ z['p'].transpose(-1, -2)
    q = z['alpha2'] * (sc * z['r'].abs().square()).sum(-2)
    gs_p = z['at'].conj().transpose(-1, -2) @ (q[..., None] * z['s'])
    gt = q[..., None] * (z['s'] @ z['p'].conj().transpose(-1, -2))
    v = sc * (z['alpha2'] * z['s2'])[:, None, :]
    gr = (v * z['r']).transpose(-1, -2) @ state.w
    gw = (v * z['r'].conj()) @ z['ar']
    gw += cfg.noise_sense * (1 / z['stotal'] - 1 / z['sden'])[..., None] * state.w
    gp = cfg.rho * gc_p + (1 - cfg.rho) * gs_p
    return dict(dh=gh, dt=gt, dr=gr,
                a=gp @ state.d.conj().transpose(-1, -2),
                d=state.a.conj().transpose(-1, -2) @ gp,
                w=(1 - cfg.rho) * gw)
