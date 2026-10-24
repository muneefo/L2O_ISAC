import torch
from torch import nn


def real_features(x):
    return torch.cat((x.real.flatten(1), x.imag.flatten(1)), dim=-1)


class BoundHead(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        layers = [nn.BatchNorm1d(inputs)]
        for width in hidden:
            layers.extend([nn.Linear(inputs, width), nn.Softplus()])
            inputs = width
        layers.extend([nn.Linear(inputs, 1), nn.Softplus()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EpsNet(nn.Module):
    """Three BN/FC/SoftPlus branches, Section IV-A and Section V-A.

    Default per-channel inputs follow Fig. 1(a), rather than concatenating all users.
    Heads are shared across users/paths; their positive outputs are averaged to the
    three common radii in (9). Sharing/mean aggregation are explicit reconstruction
    choices: the paper does not specify the aggregation across channel instances.
    'concatenated' preserves the original reconstruction and historical checkpoints.
    """
    def __init__(self, cfg, layout='per-channel'):
        super().__init__()
        if layout not in ('per-channel', 'concatenated'):
            raise ValueError('Unknown EpsNet layout')
        self.layout = layout
        self.h_head = BoundHead(2 * cfg.nt * (cfg.users if layout == 'concatenated' else 1), [128, 64])
        sensing_inputs = 2 * cfg.nr * cfg.nt * (cfg.paths if layout == 'concatenated' else 1)
        self.t_head = BoundHead(sensing_inputs, [128, 64, 32])
        self.r_head = BoundHead(sensing_inputs, [128, 64, 32])

    def forward(self, batch):
        if self.layout == 'per-channel':
            b, users, nt = batch.h.shape
            radar = batch.radar
            paths, nr = radar.shape[1:3]
            h = real_features(batch.h.reshape(b * users, nt))
            radar_features = real_features(radar.reshape(b * paths, nr, nt))
            return torch.cat([
                self.h_head(h).reshape(b, users, 1).mean(1),
                self.t_head(radar_features).reshape(b, paths, 1).mean(1),
                self.r_head(radar_features).reshape(b, paths, 1).mean(1)], -1)
        h = real_features(batch.h)
        radar = real_features(batch.radar)
        return torch.cat([self.h_head(h), self.t_head(radar), self.r_head(radar)], -1)


def eps_rmse(predicted, true):
    """Equation (18) per sample, then batch mean (not sqrt of global batch MSE)."""
    return (predicted - true).square().mean(-1).clamp_min(torch.finfo(predicted.dtype).eps ** 2).sqrt().mean()
