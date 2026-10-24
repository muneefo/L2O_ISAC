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

    Communication branch sees h; both sensing branches see flattened round-trip H.
    The paper does not specify cross-branch input sharing; this choice is explicit.
    """
    def __init__(self, cfg):
        super().__init__()
        self.h_head = BoundHead(2 * cfg.users * cfg.nt, [128, 64])
        sensing_inputs = 2 * cfg.paths * cfg.nr * cfg.nt
        self.t_head = BoundHead(sensing_inputs, [128, 64, 32])
        self.r_head = BoundHead(sensing_inputs, [128, 64, 32])

    def forward(self, batch):
        h = real_features(batch.h)
        radar = real_features(batch.radar)
        return torch.cat([self.h_head(h), self.t_head(radar), self.r_head(radar)], -1)


def eps_rmse(predicted, true):
    return (predicted - true).square().mean().clamp_min(torch.finfo(predicted.dtype).eps ** 2).sqrt()
