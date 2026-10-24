from dataclasses import dataclass, fields
import math
import torch
from .config import SystemConfig


@dataclass
class ChannelBatch:
    h: torch.Tensor       # [batch, users, nt], presumed channel vectors (no conjugation)
    at: torch.Tensor      # [batch, target+clutter, nt], transmit steering
    ar: torch.Tensor      # [batch, target+clutter, nr], receive steering
    alpha: torch.Tensor   # [batch, target+clutter]
    eps: torch.Tensor     # [batch, 3], dimension-normalized radii

    def __len__(self):
        return self.h.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            index = slice(index, index + 1)
        return ChannelBatch(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def to(self, device):
        return ChannelBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    @property
    def radar(self):
        # Equation (5): transpose, deliberately NOT Hermitian transpose.
        return self.alpha[..., None, None] * self.ar[..., :, None] * self.at[..., None, :]


def complex_normal(shape, generator, dtype=torch.complex64, device="cpu"):
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    x = torch.randn(*shape, 2, generator=generator, dtype=real)
    return torch.view_as_complex(x.contiguous()).to(device) / math.sqrt(2)


def generate_channels(cfg: SystemConfig, count, seed=0, dtype=torch.complex64,
                      device="cpu", variable_eps=False):
    """Seeded independent samples; unspecified angular distribution is documented."""
    g = torch.Generator().manual_seed(seed)
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    h = complex_normal((count, cfg.users, cfg.nt), g, dtype)
    # Uniform sorted points on compressed interval give separated angles without rejection.
    width = cfg.angle_max_deg - cfg.angle_min_deg - (cfg.paths - 1) * cfg.min_separation_deg
    angles = torch.rand(count, cfg.paths, generator=g, dtype=real).sort(-1).values * width
    angles += cfg.angle_min_deg + torch.arange(cfg.paths) * cfg.min_separation_deg
    # Randomize path roles: sorted angles must not force targets to one side of the array.
    permutations = torch.rand(count, cfg.paths, generator=g).argsort(-1)
    angles = angles.gather(1, permutations) * math.pi / 180
    def steering(n):
        return torch.exp(-1j * math.pi * angles[..., None].sin() * torch.arange(n, dtype=real))
    alpha = complex_normal((count, cfg.paths), g, dtype)
    variance = torch.tensor([10 ** (cfg.target_variance_db / 10)] * cfg.targets
                            + [10 ** (cfg.clutter_variance_db / 10)] * cfg.clutter, dtype=real)
    alpha *= variance.sqrt()
    eps = torch.tensor([cfg.eps_h, cfg.eps_t, cfg.eps_r], dtype=real).expand(count, -1).clone()
    if variable_eps:
        # Experimental only: independent random bounds are NOT identifiable from nominal CSI.
        eps = 0.001 + torch.rand(count, 3, generator=g, dtype=real) * (eps - 0.001).clamp_min(0)
    return ChannelBatch(h, steering(cfg.nt), steering(cfg.nr), alpha, eps).to(device)


def random_ball(shape, radius, generator, dtype, device):
    x = complex_normal(shape, generator, dtype, device)
    n = shape[-1]
    real = x.real.dtype
    radial = torch.rand(*shape[:-1], 1, generator=generator, dtype=real).to(device) ** (1 / (2 * n))
    return x / x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(real).tiny) * radius * radial


def initialize_state(batch: ChannelBatch, cfg: SystemConfig, seed=0, zero_errors=False):
    from .solver import SolverState, project_ball, project_beamformers
    g = torch.Generator().manual_seed(seed)
    b, dtype, device = len(batch), batch.h.dtype, batch.h.device
    a = complex_normal((b, cfg.nt, cfg.rf_chains), g, dtype, device)
    d = complex_normal((b, cfg.rf_chains, cfg.streams), g, dtype, device)
    w = complex_normal((b, cfg.targets, cfg.nr), g, dtype, device)
    a, d, w = project_beamformers(a, d, w, cfg.power)
    errors = []
    for i, (paths, n) in enumerate([(cfg.users, cfg.nt), (cfg.paths, cfg.nt), (cfg.paths, cfg.nr)]):
        radius = batch.eps[:, i, None, None] * math.sqrt(n)
        error = random_ball((b, paths, n), radius, g, dtype, device)
        errors.append(torch.zeros_like(error) if zero_errors else project_ball(error, radius))
    return SolverState(*errors, a, d, w)
