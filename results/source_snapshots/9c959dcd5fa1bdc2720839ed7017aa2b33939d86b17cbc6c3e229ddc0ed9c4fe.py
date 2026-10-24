"""Load saved networks and design beams for a ChannelBatch."""
import torch
from .config import SystemConfig
from .networks import EpsNet
from .solver import UnrolledPGDA
from .data import initialize_state


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    cfg = SystemConfig(**saved['config'])
    run = saved['run']
    # Missing metadata identifies checkpoints written before per-channel support.
    epsnet = EpsNet(cfg, layout=run.get('eps_layout', 'concatenated')).to(device)
    epsnet.load_state_dict(saved['epsnet'])
    epsnet.eval()
    model = UnrolledPGDA(cfg, layers=run['layers'],
                         learn_momentum=run['momentum'] == 'learned',
                         momentum=0 if run['momentum'] == 'none' else .01,
                         parameterization=run['parameterization']).to(device)
    model.load_state_dict(saved['unrolled'])
    model.eval()
    return cfg, epsnet, model, saved


@torch.no_grad()
def design_beams(path, batch, seed=0, extra_layers=0, device='cpu'):
    cfg, epsnet, model, saved = load_checkpoint(path, device)
    batch = batch.to(device)
    initial = initialize_state(batch, cfg, seed=seed)
    eps = epsnet(batch) if saved['run']['bounds'] == 'learned' else batch.eps
    state, trace, _ = model(batch, initial, eps=eps, extra_layers=extra_layers,
                            beta=saved['recurrence']['beta'], phi=saved['recurrence']['phi'])
    return state, trace, eps
