from dataclasses import asdict, replace
import torch
from l2o_isac import SystemConfig, generate_channels, initialize_state, EpsNet, UnrolledPGDA
from l2o_isac.data import ChannelBatch
from l2o_isac.inference import load_checkpoint
from l2o_isac.networks import eps_rmse
from l2o_isac.solver import constraint_errors


def test_equation18_averages_sample_rmse_after_square_root():
    prediction = torch.tensor([[0., 0., 0.], [3., 3., 3.]], dtype=torch.float64, requires_grad=True)
    truth = torch.zeros_like(prediction)
    loss = eps_rmse(prediction, truth)
    torch.testing.assert_close(loss, torch.tensor(1.5, dtype=torch.float64))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_per_channel_heads_are_order_invariant_at_inference():
    cfg = SystemConfig(nt=6, nr=4, users=3, targets=2, clutter=1, rf_chains=3)
    batch = generate_channels(cfg, 4)
    model = EpsNet(cfg).eval()
    u, p = torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0])
    permuted = replace(batch, h=batch.h[:, u], at=batch.at[:, p], ar=batch.ar[:, p], alpha=batch.alpha[:, p])
    torch.testing.assert_close(model(batch), model(permuted), rtol=1e-6, atol=1e-7)


def test_frozen_weights_run_across_counts_with_feasible_beams():
    torch.set_num_threads(1)
    cfg = SystemConfig(nt=8, nr=6, users=2, targets=2, clutter=1, rf_chains=4)
    epsnet, solver = EpsNet(cfg).eval(), UnrolledPGDA(cfg, layers=3).eval()
    for users, targets in ((1, 1), (4, 3), (6, 6)):
        other = replace(cfg, users=users, targets=targets)
        batch = generate_channels(other, 3)
        other_eps = EpsNet(other).eval()
        other_eps.load_state_dict(epsnet.state_dict(), strict=True)
        torch.testing.assert_close(epsnet(batch), other_eps(batch))
        other_solver = UnrolledPGDA(other, layers=3).eval()
        other_solver.load_state_dict(solver.state_dict(), strict=True)
        eps = epsnet(batch)
        state, _, _ = other_solver(batch, initialize_state(batch, other), eps=eps)
        assert max(constraint_errors(state, batch, other, eps).values()) < 1e-5


def test_historical_checkpoint_loads_original_concatenated_layout(tmp_path):
    cfg = SystemConfig(nt=6, nr=4, users=2, targets=1, clutter=1, rf_chains=3)
    batch = generate_channels(cfg, 3)
    epsnet = EpsNet(cfg, layout='concatenated').eval()
    model = UnrolledPGDA(cfg, layers=3)
    checkpoint = tmp_path / 'legacy.pt'
    torch.save(dict(config=asdict(cfg), run=dict(layers=3, momentum='learned', parameterization='direct', bounds='learned'),
                    epsnet=epsnet.state_dict(), unrolled=model.state_dict(), recurrence=dict(beta=.9, phi=.9)), checkpoint)
    _, restored, _, _ = load_checkpoint(checkpoint)
    assert restored.layout == 'concatenated'
    torch.testing.assert_close(restored(batch), epsnet(batch), rtol=0, atol=0)


def test_new_checkpoint_retains_per_channel_layout(tmp_path):
    cfg = SystemConfig(nt=6, nr=4, users=2, targets=1, clutter=1, rf_chains=3)
    epsnet, model = EpsNet(cfg).eval(), UnrolledPGDA(cfg, layers=3)
    path = tmp_path / 'new.pt'
    torch.save(dict(config=asdict(cfg), run=dict(layers=3, momentum='learned', parameterization='direct', bounds='learned', eps_layout='per-channel'),
                    epsnet=epsnet.state_dict(), unrolled=model.state_dict(), recurrence=dict(beta=.9, phi=.9)), path)
    _, restored, _, _ = load_checkpoint(path)
    assert restored.layout == 'per-channel'
    batch = generate_channels(replace(cfg, users=5, targets=3), 2)
    torch.testing.assert_close(restored(batch), epsnet(batch), rtol=0, atol=0)


def test_paper_grid_labels_do_not_change_nominal_channels_or_leak_into_encoder():
    cfg = SystemConfig(nt=6, nr=4, users=2, targets=1, clutter=1, rf_chains=3)
    fixed = generate_channels(cfg, 200, seed=42)
    varied = generate_channels(cfg, 200, seed=42, eps_mode='paper-grid')
    for key in ('h', 'at', 'ar', 'alpha'):
        torch.testing.assert_close(getattr(fixed, key), getattr(varied, key), rtol=0, atol=0)
    assert torch.unique(varied.eps[:, 0]).numel() == 6
    torch.testing.assert_close(varied.eps[:, 0], varied.eps[:, 1])
    encoder = EpsNet(cfg).eval()
    torch.testing.assert_close(encoder(fixed), encoder(varied), rtol=0, atol=0)
