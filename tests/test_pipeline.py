from dataclasses import replace
import torch
from l2o_isac import SystemConfig, generate_channels, initialize_state, EpsNet, UnrolledPGDA
from l2o_isac.physics import metrics
from l2o_isac.solver import constraint_errors, unrolled_loss
from l2o_isac.training import train_epsnet, train_unrolled, recurrent_grid_search, average_network_loss
from l2o_isac.evaluation import adversarial_evaluate


def test_independent_adversary_is_feasible_and_worsens_nominal():
    torch.set_num_threads(1)
    cfg = SystemConfig(nt=6, nr=5, users=2, targets=2, clutter=1, rf_chains=3)
    batch = generate_channels(cfg, 8, seed=9)
    state = initialize_state(batch, cfg, seed=2, zero_errors=True)
    attacked, errors = adversarial_evaluate(batch, state, cfg, steps=20, restarts=3, seed=8)
    nominal = metrics(batch, state, cfg)
    assert (attacked['comm'] <= nominal['comm'] + 1e-6).all()
    assert (attacked['sense'] <= nominal['sense'] + 1e-6).all()
    assert (attacked['objective'] < nominal['objective']).all()
    assert max(constraint_errors(errors, batch, cfg).values()) < 2e-6
    again, _ = adversarial_evaluate(batch, state, cfg, steps=20, restarts=3, seed=8)
    torch.testing.assert_close(attacked['objective'], again['objective'], rtol=0, atol=0)


def test_training_and_grid_search_without_test_leakage():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = SystemConfig(nt=6, nr=5, users=2, targets=1, clutter=1, rf_chains=3)
    train = generate_channels(cfg, 24, seed=40)
    valid = generate_channels(cfg, 8, seed=41)
    epsnet = EpsNet(cfg)
    history = train_epsnet(epsnet, train, valid, epochs=2, batch_size=8)
    assert history[-1]['validation_rmse'] < history[0]['validation_rmse']
    assert (epsnet(valid) > 0).all()
    model = UnrolledPGDA(cfg, layers=3)
    trained = train_unrolled(model, train, valid, epsnet, epochs=2, batch_size=8)
    assert trained['best_validation_loss'] <= trained['initial_validation_loss']
    original = {k: v.clone() for k, v in model.state_dict().items()}
    result = recurrent_grid_search(model, valid, epsnet, grids=2, extra_layers=2, batch_size=8)
    assert len(result['scores']) == 4
    assert result['best']['loss'] == min(r['loss'] for r in result['scores'])
    for candidate in result['scores']:
        reference = average_network_loss(model, valid, epsnet, batch_size=8, seed=0,
                                          extra_layers=2, beta=candidate['beta'], phi=candidate['phi'])
        assert abs(reference - candidate['loss']) < 1e-7
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_second_order_training_gradients_all_finite():
    torch.manual_seed(0)
    cfg = SystemConfig(nt=8, nr=6, users=3, targets=2, clutter=1, rf_chains=4)
    batch = generate_channels(cfg, 4)
    state = initialize_state(batch, cfg)
    model = UnrolledPGDA(cfg, layers=5)
    loss = unrolled_loss(model(batch, state)[1])
    loss.backward()
    assert torch.isfinite(model.raw_mu.grad).all()
    assert torch.isfinite(model.raw_eta.grad).all()
    # First-layer momentum is identically zero because previous == current.
    assert model.raw_eta.grad[0].abs().max() == 0
    assert model.raw_eta.grad[1:].abs().sum() > 0


def test_serialized_model_reproduces_inference(tmp_path):
    cfg = SystemConfig(nt=6, nr=4, users=2, targets=1, clutter=1, rf_chains=3)
    batch = generate_channels(cfg, 4)
    state = initialize_state(batch, cfg)
    model = UnrolledPGDA(cfg, layers=4)
    before = model(batch, state)[1][-1]['objective']
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    restored = UnrolledPGDA(cfg, layers=4)
    restored.load_state_dict(torch.load(path, weights_only=True))
    after = restored(batch, state)[1][-1]['objective']
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_optional_projected_training_preserves_step_sign():
    cfg = SystemConfig(nt=6, nr=4, users=2, targets=1, clutter=1, rf_chains=3)
    model = UnrolledPGDA(cfg, layers=3, parameterization='projected')
    with torch.no_grad():
        model.raw_mu.fill_(-1)
        model.raw_eta.fill_(2)
    model.project_hyperparameters()
    assert model.mu.min() == 0
    assert model.eta.max() <= 0.999001


def test_attack_against_closed_form_single_stream_worst_case():
    """A solvable special case validates the attack against an actual analytic minimum."""
    import math
    cfg = SystemConfig(nt=4, nr=3, users=1, targets=1, clutter=0, rf_chains=1)
    batch = generate_channels(cfg, 5, seed=15, dtype=torch.complex128)
    state = initialize_state(batch, cfg, seed=8, zero_errors=True)
    # Only one nonzero stream, so communication has no multi-stream interference.
    digital = torch.zeros_like(state.d)
    digital[:, 0, 0] = math.sqrt(cfg.power / cfg.nt)
    state = replace(state, d=digital)
    p = (state.a @ state.d)[..., 0]
    c_amplitude = (batch.h[:, 0].conj() * p).sum(-1).abs()
    c_min = (c_amplitude - math.sqrt(cfg.nt) * batch.eps[:, 0] * p.norm(dim=-1)).clamp_min(0)
    expected_comm = torch.log1p(c_min.square() / cfg.noise_comm)
    tx = (batch.at[:, 0] * p).sum(-1).abs()
    rx = (state.w[:, 0].conj() * batch.ar[:, 0]).sum(-1).abs()
    tx_min = (tx - math.sqrt(cfg.nt) * batch.eps[:, 1] * p.norm(dim=-1)).clamp_min(0)
    rx_min = (rx - math.sqrt(cfg.nr) * batch.eps[:, 2]).clamp_min(0)
    expected_sense = torch.log1p(batch.alpha[:, 0].abs().square() * tx_min.square() * rx_min.square() / cfg.noise_sense)
    attacked, _ = adversarial_evaluate(batch, state, cfg, steps=300, restarts=2, normalize_gradients=True)
    torch.testing.assert_close(attacked['comm'], expected_comm, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(attacked['sense'], expected_sense, rtol=1e-6, atol=1e-8)
