from dataclasses import replace
import math
import torch
import pytest
from l2o_isac import SystemConfig, generate_channels, initialize_state, metrics, gradients
from l2o_isac.solver import NAMES, project_beamformers, project_ball, PGDA, UnrolledPGDA, unrolled_loss, constraint_errors


@pytest.fixture
def problem():
    cfg = SystemConfig(nt=5, nr=4, users=2, targets=2, clutter=1, rf_chains=3)
    batch = generate_channels(cfg, 3, seed=31, dtype=torch.complex128)
    return cfg, batch, initialize_state(batch, cfg, seed=19)


def reference_metrics(batch, state, cfg):
    """Deliberately slow, independent literal sums of equations (3) and (6)."""
    results = []
    for b in range(len(batch)):
        p = state.a[b] @ state.d[b]
        h = batch.h[b] + state.dh[b]
        at = batch.at[b] + state.dt[b]
        ar = batch.ar[b] + state.dr[b]
        hc = batch.alpha[b, :, None, None] * ar[:, :, None] * at[:, None, :]
        comm = 0
        for m in range(cfg.users):
            signal = (h[m].conj() @ p[:, m]).abs().square()
            interference = sum((h[m].conj() @ p[:, k]).abs().square()
                               for k in range(cfg.streams) if k != m)
            comm += torch.log1p(signal / (interference + cfg.noise_comm))
        sense = 0
        for t in range(cfg.targets):
            signal = (state.w[b, t].conj() @ hc[t] @ p).abs().square().sum()
            interference = sum((state.w[b, t].conj() @ hc[l] @ p).abs().square().sum()
                               for l in range(cfg.paths) if l != t)
            noise = cfg.noise_sense * state.w[b, t].abs().square().sum()
            sense += torch.log1p(signal / (interference + noise))
        results.append(torch.stack([comm, sense]))
    return torch.stack(results)


def test_metrics_against_literal_equations(problem):
    cfg, batch, state = problem
    ref = reference_metrics(batch, state, cfg)
    actual = metrics(batch, state, cfg)
    torch.testing.assert_close(actual['comm'], ref[:, 0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual['sense'], ref[:, 1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('name', NAMES)
def test_analytic_gradient_against_independent_autograd(problem, name):
    cfg, batch, state = problem
    x = getattr(state, name).clone().requires_grad_()
    ref = reference_metrics(batch, replace(state, **{name: x}), cfg)
    value = ref[:, 0].sum() if name == 'dh' else ref[:, 1].sum() if name in ('dt', 'dr') else (cfg.rho * ref[:, 0] + (1 - cfg.rho) * ref[:, 1]).sum()
    expected, = torch.autograd.grad(value, x)
    actual = gradients(batch, state, cfg)[name]
    torch.testing.assert_close(actual, expected / 2, rtol=2e-10, atol=1e-11)


@pytest.mark.parametrize('name', NAMES)
def test_gradient_directional_finite_difference(problem, name):
    cfg, batch, state = problem
    x = getattr(state, name)
    torch.manual_seed(47)
    direction = torch.randn_like(x)
    direction /= direction.norm()
    def f(y):
        m = metrics(batch, replace(state, **{name: y}), cfg)
        return m['comm' if name == 'dh' else 'sense' if name in ('dt', 'dr') else 'objective'].sum()
    step = 1e-6
    numerical = (f(x + step * direction) - f(x - step * direction)) / (2 * step)
    analytic = 2 * (gradients(batch, state, cfg)[name].conj() * direction).sum().real
    torch.testing.assert_close(analytic, numerical, rtol=1e-6, atol=1e-8)


def test_all_iterate_constraints(problem):
    cfg, batch, state = problem
    model = PGDA(cfg, iterations=12).double()
    final, trace, states = model(batch, state, return_states=True)
    for s in states:
        assert max(constraint_errors(s, batch, cfg).values()) < 1e-12
    assert torch.isfinite(torch.stack([t['objective'] for t in trace])).all()


def test_zero_projections_are_feasible(problem):
    cfg, batch, state = problem
    a, d, w = project_beamformers(torch.zeros_like(state.a), torch.zeros_like(state.d), torch.zeros_like(state.w), cfg.power)
    feasible = replace(state, a=a, d=d, w=w)
    assert max(constraint_errors(feasible, batch, cfg).values()) < 1e-12
    assert project_ball(torch.zeros_like(state.dh), 0).abs().max() == 0


def test_unrolled_equals_benchmark_at_initialization(problem):
    cfg, batch, state = problem
    bench = PGDA(cfg, iterations=4).double()
    net = UnrolledPGDA(cfg, layers=4).double()
    bs, _, _ = bench(batch, state)
    ns, _, _ = net(batch, state)
    for name in NAMES:
        torch.testing.assert_close(getattr(bs, name), getattr(ns, name))


def test_zero_uncertainty_and_recurrence(problem):
    cfg, batch, state = problem
    batch = replace(batch, eps=torch.zeros_like(batch.eps))
    state = initialize_state(batch, cfg)
    net = UnrolledPGDA(cfg, layers=3).double()
    final, trace, _ = net(batch, state, extra_layers=4, beta=1, phi=1)
    expected, _, _ = PGDA(cfg, iterations=7).double()(batch, state)
    assert len(trace) == 7
    assert final.dh.abs().max() == final.dt.abs().max() == final.dr.abs().max() == 0
    for name in NAMES:
        torch.testing.assert_close(getattr(final, name), getattr(expected, name))


def test_learning_derivative_through_gradients_and_projections(problem):
    cfg, batch, state = problem
    model = UnrolledPGDA(cfg, layers=3).double()
    loss = unrolled_loss(model(batch, state)[1])
    analytic, = torch.autograd.grad(loss, model.raw_mu)
    index = (1, 4)
    original = model.raw_mu[index].item()
    delta = 1e-6
    values = []
    with torch.no_grad():
        for sign in [1, -1]:
            model.raw_mu[index] = original + sign * delta
            values.append(unrolled_loss(model(batch, state)[1]).item())
        model.raw_mu[index] = original
    finite = (values[0] - values[1]) / (2 * delta)
    assert analytic[index].item() == pytest.approx(finite, rel=2e-5, abs=1e-7)
    assert torch.isfinite(analytic).all()


def test_tiny_unprojected_steps_have_correct_sign(problem):
    cfg, batch, state = problem
    g = gradients(batch, state, cfg)
    inner = replace(state, **{name: getattr(state, name) - 1e-7 * g[name] for name in NAMES[:3]})
    outer = replace(state, **{name: getattr(state, name) + 1e-7 * g[name] for name in NAMES[3:]})
    before = metrics(batch, state, cfg)
    assert (metrics(batch, inner, cfg)['comm'] <= before['comm'] + 1e-12).all()
    assert (metrics(batch, inner, cfg)['sense'] <= before['sense'] + 1e-12).all()
    assert (metrics(batch, outer, cfg)['objective'] >= before['objective'] - 1e-12).all()


@pytest.mark.parametrize('rho', [0., 1.])
def test_objective_endpoints(problem, rho):
    cfg, batch, state = problem
    m = metrics(batch, state, replace(cfg, rho=rho))
    torch.testing.assert_close(m['objective'], m['comm' if rho == 1 else 'sense'])


def test_seeds_and_slice(problem):
    cfg, batch, state = problem
    other = generate_channels(cfg, len(batch), seed=31, dtype=torch.complex128)
    torch.testing.assert_close(batch.h, other.h)
    assert len(batch[0]) == 1
    assert len(batch[torch.tensor([0, 2])]) == 2


def test_no_clutter_unequal_dimensions():
    cfg = SystemConfig(nt=6, nr=3, users=3, targets=1, clutter=0, rf_chains=2)
    batch = generate_channels(cfg, 2)
    state = initialize_state(batch, cfg)
    result = metrics(batch, state, cfg)
    literal = reference_metrics(batch, state, cfg)
    torch.testing.assert_close(result['sense'], literal[:, 1])
    assert torch.isfinite(gradients(batch, state, cfg)['w']).all()
