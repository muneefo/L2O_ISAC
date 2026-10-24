"""Measure this reconstruction separately from the paper's published reference data.

Includes Table-I epsilon settings, Fig.-9 count settings with explicitly frozen
weights, and warm batch-one latency. The authors do not specify the same-weight
transfer protocol used here; this script does not label it as their experiment.
"""
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .data import generate_channels, initialize_state, PAPER_EPSILON_GRID
from .inference import load_checkpoint
from .networks import EpsNet
from .physics import metrics
from .solver import PGDA, UnrolledPGDA, constraint_errors
from .training import bound_metrics
from .evaluation import adversarial_evaluate, statistics


def timed(fn, repeats, warmups):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return dict(samples_ms=samples, repeats=repeats, warmups=warmups,
                median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))


@torch.no_grad()
def check_checkpoint(checkpoint, samples=128, repeats=30, warmups=5, seed=39001):
    cfg, epsnet, model, saved = load_checkpoint(checkpoint)
    use_epsnet = saved['run']['bounds'] == 'learned'
    bound_provider = epsnet if use_epsnet else lambda batch: batch.eps
    # This is an evaluation sweep of one fixed model; no refitting or label input.
    eps_sweep = []
    for epsilon in PAPER_EPSILON_GRID:
        scenario = replace(cfg, eps_h=epsilon, eps_t=epsilon, eps_r=epsilon)
        batch = generate_channels(scenario, samples, seed=seed)
        eps_sweep.append(dict(epsilon=epsilon, **bound_metrics(epsnet, batch)))
    transfer = []
    for users, targets in ((2, 2), (4, 4), (6, 6)):
        scenario = replace(cfg, users=users, targets=targets)
        if epsnet.layout == 'concatenated' and (users != cfg.users or targets != cfg.targets):
            transfer.append(dict(users=users, targets=targets, status='unsupported by legacy concatenated EpsNet'))
            continue
        batch = generate_channels(scenario, samples, seed=seed + 1)
        initial = initialize_state(batch, scenario, seed=seed + 2)
        candidate = UnrolledPGDA(scenario, layers=model.layers,
                                 parameterization=model.parameterization,
                                 learn_momentum=saved['run']['momentum'] == 'learned',
                                 momentum=0 if saved['run']['momentum'] == 'none' else .01)
        candidate.load_state_dict(model.state_dict())
        candidate.eval()
        predicted = bound_provider(batch)
        design, trace, _ = candidate(batch, initial, eps=predicted)
        bench, btrace, _ = PGDA(scenario, iterations=180)(batch, initial)
        attacked, _ = adversarial_evaluate(batch, design, scenario, steps=80, restarts=4, seed=seed + 3)
        b_attacked, _ = adversarial_evaluate(batch, bench, scenario, steps=80, restarts=4, seed=seed + 3)
        residuals = constraint_errors(design, batch, scenario, predicted)
        assert max(residuals.values()) < 1e-5
        assert torch.isfinite(attacked['objective']).all()
        transfer.append(dict(users=users, targets=targets, status='measured with frozen weights',
                             internal_objective=statistics(trace[-1]['objective']),
                             benchmark_internal_objective=statistics(btrace[-1]['objective']),
                             attacked_objective=statistics(attacked['objective']),
                             benchmark_attacked_objective=statistics(b_attacked['objective']),
                             paired_attacked_difference=statistics(attacked['objective'] - b_attacked['objective']),
                             constraints=residuals))
    one = generate_channels(cfg, 1, seed=seed + 4)
    initial = initialize_state(one, cfg, seed=seed + 5)
    eps = bound_provider(one)
    latency = {}
    latency['EpsNet'] = timed(lambda: epsnet(one), repeats, warmups)
    for name, solver, kwargs in [
        ('U-PGDA', model, {}),
        ('RU-PGDA+10', model, dict(extra_layers=10, beta=saved['recurrence']['beta'], phi=saved['recurrence']['phi'])),
        ('B-PGDA', PGDA(cfg, iterations=180), {})]:
        is_benchmark = name == 'B-PGDA'
        fixed_eps = one.eps if is_benchmark else eps
        solver_samples = min(repeats, 10) if is_benchmark else repeats
        latency[name + '/solver_only'] = timed(
            lambda: solver(one, initial, eps=fixed_eps, **kwargs), solver_samples, warmups)
        def decision():
            start = initialize_state(one, cfg, seed=seed + 5)
            current_eps = one.eps if is_benchmark else bound_provider(one)
            return solver(one, start, eps=current_eps, **kwargs)
        latency[name + '/including_initialization_and_bounds'] = timed(decision, solver_samples, warmups)
    return dict(checkpoint=str(checkpoint), config=asdict(cfg), eps_layout=epsnet.layout,
                uncertainty_protocol='One frozen EpsNet evaluated on Table I epsilon values; author training-label protocol unspecified.',
                epsilon_sweep=eps_sweep,
                transfer_protocol='Same saved weights, no retraining, at Fig. 9 user/target counts. This is an added transfer audit, not a claimed reproduction of their unspecified sharing protocol.',
                frozen_weight_count_transfer=transfer, latency=latency,
                latency_protocol=dict(batch_size=1, threads=torch.get_num_threads(), device='CPU',
                                      torch=str(torch.__version__), python=platform.python_version(),
                                      includes='fixed-depth inference and diagnostic layer metrics',
                                      excludes='CSI acquisition, external transfers, RF programming',
                                      interpretation='Measured inference latency; not a certified time to convergence'),
                seed=seed, samples_per_scenario=samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--seed', type=int, default=39001)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error('Need positive samples/repeats and nonnegative warmups')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    result = check_checkpoint(args.checkpoint, args.samples, args.repeats, args.warmups, args.seed)
    reference_file = Path(__file__).resolve().parents[1] / 'configs/paper_reference.json'
    result['published_reference'] = json.loads(reference_file.read_text()) if reference_file.exists() else None
    (args.output / 'measured_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Measured reconstruction checks', '',
             'Published reference values are separately labeled in measured_checks.json and configs/paper_reference.json. '
             'They are not outputs of the reconstructed solver.', '',
             f'EpsNet layout: `{result["eps_layout"]}`.', '',
             '## Frozen-weight user/target-count audit', '', result['transfer_protocol'], '',
             '| Users | Targets | Attacked objective | Benchmark |', '|---:|---:|---:|---:|']
    for r in result['frozen_weight_count_transfer']:
        if 'attacked_objective' in r:
            lines.append(f"| {r['users']} | {r['targets']} | {r['attacked_objective']['mean']:.5f} | {r['benchmark_attacked_objective']['mean']:.5f} |")
    lines.extend(['', '## Batch-one latency', '',
                  'Warm CPU timings, including layer diagnostics. Initialization/bounds are included only where named. '
                  'Acquisition, external transfer, and RF programming are excluded. These are fixed-depth inference times.', '',
                  '| Scope | Median ms | p95 ms |', '|---|---:|---:|'])
    for name, r in result['latency'].items():
        lines.append(f"| {name} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} |")
    lines.extend(['', '## Table-I settings with one frozen EpsNet', '', result['uncertainty_protocol'], '',
                  '| True epsilon | Mean predicted h/t/r | Mean per-sample RMSE |', '|---:|---|---:|'])
    for r in result['epsilon_sweep']:
        lines.append(f"| {r['epsilon']} | {', '.join(f'{v:.5f}' for v in r['mean_predicted'])} | {r['rmse']:.6f} |")
    lines.extend(['', 'The nominal-channel generator is independent of the radius label in this reconstruction. '
                  'A fixed model cannot infer an independently changed radius from identical nominal inputs. '
                  'This limitation does not establish how the authors generated their training data.'])
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(args.output / 'README.md')


if __name__ == '__main__':
    main()
