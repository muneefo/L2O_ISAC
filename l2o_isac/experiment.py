"""Run the complete EpsNet -> U-PGDA -> recurrence -> held-out evaluation pipeline."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import numpy as np
import torch
from .config import SystemConfig
from .data import generate_channels
from .networks import EpsNet
from .solver import UnrolledPGDA
from .training import train_epsnet, train_unrolled, recurrent_grid_search, bound_metrics
from .evaluation import evaluate_methods


def plot_results(report, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'l2o-isac-mpl'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'l2o-isac-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in report['evaluation'].items():
        if name != 'Nonrobust-PGA':
            curve = r['objective_curve']
            ax[0].plot(range(1, len(curve) + 1), curve, label=name)
    ax[0].set(xlabel='Iteration / layer', ylabel='Internal weighted objective (nat)', title='Held-out solver trajectories')
    ax[0].legend(fontsize=8)
    names = list(report['evaluation'])
    x = np.arange(len(names))
    for offset, key in [(-0.2, 'nominal'), (0.2, 'adversarial')]:
        vals = [report['evaluation'][n][key]['objective']['mean'] for n in names]
        errs = [1.96 * report['evaluation'][n][key]['objective']['sem'] for n in names]
        ax[1].bar(x + offset, vals, 0.4, yerr=errs, label=key, capsize=2)
    ax[1].set_xticks(x, names, rotation=25, ha='right', fontsize=8)
    ax[1].set(ylabel='Weighted objective (nat)', title='Nominal and feasible multistart attacks')
    ax[1].legend()
    hist = report['unrolled_training']['epochs']
    ax[2].plot([h['epoch'] for h in hist], [-h['train_loss'] for h in hist], label='training')
    ax[2].plot([h['epoch'] for h in hist], [-h['validation_loss'] for h in hist], label='validation')
    ax[2].set(xlabel='Epoch', ylabel='Mean layer objective (nat)', title='Unrolled training')
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(output / 'results.png', dpi=180)
    fig.savefig(output / 'results.pdf')
    plt.close(fig)


def write_markdown(report, output):
    lines = ['# Validation run', '', f"Configuration: `{json.dumps(report['system'])}`", '',
             f"Train / validation / test: {report['run']['train']} / {report['run']['valid']} / {report['run']['test']}; "
             f"epochs: {report['run']['epochs']}; unrolled layers: {report['run']['layers']}.", '',
             '| Method | Internal objective | Nominal objective | Attacked objective ± 95% SEM | Solver ms/sample |',
             '|---|---:|---:|---:|---:|']
    for name, r in report['evaluation'].items():
        a = r['adversarial']['objective']
        lines.append(f"| {name} | {r['internal']['objective']['mean']:.5f} | {r['nominal']['objective']['mean']:.5f} | "
                     f"{a['mean']:.5f} ± {1.96*a['sem']:.5f} | {r['solver_ms_per_sample']:.3f} |")
    lines.extend(['', 'Internal objectives use each solver’s own uncertainty variables and possibly predicted radii. '
                  'They do not establish comparative robustness. Independent attacks use the same true radii and seeded '
                  'starting errors for all methods, on the first held-out attack_samples examples. '
                  'An attack finds feasible errors; it does not certify the global worst case. '
                  'Confidence intervals summarize channel-sample variation for one training seed, not training-seed variation.', '',
                  f"EpsNet test RMSE: {report['epsnet_test']['rmse']:.6f}.",
                  f"Recurrence: beta={report['recurrence']['best']['beta']:.4f}, phi={report['recurrence']['best']['phi']:.4f}.", '',
                  'See report.json for constraints, per-metric statistics, training histories, and provenance.'])
    negative_inner = sum(x < 0 for row in report['learned_parameters']['mu'] for x in row[:3])
    if negative_inner:
        lines.extend(['', f'**Semantic caveat:** {negative_inner} learned inner step sizes are negative. '
                      'The literal unconstrained training procedure has learned ascent updates for those entries; '
                      'its internal objective is not evidence of worst-case minimization. '
                      'See the separately labeled projected-hyperparameter run for nonnegative steps.'])
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['smoke', 'paper'], default='smoke')
    p.add_argument('--output', type=Path, default=Path('reports/run'))
    p.add_argument('--nt', type=int, default=16)
    p.add_argument('--nr', type=int)
    p.add_argument('--users', type=int, default=4)
    p.add_argument('--targets', type=int, default=2)
    p.add_argument('--clutter', type=int, default=2)
    p.add_argument('--rf-chains', type=int, default=8)
    p.add_argument('--layers', type=int, default=15)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--rho', type=float, default=0.8)
    p.add_argument('--snr-db', type=float, default=5.)
    p.add_argument('--epsilon', type=float, default=0.05)
    p.add_argument('--momentum', choices=['learned', 'fixed', 'none'], default='learned')
    p.add_argument('--parameterization', choices=['direct', 'positive', 'projected'], default='direct')
    p.add_argument('--bounds', choices=['learned', 'oracle'], default='learned')
    p.add_argument('--eps-layout', choices=['per-channel', 'concatenated'], default='per-channel')
    p.add_argument('--eps-label-mode', choices=['fixed', 'paper-grid', 'independent-uniform'], default='fixed',
                   help='Reconstruction label protocol; authors do not specify the EpsNet training-label distribution')
    p.add_argument('--variable-eps', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    for name in ['train', 'valid', 'test', 'epochs', 'batch-size', 'grids', 'attack-steps', 'attack-restarts', 'attack-samples']:
        p.add_argument('--' + name, type=int)
    args = p.parse_args(argv)
    if args.variable_eps:
        if args.eps_label_mode != 'fixed':
            p.error('Use --variable-eps or --eps-label-mode, not both')
        args.eps_label_mode = 'independent-uniform'
    defaults = dict(train=5000, valid=1000, test=1000, epochs=10, batch_size=250, grids=10,
                    attack_steps=80, attack_restarts=4, attack_samples=128) if args.profile == 'paper' else dict(
                    train=128, valid=32, test=32, epochs=3, batch_size=32, grids=3,
                    attack_steps=30, attack_restarts=2, attack_samples=32)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if min(args.train, args.valid, args.test, args.epochs, args.grids, args.attack_samples) < 1:
        p.error('Dataset sizes, epochs, grid count and attack samples must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg = SystemConfig(nt=args.nt, nr=args.nr or args.nt, users=args.users, targets=args.targets,
                       clutter=args.clutter, rf_chains=args.rf_chains, snr_db=args.snr_db,
                       rho=args.rho, eps_h=args.epsilon, eps_t=args.epsilon, eps_r=args.epsilon)
    train = generate_channels(cfg, args.train, seed=args.seed, device=args.device, eps_mode=args.eps_label_mode)
    valid = generate_channels(cfg, args.valid, seed=args.seed + 1, device=args.device, eps_mode=args.eps_label_mode)
    test = generate_channels(cfg, args.test, seed=args.seed + 2, device=args.device, eps_mode=args.eps_label_mode)
    started = time.perf_counter()
    epsnet = EpsNet(cfg, layout=args.eps_layout).to(args.device)
    ehistory = train_epsnet(epsnet, train, valid, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    etest = bound_metrics(epsnet, test)
    model = UnrolledPGDA(cfg, layers=args.layers, learn_momentum=args.momentum == 'learned',
                         momentum=0. if args.momentum == 'none' else 0.01,
                         parameterization=args.parameterization).to(args.device)
    bound_model = epsnet if args.bounds == 'learned' else None
    history = train_unrolled(model, train, valid, bound_model, epochs=args.epochs,
                             batch_size=args.batch_size, seed=args.seed)
    grid = recurrent_grid_search(model, valid, bound_model, grids=args.grids, batch_size=args.batch_size, seed=args.seed)
    print('RecNet best:', grid['best'], flush=True)
    # Evaluator accepts an oracle bound provider with the same call interface.
    provider = bound_model if bound_model is not None else lambda batch: batch.eps
    evaluation, arrays = evaluate_methods(cfg, test, model, provider,
                                          {k: grid['best'][k] for k in ('beta', 'phi')},
                                          batch_size=args.batch_size, attack_steps=args.attack_steps,
                                          attack_restarts=args.attack_restarts, attack_samples=args.attack_samples,
                                          seed=args.seed + 100000)
    # Paired differences use common test channels and initialization, reducing comparison variance.
    paired = {}
    for name in arrays:
        difference = arrays[name]['adversarial']['objective'] - arrays['B-PGDA']['adversarial']['objective']
        paired[name] = {"mean_difference": difference.mean().item(),
                        "sem": (difference.std(unbiased=len(difference) > 1) / len(difference) ** .5).item()}
    run = vars(args).copy()
    run['output'] = str(output)
    report = {"system": asdict(cfg), "run": run, "epsnet_training": ehistory, "epsnet_test": etest,
              "constant_bound_baseline_test_rmse": (test.eps - train.eps.mean(0)).square().mean().sqrt().item(),
              "unrolled_training": history, "recurrence": grid, "evaluation": evaluation,
              "paired_attack_difference_vs_benchmark": paired,
              "learned_parameters": {"mu": model.mu.detach().cpu().tolist(), "eta": model.eta.detach().cpu().tolist()},
              "parameter_diagnostics": {"negative_step_count": (model.mu < 0).sum().item(),
                                        "zero_step_count": (model.mu == 0).sum().item(),
                                        "negative_momentum_count": (model.eta < 0).sum().item(),
                                        "inner_step_min": model.mu[:, :3].min().item(),
                                        "outer_step_min": model.mu[:, 3:].min().item()},
              "reconstruction": {"schema_version": 2, "eps_layout": args.eps_layout,
                                  "eps_aggregation": 'shared heads followed by arithmetic mean' if args.eps_layout == 'per-channel' else 'concatenation',
                                  "eps_aggregation_specified_by_paper": False,
                                  "eps_training_label_distribution_specified_by_paper": False,
                                  "eps_loss": 'mean per-sample RMSE, Eq. (18)',
                                  "timing_scope": 'batched solver throughput; excludes EpsNet; not single-decision latency'},
              "provenance": {"paper_doi": "10.1109/TWC.2026.3702925", "python": platform.python_version(),
                             "torch": torch.__version__, "numpy": np.__version__, "device": args.device,
                             "implementation": "independent; not author-provided code"},
              "seconds_total": time.perf_counter() - started}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    torch.save({"config": asdict(cfg), "run": run, "epsnet": epsnet.state_dict(),
                "unrolled": model.state_dict(), "recurrence": grid['best']}, output / 'checkpoint.pt')
    np.savez_compressed(output / 'test_metrics.npz', **{f"{name}/{kind}/{metric}": t.numpy()
                        for name, kinds in arrays.items() for kind, metrics in kinds.items() for metric, t in metrics.items()})
    plot_results(report, output)
    write_markdown(report, output)
    print(f"Saved results to {output} ({report['seconds_total']:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
