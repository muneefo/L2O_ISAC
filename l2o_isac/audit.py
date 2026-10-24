"""Post-training robustness audit with stronger, independent attacks.

Usage: python -m l2o_isac.audit --checkpoint reports/paper_n16_l15/checkpoint.pt
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from .data import generate_channels, initialize_state
from .inference import load_checkpoint
from .solver import PGDA, constraint_errors
from .physics import metrics
from .evaluation import adversarial_evaluate, statistics


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--restarts', type=int, default=8)
    p.add_argument('--seed', type=int, default=50000)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg, epsnet, model, saved = load_checkpoint(args.checkpoint)
    batch = generate_channels(cfg, args.samples, seed=args.seed)
    initial = initialize_state(batch, cfg, seed=args.seed + 1)
    eps = epsnet(batch) if saved['run']['bounds'] == 'learned' else batch.eps
    methods = {'B-PGDA': PGDA(cfg, iterations=180), 'U-PGDA': model, 'RU-PGDA+10': model}
    results, vectors = {}, {}
    for name, solver in methods.items():
        kwargs = dict(extra_layers=10, beta=saved['recurrence']['beta'], phi=saved['recurrence']['phi']) if name == 'RU-PGDA+10' else {}
        design, _, _ = solver(batch, initial, eps=batch.eps if name == 'B-PGDA' else eps, **kwargs)
        nominal_state = replace(design, dh=design.dh * 0, dt=design.dt * 0, dr=design.dr * 0)
        nominal = metrics(batch, nominal_state, cfg)
        weak, weak_state = adversarial_evaluate(batch, design, cfg, steps=80, restarts=4, seed=args.seed + 100)
        strong, strong_state = adversarial_evaluate(batch, design, cfg, steps=args.steps, restarts=args.restarts, seed=args.seed + 100)
        # Strong settings include identical initial starts and more monotone attack iterations.
        if args.steps >= 80 and args.restarts >= 4:
            if (strong['objective'] > weak['objective'] + 1e-5).any():
                raise AssertionError('Stronger nested attack unexpectedly increased objective')
        if (strong['objective'] > nominal['objective'] + 1e-5).any():
            raise AssertionError('Attack unexpectedly improved on nominal')
        results[name] = {"nominal": statistics(nominal['objective']), "weak_attack": statistics(weak['objective']),
                         "strong_attack": statistics(strong['objective']),
                         "strong_minus_weak": statistics(strong['objective'] - weak['objective']),
                         "strong_attack_constraints": constraint_errors(strong_state, batch, cfg)}
        vectors[name] = strong['objective']
    paired = {name: statistics(value - vectors['B-PGDA']) for name, value in vectors.items()}
    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "results": results, "paired_difference_vs_benchmark": paired,
              "warning": "Feasible local attacks, not a global worst-case certificate"}
    output = args.output or args.checkpoint.parent / 'robustness_audit.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
