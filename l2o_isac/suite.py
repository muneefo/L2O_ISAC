"""Reproducible parameter sweeps. Every setting retrains without test-set selection."""
import argparse
import json
from pathlib import Path
from .experiment import main as run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['core', 'momentum', 'robustness', 'tradeoff', 'snr', 'scenarios'], default='core')
    p.add_argument('--profile', choices=['smoke', 'paper'], default='paper')
    p.add_argument('--output', type=Path, default=Path('reports/suite'))
    p.add_argument('--seeds', type=int, nargs='+', default=[2026])
    args, overrides = p.parse_known_args(argv)
    settings = {
        'core': [(f'n{n}_l{l}', ['--nt', str(n), '--layers', str(l)]) for n in (16, 32) for l in (15, 20)],
        'momentum': [(kind, ['--momentum', kind]) for kind in ('none', 'fixed', 'learned')],
        'robustness': [(f'eps{e}', ['--epsilon', str(e)]) for e in (.001, .005, .01, .025, .05)],
        'tradeoff': [(f'rho{r}', ['--rho', str(r)]) for r in (.1, .2, .3, .4, .5, .6, .7, .8, .9)],
        'snr': [(f'snr{s}', ['--snr-db', str(s)]) for s in (-6, -3, 0, 3, 6)],
        'scenarios': [(f'm{m}_t{m}', ['--users', str(m), '--targets', str(m)]) for m in (2, 4, 6)],
    }[args.family]
    index = []
    for seed in args.seeds:
        for name, flags in settings:
            output = args.output / f'{name}_seed{seed}'
            run(['--profile', args.profile, '--output', str(output), '--seed', str(seed), *flags, *overrides])
            index.append(dict(setting=name, seed=seed, report=str(output / 'report.json')))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
