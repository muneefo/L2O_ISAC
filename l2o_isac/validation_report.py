"""Summarize completed experiments without overstating scientific reproduction."""
import json
from pathlib import Path


def main():
    root = Path('reports')
    reports = sorted(root.glob('*/report.json'))
    lines = ['# Implementation and validation', '',
             'This is an independent implementation of the proposed PGDA/EpsNet/U-PGDA/RU-PGDA pipeline. '
             'Numerical correctness and empirical training have been checked; exact reproduction of all paper '
             'figures and comparator methods is not claimed.', '', '## Numerical checks', '',
             '```text', (root / 'tests.txt').read_text().strip(), '```', '',
             'Checks cover all six analytical gradients against independently written signal equations and '
             'autograd, central finite differences, a training derivative through the iterative algorithm, '
             'projection feasibility, deterministic serialization, zero uncertainty, unequal array dimensions, '
             'recurrence, and adversarial search. A one-stream special case compares normalized-gradient '
             'adversarial search with a closed-form global worst-case solution.', '',
             '## Completed training runs', '',
             '| Run | Train/val/test | Layers | Validation objective: initial → best | EpsNet test RMSE | Negative inner steps |',
             '|---|---:|---:|---:|---:|---:|']
    selected = []
    max_constraints = {}
    for path in reports:
        r = json.loads(path.read_text())
        if r['run']['profile'] != 'paper':
            continue
        selected.append((path.parent.name, r))
        nnegative = sum(x < 0 for row in r['learned_parameters']['mu'] for x in row[:3])
        training = r['unrolled_training']
        lines.append(f"| [{path.parent.name}]({path.parent.name}/summary.md) | "
                     f"{r['run']['train']}/{r['run']['valid']}/{r['run']['test']} | {r['run']['layers']} | "
                     f"{-training['initial_validation_loss']:.4f} → {-training['best_validation_loss']:.4f} | "
                     f"{r['epsnet_test']['rmse']:.6f} | {nnegative} |")
        for method in r['evaluation'].values():
            for key, value in method['max_constraint_errors'].items():
                max_constraints[key] = max(max_constraints.get(key, 0), value)
    lines.extend(['', 'All paper-profile runs use 10 epochs, batch size 250, the stated SGD learning rates, '
                  'and validation-only 10×10 recurrence grid search. Nonnegative-step runs are explicitly '
                  'labeled as an implementation safeguard absent from the paper’s specified optimizer.', '',
                  '## Independent robustness evaluation', '',
                  'Each beamformer is frozen and attacked using true uncertainty radii, with 80 projected '
                  'gradient steps and 4 common seeded starts on 128 test channels. The table shows paired '
                  'objective differences against B-PGDA and approximate 95% intervals from channel sampling.', '',
                  '| Run | Method | Attacked objective | Difference vs B-PGDA ± 95% SEM |',
                  '|---|---|---:|---:|'])
    for name, r in selected:
        for method in ('B-PGDA', 'U-PGDA', 'RU-PGDA+10'):
            value = r['evaluation'][method]['adversarial']['objective']['mean']
            pair = r['paired_attack_difference_vs_benchmark'][method]
            lines.append(f"| {name} | {method} | {value:.5f} | {pair['mean_difference']:+.5f} ± {1.96*pair['sem']:.5f} |")
    for path in sorted(root.glob('*/robustness_audit.json')):
        r = json.loads(path.read_text())
        lines.extend(['', f"Stronger audit: [{path.parent.name}]({path.parent.name}/robustness_audit.json), "
                      f"{r['settings']['samples']} fresh channels, {r['settings']['steps']} steps and "
                      f"{r['settings']['restarts']} starts."])
        for method, stats in r['results'].items():
            pair = r['paired_difference_vs_benchmark'][method]
            lines.append(f"- {method}: attacked objective {stats['strong_attack']['mean']:.5f}; "
                         f"paired difference {pair['mean']:+.5f} ± {1.96*pair['sem']:.5f}.")
    lines.extend(['', 'These attacks are feasible local searches, not certificates of the global worst case. '
                  'Intervals do not include variation over training seeds. Solver internal objectives can be '
                  'higher because their own error iterates have not solved the inner minimization.', '',
                  '## Constraint residuals', '', 'Maximum absolute errors across completed runs:', '', '```json',
                  json.dumps(max_constraints, indent=2), '```', '',
                  '## Findings that matter before extensions', '',
                  '1. Literal unconstrained SGD can learn negative inner step sizes. Such entries perform '
                  'ascent rather than minimization; this is visible in the saved parameters. The projected '
                  'variant prevents negative steps, but cannot fix the objective’s incentive to weaken '
                  'the inner adversary, including by shrinking step sizes to zero.',
                  '2. With constant 0.05 labels, EpsNet can learn the constant bound, but a constant predictor '
                  'already has zero error. Independent random uncertainty bounds cannot be inferred from '
                  'nominal channels that contain no information about those bounds.',
                  '3. The paper omits angular sampling and initialization details. Explicit choices are '
                  'documented in README.md. All proposed architectures are implemented; the SCA/CNN/DNN '
                  'comparison baselines from other papers are excluded because their adaptations are insufficiently specified.',
                  '4. The current evidence supports the implementation’s numerical correctness and the '
                  'reported empirical results, not exact replication, a general superiority claim, or a '
                  'global min-max convergence theorem.', '',
                  'Raw per-sample metrics, checkpoints, full histories, plots, and every learned hyperparameter '
                  'are retained beside each report.'])
    (root / 'VALIDATION.md').write_text('\n'.join(lines) + '\n')
    print(root / 'VALIDATION.md')


if __name__ == '__main__':
    main()
