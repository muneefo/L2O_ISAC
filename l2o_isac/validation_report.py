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
             '| Run | EpsNet layout | Train/val/test | Layers | Validation objective: initial → best | EpsNet loss | Negative inner steps |',
             '|---|---|---:|---:|---:|---:|---:|']
    selected = []
    max_constraints = {}
    for path in reports:
        r = json.loads(path.read_text())
        if r['run']['profile'] != 'paper':
            continue
        selected.append((path.parent.name, r))
        nnegative = sum(x < 0 for row in r['learned_parameters']['mu'] for x in row[:3])
        training = r['unrolled_training']
        layout = r['run'].get('eps_layout', 'concatenated (historical)')
        lines.append(f"| [{path.parent.name}]({path.parent.name}/summary.md) | {layout} | "
                     f"{r['run']['train']}/{r['run']['valid']}/{r['run']['test']} | {r['run']['layers']} | "
                     f"{-training['initial_validation_loss']:.4f} → {-training['best_validation_loss']:.4f} | "
                     f"{r['epsnet_test']['rmse']:.6f} | {nnegative} |")
        for method in r['evaluation'].values():
            for key, value in method['max_constraint_errors'].items():
                max_constraints[key] = max(max_constraints.get(key, 0), value)
    lines.extend(['', 'Each report records its epochs, batch size, grid, and label protocol. Historical concatenated '
                  'runs used a batch-global RMSE; corrected runs average the per-sample Eq. (18) loss. '
                  'Those loss definitions must not be silently compared as identical. Historical results and '
                  'checkpoints have not been overwritten. See [reproduction status](../docs/REPRODUCTION_STATUS.md).', '',
                  '## Independent robustness evaluation', '',
                  'Each beamformer is frozen and attacked using true uncertainty radii and common seeded starts. '
                  'Attack steps, restart counts and sample counts are recorded in each report. The table shows paired '
                  'differences against B-PGDA and approximate 95% intervals from channel sampling.', '',
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
                      f"{r['settings']['restarts']} starts.", ''])
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
                  '1. In our reconstruction, literal unconstrained SGD can learn negative inner step sizes. Such entries perform '
                  'ascent rather than minimization; this is visible in the saved parameters. The projected '
                  'variant prevents negative steps, but cannot fix the objective’s incentive to weaken '
                  'the inner adversary, including by shrinking step sizes to zero. This is not evidence that the '
                  'authors learned the same coefficients.',
                  '2. Our fixed-label training runs admit a constant predictor. That is our explicit data protocol, '
                  'not a claim about the authors\' EpsNet training labels. The paper evaluates multiple levels '
                  'in Table I and miscalibration in Fig. 6. Its label-generation distribution is unspecified. '
                  'Our optional independently sampled labels cannot be inferred from nominal inputs that do not encode them.',
                  '3. The paper omits angular sampling and initialization details. Explicit choices are '
                  'documented in README.md. EpsNet sharing/mean aggregation are additional explicit choices. '
                  'The proposed pipeline is reconstructed; the SCA/CNN/DNN comparison baselines remain unimplemented.',
                  '4. The current evidence supports the implementation’s numerical correctness and the '
                  'reported empirical results, not exact replication, a general superiority claim, or a '
                  'global min-max convergence theorem. The paper itself does include a convergence discussion in IV-E.', '',
                  '## Paper evidence and configuration/latency checks', '',
                  'The paper already evaluates M=T=2,4,6 in Fig. 9 and N=16,32 in Fig. 3/Table II. '
                  'Its Table II reports U-PGDA (15/15) runtimes of 7.56/12.12 ms, respectively, on an '
                  'Intel Core Ultra 7 155H with 32 GB RAM. These are published reference values, not our measurements. '
                  'See [reference data](../configs/paper_reference.json) and the [paper reading](../docs/PAPER_READING.md).', '',
                  'The corrected full-size run\'s source snapshot is commit `b6a289d`; its execution manifest '
                  'records those source hashes. Historical runs originate from `4425f7a`. Later reporting and '
                  'experiment-script changes do not retroactively change either source snapshot.', '',
                  'The additional [frozen-weight/epsilon/latency audit](corrected_per_channel_checks/README.md) '
                  'keeps one checkpoint fixed and tests the Fig.-9 count settings, Table-I epsilon settings, '
                  'and warm batch-one latency. This sharing protocol is explicitly ours; the authors do not '
                  'specify whether Fig. 9 reused the same weights. Batched throughput is not single-decision '
                  'latency, and fixed-depth inference is not certified time to convergence.', '',
                  'Raw per-sample metrics, checkpoints, full histories, plots, and every learned hyperparameter '
                  'are retained beside each report.'])
    (root / 'VALIDATION.md').write_text('\n'.join(lines) + '\n')
    print(root / 'VALIDATION.md')


if __name__ == '__main__':
    main()
