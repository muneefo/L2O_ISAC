# Implementation and validation

This is an independent implementation of the proposed PGDA/EpsNet/U-PGDA/RU-PGDA pipeline. Numerical correctness and empirical training have been checked; exact reproduction of all paper figures and comparator methods is not claimed.

## Numerical checks

```text
...................................                                      [100%]
35 passed in 2.76s
```

Checks cover all six analytical gradients against independently written signal equations and autograd, central finite differences, a training derivative through the iterative algorithm, projection feasibility, deterministic serialization, zero uncertainty, unequal array dimensions, recurrence, and adversarial search. A one-stream special case compares normalized-gradient adversarial search with a closed-form global worst-case solution.

## Completed training runs

| Run | EpsNet layout | Train/val/test | Layers | Validation objective: initial → best | EpsNet loss | Negative inner steps |
|---|---|---:|---:|---:|---:|---:|
| [corrected_per_channel_seed73129](corrected_per_channel_seed73129/summary.md) | per-channel | 5000/1000/1000 | 15 | 0.7675 → 4.2102 | 0.001013 | 16 |
| [fresh_execution_seed73129](fresh_execution_seed73129/summary.md) | concatenated (historical) | 5000/1000/1000 | 15 | 0.7676 → 4.2102 | 0.002028 | 15 |
| [paper_n16_l15](paper_n16_l15/summary.md) | concatenated (historical) | 5000/1000/1000 | 15 | 0.7704 → 4.2252 | 0.002166 | 14 |
| [paper_n16_l20](paper_n16_l20/summary.md) | concatenated (historical) | 5000/1000/1000 | 20 | 0.8588 → 4.3529 | 0.002166 | 21 |
| [paper_n32_l15](paper_n32_l15/summary.md) | concatenated (historical) | 5000/1000/1000 | 15 | 1.1206 → 4.7951 | 0.002209 | 17 |
| [projected_n16_l15](projected_n16_l15/summary.md) | concatenated (historical) | 5000/1000/1000 | 15 | 0.7704 → 4.2061 | 0.002166 | 0 |

Each report records its epochs, batch size, grid, and label protocol. Historical concatenated runs used a batch-global RMSE; corrected runs average the per-sample Eq. (18) loss. Those loss definitions must not be silently compared as identical. Historical results and checkpoints have not been overwritten. See [reproduction status](../docs/REPRODUCTION_STATUS.md).

## Independent robustness evaluation

Each beamformer is frozen and attacked using true uncertainty radii and common seeded starts. Attack steps, restart counts and sample counts are recorded in each report. The table shows paired differences against B-PGDA and approximate 95% intervals from channel sampling.

| Run | Method | Attacked objective | Difference vs B-PGDA ± 95% SEM |
|---|---|---:|---:|
| corrected_per_channel_seed73129 | B-PGDA | 4.06798 | +0.00000 ± 0.00000 |
| corrected_per_channel_seed73129 | U-PGDA | 4.07477 | +0.00679 ± 0.02262 |
| corrected_per_channel_seed73129 | RU-PGDA+10 | 4.10791 | +0.03994 ± 0.02600 |
| fresh_execution_seed73129 | B-PGDA | 4.06798 | +0.00000 ± 0.00000 |
| fresh_execution_seed73129 | U-PGDA | 4.07495 | +0.00698 ± 0.02260 |
| fresh_execution_seed73129 | RU-PGDA+10 | 4.10803 | +0.04006 ± 0.02598 |
| paper_n16_l15 | B-PGDA | 4.00964 | +0.00000 ± 0.00000 |
| paper_n16_l15 | U-PGDA | 4.03400 | +0.02436 ± 0.02430 |
| paper_n16_l15 | RU-PGDA+10 | 4.07240 | +0.06276 ± 0.03091 |
| paper_n16_l20 | B-PGDA | 4.00964 | +0.00000 ± 0.00000 |
| paper_n16_l20 | U-PGDA | 4.07234 | +0.06270 ± 0.02946 |
| paper_n16_l20 | RU-PGDA+10 | 4.09275 | +0.08312 ± 0.03198 |
| paper_n32_l15 | B-PGDA | 4.41743 | +0.00000 ± 0.00000 |
| paper_n32_l15 | U-PGDA | 4.33685 | -0.08058 ± 0.01351 |
| paper_n32_l15 | RU-PGDA+10 | 4.38176 | -0.03567 ± 0.01353 |
| projected_n16_l15 | B-PGDA | 4.00964 | +0.00000 ± 0.00000 |
| projected_n16_l15 | U-PGDA | 4.03964 | +0.03000 ± 0.02430 |
| projected_n16_l15 | RU-PGDA+10 | 4.07944 | +0.06980 ± 0.03097 |

Stronger audit: [paper_n16_l15](paper_n16_l15/robustness_audit.json), 128 fresh channels, 200 steps and 8 starts.

- B-PGDA: attacked objective 4.02328; paired difference +0.00000 ± 0.00000.
- U-PGDA: attacked objective 4.01659; paired difference -0.00669 ± 0.01840.
- RU-PGDA+10: attacked objective 4.05478; paired difference +0.03150 ± 0.02218.

These attacks are feasible local searches, not certificates of the global worst case. Intervals do not include variation over training seeds. Solver internal objectives can be higher because their own error iterates have not solved the inner minimization.

## Constraint residuals

Maximum absolute errors across completed runs:

```json
{
  "analog_modulus": 1.1920928955078125e-07,
  "transmit_power": 2.6226043701171875e-06,
  "receive_norm": 2.384185791015625e-07,
  "dh_violation": 8.940696716308594e-08,
  "dt_violation": 8.940696716308594e-08,
  "dr_violation": 8.940696716308594e-08
}
```

## Findings that matter before extensions

1. In our reconstruction, literal unconstrained SGD can learn negative inner step sizes. Such entries perform ascent rather than minimization; this is visible in the saved parameters. The projected variant prevents negative steps, but cannot fix the objective’s incentive to weaken the inner adversary, including by shrinking step sizes to zero. This is not evidence that the authors learned the same coefficients.
2. Our fixed-label training runs admit a constant predictor. That is our explicit data protocol, not a claim about the authors' EpsNet training labels. The paper evaluates multiple levels in Table I and miscalibration in Fig. 6. Its label-generation distribution is unspecified. Our optional independently sampled labels cannot be inferred from nominal inputs that do not encode them.
3. The paper omits angular sampling and initialization details. Explicit choices are documented in README.md. EpsNet sharing/mean aggregation are additional explicit choices. The proposed pipeline is reconstructed; the SCA/CNN/DNN comparison baselines remain unimplemented.
4. The current evidence supports the implementation’s numerical correctness and the reported empirical results, not exact replication, a general superiority claim, or a global min-max convergence theorem. The paper itself does include a convergence discussion in IV-E.

## Paper evidence and configuration/latency checks

The paper already evaluates M=T=2,4,6 in Fig. 9 and N=16,32 in Fig. 3/Table II. Its Table II reports U-PGDA (15/15) runtimes of 7.56/12.12 ms, respectively, on an Intel Core Ultra 7 155H with 32 GB RAM. These are published reference values, not our measurements. See [reference data](../configs/paper_reference.json) and the [paper reading](../docs/PAPER_READING.md).

The corrected full-size run's source snapshot is commit `b6a289d`; its execution manifest records those source hashes. Historical runs originate from `4425f7a`. Later reporting and experiment-script changes do not retroactively change either source snapshot.

The additional [frozen-weight/epsilon/latency audit](corrected_per_channel_checks/README.md) keeps one checkpoint fixed and tests the Fig.-9 count settings, Table-I epsilon settings, and warm batch-one latency. This sharing protocol is explicitly ours; the authors do not specify whether Fig. 9 reused the same weights. Batched throughput is not single-decision latency, and fixed-depth inference is not certified time to convergence.

Raw per-sample metrics, checkpoints, full histories, plots, and every learned hyperparameter are retained beside each report.
