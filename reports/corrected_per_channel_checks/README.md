# Measured reconstruction checks

Published reference values are separately labeled in measured_checks.json and configs/paper_reference.json. They are not outputs of the reconstructed solver.

EpsNet layout: `per-channel`.

## Frozen-weight user/target-count audit

Same saved weights, no retraining, at Fig. 9 user/target counts. This is an added transfer audit, not a claimed reproduction of their unspecified sharing protocol.

| Users | Targets | Attacked objective | Benchmark |
|---:|---:|---:|---:|
| 2 | 2 | 3.22502 | 3.21616 |
| 4 | 4 | 4.09082 | 4.05631 |
| 6 | 6 | 4.54146 | 4.50422 |

## Batch-one latency

Warm CPU timings, including layer diagnostics. Initialization/bounds are included only where named. Acquisition, external transfer, and RF programming are excluded. These are fixed-depth inference times.

| Scope | Median ms | p95 ms |
|---|---:|---:|
| EpsNet | 0.215 | 0.248 |
| U-PGDA/solver_only | 15.436 | 20.424 |
| U-PGDA/including_initialization_and_bounds | 15.987 | 16.589 |
| RU-PGDA+10/solver_only | 25.465 | 26.241 |
| RU-PGDA+10/including_initialization_and_bounds | 26.728 | 27.543 |
| B-PGDA/solver_only | 186.469 | 190.662 |
| B-PGDA/including_initialization_and_bounds | 187.625 | 196.962 |

## Table-I settings with one frozen EpsNet

One frozen EpsNet evaluated on Table I epsilon values; author training-label protocol unspecified.

| True epsilon | Mean predicted h/t/r | Mean per-sample RMSE |
|---:|---|---:|
| 0.001 | 0.04979, 0.04990, 0.04999 | 0.048900 |
| 0.01 | 0.04979, 0.04990, 0.04999 | 0.039902 |
| 0.02 | 0.04979, 0.04990, 0.04999 | 0.029906 |
| 0.03 | 0.04979, 0.04990, 0.04999 | 0.019913 |
| 0.04 | 0.04979, 0.04990, 0.04999 | 0.009936 |
| 0.05 | 0.04979, 0.04990, 0.04999 | 0.001028 |

The nominal-channel generator is independent of the radius label in this reconstruction. A fixed model cannot infer an independently changed radius from identical nominal inputs. This limitation does not establish how the authors generated their training data.
