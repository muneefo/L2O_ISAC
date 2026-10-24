# Validation run

Configuration: `{"nt": 32, "nr": 32, "users": 4, "targets": 2, "clutter": 2, "rf_chains": 8, "snr_db": 5.0, "noise_comm": 1.0, "noise_sense": 1.0, "rho": 0.8, "eps_h": 0.05, "eps_t": 0.05, "eps_r": 0.05, "target_variance_db": 2.0, "clutter_variance_db": 3.0, "angle_min_deg": -60.0, "angle_max_deg": 60.0, "min_separation_deg": 5.0}`

Train / validation / test: 5000 / 1000 / 1000; epochs: 10; unrolled layers: 15.

| Method | Internal objective | Nominal objective | Attacked objective ± 95% SEM | Solver ms/sample |
|---|---:|---:|---:|---:|
| B-PGDA | 4.92111 | 5.28514 | 4.41743 ± 0.10070 | 7.351 |
| U-PGDA | 5.28690 | 5.22767 | 4.33685 ± 0.10359 | 0.583 |
| RU-PGDA+5 | 5.31749 | 5.26746 | 4.36131 ± 0.10352 | 0.816 |
| RU-PGDA+10 | 5.34018 | 5.29910 | 4.38176 ± 0.10354 | 1.019 |
| Nonrobust-PGA | 5.30362 | 5.30362 | 4.39927 ± 0.10250 | 7.490 |

Internal objectives use each solver’s own uncertainty variables and possibly predicted radii. They do not establish comparative robustness. Independent attacks use the same true radii and seeded starting errors for all methods, on the first held-out attack_samples examples. An attack finds feasible errors; it does not certify the global worst case. Confidence intervals summarize channel-sample variation for one training seed, not training-seed variation.

EpsNet test RMSE: 0.002209.
Recurrence: beta=0.9900, phi=0.9900.

See report.json for constraints, per-metric statistics, training histories, and provenance.

**Semantic caveat:** 17 learned inner step sizes are negative. The literal unconstrained training procedure has learned ascent updates for those entries; its internal objective is not evidence of worst-case minimization. See the separately labeled projected-hyperparameter run for nonnegative steps.
