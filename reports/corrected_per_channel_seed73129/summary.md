# Validation run

Configuration: `{"nt": 16, "nr": 16, "users": 4, "targets": 2, "clutter": 2, "rf_chains": 8, "snr_db": 5.0, "noise_comm": 1.0, "noise_sense": 1.0, "rho": 0.8, "eps_h": 0.05, "eps_t": 0.05, "eps_r": 0.05, "target_variance_db": 2.0, "clutter_variance_db": 3.0, "angle_min_deg": -60.0, "angle_max_deg": 60.0, "min_separation_deg": 5.0}`

Train / validation / test: 5000 / 1000 / 1000; epochs: 10; unrolled layers: 15.

| Method | Internal objective | Nominal objective | Attacked objective ± 95% SEM | Solver ms/sample |
|---|---:|---:|---:|---:|
| B-PGDA | 4.34053 | 4.61795 | 4.06798 ± 0.10687 | 4.956 |
| U-PGDA | 4.69362 | 4.64514 | 4.07477 ± 0.10190 | 0.419 |
| RU-PGDA+5 | 4.70768 | 4.66861 | 4.09366 ± 0.10164 | 0.567 |
| RU-PGDA+10 | 4.71730 | 4.68729 | 4.10791 ± 0.10141 | 0.714 |
| Nonrobust-PGA | 4.62359 | 4.62359 | 4.04766 ± 0.10882 | 4.999 |

Internal objectives use each solver’s own uncertainty variables and possibly predicted radii. They do not establish comparative robustness. Independent attacks use the same true radii and seeded starting errors for all methods, on the first held-out attack_samples examples. An attack finds feasible errors; it does not certify the global worst case. Confidence intervals summarize channel-sample variation for one training seed, not training-seed variation.

EpsNet test RMSE: 0.001013.
Recurrence: beta=0.9900, phi=0.9900.

See report.json for constraints, per-metric statistics, training histories, and provenance.

**Semantic caveat:** 16 learned inner step sizes are negative. The literal unconstrained training procedure has learned ascent updates for those entries; its internal objective is not evidence of worst-case minimization. See the separately labeled projected-hyperparameter run for nonnegative steps.
