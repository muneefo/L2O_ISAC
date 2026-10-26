# Validation run

Configuration: `{"nt": 16, "nr": 16, "users": 4, "targets": 2, "clutter": 2, "rf_chains": 8, "snr_db": 5.0, "noise_comm": 1.0, "noise_sense": 1.0, "rho": 0.8, "eps_h": 0.05, "eps_t": 0.05, "eps_r": 0.05, "target_variance_db": 2.0, "clutter_variance_db": 3.0, "angle_min_deg": -60.0, "angle_max_deg": 60.0, "min_separation_deg": 5.0}`

Train / validation / test: 5000 / 1000 / 1000; epochs: 10; unrolled layers: 15.

| Method | Internal objective | Nominal objective | Attacked objective ± 95% SEM | Solver ms/sample |
|---|---:|---:|---:|---:|
| B-PGDA | 4.33037 | 4.60757 | 4.01801 ± 0.10180 | 5.007 |
| U-PGDA | 4.68614 | 4.63745 | 4.01832 ± 0.09771 | 0.446 |
| RU-PGDA+5 | 4.70329 | 4.66377 | 4.03850 ± 0.09763 | 0.574 |
| RU-PGDA+10 | 4.71533 | 4.68462 | 4.05501 ± 0.09726 | 0.736 |
| Nonrobust-PGA | 4.61267 | 4.61267 | 4.00734 ± 0.10202 | 5.224 |

Internal objectives use each solver’s own uncertainty variables and possibly predicted radii. They do not establish comparative robustness. Independent attacks use the same true radii and seeded starting errors for all methods, on the first held-out attack_samples examples. An attack finds feasible errors; it does not certify the global worst case. Confidence intervals summarize channel-sample variation for one training seed, not training-seed variation.

EpsNet test RMSE: 0.001306.
Recurrence: beta=0.9900, phi=0.9900.

See report.json for constraints, per-metric statistics, training histories, and provenance.

**Semantic caveat:** 16 learned inner step sizes are negative. The literal unconstrained training procedure has learned ascent updates for those entries; its internal objective is not evidence of worst-case minimization. See the separately labeled projected-hyperparameter run for nonnegative steps.
