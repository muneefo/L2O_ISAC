# Validation run

Configuration: `{"nt": 16, "nr": 16, "users": 4, "targets": 2, "clutter": 2, "rf_chains": 8, "snr_db": 5.0, "noise_comm": 1.0, "noise_sense": 1.0, "rho": 0.8, "eps_h": 0.05, "eps_t": 0.05, "eps_r": 0.05, "target_variance_db": 2.0, "clutter_variance_db": 3.0, "angle_min_deg": -60.0, "angle_max_deg": 60.0, "min_separation_deg": 5.0}`

Train / validation / test: 128 / 32 / 32; epochs: 3; unrolled layers: 15.

| Method | Internal objective | Nominal objective | Attacked objective ± 95% SEM | Solver ms/sample |
|---|---:|---:|---:|---:|
| B-PGDA | 4.21821 | 4.47733 | 3.94308 ± 0.17967 | 11.893 |
| U-PGDA | 4.30174 | 4.30349 | 3.77751 ± 0.19552 | 0.982 |
| RU-PGDA+5 | 4.33360 | 4.34711 | 3.81617 ± 0.18853 | 1.258 |
| RU-PGDA+10 | 4.36013 | 4.38504 | 3.84968 ± 0.18090 | 1.677 |
| Nonrobust-PGA | 4.46391 | 4.46391 | 3.91232 ± 0.18888 | 11.583 |

Internal objectives use each solver’s own uncertainty variables and possibly predicted radii. They do not establish comparative robustness. Independent attacks use the same true radii and seeded starting errors for all methods, on the first held-out attack_samples examples. An attack finds feasible errors; it does not certify the global worst case. Confidence intervals summarize channel-sample variation for one training seed, not training-seed variation.

EpsNet test RMSE: 0.118227.
Recurrence: beta=0.9900, phi=0.9900.

See report.json for constraints, per-metric statistics, training histories, and provenance.
