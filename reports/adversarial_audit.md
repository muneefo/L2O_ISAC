# Independent frozen-beamformer audit (Gate D)

Executed on 128 new held-out paired audit channels, oracle radii .05, rho=.8, unchanged physical equations. Independent autograd PGD uses five random starts plus zero/terminal, 50/200 nested steps and 50 projected-Adam steps. Beamformers remain frozen. Values are empirical feasible-candidate minima, upper bounds on unknown true minima; no certificate.

| Method | Terminal | 50 steps | 200 steps | Gap | Comm at 200 | Sensing MI upper bound at 200 |
|---|---:|---:|---:|---:|---:|---:|
| PGDA15 | 3.954765 | 3.871159 | 3.871086 | 0.083679 | 4.663906 | 0.699805 |
| PGDA180 | 5.198711 | 4.981251 | 4.981233 | 0.217478 | 4.822395 | 5.616582 |
| StrongInnerMatched | 3.585009 | 3.474894 | 3.474828 | 0.110181 | 4.165841 | 0.710779 |
| StrongInner15 | 3.944215 | 3.864188 | 3.864115 | 0.080100 | 4.655636 | 0.698034 |
| U15 | 4.665152 | 4.039850 | 4.039769 | 0.625383 | 4.873873 | 0.703350 |
| RU25 | 4.692201 | 4.076830 | 4.076737 | 0.615463 | 4.907509 | 0.753650 |
| U15_EpsNet | 4.664982 | 4.040064 | 4.039982 | 0.625000 | 4.874138 | 0.703355 |

H1 is supported for **our checkpoint**: its terminal perturbations substantially overestimate utility relative to independent feasible search. H2 graph evidence is consistent with this finding (16/45 signed inner step entries are negative), but attack gaps alone do not establish causation or an author-code defect.

Gate B: 52 tests passed. Gate C: exact equality to eight saved raw states and traces. All attacked hardware/power/ball residuals are below 1e-5, no failed scenarios. Union-candidate results and raw errors are retained in the adjacent JSON/PT files. Larger budgets cannot worsen the found minimum by construction. Small remaining 50→200 changes do not prove global optimality.

Validation tuning found a much stronger 180-step classical baseline. It uses 360 full derivative calls versus 30 for U15, so its higher utility is not an equal-budget comparison. Stronger inner search alone is separately reported, including a 30-call variant.

This is an exploratory single-reference-checkpoint audit. Phase E is now authorized by passed A–D gates; independent role-training results will be added separately.
