# Fluid transmit pilot — completed

Isolated at experiments/fluid_tx_single_user/. Baseline source snapshot:6e037b3.
No original reproduction source,checkpoint,dataset,configuration or published numerical output was changed.

- PASS:all70 saved method designs replay exactly with final source (zero maximum difference); all70 metrics independently checked with NumPy; all767 pre-existing tracked files unchanged.
- PASS:72 tests (existing suite plus13 fluid tests), including read-only checkpoint replay.
- PASS:exact bounded-isotonic/PAV projection versus SciPy convex QP and exhaustive tiny partitions.
- PASS:float64 spatial/position gradients,physical-unit/conjugation checks and full-unroll derivative tests.
- PASS:full-layer hardware,power,error-ball and position constraints; maximum observed residual1.43e-6 (float32 tolerance1e-5).
- EXECUTED:100 paired test scenarios in each of three regimes,64 train/32 validation,common aperture3.5lambda/minimum spacing.25lambda,Nt=Nr8,L4.
- EXECUTED:fixed ULA,one static random array,per-scenario random control,joint fluid;60/180/360 gradient budgets plus validation-runtime-matched points.
- EXECUTED:per-budget long-ULA/static validation retuning; nominal,terminal,50/200/1000 independent attacks and shared pools.
- EXECUTED:three-seed position learning,one-scalar equal-training control versus20-scalar schedule; no larger NN or BF-coefficient training.
- EXECUTED:rho.2,.5,.8 communication/sensing tradeoff with fresh validation tuning and unchanged scenarios.
- PASS:classical smoke command; actual logs and one resolved initial precision-matching test failure preserved.

Measured geometry comparison (long matched360 helper calls, A uses analytic ULA reference):

| Case | Fixed ULA | Fluid | Difference |
|---|---:|---:|---:|
| A | 3.527760 | 3.522612 | -0.005148 |
| B | 2.968832 | 3.318646 | 0.349814 |
| C | 2.660195 | 2.940933 | 0.280739 |

A has no intrinsic geometry gain; finite inherited PGDA can leave its analytic optimum. B/C show positive paired gains under this spatial model. The20-step learned schedule does not beat an equally trained scalar in a useful way, and does not recover longer classical fluid performance at lower budget. Some clutter runs select the untrained initialization or worsen held-out utility; none are removed.

Source/model limits:known fixed spatial path parameters,one-dimensional far-field movement,no coupling/position error/actuation cost. This is not the old iid channel distribution. CPU computation times are not physical movement or RF latency. Found minima are empirical,not globally certified except the separately derived solvable case-A check. No publication novelty is claimed.

Exact commands and dependencies:README.md. Measured report/plots:REPORT.md and outputs/pilot/. Raw states and metrics,training optimizer/RNG states,configs,CSV,software versions,independent NumPy checks and hashes are retained.

Unresolved research question:layer-specific learned position updates have no demonstrated benefit beyond scalar step calibration in this pilot. Do not scale up the network based on these results; stronger optimization/data/generalization studies would be separate work.
