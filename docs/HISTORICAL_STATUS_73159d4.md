# Robust ISAC architecture audit — completed executed pilot

Pinned independent reference: `83ba89148c8d89a7ab60105da0fa5cbfe6105d9e`.
Reference checkpoint SHA256:
`58845000260beae656c91f30217fc75a27c11a94eba116c766b5055a7079f77e`.
Reference physics/solver/data/networks/training/inference, checkpoint and existing
raw states are byte-identical to that commit. No reference training or data
regeneration was performed. See reports/extension_results.md for the full result.

| Gate | Status and measured evidence |
|---|---|
| A | PASS: equation map and full graph/objective interpretation written before extension training; ambiguities explicit. Retained complete paper text reread; original PDF attachment absent, prior PDF inspection and current author portal verify identity. |
| B | PASS: 53 tests in 3.67s; independent NumPy SINRs/components, all complex block coordinate/directional gradients, short-unroll higher derivatives, projections, unequal arrays, full-layer feasibility, independent attacker and frozen role blocks. |
| C | PASS: eight saved raw channel/state/trace cases replay exactly, zero max difference. No retraining. |
| D | PASS: 128 frozen held-out audit scenarios; 50/200 nested PGD steps, five random plus zero/terminal starts, 50 Adam steps; U15 terminal4.66515 versus independent4.03977, gap.62538. All feasible. |
| E | PASS at pilot scope: separate objectives, same180 parameters/30 online full analytic calls, three shuffle seeds, validation selection; independently attacked role utility4.09496,4.09497,4.09489. |
| Extra strength | PASS: 1,000 steps on all128 scenarios for reference/RU25/tunedPGDA180/three role variants; gain remains~.05519/~1.37%; all feasible. Five random restarts remain a pilot budget, not the full target20. |
| Controls | Executed three-seed inner-only and outer-only variants, raw outputs and training histories retained. Both improve; combined is best among these controls. No equal-extra-training shared-loss control. |
| Secondary | Executed32-channel rho=.2,.5,.8 / epsilon=.01,.05 sensitivity with fixed reference and extensionseed101. Single-seed exploratory scope. |
| F/G | Deferred optional controller/calibration. No state-conditioned or calibrated-coverage result claimed; no new metadata or channel model introduced. |

A deployed-stage audit found communication utility increasing in80% of reference
inner sample/layer steps; role variants had no increases above1e-6 in these
samples. This supports a role-conflict mechanism in our reconstruction, not an
author-code defect or a general monotonicity theorem. Learned terminal errors
remain insufficiently adverse after role training (gap~.363); independent attacks
are still necessary. Found minima are upper bounds on unknown global minima.

Validation-tuned180-step classical PGDA reaches4.98123 at360 full derivative
calls and~180ms median; U15/role use30 calls and~15–16ms. The extension does not
beat unrestricted classical search or establish sub-millisecond convergence.
All recorded scientific evaluation scenarios have zero observed numerical or
hardware/ball failures at complex64 tolerance1e-5. Per-scenario residuals/failure
flags are retained in results/architecture_audit/feasibility_per_scenario.npz.

Resources: AMD EPYC9V74 virtual CPU, cgroup quota two CPUs, PyTorch one thread,
complex64 experiments/complex128 reference tests. No GPU used. This study reuses
one pretrained reference; three fine-tuning shuffle seeds do not represent
independent pretraining seeds. Paired channel bootstrap intervals and seed SD
are reported separately.

Exact executed commands from repository root (Python executable
`/workspace/.venv/bin/python`; defaults point to saved pilot/reference files):

```bash
python -m l2o_isac.checkpoint_regression
python -m pytest -q
python -m l2o_isac.architecture_audit --stage tune
python -m l2o_isac.architecture_audit --stage audit
python -m l2o_isac.architecture_audit --stage train
python -m l2o_isac.architecture_audit --stage evaluate
python -m l2o_isac.strong_attack_check
python -m l2o_isac.role_ablations
python -m l2o_isac.secondary_tradeoff
python -m l2o_isac.minnet_direction_audit
python -m l2o_isac.audit_artifacts
python -m l2o_isac.architecture_audit --stage figures
python -m l2o_isac.audit_report
python -m l2o_isac.architecture_audit --config configs/pilot/architecture_smoke.json --output results/architecture_smoke --stage all
```

Tiny smoke initially failed only while serializing undefined one-run sample SD
(ddof1); corrected to null and rerun successfully with cached states/splits. The
failed attempt log is preserved. Scientific three-seed results were unaffected.
Final smoke/resume and test logs are retained. Source snapshots reproduce recorded
checkpoint hashes, including intermediate reporting revisions.

Unexecuted full-study command (roughly1–2CPU hours of primary attacks before
additional training/tuning/controls; an estimate, not a benchmark):

```bash
python -m l2o_isac.architecture_audit --config configs/full/architecture_audit.json --output results/full_architecture_audit --stage all
```

Next action: run that five-seed/1,000-scenario/20-restart study, add a matched
extra-training shared-loss control and broader configurations, then assess a
state controller only against this strengthened evaluation. Literature shows
separate minimax roles and trajectory-conditioned optimizers already exist.
No publication novelty, globally certified robustness, actual detection gain or
general convergence guarantee is claimed. See docs/related_work_audit.md.
