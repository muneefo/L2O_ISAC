# Prespecified pilot protocol

Fixed source problem: N_t=N_r=16, M=4,T=2,J=2,L=8, rho=.8,
SNR=5dB, all simulator radii=.05, original physical metric/constraints.
Primary comparison uses **oracle radii** for every solver and the same true sets
for every independent attack. An EpsNet deployment branch is a separate baseline;
its internal terminal errors are projected into common evaluation balls, and its
unprojected internal utility is separately recorded. No smaller predicted radius
makes its final attack easier. Such projection is disclosed in raw results.

The completed reference checkpoint, source files, raw values and reports are
reused without training. Original channel tensors for all 1000 reference samples
were not persisted, so this audit does **not regenerate** that reference split.
It instead saves separate new audit train/validation/test scenarios before any
augmentations or tuning. Independent scenarios have no correlated groups; if
later introducing correlated snapshots, groups must be split together. Audit
seeds are disjoint from source manifest seeds. Training never reads audit test.
128 held-out audit test channels are frozen on disk and hashed. These were not
used to select the original model. A short regression reuses its eight saved raw
channels exactly; see results/architecture_audit/checkpoint_regression.json.

Validation selection (before opening test evaluation): choose PGD attack step
size from {.01,.05,.2}, minimizing validation utility. Adam epsilon=1e-8 is an
optimizer normalization stabilizer, not a physics denominator clamp. Tune each
classical family on the same validation channels over the 3×3 inner/outer grid.
Store choices on disk. Test PGD budgets are nested 50/200 steps, five random
restarts plus zero and learned-terminal starts; use both interior/boundary starts.
Also retain 50 projected-Adam steps. Sensing error pairs are always joint and
shared across target SINRs. Best candidates are retained at every step and then
across budgets. Independent final attacks use autograd metrics, not MinNet
schedules/analytic update code. Freeze all beamformers throughout every attack.

Report internal perturbed, nominal, independent lowest-found and shared-candidate
union results, each split into comm and sensing MI-upper-bound components. Attack
gap uses common-set feasible terminal errors and is nonnegative by inclusion.
Per-scenario tensors include errors, beams, failure counts, residuals and all
budget curves. Independent search is NOT a certificate; found minima are upper
bounds on the unknown global minima. Attack every method, then cross-evaluate the
union of all found candidates plus terminal candidates for every method.

Role objectives are fixed in audit_hypotheses.md. New training uses 256 scenarios,
64 validation and three SGD shuffle seeds. Two outer learning rates are selected
on validation with seed 101 before the final three runs; initial weights remain
eligible. These runs fine-tune the **same** reference and do not quantify pretraining
seed variability. Training cost and optimizer/RNG states are saved. Optional
controllers/calibration are deferred, avoiding confounded primary results.

Matched derivative budget: U15=30 full analytic calls, fixed PGDA15=30,
stronger-inner k=4 with six outer steps=30; RU25=50, PGDA180=360;
stronger-inner k=4 with 15 outer steps=75. The analytic helper computes all six
blocks even when only one role is used; charge full calls accordingly. The
independent autograd attack computes only three error blocks, so report its calls
separately rather than pretending identical FLOPs. Timings include projections
and controller cost (none here), report warm CPU batch-one median/p95, plus batch
throughput. Initialization and EpsNet timing are separately identified. No claim
of matched runtime is made without actual matched-runtime experiments.

Confidence intervals use paired channel bootstrap for each fixed checkpoint;
report the three extension-run mean differences and SD separately. Do not pool
128×3 observations as independent training runs. There is only one reference
training seed. Failure logs retain failed configurations; nonfinite values abort
rather than being removed. Tests must pass before Gate D; Gate E begins only
following a recorded Gate D report. Optional F/G and secondary rho/dimension/SNR
sweeps are unexecuted unless explicitly saved. Pilot cannot support a publication
claim or actual detection/estimation improvement.
