# One-dimensional fluid transmit-array pilot

**Finding:** geometry optimization improves the independently attacked objective in the tested multipath regimes. Aligned LOS with no clutter has no intrinsic geometry gain: a feasible analytic fixed-ULA solution already reaches the geometry-independent optimum. Learning20 layer-specific position steps does not show a useful advantage over an equally trained single scalar.

## Isolation and model

This workflow lives entirely in experiments/fluid_tx_single_user/. The completed reproduction and controlled studies are pinned before this experiment; outputs/baseline_preservation.json hashes all767 pre-existing tracked files. Shared physics, gradients, original PGDA and independent attackers are imported unchanged. No old checkpoint is transferred to the new channel distribution or overwritten.

M=T1,Nt=Nr8,L4,communication plus dedicated sensing streams, SNR5dB,rho.8,noise1,common normalized uncertainty radii.05. Transmit aperture3.5lambda,min spacing.25lambda; half-wave ULA fits in the same aperture with genuine movement freedom. Coordinates are x=p/lambda; wavelength.01m converts units only. Receive ULA is fixed. All hardware/power/receive-filter constraints and norm-ball radii eps*sqrt(Nt/Nr) are retained. Sensing uses the inherited sum-of-clutter-powers approximation and natural-log MI metric; no coherent-clutter substitution or detection-accuracy claim.

The communication field is h(p)^H=sum beta_k*a_t(vartheta_k;p)^T, with fixed gains/angles while positions change. Stored h is conjugated consistently with the shared receiver equation. This is a necessary new spatial-channel assumption, **not** the paper’s position-independent CN(0,I) distribution. Fresh fixed-ULA and fluid baselines share identical spatial scenarios; no arbitrary old vector was fitted to a speculative spatial model, redrawn after movement, or normalized separately by position.

Cases A: unit-magnitude LOS user aligned with the target,J0; B: four fixed multipath gains with dominant user direction near-25deg and target near+25deg,J0; C: multipath,one target and one competing clutter. Clutter count is configurable and tests cover two clutter paths. Each has64 train/32 validation/100 test scenarios with disjoint saved seeds. Positions/weights are optimized per scenario. Random is fixed within a scenario but independently resampled across scenarios; **StaticRandom** is one draw(seed909999) held fixed across all splits/scenarios and is the primary fixed-random comparator. Both share aperture/endpoints/spacing.

## Classical result at360 full gradient-helper calls

| Case | Fixed ULA reference | Static random array | Joint fluid | Fluid minus ULA | Paired95% channel interval |
|---|---:|---:|---:|---:|---|
| A | 3.527760 | 3.513154 | 3.522612 | -0.005148 | [-0.005699,-0.004568] |
| B | 2.968832 | 3.002079 | 3.318646 | 0.349815 | [0.329902,0.370277] |
| C | 2.660195 | 2.680411 | 2.940933 | 0.280739 | [0.255666,0.305106] |

For B/C, ULA and StaticRandom use180 iterations and fluid120: all360 calls to the same helper that computes six variable-block gradients. A uses its analytic feasible MRT/MRC reference instead, not an iteratively degraded comparator. The100 paired channels, not a selected geometry, support the B/C observation. Scalar components and all shorter-budget points are in metrics.csv/summary.json.

Case A is a crucial negative control. Its analytic robust value is independent of positions under these idealized aligned/single-stream assumptions. The inherited finite PGDA updates can move away from that optimal initialization; apparent fluid recovery relative to a degraded finite-iteration ULA is an optimizer effect, **not intrinsic geometry benefit**. All A methods receive an exact feasible error candidate as an added validation control; this does not certify B/C searches.

## Matched derivative and runtime comparisons

| Case/method | Communication | Sensing MI metric | Utility | Full helper calls | Warm batch-one median/p95ms |
|---|---:|---:|---:|---:|---|
| A/ULA_budget60 | 3.170124 | 4.953835 | 3.526866 | 60 | 38.01/39.98 |
| A/StaticRandom_budget60 | 3.170137 | 4.953852 | 3.526880 | 60 | 48.22/75.33 |
| A/Fluid_budget60 | 3.170735 | 4.954461 | 3.527481 | 60 | 49.15/51.45 |
| A/ULA_runtime_match | 3.170124 | 4.953835 | 3.526866 | 60 | 36.41/37.99 |
| B/ULA_budget60 | 2.828969 | 3.515053 | 2.966186 | 60 | 35.86/37.32 |
| B/StaticRandom_budget60 | 2.849082 | 3.594444 | 2.998154 | 60 | 42.31/51.29 |
| B/Fluid_budget60 | 3.118546 | 3.708707 | 3.236578 | 60 | 52.95/59.39 |
| B/ULA_runtime_match | 2.808732 | 3.614875 | 2.969960 | 90 | 63.39/143.17 |
| C/ULA_budget60 | 2.591414 | 2.508940 | 2.574920 | 60 | 40.81/48.91 |
| C/StaticRandom_budget60 | 2.623853 | 2.399335 | 2.578949 | 60 | 41.39/43.87 |
| C/Fluid_budget60 | 2.864349 | 2.743414 | 2.840162 | 60 | 47.25/49.87 |
| C/ULA_runtime_match | 2.571276 | 2.697335 | 2.596488 | 90 | 55.87/57.81 |

A fixed macro uses2 full analytic calls; a fluid macro uses3, with the third supplying communication/steering partials for the real position chain rule. Budgets60/180/360 mean fixed30/90/180 versus fluid20/60/120. Fluid also pays for field reconstruction, exact PAV projections and line-search metric evaluations. Counts and measured wall time are both reported; equal helper count is not equal FLOPs. A validation-timing-selected ULA budget is included near fluid20 runtime. It is approximate timing matching, not a guaranteed equal deadline.

BF coefficients are selected on validation with a3x3 grid, separately for ULA/random; fluid reuses ULA BF coefficients and tunes3 position steps. Strong ULA90/180 and StaticRandom budgets are separately retuned on validation. No test performance selects these coefficients. Positions use zero momentum and fixed-error Armijo backtracking; a surrogate improvement is not monotonic improvement after an independently reoptimized adversary. Long runs can lose utility, and those results remain saved.

Timing uses one CPU thread,complex64/float32,batch1,three warmups and first15 paired test scenarios, with raw times,median/p95. It includes gradients,field reconstruction,projection/backtracking and static learned-parameter access. Dataset construction and initialization are excluded. Independent attacks are timed separately. No antenna movement,RF programming,CSI acquisition,energy,mechanical feasibility or hardware-realistic latency is measured.

## Learned-position result: small schedules and equal-extra-training control

There is no compatible pretrained original single-user/spatial checkpoint. New position-only models retain the original physical MinNet/MaxNet structure with frozen validation-tuned **update coefficients**; the beamformer/error iterates still update. We train either one shared step or20 layer-specific positive steps, with position momentum0. No larger NN, EpsNet,RecNet or joint BF fine-tuning is introduced. Models are trained separately per regime; this is not one-model transfer across regimes.

Both learned variants start from the same validation-tuned no-backtracking constant step and use identical paired batch orders,64 training channels,four Adam epochs,lr.002,batch32,eight updates,three seeds. Each online solve has20 macros/60 full calls and20 position projections. Full derivatives pass through spatial responses,analytic gradient updates and PAV’s exact active-face Jacobian; no straight-through geometry derivative is used. The terminal10-step training attacker is detached, an approximate surrogate rather than an exact envelope/global-minimum derivative. Validation uses independent50-step search with initialization eligible. Actual training histories,optimizer/RNG states,source/data hashes and selected epochs are retained.

| Case | Fixed tuned plain step | Learned scalar | Learned20-step schedule | Schedule minus scalar | Selected epochs scalar / schedule |
|---|---:|---:|---:|---:|---|
| A | 3.527481 | 3.527486 | 3.527486 | -0.00000000 | [4, 4, 4] / [4, 4, 4] |
| B | 3.236578 | 3.241049 | 3.241042 | -0.00000716 | [4, 4, 4] / [4, 4, 4] |
| C | 2.830243 | 2.829061 | 2.828979 | -0.00008139 | [4, 0, 0] / [0, 3, 0] |

**Decision:** a20-parameter schedule is not justified by this pilot. In B, the small gain over a coarse fixed-step grid is reproduced by the equally trained single scalar. In C, several runs select the untrained initialization, and trained selections can worsen test utility. All seeds, including losses, remain in the averages. Learned20-step designs also do not recover the higher utility of longer classical fluid solves. Faster no-backtracking inference must not be credited to learning when the fixed no-backtracking control has the same overhead. This does not rule out all future learning methods; it does not establish learned acceleration here.

Channel intervals average paired differences over the three trained runs before resampling100 channel scenarios. Per-seed means/SD are reported separately;300 model/channel pairs are not300 independent test scenarios. Training wall times include epoch validation but exclude initial validation/grid setup; inference comparisons do not hide that offline cost.

## Validation, plots and limitations

The tests cover steering recovery,field repeatability,independent NumPy signal equations,transpose/conjugation/physical units,float64 position finite differences,projection against SciPy convex QPs and exhaustive tiny partitions,PAV first/second derivatives away from kinks,full-layer feasibility,disabled-position equivalence,frozen variables,zero-channel handling,short-unroll parameter finite differences and read-only original checkpoint regression. The initial disabled-position mismatch was a coefficient-initialization precision issue; matching the shared PGDA buffer convention fixed it without relaxing the equality test.

The independent attacker freezes geometry and beamformers, searches only the original separate error sets, includes zero/terminal/five random interior/boundary starts,50 Adam steps and nested50/200/1000 PGD. Best candidates are retained and every method receives a shared feasible candidate pool evaluated at its own frozen positions. Raw budget curves,complex states,positions,per-scenario metrics/residuals and independent NumPy checks are saved. Lowest utility found is an upper bound on the true minimum, not certified robust performance.

Position and beampattern diagnostics use scenario index0 fixed before inspection; they illustrate rather than prove the average gain. Patterns share the same Pt normalization (power/Pt), not each curve’s peak. Tradeoff results at rho.2,.5,.8 use the same100 spatial scenarios with validation retuning at each rho and matched60 gradient calls. They are observed tradeoffs, not a globally optimized Pareto frontier or measured sensing detection accuracy.

This is an idealized known-field-response experiment. Coupling,position errors,channel aging,movement cost/calibration overhead and hardware latency are not modeled. The original constant-radius uncertainty assumptions are retained rather than statistically calibrated. The angular/gain distributions and optimizer choices condition the results. More apertures/path laws,initializations,stronger BF optimization and independent larger studies are needed before broad/publishable claims. No novelty is claimed.

## Reproduce

Figures: [gradient budgets](outputs/pilot/utility_vs_gradient_budget.png), [runtime](outputs/pilot/utility_vs_runtime.png), [positions/common-reference patterns](outputs/pilot/positions_and_common_reference_patterns.png), [all100 geometries](outputs/pilot/all_scenario_positions.png), [tradeoff](outputs/pilot/communication_sensing_tradeoff.png), [attack saturation](outputs/pilot/attack_budget_saturation.png).

Independent evidence replays every saved design using the final source with zero maximum difference and verifies all767 pre-existing tracked files are unchanged.

See README.md for exact commands,configs,dependencies and the isolated STATUS.md. Primary raw data are outputs/pilot/{scenarios.pt,data_manifest.json,A/,B/,C/}; learning checkpoints are under outputs/pilot/learning/; tradeoff outputs under outputs/pilot/tradeoff/. Evidence verifies every pinned pre-existing file remains byte-identical.
