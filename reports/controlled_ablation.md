# Controlled training and computation audit

This study continues the completed independent reproduction and adversarial audit; the original checkpoint and prior outputs remain unchanged. It distinguishes extra training, positive inner steps and separated roles. The source PDF is **not present** in the accessible workspace/attachments; `/workspace/reports/paper_extracted.txt` and earlier source records exist. Reattachment is requested. No new paper-equation verification or verified author-code claim is made.

## Preserved evidence and design

Pre-study source `73159d4`; immutable full archive `/workspace/evidence_archives/pre_controls_73159d4.zip` is hashed in results/controlled_ablation/preservation.json. Prior saved values were verified: terminal4.66515, independent reference4.03977, earlier separated4.09496, tunedPGDA1804.98123. These concern our implementation. Earlier three fine-tuning seeds share one pretrained reference.

Now use **three independently pretrained corrected-reference seeds73129,74129,75129**, each with paired fine-tuning seeds101,202,303. New references reuse the unchanged established `l2o_isac.experiment --profile paper` pipeline,5000/1000/1000 channels and10epochs, in fresh directories. Earlier checkpoints were inventoried; remaining ones use legacy EpsNet layouts, different depths, parameterization or array dimensions and are not silently counted as matched corrected references.

The physical problem and saved audit splits remain fixed: Nt=Nr16,M4,T2,J2,L8,rho.8,SNR5dB,oracle radii.05;256 train/64 validation/128 test channels. Same channel tensors, per-scenario initial states, paired epoch permutations, complex64, three continuation epochs, SGD lr1e-4 with decay.98. Per-case verification asserts identical C/D mapped initial tensors and B/C/D batch-order hashes. No CRB/controller or physical-model change.

A=frozen reference. B=original joint negative layer-average utility loss. C=same joint loss with direct inner steps constrained to[1e-6,infinity) by post-update projection. D=separate inner/outer training with **exactly C’s parameterization and outer layer-average loss**, same LR and unchanged momentum treatment. Only inner training has its adversarial objective; inner-policy parameters are frozen in the outer phase, retaining gradients through its iterates. Mapping flips negative inner steps to their absolute magnitudes, floors exact zeros, and leaves all momentum/outer parameters unchanged. Mapping-only performance is evaluated separately before any weight training. All conditions use the same independent validation-selection protocol, including initialization. This is a controlled continuation under oracle radii and saved initial states, not a claim of replaying the authors’ unavailable training.

## Main controlled results

| Condition | Nominal utility | Communication | Sensing MI upper-bound metric | Independently attacked shared-union utility |
|---|---:|---:|---:|---:|
| A | 4.620290 | 4.873517 | 0.704253 | 4.039664 |
| B | 4.638396 | 4.875548 | 0.757362 | 4.051911 |
| C | 4.645097 | 4.889360 | 0.758980 | 4.063284 |
| D | 4.639626 | 4.893629 | 0.736904 | 4.062284 |

| Paired contrast | Weighted gain | Per-pretrained-model mean gains | Crossed hierarchical95% interval |
|---|---:|---|---|
| B_minus_A | 0.012247 | 0.012453, 0.012169, 0.012119 | [0.010075, 0.014617] |
| C_minus_B | 0.011373 | 0.011342, 0.011510, 0.011267 | [0.010078, 0.012783] |
| D_minus_C | -0.001000 | -0.000958, -0.000972, -0.001070 | [-0.002123, 0.000032] |
| D_minus_A | 0.022620 | 0.022838, 0.022707, 0.022316 | [0.020339, 0.025043] |

Role separation does not outperform the positive-step equal-extra-training control on average in this experiment. The paired interval includes zero; this does not establish a reliable improvement. The gain/interval and per-pretraining results above determine its strength; no positive narrative is forced. These are exploratory results with only three pretrained models and128 shared scenarios. Bootstrap reports separate conditional channel, pretraining-only and crossed intervals in summary.json; fine-tuning variability within each pretrained model is also saved. Thousands of channel evaluations are not independent pretrained models.

| Pretrained seed | A | B | C | D | Mapping-only |
|---|---:|---:|---:|---:|---:|
| 73129 | 4.039765 | 4.052218 | 4.063560 | 4.062603 | 4.050271 |
| 74129 | 4.040508 | 4.052677 | 4.064187 | 4.063215 | 4.051085 |
| 75129 | 4.038719 | 4.050838 | 4.062105 | 4.061035 | 4.049109 |

## Training computation and role verification

Separate validation profiling of the loss-matched D cycle determines the fractional joint repeat count saved in compute_matching_clean.json. Primary B/C use that deterministic mix of adjacent integer repeat counts; D makes one inner and one outer update. The earlier integer repeats, terminal-adversary and terminal-budget-balanced variants remain supplemental outputs. This matches compute approximately, not optimizer update counts. All batches/permutations/LRs are shared; repeats reuse the same minibatch. Actual measured time and distinct derivative counts are reported rather than called exactly equal.

| Condition | Mean training update seconds | Full analytic helper calls | Error-only gradient calls | Differentiated layers | Differentiated full calls | Backward passes | Optimizer calls | Inner/outer block updates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B | 3.649 | 1230 | 0 | 615 | 1230 | 41 | 41 | 41/41 |
| C | 3.550 | 1230 | 0 | 615 | 1230 | 41 | 41 | 41/41 |
| D | 3.569 | 1800 | 0 | 720 | 1080 | 48 | 48 | 24/24 |

One full analytic helper call computes all six physical variable gradients; each alternating layer calls it twice. Fifteen online layers cost30 calls for every learned condition. Independent attackers use three-error autograd derivatives, counted separately. Differentiated inner-only layers and alternating layers have different costs; higher derivatives and projections are included in wall time. Raw per-minibatch records, optimizer states, RNG states and source/data hashes are saved. Training-update time excludes common validation/model selection, whose cost is recorded separately. CPU concurrency can affect wall times; derivative counts remain explicit.

D’s inner training freezes A/deployed beamformers and differentiates its complete15-step search. Outer training freezes attack-policy parameters and retains full derivatives through deployed adversarial iterates, using exactly C’s negative layer-average utility loss. It introduces no new terminal training attacker. This finite-unroll gradient is not an exact envelope gradient through a solved global minimum. The earlier terminal-adversary outer loss is retained as a distinct supplementary variant. Tests verify intended parameter blocks and independent NumPy whole-search utility, not merely step signs.

On identical frozen A validation beams, mapped-initial and trained D policy utilities are independently recomputed with NumPy, then refined using independent search. Complete tables are in summary.json. A lowered policy-candidate utility indicates stronger learned search. With beams fixed, the true minimum does not change; independent refinement can therefore return the same value before/after training. Negative steps and uphill stages alone are not sufficient proof of failure.

## Whole inner-search validation (same frozen A beams)

| Pretraining seed | Mapping-only policy | Positive joint policy C | Separated policy D | Independent refinement of D |
|---|---:|---:|---:|---:|
| 73129 | 4.409606 | 4.428167 | 4.375435 | 3.896123 |
| 74129 | 4.410337 | 4.428920 | 4.376136 | 3.896851 |
| 75129 | 4.408014 | 4.426561 | 4.373917 | 3.894716 |

These complete15-step policy outputs are measured independently with NumPy. Stronger learned minimization on fixed beams need not improve the final independently attacked beamformer design. A candidate value and its independent refinement are different quantities; neither is a certified minimum.

## Classical PGDA at several computation budgets

Each budget is tuned on24 saved validation scenarios using the same3×3 inner/outer grid and30-step independent validation search:63 configurations total. Test results never choose steps. Generation and adversarial-evaluation time are separated.

| PGDA iterations | Full analytic calls | Comm | Sensing MI upper bound | Lowest-found utility | Batch-one median/p95ms |
|---|---:|---:|---:|---:|---|
| 5 | 10 | 3.965494 | 0.710428 | 3.314481 | 5.81/6.65 |
| 10 | 20 | 4.199645 | 1.088400 | 3.577396 | 12.43/13.91 |
| 15 | 30 | 4.663906 | 0.699788 | 3.871083 | 18.26/19.06 |
| 20 | 40 | 4.902650 | 0.818219 | 4.085763 | 24.51/28.50 |
| 30 | 60 | 5.074924 | 1.067901 | 4.273519 | 34.80/36.22 |
| 60 | 120 | 5.157726 | 1.913478 | 4.508877 | 69.92/72.02 |
| 180 | 360 | 4.822395 | 5.616575 | 4.981232 | 211.02/216.63 |

A:30 online full calls, median17.83ms,p9518.74ms.

B_clean_s101:30 online full calls, median17.26ms,p9517.94ms.

C_clean_s101:30 online full calls, median17.09ms,p9517.87ms.

D_clean_s101:30 online full calls, median17.00ms,p9518.10ms.

These are warm CPU generation measurements: AMD EPYC9V74 virtual CPU, quota two CPUs, one PyTorch thread, complex64,batch1,five warmups,20 repeats. No GPU synchronization is applicable; no GPU was used. Static scalar schedules have no neural-controller overhead. Initialization and EpsNet are timed separately; RF/CSI acquisition is excluded. No acceleration is claimed solely against180 iterations; nearby15/20-step points and the full utility/runtime/gradient curves are shown.

## Independent evaluation, feasibility and limits

Every method uses frozen beamformers, common balls, zero/terminal/five random interior/boundary starts,50 Adam steps and nested200/1,000-step PGD, retaining best candidates and a shared union across all methods/pretrained models. The original independent attacker is preserved. Folding starts into a batch was validated against it (exact component agreement on saved validation cases and complex128 equivalence tests) without changing seeds, domains or schedules. Logical per-start gradient counts and actual batched tensor calls are distinct. There are0 primary numerical/feasibility failures, maximum primary residual2.15e-06 at tolerance1e-5; raw flags/residuals are retained. Failures would remain in accounting rather than be dropped.

Results are lowest utility found, not certified global worst cases. Attack saturation from200→1,000 is saved and plotted; small changes do not prove global optimality. Nominal/terminal/attacked components, full complex designs/errors, per-scenario results and candidate unions are retained. Sensing improvements refer only to the MI upper-bound metric, not measured detection or estimation accuracy.

The supplementary terminal-robust outer-loss variant has per-pretraining mean utilities {'73129': 4.0755917231241865, '74129': 4.0763045946757, '75129': 4.075026671091716}. It changes the outer objective and adds training-time independent search, so it is not the pure role-separation contrast. E_previous is the prior unconstrained separated method, with inner lr1e-3 and one pretrained reference. It remains a separate additional condition; its earlier1.37% gain must not be attributed solely to D’s role contrast. All observed mapping, extra-training and sign effects remain explicit. Optional controllers, physical extensions, broad dimension/channel transfer and publication novelty are outside this experiment. This is a sequential exploratory study on a previously used fixed audit test; recipes were frozen before their own comparisons and selection used validation only. Fresh confirmatory scenarios and larger budgets are still needed for publication claims.

Executable commands: [CONTROLLED_COMMANDS.md](../docs/CONTROLLED_COMMANDS.md). Protocol: [CONTROLLED_PROTOCOL.md](../docs/CONTROLLED_PROTOCOL.md). Raw outputs and plots: results/controlled_ablation/. Reattach the source PDF for direct source verification; implementation experiments proceeded without assuming access.
