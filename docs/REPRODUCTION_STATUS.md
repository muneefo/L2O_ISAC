# Reproduction status after reading the complete paper

The scientific core is an independent reconstruction. It is not the authors' code,
and numerical agreement with every figure/table has not been established.

## Corrected implementation choices

| Item | Original reconstruction | Current implementation | Remaining ambiguity |
|---|---|---|---|
| EpsNet input | All users/paths concatenated | Shared per-channel heads by default; legacy layout retained | Fig. 1 depicts h_m, but parameter sharing and aggregation to a common radius are not specified. We use an explicitly documented arithmetic mean. |
| EpsNet loss | sqrt of batch-global MSE | Mean of Eq. (18)'s per-sample three-bound RMSE | Batch aggregation is not explicitly written in (18); averaging the sample loss follows its per-sample expression. |
| Uncertainty experiments | Optional grid differed from Table I | Six exact Table-I epsilon values available | Authors' training-label/data-generation protocol remains unspecified. |
| Counts/scalability | Scenario sweeps only | Scenario sweeps plus a distinct frozen-weight transfer audit | Fig. 9 tests counts; it does not explicitly identify shared weights or retraining. |
| Timing | Saved batched throughput | Separate warm batch-one timing with raw samples, plus published reference data | The paper's complete timing protocol is unspecified; neither timing measures the entire RF pipeline. |

The common-bound pooling operation is not presented as an additional recovered
author detail. It makes the implemented tensor contract explicit and testable.
Changing the input layout requires training new EpsNet weights. Historical
checkpoints load automatically with the concatenated architecture and are not
silently reinterpreted as corrected models.

## What the paper already does

- Fig. 9 evaluates M=T=2, 4, 6; Fig. 3/Table II evaluate N=16 and 32.
- Section IV-C/Fig. 4 demonstrate extending 15 trained layers to 20/25 at inference.
- Section V-F/Fig. 8 study the communication–sensing trade-off over rho=0.1…0.9.
- Section II-D accounts for estimation/quantization/calibration/array-shape errors
  through separate norm-bounded channel and steering-vector uncertainty sets.
- Table I and Fig. 6 study multiple uncertainty levels and bound miscalibration.
- Section IV-E discusses convergence and claims stationarity.
- Table II reports millisecond runtimes, reproduced separately as published
  reference values in `configs/paper_reference.json`.

Our tests and measurements do not establish that the authors' implementation had
negative learned steps, constant-only labels, or the original concatenated layout.

## What is still a reconstruction assumption

Angle sampling/separation, initial-error distribution, prior-iterate initialization,
per-channel sharing/aggregation, exact EpsNet input assembly, EpsNet label distribution,
and validation checkpoint selection are explicitly recorded choices. Figure 2's
staged training procedure is followed; V-A also uses joint/end-to-end wording.
The paper does not fully resolve every implementation detail between those descriptions.

The SCA, DNN, and CNN comparator methods remain unimplemented. V-H gives the DNN
widths and parts of the CNN architecture, but complete baseline training targets/
losses and SCA reformulations are not supplied. We do not label the proposed-method
implementation as a full reproduction of every comparator and published figure.

## Measured checks

`l2o_isac.paper_checks` loads one checkpoint, keeps its weights fixed, and:

1. Evaluates EpsNet at the exact six Table-I uncertainty levels.
2. Tests the Fig.-9 user/target counts, with independent attacks and constraint checks.
3. Records warm batch-one latency, both solver-only and with initialization/bounds.

The frozen-weight protocol is an added audit, not an assertion about how Fig. 9
was generated. The epsilon audit does not secretly retrain on test labels. If a
nominal generator contains no information about independently changed radius
labels, the report must show that limitation, rather than force agreement with Table I.

The source paper and detailed page-by-page evidence are cited in `PAPER_READING.md`.

## Source snapshots and actual corrected execution

The full-size corrected run at `reports/corrected_per_channel_seed73129` records
the source hashes corresponding to commit `b6a289d`. It trained on 5000/1000/1000
channels for 10 epochs and saved new weights, logs and raw complex beamformers.
Later changes to report formatting and sweep defaults are not changes to the
numerical training code used in that run. Historical concatenated runs originate
from commit `4425f7a`. Execution-manifest hashes refer to their source snapshot;
they are not expected to match every later repository revision.

The primary CLI implementation is still an independent reconstruction: the new
run must not be described as exact numerical reproduction of Tables I or II.
