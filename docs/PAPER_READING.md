# Paper-grounded reading and corrections

Source: Lei Wang, Sergiy A. Vorobyov, and Esa Ollila, “L2O Robust Hybrid Beamforming
for ISAC,” IEEE Transactions on Wireless Communications, vol. 25, pp. 18778–18794,
2026. DOI: https://doi.org/10.1109/TWC.2026.3702925.

This note is based on a fresh reading of the supplied 17-page PDF, including the
technical body, appendices, architecture figures, numerical figures, and tables.
It does not use our implementation to establish what the authors assumed or tested.
Printed page numbers below are the IEEE page numbers; PDF page = printed page − 18777.

## Changing user counts and configurations

| Finding | Paper location | Evidence and interpretation |
|---|---|---|
| Different user/target counts are explicitly evaluated. | Section V-G, Fig. 9, p. 18788 (PDF 11). | Cases M=T=2, M=T=4, M=T=6, with Nt=Nr=16. The authors describe this as scalability with system dimensionality. |
| Different array sizes are explicitly evaluated. | Section V-A/B, Fig. 3, p. 18786; Table II, p. 18789. | Nt=Nr=16 and Nt=Nr=32. |
| The default RF-chain count is L=8. | Section V-A, p. 18786. | No corresponding RF-chain-count transfer experiment is specified in the numerical evaluation. |
| RecNet extends inference depth without retraining the U-PGDA prefix. | Section IV-C, Eq. (24), p. 18784; Section V-C, Fig. 4, p. 18787. | A 15-layer prefix is extended to 20 and 25 inference layers. The no-retraining claim is explicitly explained in this variable-depth context. |
| One common set of trained weights across all user/array configurations is not explicitly demonstrated. | Section V-G and associated training descriptions. | The paper does not state a train-on-one-count/test-on-other-count protocol or identify shared checkpoints across Fig. 9 cases. This is unspecified, not proof that separate retraining was performed. |
| The learned U-PGDA coefficients do not inherently grow with user count. | Eqs. (19)–(21), p. 18784. | Six step sizes and six momentum coefficients per layer. Dimension compatibility follows from the formulation, but transfer quality is an empirical question. |
| Online user arrivals/departures and temporal tracking are not evaluated. | System model, numerical evaluation, conclusion. | No explicit time evolution, channel-aging process, arrival/departure sequence, or deadline-aware online adaptation experiment is supplied. New channel inputs are supported; that differs from a demonstrated temporal tracking policy. |

Correct description: the authors evaluate multiple system sizes and demonstrate
variable-depth inference. They do not explicitly establish that one complete
trained EpsNet/U-PGDA system transfers across changing user/array configurations
without retraining. It would be inaccurate to say simply that the method cannot
handle different user counts.

## EpsNet: correction to an earlier architectural inference

Figure 1(a), p. 18783 (PDF 6), depicts an individual channel h_m as input, split
into real and imaginary components. Section IV-A describes the communication
channel and flattened sensing channel, with BN, FC, and SoftPlus layers. Footnote 6
on p. 18784 says that only the epsilon_h, delta_h,m, and Fa architectures are drawn
for brevity, with the other branches described as similar.

Consequently, the paper does NOT explicitly say to concatenate all M communication
users into one fixed-length input. That was a documented implementation choice in
our reconstruction. It cannot be used to prove that the authors' EpsNet input must
change when M changes. Applying a channel encoder across users is compatible with
the schematic, although the exact sharing/aggregation protocol is not specified.

The depicted FC architecture does depend on the lengths of the channel vector or
flattened channel matrix supplied to it. No explicit variable-antenna encoder,
padding/masking mechanism, or cross-array weight-sharing protocol is described.

Section IV-A2 explicitly says that ground-truth bounds are available by deliberate
simulation design. Section V-D, Fig. 5, and Table I, p. 18787, evaluate uncertainty
levels 0.001, 0.01, 0.02, 0.03, 0.04, and 0.05. Figure 6 evaluates under/overestimation
by scaling the uncertainty bound. The default value 0.05 in V-A must not be used
to assert that all the authors' EpsNet training/evaluation labels are constant.
The exact bound-generation/training distribution is not fully specified.

## Authors' reported runtime: Table II

Table II, p. 18789 (PDF 12), explicitly labels runtime in milliseconds:

| Method | Training/testing depth | N=16 runtime (ms) | N=32 runtime (ms) |
|---|---|---:|---:|
| U-PGDA | 20 / 20 | 9.68 | 13.47 |
| U-PGDA | 15 / 15 | 7.56 | 12.12 |
| RU-PGDA | 15 / 20 | 9.98 | 15.44 |
| RU-PGDA | 15 / 25 | 12.64 | 18.51 |
| B-PGDA | no training / 180 iterations | 40.03 | 94.31 |

Footnote 8 specifies an Intel Core Ultra 7 155H processor and 32 GB RAM.
Section IV-D states that inference requires neither learning nor grid search.

These are the authors' reported values, not measurements of our code. The table
and surrounding text do not specify a complete batch-size-one timing protocol,
warm-up methodology, latency percentiles, or inclusion of acquisition, transfer,
and hardware-programming delays. Batch size 250 is specified for training in V-A;
it must not be assumed to define the timing protocol.

The paper therefore already reports millisecond computation. A new contribution
cannot be defined merely as “achieving milliseconds.” A tighter latency target,
online configuration transfer, or deadline/reliability requirement would need to
be defined and demonstrated.

## System model and practical assumptions

| Topic | What the paper actually specifies |
|---|---|
| Architecture | Section II-A, Eq. (1), p. 18780: common analog transmit matrix Fa for communication and sensing, with separate digital matrices combined as Fd. |
| Users, targets, clutter | Section II: single-antenna users, point-like targets, and interference sources; footnote 1 identifies the interference as environmental clutter. |
| Waveform statistics | Section II-A: communication symbols CN(0,I); sensing streams zero-mean and mutually uncorrelated with identity covariance; communication and sensing symbols uncorrelated. |
| Arrays | Section II-C: transmit and receive ULAs with the stated plane-wave steering vectors. The optimization is a spatial frequency-flat model; explicit delay/Doppler waveform processing is not developed. |
| Interference | Eq. (3) includes other communication streams and sensing streams at a user. Eq. (6) includes other targets and clutter at the sensing receiver. It is incorrect to say the paper ignores interference. |
| Cancellation | Eqs. (2)–(7) contain no explicit successive-decoding/cancellation stages, residual self-interference channel, analog/digital canceller, or clipping process. This is an omission from the modeled processing chain; the text does not explicitly establish an assumption of perfect SIC. |
| Hardware uncertainty | Section II-D, Eqs. (8)–(9), p. 18781, explicitly motivates steering errors using pointing, calibration, and distorted-array effects, and channel errors using estimation/quantization. It is incorrect to say hardware imperfections are wholly absent. |
| Unified uncertainty | Section II-D explicitly distinguishes a common mathematical treatment from identical physical origins; channel, transmit-steering, and receive-steering sets have separate radii. |
| Detailed hardware | No explicit PA input/output law, ADC saturation/quantization chain, oscillator phase-noise process, or discrete phase-shifter codebook is introduced in the proposed model. Section II-E explicitly characterizes the analog constraint as continuous rather than discrete. |
| Communication–sensing balance | Eq. (10a) optimizes a weighted composition of worst-case sum rate and sensing MI. Section V-F/Fig. 8 explicitly evaluates rho from 0.1 to 0.9. It is incorrect to say the paper does not investigate the trade-off. |
| Service guarantees | Eq. (10b)–(10d) imposes unit-modulus analog weights, unit-norm receive filters, and exact total transmit power. It does not impose per-user minimum rate or per-target detection/estimation thresholds. |
| Sensing surrogate | Section II-C explicitly calls the sensing metric an upper bound and discusses tightness under cross-target decorrelation. It also discusses a small cross-correlation threshold, but that threshold is not explicitly present among Eq. (10)'s listed constraints. |
| Simulation channels | Section V-A draws presumed communication channels from CN(0,I), and target/clutter reflection coefficients from complex Gaussian distributions with the stated variances. This is the specified numerical setup, not a restriction on every channel to which the algebra could be applied. |

## Optimization, training, and convergence

- Algorithm 1 and Eqs. (11)–(16), p. 18782, explicitly implement alternating
  inner minimization, outer maximization, momentum, and projections.
- Figure 1 and Eqs. (19)–(23) specify learned step sizes/momentum and a negative
  weighted-objective loss averaged over layers and data.
- The paper does not explicitly state positivity/bounding transforms for these
  learned coefficients in those equations. SoftPlus is explicitly described for
  EpsNet. This does not establish that the authors' actual trained coefficients
  became negative; that observation belongs only to our reconstruction.
- Figure 2 and Section IV-D describe three stages: train EpsNet; fix its output
  and train U-PGDA; freeze U-PGDA and grid-search recurrence parameters.
  Section V-A also uses cascaded joint/end-to-end training wording. Exact
  implementation details beyond the staged description should not be invented.
- Eq. (24) uses two shared recurrence factors for all six step sizes and all six
  momentum coefficients. Footnote 7 explicitly identifies more fine-grained
  updates as a possible alternative with greater complexity.
- Section IV-E, p. 18785, DOES contain a convergence discussion: the authors
  assert monotonic behavior and convergence to a KKT stationary point, citing
  reference [47]. It would be incorrect to say they provide no convergence
  analysis. Whether the cited theorem covers the exact projected, momentum,
  learned-parameter method requires a separate mathematical applicability audit.
- Footnote 4 explains the fixed-depth stopping issue for unrolled networks;
  Algorithm 1 uses a prescribed iteration count. Table II uses 180 benchmark
  iterations. The tolerance 1e-6 and 2000-iteration cap in V-H describe the SCA
  comparator, not a demonstrated stopping rule for U-PGDA.
- Section VI discusses unrolling/DRL combinations for adaptive robustness under
  dynamic environments as future work. This does not negate the adaptability
  to new channel inputs that the current inference pipeline already supports.

## Correct research-gap wording

“The paper demonstrates robust hybrid beamforming across several system sizes,
variable-depth inference, a communication–sensing trade-off, and millisecond
reported runtimes. It leaves an explicit single-model, no-retraining transfer
protocol across changing configurations, temporal adaptation with arrivals and
departures, and detailed hardware/cancellation constraints unestablished.”

Evidence from our implementation must be labeled separately. It cannot be used
to attribute unstated architectural choices, timing protocols, or observed
failure modes to the authors.
