# Source and implementation map (audit Gate A)

Source: Lei Wang, Sergiy A. Vorobyov, Esa Ollila, **L2O Robust Hybrid
Beamforming for ISAC**, IEEE TWC 25, 18778–18794 (2026),
https://doi.org/10.1109/TWC.2026.3702925. Identity agrees with the previously
verified PDF and its retained complete text in `/workspace/reports/paper_extracted.txt`.
The original attachment PDF is no longer present in this session. This session
re-read the retained formulation, Algorithms 1–3, training, Appendix and settings;
it does not claim a new direct PDF metadata verification. Previous page/figure
inspection is recorded in [PAPER_READING.md](PAPER_READING.md).

**Classification:** independent reimplementation, subsequently corrected interpretation;
not official author software and not exact numerical reproduction. No official code
or author checkpoint has been verified. Public repository:
https://github.com/muneefo/L2O_ISAC ; preserved reference commit
`83ba89148c8d89a7ab60105da0fa5cbfe6105d9e`. Reconstructed checkpoint
`reports/corrected_per_channel_seed73129/checkpoint.pt` was trained using source
`b6a289d`; its existing execution manifest records source and artifact hashes.
No repository LICENSE file is present at the pinned snapshot; open visibility
does not establish a reuse license. No license for unavailable author code is inferred. Reference files are never overwritten by this audit.

Shapes below omit leading batch dimension B. Q=T+J; K=M+T. All vectors/matrices
are complex; eps has three real dimension-normalized radii.

| Quantity / rule | Meaning / dimensions | Paper equation; printed page | Implementation | Existing/new validation |
|---|---|---|---|---|
| z_c,z_s; F_a,F_d,w | independent identity-covariance symbols M,T; N_t×L, L×K, T×N_r | (1), II-A;18780 | data.initialize_state; physics.components | literal signal equations; NumPy audit |
| h; SINR_c; gamma_c | M×N_t; one desired column, all other K columns interfere; sum natural log(1+SINR) | (2)–(4);18780 | physics.components/metrics | test_physics literal/autograd; test_architecture_audit NumPy SINRs |
| a_t,a_r; H_l | Q×N_t,Q×N_r; alpha_l a_r,l a_t,l^T, **transpose not Hermitian** | (5);18780 | data.ChannelBatch.radar; physics.components | literal radar matrix and gradients |
| SINR_s; gamma_s | T filter SINRs; all other targets and clutter interfere; sum-log MI **upper-bound metric** | (6),(7);18780–81 | physics.components/metrics | NumPy per-target SINRs, shared-error attacks |
| steering convention | exp(-j*pi*n*sin(theta)); unnormalized half-wave ULA | II-C;18780 | data.generate_channels | literal metrics, unequal dimensions |
| noise, alpha, power, SNR | noise 1; complex reflection variances 10^(2/10),10^(3/10); P_t/noise=10^(SNR_dB/10) | V-A;18786 | config.SystemConfig; data.generate_channels | saved config; zero-signal test |
| presumed + errors | actual h=hhat+dh; a_t=ahat_t+dt; a_r=ahat_r+dr | (8);18781 | physics.components | zero/perturbed metrics |
| A_h,A_t,A_r | separate per-user/path balls radii eps_h*sqrt(N_t), eps_t*sqrt(N_t), eps_r*sqrt(N_r) | (9);18781 | solver.project_ball/constraint_errors | inside/boundary/zero/idempotence; unequal Nt,Nr |
| shared sensing errors | all Q targets **and clutter** perturbed; same dt_l,dr_l used in every target SINR | (6),(8)–(10) | physics.components; attacks | frozen design, separable candidate retention |
| objective | rho*min gamma_c +(1-rho)*min gamma_s; separate independent sets | (10a);18781 | physics.metrics; attacks | componentwise minima may combine h with jointly retained (t,r) |
| hardware sets | abs(F_a)=1; norm(w_t)=1; norm(F_a F_d)^2=P_t | (10b–d);18781 | solver.project_beamformers | residual after every layer, zero fallback |
| MinNet | simultaneous inner gradient descent, momentum from prior **projected** errors, then ball projection | (11)–(13);18782 | solver.pgda_step | six gradient checks, short-unroll derivatives |
| MaxNet | gradients at new errors; simultaneous outer proposals, momentum, project a before normalizing d with new a; normalize w | (14)–(16);18782 | solver.pgda_step/project_beamformers | all-iterate feasibility, regression |
| complex derivative | df/dz*; PyTorch real loss complex gradient is twice this | (25)–(70);18790–92 | physics.gradients | literal autograd /2 and complex128 central differences |
| initialization | feasible random a,d,w and ball errors; previous=current so first momentum zero | Algorithm 1;18782 | data.initialize_state | repeatability; initialization convention is reconstruction choice |
| EpsNet | individual real/imag channel inputs; radar matrix flattened; BN/FC/SoftPlus; labels known by simulator design | (17),(18), Fig1;18783 | networks.EpsNet/eps_rmse | test_paper_alignment |
| Eps ambiguities | shared per-channel heads and arithmetic pooling are our choices; independent changing radius labels cannot be inferred from unchanged nominal input law | IV-A, V-A | networks; data.generate_channels | previous fixed-weight epsilon audit |
| U-PGDA parameters | L_u×6 mu and eta (180 scalars for 15 layers); same negative utility over samples and **all layers** | (19)–(23), Algorithm2;18784 | solver.UnrolledPGDA/unrolled_loss; training.train_unrolled | no detach in unroll; second-order/finite-difference tests |
| training inputs/graph | implementation evaluates presumed+current-error channels; EpsNet frozen, eps calculated under no_grad; gradients flow through inner/outer analytic gradients and projections | (22), Fig2;18784–85 | training.train_unrolled | graph audit in audit_hypotheses.md |
| RecNet | mu_i=beta*mu_(i-1), eta_i=phi*eta_(i-1); two shared scalars; freeze prefix, 10×10 grid [.01,.99], validate 5 extra layers | (24), Algorithm3;18784–86 | training.recurrent_grid_search; solver.UnrolledPGDA | cached/full grid equality; 20/25-layer inference |

Numerical policy: positive physical receiver noise is part of the model, not a
clamp. Projection division uses dtype `tiny` only to avoid division by zero;
exact-zero vectors use explicit feasible fallbacks. A zero receive vector is
unsuitable for direct metrics (0/0); project it first. Nonfinite solver/attack
values cause explicit failures. No SINR denominator floor is added. Complex128
reference feasibility tolerance is 1e-8, component agreement 1e-10, gradient
relative tolerance 1e-4 (usually much tighter). Complex64 feasibility uses 1e-5.
See REPRODUCTION_STATUS.md for unresolved author settings and omitted comparators.
