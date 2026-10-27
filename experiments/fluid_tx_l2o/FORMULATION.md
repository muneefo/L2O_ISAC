# Formulation and source mapping

Original source: Wang, Vorobyov, Ollila, TWC25:18778–18794 (2026), DOI10.1109/TWC.2026.3702925. Read from the retained original full-text extraction in sources/paper_preexisting_extract.txt; sections II–IV, Algorithm1/2, Eq22, experimental setup, and appendices A/B checked against the implementation. The institutional PDF link was identified but returned403; this limitation is recorded in STATUS, rather than claiming a newly inspected PDF or official code.

| Original formulation | Implementation reused unchanged | Specialization/addition |
|---|---|---|
| Eqs1–7 hybrid signal, communication SR, radar channel and sensing MI surrogate | l2o_isac/physics.py components/metrics | M=T=1; communication and dedicated sensing streams remain, sensing interferes at the user; target+J clutter at radar; no SIC |
| Eq8 additive channel/transmit/receive errors | physics.components adds errors before products | Full radar product (ar+dr)(at+dt)^T retains dr dt^T; no dropped bilinear error term |
| Eq9 radius normalization | solver.project_ball and shared attacker | each radius eps_h/t*sqrt(Nt) or eps_r*sqrt(Nr); common fixed eps=.05 |
| Eq10 separate worst communication and sensing weighted objective | metrics objective=rho*comm+(1-rho)*sense; independent attacks retain component minima separately | rho=.8 initially; no new QoS or CRB objective |
| Eqs11–13 MinNet uncertainty steps and momentum/projection | solver.pgda_step first stage | all three layerwise steps and momenta learned jointly as inherited |
| Eqs14–16 MaxNet beamformer/filter steps and feasibility | solver.pgda_step second stage | same simultaneous outer block gradients after inner update; analog phase projection, digital effective-power renormalization using updated analog matrix, filter normalization |
| Appendix A/B (25–70) complex gradients | physics.gradients | Wirtinger df/dz*, equal to half Torch's real-loss complex gradient; no added ln2 (original paper explicitly uses nats) |
| Eqs19–21 layerwise learned scalar schedules | fluid_tx_l2o.model FluidUnroll | six inherited mu and eta plus real position mu_p/eta_p per layer; 280 parameters at20 layers, fixed-array240; direct unconstrained parameterization retained |
| Eq22 negative layer/scenario-average utility, Algorithm2 | FluidUnroll.loss and run.Study.train_one | full differentiation through MinNet, MaxNet, positions, and PAV; SGD lr1e-4, exponentialdecay.98; no detached terminal surrogate replaces Eq22 |
| EpsNet/RecNet | not enabled | supplied common radii and fixed20layers; no count/depth transfer claim |

Communication field is hhat(p)^H=sum beta_k at(vartheta_k;p)^T. beta/angles are immutable per scenario. Shared ChannelBatch stores h, so the sum is conjugated before storage. This necessarily replaces the paper's position-independent CN(0,I) communication draws. Fresh ULA/fluid baselines use exactly the same new field response. Reflection coefficients retain zero-mean complex Gaussian target/clutter variances2/3dB. No original checkpoint is reused or retrained.

Positions are x=p/lambda in wavelength units, angles in radians from broadside, at_n=exp(-j2pi x_n sin(theta)), and the receive half-wave ULA is fixed. lambda=.01m converts physical units. A=3.5lambda, dmin=.25lambda, Nt8 leave movement freedom since3.5>7*.25. The halfwave transmit ULA spans the same allowed aperture. Illegal A<(Nt-1)dmin is rejected by the reused geometry implementation.

## Real position partial

For fixed errors and beamformers, d at_n/d x_n=-j2pi sin(theta) at_n. For stored h, d h_n/dx_n=conj(sum beta_k [-j2pi sin(vartheta_k)] at_kn). A physical-position derivative divides the x derivative by lambda.

Let gh=d gamma_c/dh* and gt_l=d gamma_s/dat_l* from the shared analytic gradients. Then

    du/dx_n = 2 Re[rho conj(gh_n) dh_n/dx_n
                  +(1-rho) sum_l conj(gt_ln) dat_ln/dx_n].

Equivalently, for N/D power SINR, d log(1+N/D)=(D dN-N dD)/(D(D+N)), with d|b|²=2Re(conj(b)db). This propagates through both streams, the full radar product, target and clutter denominator. Additive uncertainty vectors are fixed in this partial, not re-estimated or differentiated as explicit functions of x. Training does retain all dependencies between layer states and learned coefficients when differentiating the assembled unroll. The partial is not the exact derivative of an unknown globally minimized robust value.

## Layer ordering

1. Rebuild hhat(x), athat(x), fixed arhat.
2. Evaluate all six gradient blocks; simultaneously descend the three error blocks and project their balls, using current and previous error states.
3. Reevaluate all six gradients at the new errors and old beamformers; simultaneously ascend Fa/Fd/w with momentum, then project/normalize.
4. Evaluate the new position partial at the updated beamformers/errors and old positions; ascend positions plus optional position momentum, then exact bounded-isotonic projection.
5. Reconstruct spatial responses at the new positions for the layer utility and next layer.

The position-disabled mode is exactly the existing UnrolledPGDA on fixed spatial channels. Zero position momentum is the classical default and a trained ablation. A full helper call computes six variable-block analytic gradients; fixed layers use2, fluid3. Equal calls are not equal FLOPs and all online initialization/field/projection/metric work is included in timing. Learned coefficient loading is offline setup; forward coefficient access is online. Attacks, training, movement/RF settling, and CSI acquisition are separated.

## Feasibility and assumptions

Position projection subtracts (n-1)dmin, performs bounded isotonic regression on [0,A-(Nt-1)dmin], then restores offsets. PAV's piecewise-exact pooled-block Jacobian is used, not STE. Analog zero proposals map to+1; zero effective-power proposals use the existing deterministic feasible fallback; zero receive filters also use its fallback.

The inherited sensing model sums effective clutter powers; it assumes negligible echo cross-correlations and is not a coherent-clutter model. The communication stream is counted in sensing illumination; the dedicated sensing stream is interference at the user. Both metrics use natural logarithms. Continuous phase, ideal accurate positioning, negligible coupling and fixed path gains/angles within the block remain assumptions.

## Documented differences and limits

Training uses smaller new spatial sets128/32/100 per regime, batch32,10epochs, three paired shuffle seeds rather than the original5000/1000/1000 multiuser distribution. Scalar initialization is selected by independent validation attacks on the new channel law, with paired untrained controls; no old model transfer. Validation checkpoint selection retains inherited Eq22 loss (initialization eligible); independently attacked validation is logged diagnostically, not used to redefine the loss silently. Direct learned inner steps can become negative; signs and terminal error utility must not substitute for independent frozen-design robustness evaluation. We report numerical failures if any, rather than remove unsuccessful models.
