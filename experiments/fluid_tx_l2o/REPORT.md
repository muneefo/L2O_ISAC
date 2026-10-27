# Coupled fluid-transmit L2O: measured study

Independent shared-union attack estimates; not certified minima. Seeds are paired training shuffles, not additional independent test channels.

## Main comparisons

|Case/method|Comm|Sensing|Utility|Full calls|Median ms|
|---|---:|---:|---:|---:|---:|
|A/ULA_G60|3.170068|4.862593|3.508573|60|35.75|
|A/Fluid_G60|3.170720|4.863260|3.509228|60|50.83|
|A/FixedL2O_s401|3.170615|4.863153|3.509122|40|25.50|
|A/FluidL2O_s401|3.170740|4.863281|3.509248|60|35.75|
|A/FluidUntrained|3.170720|4.863260|3.509228|60|38.30|
|A/FluidNoPositionMomentum_s401|3.170740|4.863281|3.509248|60|37.20|
|A/TrainOptimizedFixed|3.170569|4.863106|3.509077|40|25.46|
|A/StaticRandom|3.170577|4.863114|3.509084|40|25.97|
|A/ULA_G360|3.155485|4.847670|3.493922|360|221.29|
|A/Fluid_G360|3.165584|4.857999|3.504067|360|283.30|
|B/ULA_G60|2.705706|3.543994|2.873363|60|35.82|
|B/Fluid_G60|3.015493|3.738879|3.160171|60|51.55|
|B/FixedL2O_s401|2.713721|3.488211|2.868619|40|26.96|
|B/FluidL2O_s401|3.014066|3.795704|3.170394|60|37.21|
|B/FluidUntrained|3.015493|3.738879|3.160171|60|40.12|
|B/FluidNoPositionMomentum_s401|3.014066|3.795704|3.170394|60|39.57|
|B/TrainOptimizedFixed|2.720235|3.457038|2.867596|40|27.29|
|B/StaticRandom|2.755627|3.731943|2.950890|40|25.38|
|B/ULA_G360|2.697826|3.589107|2.876082|360|229.20|
|B/Fluid_G360|3.001702|4.234139|3.248190|360|301.52|
|C/ULA_G60|2.619456|2.449508|2.585466|60|37.59|
|C/Fluid_G60|2.856447|2.693524|2.823862|60|50.78|
|C/FixedL2O_s401|2.642470|2.289834|2.571943|40|25.89|
|C/FluidL2O_s401|2.858735|2.615977|2.810184|60|38.70|
|C/FluidUntrained|2.858735|2.615977|2.810184|60|39.17|
|C/FluidNoPositionMomentum_s401|2.858735|2.615977|2.810184|60|39.27|
|C/TrainOptimizedFixed|2.634442|2.309251|2.569404|40|26.88|
|C/StaticRandom|2.666863|2.414639|2.616418|40|26.16|
|C/ULA_G360|2.545493|3.098078|2.656010|360|241.11|
|C/Fluid_G360|2.825101|3.351691|2.930420|360|298.91|

## Paired channel differences

L2O contrasts average the three trained runs before resampling100 paired scenarios. These intervals condition on this training sample and the three fitted models.

- A/FluidL2O_minus_FixedL2O: +0.000126, paired95% [0.00010505175590515137, 0.00014748841524124146].
- A/FluidL2O_minus_classical_Fluid60: +0.000020, paired95% [1.7280519008636477e-05, 2.305281162261963e-05].
- A/FluidL2O_minus_untrained: +0.000020, paired95% [1.7280519008636477e-05, 2.305281162261963e-05].
- A/FixedL2O_minus_classical_ULA40: +0.000046, paired95% [4.047560691833497e-05, 5.073577165603638e-05].
- A/Fluid_classical60_minus_ULA60: +0.000655, paired95% [0.0005828306078910827, 0.0007279813289642334].
- A/FluidL2O_minus_no_position_momentum_seed401: +0.000000, paired95% [0.0, 0.0].
- A/FluidL2O_minus_ULA_runtime_match: +0.000675, paired95% [0.0006010320782661438, 0.0007500350475311279].
- A/FluidL2O_minus_Fluid_runtime_match: -0.000099, paired95% [-0.00011407887935638429, -8.508509397506714e-05].
- B/FluidL2O_minus_FixedL2O: +0.301774, paired95% [0.2733703927397728, 0.32917792147397995].
- B/FluidL2O_minus_classical_Fluid60: +0.010223, paired95% [0.008499391078948974, 0.011952931344509126].
- B/FluidL2O_minus_untrained: +0.010223, paired95% [0.008499391078948974, 0.011952931344509126].
- B/FixedL2O_minus_classical_ULA40: +0.004041, paired95% [0.003110668361186981, 0.0051081210970878605].
- B/Fluid_classical60_minus_ULA60: +0.286807, paired95% [0.25916204112768176, 0.31404048442840576].
- B/FluidL2O_minus_no_position_momentum_seed401: +0.000000, paired95% [5.7220458984375e-08, 1.71661376953125e-07].
- B/FluidL2O_minus_ULA_runtime_match: +0.297031, paired95% [0.26922509247064585, 0.32427124667167667].
- B/FluidL2O_minus_Fluid_runtime_match: +0.029315, paired95% [0.025386392533779147, 0.033234038710594174].
- C/FluidL2O_minus_FixedL2O: +0.237585, paired95% [0.20895368389785288, 0.2668348693102598].
- C/FluidL2O_minus_classical_Fluid60: -0.013678, paired95% [-0.02630274114012718, -0.003575173884630202].
- C/FluidL2O_minus_untrained: -0.000000, paired95% [-2.384185791015625e-08, 1.430511474609375e-08].
- C/FixedL2O_minus_classical_ULA40: +0.002164, paired95% [-0.0008141469061374664, 0.00449064713716507].
- C/Fluid_classical60_minus_ULA60: +0.238396, paired95% [0.20967665448784828, 0.268449550896883].
- C/FluidL2O_minus_no_position_momentum_seed401: +0.000000, paired95% [0.0, 0.0].
- C/FluidL2O_minus_ULA_runtime_match: +0.224718, paired95% [0.19582220482826232, 0.254124753177166].
- C/FluidL2O_minus_Fluid_runtime_match: +0.006444, paired95% [-0.01755709400773048, 0.03250091657042503].

## Training and interpretation

Inherited Eq22 loss, full end-to-end derivatives, direct unconstrained coefficients, SGD1e-4, decay.98,10epochs,128 train/32 validation/100 test scenarios per regime;40 optimizer updates per model. Fixed models have240 active scalar parameters, fluid280, no-position-momentum260. Three paired shuffle seeds401/402/403 for the primary fixed/fluid models; the momentum ablation uses401 only. No original checkpoint is transferred.

Classical settings are validation-tuned. Fresh split seeds926101/926102/926103 (+case offsets) differ from the earlier fluid pilot. Epoch0 is eligible under the inherited validation layer loss; an epoch0-selected model cannot support a claim of learning improvement. Negative or unsuccessful outcomes remain in all aggregates.

Matched runtime uses validation-scene timing only to select nearby classical iteration counts; reported test medians/p95/raw samples reveal actual mismatch. Same helper budget is not equal FLOPs. Online times include initialization and full forward; attacks/training separate. Physical movement, CSI acquisition, RF programming and hardware latency are unmeasured.

The aligned LOS analytic reference is geometry invariant. Its exact adverse candidate is included where the single active stream condition holds. This is not a certificate for multipath/clutter. Weighted improvement at rho=.8 is not a Pareto frontier or a minimum QoS guarantee.

See summary.json,per_scenario.csv,training/*/*/complete.json and raw PT outputs for all seeds, selected epochs, costs, parameters, attack budgets and residuals. The classical geometry gains in the earlier pilot are not relabeled as L2O gains here. Wider physical/configuration/initialization studies and competitive cross-literature baselines remain unresolved.

### A selection and seed coverage
- FixedL2O_s401: selected epoch10; 40 updates; 5.48s; mu changes [7.128715515136719e-05, 1.0728836059570312e-05, 1.0728836059570312e-05, 2.3283064365386963e-10, 8.24516755528748e-05, 0.0].
- FixedL2O_s402: selected epoch10; 40 updates; 5.53s; mu changes [7.128715515136719e-05, 1.068413257598877e-05, 1.068413257598877e-05, 5.820766091346741e-10, 8.245237404480577e-05, 0.0].
- FixedL2O_s403: selected epoch10; 40 updates; 6.15s; mu changes [7.128715515136719e-05, 1.0758638381958008e-05, 1.0758638381958008e-05, 2.3283064365386963e-10, 8.245155913755298e-05, 0.0].
- FluidL2O_s401: selected epoch10; 40 updates; 12.87s; mu changes [7.128715515136719e-05, 1.0728836059570312e-05, 1.0728836059570312e-05, 1.5133991837501526e-09, 5.345145473256707e-05, 0.0, 0.0].
- FluidL2O_s402: selected epoch10; 40 updates; 12.39s; mu changes [7.128715515136719e-05, 1.068413257598877e-05, 1.068413257598877e-05, 1.6298145055770874e-09, 5.3451687563210726e-05, 0.0, 0.0].
- FluidL2O_s403: selected epoch10; 40 updates; 12.37s; mu changes [7.128715515136719e-05, 1.0758638381958008e-05, 1.0758638381958008e-05, 1.5133991837501526e-09, 5.345162935554981e-05, 0.0, 0.0].
- FluidNoPositionMomentum_s401: selected epoch10; 40 updates; 12.74s; mu changes [7.128715515136719e-05, 1.0728836059570312e-05, 1.0728836059570312e-05, 1.5133991837501526e-09, 5.345145473256707e-05, 0.0, 0.0].

### B selection and seed coverage
- FixedL2O_s401: selected epoch10; 40 updates; 5.99s; mu changes [0.0002956762909889221, 8.966773748397827e-05, 6.385147571563721e-05, 1.5484634786844254e-05, 0.003449171781539917, 0.0].
- FixedL2O_s402: selected epoch10; 40 updates; 6.29s; mu changes [0.0002956688404083252, 8.965283632278442e-05, 6.385892629623413e-05, 1.548929139971733e-05, 0.0034494027495384216, 0.0].
- FixedL2O_s403: selected epoch10; 40 updates; 5.57s; mu changes [0.00029569119215011597, 8.966028690338135e-05, 6.384402513504028e-05, 1.547904685139656e-05, 0.003448253497481346, 0.0].
- FluidL2O_s401: selected epoch10; 40 updates; 12.43s; mu changes [0.0002721063792705536, 0.00010162964463233948, 6.394088268280029e-05, 4.667043685913086e-05, 0.003994273953139782, 0.0, 0.0005522742867469788].
- FluidL2O_s402: selected epoch10; 40 updates; 12.13s; mu changes [0.00027213990688323975, 0.0001016370952129364, 6.39595091342926e-05, 4.666997119784355e-05, 0.003992716781795025, 0.0, 0.0005522444844245911].
- FluidL2O_s403: selected epoch10; 40 updates; 11.45s; mu changes [0.00027218088507652283, 0.00010162591934204102, 6.396323442459106e-05, 4.668114706873894e-05, 0.003995245322585106, 0.0, 0.0005524232983589172].
- FluidNoPositionMomentum_s401: selected epoch10; 40 updates; 11.67s; mu changes [0.0002721063792705536, 0.00010162964463233948, 6.394088268280029e-05, 4.666997119784355e-05, 0.003994272090494633, 0.0, 0.0005522742867469788].

### C selection and seed coverage
- FixedL2O_s401: selected epoch10; 40 updates; 5.85s; mu changes [4.8220157623291016e-05, 3.662705421447754e-05, 2.7686357498168945e-05, 3.486126661300659e-05, 0.000953899696469307, 0.00022921152412891388].
- FixedL2O_s402: selected epoch10; 40 updates; 5.94s; mu changes [4.665553569793701e-05, 3.612041473388672e-05, 2.7805566787719727e-05, 3.5667791962623596e-05, 0.0005883742123842239, 0.0002299472689628601].
- FixedL2O_s403: selected epoch10; 40 updates; 5.98s; mu changes [4.7087669372558594e-05, 3.6135315895080566e-05, 2.785027027130127e-05, 3.5705044865608215e-05, 0.0006802696734666824, 0.00023001432418823242].
- FluidL2O_s401: selected epoch0; 40 updates; 11.89s; mu changes [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
- FluidL2O_s402: selected epoch0; 40 updates; 11.55s; mu changes [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
- FluidL2O_s403: selected epoch0; 40 updates; 12.69s; mu changes [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
- FluidNoPositionMomentum_s401: selected epoch0; 40 updates; 11.64s; mu changes [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
