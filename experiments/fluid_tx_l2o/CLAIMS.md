# Claim-to-evidence map

|Claim|Evidence|Status / limit|
|---|---|---|
|The coupled extension trains inherited and position schedules|model.py, training/*/{initial,latest,selected}.pt, complete.json, gradient-block tests|Implemented. All21 runs completed; C selected initialization. Some LOS blocks have zero gradients by physical symmetry.|
|Continuous movement uses consistent spatial channels|FORMULATION.md, spatial imports, scenario PT and geometry/physics tests|Fixed path gains/angles; original iid channel law intentionally replaced.|
|Numerical feasibility|validation.json and feasibility_per_scenario.npz|Maximum residual 1.67e-06; tolerance1e-5. Does not certify robust optimality.|
|Reproducible saved designs|Independent NumPy and complete replay of 67 method outputs|Maximum complete replay difference 0.0; original 1041 tracked files unchanged.|
|Novel fluid-antenna/learned-position concept|LITERATURE.md|Unsupported: substantial prior art already exists. Full novelty assessment remains open.|
|General L2O acceleration, sub-ms operation or configuration transfer|Online timing, untrained controls and limited fixed configuration|Not established. No hardware movement/CSI/RF timing or unseen-size evaluation.|
|A/Fluid_classical60_minus_ULA60|summary.json; raw shared_union states; per_scenario.csv|Mean +0.000655; paired channel95% [0.0005828306078910827, 0.0007279813289642334]. Conditional on spatial distribution and fitted runs.|
|A/FluidL2O_minus_untrained|summary.json; raw shared_union states; per_scenario.csv|Mean +0.000020; paired channel95% [1.7280519008636477e-05, 2.305281162261963e-05]. Conditional on spatial distribution and fitted runs.|
|A/FluidL2O_minus_Fluid_runtime_match|summary.json; raw shared_union states; per_scenario.csv|Mean -0.000099; paired channel95% [-0.00011407887935638429, -8.508509397506714e-05]. Conditional on spatial distribution and fitted runs.|
|B/Fluid_classical60_minus_ULA60|summary.json; raw shared_union states; per_scenario.csv|Mean +0.286807; paired channel95% [0.25916204112768176, 0.31404048442840576]. Conditional on spatial distribution and fitted runs.|
|B/FluidL2O_minus_untrained|summary.json; raw shared_union states; per_scenario.csv|Mean +0.010223; paired channel95% [0.008499391078948974, 0.011952931344509126]. Conditional on spatial distribution and fitted runs.|
|B/FluidL2O_minus_Fluid_runtime_match|summary.json; raw shared_union states; per_scenario.csv|Mean +0.029315; paired channel95% [0.025386392533779147, 0.033234038710594174]. Conditional on spatial distribution and fitted runs.|
|C/Fluid_classical60_minus_ULA60|summary.json; raw shared_union states; per_scenario.csv|Mean +0.238396; paired channel95% [0.20967665448784828, 0.268449550896883]. Conditional on spatial distribution and fitted runs.|
|C/FluidL2O_minus_untrained|summary.json; raw shared_union states; per_scenario.csv|Mean -0.000000; paired channel95% [-2.384185791015625e-08, 1.430511474609375e-08]. Conditional on spatial distribution and fitted runs.|
|C/FluidL2O_minus_Fluid_runtime_match|summary.json; raw shared_union states; per_scenario.csv|Mean +0.006444; paired channel95% [-0.01755709400773048, 0.03250091657042503]. Conditional on spatial distribution and fitted runs.|

A: aligned LOS has a geometry-independent analytic optimum. B: learned fluid improves over its identical untrained policy by about0.32%, a modest learning gain. C: learned fluid retains epoch0, giving no demonstrated training benefit. A shared-pool scoring bug was detected and corrected before final reporting; rejected initial scores are archived and excluded. See sources/WORK_HISTORY.md.

No test scenarios selected training loss, initialization steps, model architecture, attack settings, or classical runtime iteration counts. The test results have not been used to retune unsuccessful C models. No original-paper training was repeated.
