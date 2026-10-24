# Architecture hypotheses and graph audit

Gate A interpretation: the pinned reference is an independent corrected
reimplementation. Findings below concern that implementation, never unavailable
author weights. Physical equations and uncertainty sets remain unchanged.

| Hypothesis | Prior evidence / status before new experiment |
|---|---|
| H1: learned inner errors insufficiently minimize utility | Previous 80-step attacks found lower values; new nested independent attack must quantify gaps including terminal candidate. Open pending Gate D. |
| H2: shared utility training weakens opposing roles | Graph evidence: all 180 mu/eta entries receive gradients of **negative** average perturbed utility. Direct scalars have no positivity constraint. Inner schedule may therefore learn to increase evaluated utility. This mechanism is possible, not proof of adversarial-role failure or author's defect. Requires attack evidence and controlled training. |
| H3: state-dependent schedules improve difficult cases | Untested; optional phase F deferred until A–E. |
| H4: progress-dependent continuation beats geometric scaling | Untested; existing RecNet beta=phi=.99 is retained as baseline. Geometric continuation does not itself certify convergence. |
| H5: uncertainty labels lack coverage/identifiability | Prior fixed-weight audit shows nearly constant .05 predictions across independently labeled nominal data. Our generator's nominal inputs do not identify independently sampled radii. Authors' input/label relationship is unspecified. No calibration guarantee follows from RMSE. |

## Actual computation graph

`train_unrolled`: freezes EpsNet, computes eps under no_grad, then calls
UnrolledPGDA. Each pgda_step uses analytic df/dz* expressions that remain
PyTorch differentiable. MinNet uses **unweighted** comm/sense derivatives;
MaxNet uses rho-weighted utility derivatives at the newly perturbed channels.
All intermediate objectives use presumed channels **plus the current learned
perturbations**, rather than sampled hidden actual channels or nominal-only
channels. Loss = -mean_(sample,layer) utility. There are no detach calls inside
this recurrence. Full backpropagation differentiates gradient updates (including
second derivative terms). `SolverState.detach` exists but is not invoked there.
Direct learned mu/eta can have either sign; optional positive/projected variants
are separate variants. Saved checkpoint graph has no author-code provenance.

## Role-separated extension objectives (specified before implementation)

Keep 180 scalar parameters, split into psi=(mu_in,eta_in) and
 theta=(mu_out,eta_out), each 15×3×2. Initialize exactly from the existing
reference, including signed coefficients; do not impose a new parameterization.
Deploy with identical 15 alternating projected layers (30 full analytic-gradient
calls), so architecture and online derivative budget match reference U-PGDA.

Inner phase: freeze current deployed beamformers B=stopgrad(U_(psi,theta)(X)).
Start from zero errors and run 15 MinNet-only steps A_psi(B), using physical
gradients/momentum/projections. Minimize E[U(B,A_psi(B))] over psi only. Theta
and beamformers remain unchanged. Full derivatives through this inner unroll.

Outer phase: freeze psi parameters, retain their dependence on beamformers
inside the deployed unroll. Construct B_(psi,theta) by the same 15 layers. Run a
fixed classical independent PGD adversary at stopgrad(B) and oracle radii;
then set delta_bar=stopgrad(attack output). Minimize
-E[U(B_(psi,theta),delta_bar)] over theta. This is a **stop-gradient approximate
adversary surrogate**, not an exact envelope/Danskin gradient. Full gradient
through the deployed inner/outer trajectory is retained; only the terminal
independent attack is detached. It is biased for an unsolved nonconvex minimum.
We evaluate any improvement independently, under a stronger fixed final attacker.

No extra terminal learned attack is executed online. Inner-training strength is
also measured separately on fixed validation beamformers. Model selection uses
validation adversarial performance, including the initial checkpoint. Keeping the
initial weights if training loses is reported explicitly, not as a new gain.

A stronger-classical-inner variant uses k>1 classical inner gradient steps for
each outer update; its extra calls and runtime are charged. A matched-budget
version reduces outer steps. Learning is not credited for extra search effort.

## Executed evidence after the pilot

Independent frozen-design attacks support H1 for this checkpoint: terminal
utility 4.66515 versus found 4.03977. A separate actual-stage audit reports
communication utility increasing in 80% of reference inner sample/layer steps,
and weighted utility increasing in 76.77% (threshold 1e-6). Role-separated
variants have no observed component increases above 1e-6 across these same
128×15 inner stages. This is a finite sample observation, not a monotonicity proof.
The reference has 16 negative inner mu entries; all three role variants have zero,
without an imposed sign constraint. These observations, graph evidence and the
inner-only ablation support H2's role-conflict mechanism in our reconstruction.
They do not establish that it is the only source of gain: the extension receives
additional optimization/training, and no identical-budget shared-loss fine-tuning
control was run. No conclusion about author's trained coefficients follows.

For one interior inner step with frozen beamformers, zero momentum and mu=0,
let g=df/d(delta*) and delta'=delta-mu*g. Then
 d f(delta')/d mu = -2||g||², while d(-f)/d mu = +2||g||².
Thus the direct negative-utility path encourages a smaller inner step. In a full
alternating projected unroll, other indirect paths can change this sign; this
local calculation is a mechanism, not a theorem that end-to-end training fails.

H3/H4 remain untested. H5 is supported for our independent-label simulator input
law only; no additional calibration method or real metadata was introduced.
