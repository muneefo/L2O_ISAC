# Isolated fluid-transmit pilot: fixed assumptions before test evaluation

All extension files live here. Shared l2o_isac physics, analytic Wirtinger
gradients, projections, PGDA and independent attackers are imported unchanged.
No source-paper checkpoint is adapted or overwritten. This workflow uses a new
spatial channel distribution, not the paper's position-independent CN(0,I) law.

Positions x=p/lambda are in wavelengths. Lambda=.01m is only a unit conversion.
Nt=Nr8,L4,M=T1; aperture3.5lambda, minimum spacing.25lambda. The feasible half-wave
ULA spans[0,3.5lambda], but internal coordinates have genuine freedom. Ordering is
fixed; no antenna permutation, minimum-distance change, or aperture expansion.

Steering is exp(-j2pi*x*sin(theta)). Communication h^H=sum beta_k*a_t(theta_k)^T,
so the stored ChannelBatch.h is the **conjugate** of this row. Gains and angles
are fixed per scenario, including while positions change. Radar H_l=alpha_l*a_r,l*
a_t,l^T (transpose, not Hermitian). Receive array remains half-wave. Symbol/noise/
power conventions, communication plus dedicated sensing streams, incoherent sum
of clutter powers, log natural/nats, uncertainty radii eps*sqrt(Nt/Nr), all path
errors and hardware constraints are inherited. Eps=.05 for all methods/positions.
There is no position error, mutual coupling, movement cost or hardware latency model.

Cases: A one unit-magnitude LOS path aligned with the target, J0; B four fixed
multipath gains (normalized total expected path power1) with user dominant direction
near-25deg and target near+25deg,J0; C same multipath with target near+20deg and
one competing clutter near target+10deg (J configurable). Path/gain draws, angles,
reflectivity gains and initial seeds are saved before optimization; test100 per
case is distinct from train64/validation32. No speculative fitting of old channels.

A is a null control: MRT/MRC with zero dedicated-sensing power simultaneously
maximizes both components for aligned LOS and no clutter. It is a feasible choice
of the inherited two-stream architecture. Its analytic robust value is independent
of geometry under these assumptions and validates numerical attacks. Case A starts
at that solution for every geometry; B/C use the same nominal-CSI phase initializer
and90/10 communication/sensing power split for every method.

Geometry projection subtracts index*dmin, applies pooled-adjacent-violators
isotonic regression and clips common bounds[0,A-(Nt-1)dmin], then restores offsets.
It is checked against SciPy SLSQP convex QPs and exhaustive tiny active partitions.
PAV Jacobian is block averaging on free pooled blocks and zero on clamped blocks,
valid away from active-set kinks. No straight-through or detached surrogate is used
if learned schedules are attempted.

Position gradient is the **partial** with frozen errors/beamformers, using exact
channel derivatives and shared analytic Wirtinger gradients (factor2 real chain
rule), not a total derivative through the preceding beamformer update. It remains
differentiable for a learned unroll. It is not an envelope gradient through a solved
nonconvex adversary. A macro: reconstruct; original error descent; original
beamformer ascent/projections; fixed-error position partial ascent; exact geometry
projection; reconstruct. Position momentum0. Classical safeguards accept per-case
fixed-error Armijo improvement with at most6 halving trials, otherwise keep x.
This does not guarantee independently reattacked utility improves.

Validation: tune fixed-ULA and fixed-random BF steps each over3x3 grid at30
iterations and independent50-step validation attacks. Fluid reuses ULA-tuned BF
coefficients (no hidden retuning advantage), then tunes3 position steps. Test
never selects these choices. Feasible random geometry is fixed per scenario and
initialized by the same nominal-CSI rule. Count full six-block gradient helper
calls: fixed2 per macro, fluid3 (last call provides h/at partials for position chain).
Derivative budgets60/180/360 therefore mean fixed30/90/180 vs fluid20/60/120.
Extra field reconstruction, PAV, backtracking and projections are counted/timed;
equal derivative calls do not guarantee equal FLOPs. Additional runtime-matched
fixed budgets are selected on validation timing before final comparison.

Final evaluation freezes positions and beamformers; nominal channels are rebuilt
once, then the unchanged independently validated batched attacker searches only
errors: zero/terminal/five random interior/boundary starts,50 Adam steps and nested
50/200/1000 PGD, best retention, separable comm and joint shared sensing errors.
All methods receive a shared union of candidates per paired scenario. Found minima
are upper bounds on unknown true minima, not certified robust-performance bounds.

Only after classical gates pass: optionally train20 scalar position steps, keeping
validated BF/MinNet coefficients frozen and momentum0. No compatible original
single-user/spatial pretrained checkpoint exists. Any learned model is new and
must be labeled as such. Compare with its fixed-step no-backtracking initializer,
validation-tuned no-backtracking baseline and safeguarded classical curves at equal
budgets; removing backtracking is not a learning speedup. No larger NN or joint BF
fine-tuning is included. Preserve losses or negative findings and training costs.

Case A additionally supplies a closed-form feasible error candidate for its single
active stream to every final candidate pool. This validates the null control and
prevents incomplete generic attacks from appearing as geometry gain. Generic attack
budget curves remain saved separately. No such exact-minimum claim is made for B/C.

Stronger controls: separately retune fixed ULA at90/180 iterations on validation
before comparing its longer attainable performance. Case A also includes the
feasible analytic fixed-ULA MRT/MRC initialization as a no-iteration reference.
Its gradient calls are0 but initializer computation is not claimed as zero latency.
Finite inherited PGDA can move away from this optimum; such losses or fluid recovery
are optimizer effects, not an intrinsic geometry advantage in the aligned case.

Learning control: per regime, initialize from a separately validation-tuned constant
position step with backtracking disabled. Compare a one-scalar learned constant
against a20-scalar layer schedule using identical batches, four Adam epochs, eight
updates, learning rate.002, independent stop-gradient10-step training attacks and
independent50-step validation selection (initialization eligible). The exact shared
physical unroll is differentiated, including geometry-dependent BF/error updates
and PAV's active-face Jacobian. BF/MinNet **coefficients** stay fixed; their physical
iterates still update. Full independent final evaluation is unchanged. A gain over
a fixed initializer is not evidence for a layer schedule if equal-training scalar
calibration explains it. Training/validation cost and selection epoch are retained.

Random geometry controls are distinguished explicitly: `Random` is drawn
independently per scenario and then held fixed during optimization; `StaticRandom`
is one draw(seed909999) held fixed across every scenario and split. Both use the
same full aperture/endpoints and minimum spacing. The latter is the primary fixed
random-array comparator, independently validation-tuned at each reported budget.
No random geometry is selected using test performance.
