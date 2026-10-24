# Controlled continuation protocol, fixed before final comparisons

Start: commit73159d4, immutable archive outside checkout at
/workspace/evidence_archives/pre_controls_73159d4.zip, hashed in preservation.json.
Existing findings are verified against saved outputs there. Source PDF not found
in accessible workspace/tmp inputs; retained text and prior records are available.
No new author-equation verification is claimed. Missing PDF does not block these
implementation-only experiments.

A: frozen existing reference. B: continuation using original joint negative
mean-over-layers utility loss and SGD. C: same joint procedure with positive
**inner** steps. D: separate inner/outer losses with exactly C's positive inner
parameterization. E: previous unconstrained separated method, preserved separately.

All B–D use one common schedule class, identical pretrained values before mapping,
180 scalars,15 deployed layers, same momentum treatment (direct trainable signed
coefficients with no newly imposed bounds), same outer steps, oracle evaluation
radii, same precision, saved train/validation/test tensors and initial states,
paired minibatch permutations, three epochs, SGD lr1e-4 and decay.98 for both
roles. E's earlier inner lr1e-3 and signed steps differ and are explicitly not D.

Positive parameterization: optimize direct inner mu constrained to [1e-6,infinity)
by projection after each SGD update, with **no momentum or outer constraint**.
Mapping takes max(abs(mu_pretrained),1e-6), flipping negative signs while preserving
magnitudes. Record mapped checkpoint and evaluate before any training (C0/D0),
including independent validation/test attacks. Mapping is a separate intervention;
C−B includes both its initial effect and subsequent sign-restricted optimization.
No softplus gradient-rescaling confounder is introduced.

Training compute: profile one joint update versus one separated cycle on validation
batches only, including optimizer/projection and higher-order differentiation.
Choose an integer joint repeat count1–4 closest to role-cycle wall time, using the
same count for B and C and freezing it before test. Joint repeats use the identical
minibatch, LR and original loss. D performs one inner and one outer update per
minibatch. Record actual per-block updates, differentiated lengths, full six-block
analytic helper calls, independent three-error autograd calls, backward passes and
wall time. This approximates equal training compute; it is not equal update count.
Add B/C one-update controls for original reference/seed101 to expose this tradeoff.
No dummy work or omitted overhead is used to manufacture compute matching.

Inner training freezes deployed beamformers (stop-gradient) and minimizes full
15-step inner-search utility. Outer training freezes inner-policy parameters,
retains derivatives through their deployed iterates and physical gradients, then
uses a separately computed classical terminal attack with detached errors. Both
losses and the positive mapping are explicitly tested. Independently evaluate
learned inner-search outputs and their refinement on validation beams frozen from
A. This distinguishes lowering candidate utility from improving the final attack
minimum; uphill steps alone are not proof of failure.

Validation selection is identical across B–D: same fixed30-step independent search,
initial checkpoint eligible, then three epoch candidates. LR is fixed, not selected
using test. Test data/attack config stay fixed from prior audit; numerical attack
batching is optimized only after equivalence tests, without changing candidates,
seeds, uncertainty domains, or200/1000-step budgets. Every method is attacked with
frozen beams, then all candidates are cross-evaluated in a common shared union.
Found minima are not globally certified. Original E remains a historical additional
condition with its own exact training provenance.

Classical PGDA: tune each iteration budget5,10,15,20,30,60,180 on first24 saved
validation scenarios, same3×3 step grid and30-step independent validation attack.
Record63 searched configurations. Two full six-variable analytic helper calls per
PGDA iteration; one helper computes all blocks even if only three are used in that
stage. Fifteen learned layers similarly cost30 calls. Independent error-only
attacker calls are counted separately. Time generation alone, warm CPU batch1,
one thread/complex64, five warmups and20 repeats; no GPU sync is needed on CPU.
Also report initialization/EpsNet costs separately and attacker wall time.

After original-reference controls pass, authorize two additional pretrained seeds
using unchanged existing `l2o_isac.experiment --profile paper`, new unique folders,
5000/1000/1000 channels,10epochs250batch, same learned bounds/staging. This is
additional seed coverage explicitly requested, not restarting/replacing the prior
reproduction. Use identical saved audit scenarios and paired fine-tuning seeds for
all references. Report pretraining variability and conditional fine-tuning
variability separately; paired bootstrap resamples pretrained references,
fine-tuning runs and channels at their corresponding levels, with only three
pretrained seeds acknowledged as exploratory. Preserve failures.

### Compute-profile amendment (before balanced-condition test evaluation)

The nearest integer repeat count2 predicted only82.6% of separated-cycle training
wall time. Preserve those original B/C runs as `B_s*`/`C_s*` diagnostics. The
primary compute-matched controls additionally use deterministic balanced repeats:
ratio = measured separated median / mean joint B,C median; each minibatch k uses
floor((k+1)*ratio)-floor(k*ratio) joint updates. For ratio~2.42, this mixes2 and3
updates, rather than wasting dummy work. The schedule depends only on validation
profiling, never test scores. Save as B_balanced/C_balanced; compare these with D
in the primary report, while retaining2-repeat and1-repeat controls. All variants
share identical epoch minibatch permutations and LR schedules. Actual time can
still differ from the short profile and must be reported. Additional pretrained
references use balanced B/C and D, not redundant integer-repeat controls.
