# L2O robust hybrid beamforming for ISAC

Independent PyTorch implementation of Lei Wang, Sergiy A. Vorobyov, and Esa Ollila,
“L2O Robust Hybrid Beamforming for ISAC,” *IEEE Transactions on Wireless
Communications*, vol. 25, pp. 18778–18794, 2026.
[DOI: 10.1109/TWC.2026.3702925](https://doi.org/10.1109/TWC.2026.3702925).
This is reconstructed from the supplied paper, not the authors’ software.

Completed runs and findings: [validation report](reports/VALIDATION.md).

Fresh execution with paper and previous-result reads blocked:
[execution evidence, logs, raw channels, and beamformers](reports/fresh_execution_seed73129/README.md).
This run starts from new weights with seed 73129 and checks computed objectives with separate
NumPy signal equations. Its results are produced by the code, not taken from the paper.

## Implemented

- Complex-valued communication and multi-target/clutter sensing models, equations (1)–(9), in nats.
- B-PGDA, Algorithm 1: alternating simultaneous inner/outer gradient updates, six momentum terms,
  uncertainty-ball projections, unit-modulus analog beamforming, exact transmit-power normalization,
  and unit-norm receive filters.
- U-PGDA, Algorithm 2: six learned step sizes and six learned momentum coefficients per layer,
  differentiated end to end through the analytical gradients and projections. Loss averages all layers.
- EpsNet: three BN/FC/SoftPlus branches with the stated hidden sizes and supervised RMSE.
- RU-PGDA, Algorithm 3: validation-only grid search over shared beta/phi, with arbitrary finite
  inference extension depth. No claim of stable behavior at infinite depth.
- Paper training defaults, checkpoint reload/inference, momentum ablations, parameter sweeps,
  held-out metrics, paired comparisons, plots, and independent multistart adversarial testing.

## Run

The commands below assume a Python virtual environment at `.venv`; installation instructions follow.

```bash
cd L2O_ISAC
.venv/bin/python -m pytest -q
.venv/bin/python -m l2o_isac.experiment --profile smoke --output reports/my_smoke
.venv/bin/python -m l2o_isac.experiment --profile paper --nt 16 --layers 15 --output reports/my_paper
.venv/bin/python -m l2o_isac.execution_evidence --seed 73129 --output reports/my_fresh_execution
```

On another machine:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test]'
```

`--profile paper` uses 5000/1000/1000 independent training/validation/test channels, 10 epochs,
batch size 250, 15 layers, SGD learning rates 0.05 / 0.0001, epoch decay 0.98, a 10×10 recurrent
grid and 5 validation extension layers. Evaluation includes 180 benchmark iterations, recurrent
extensions of 5 and 10 layers, and multistart attacks on 128 held-out examples. Set
`--attack-samples 1000 --attack-steps 200 --attack-restarts 8` for stronger evaluation on the entire
test set. Training and grid search never use the test set. `configs/paper.json` records the defaults.

Other useful flags: `--bounds oracle`, `--momentum fixed`, `--momentum none`, `--nt 32`,
`--nr 16`, `--layers 20`, `--rho 0.5`, `--snr-db 0`, `--epsilon 0.01`, `--device cuda`.
Use `--help` for all options. The `smoke` profile exercises the pipeline on small datasets;
its EpsNet is deliberately undertrained and it is not a performance reproduction.
CPU runs default to one PyTorch thread: small complex operations slow down substantially when
multiple thread pools compete under a two-CPU quota. Timings from concurrent runs have additional
resource-contention noise and should not be treated as precise latency benchmarks.

Run complete parameter families (each configuration is trained separately):

```bash
.venv/bin/python -m l2o_isac.suite --family core --profile paper --output reports/core
.venv/bin/python -m l2o_isac.suite --family momentum --profile paper --output reports/momentum
.venv/bin/python -m l2o_isac.suite --family robustness --profile paper --output reports/robustness
.venv/bin/python -m l2o_isac.suite --family tradeoff --profile paper --output reports/tradeoff
.venv/bin/python -m l2o_isac.suite --family snr --profile paper --output reports/snr
.venv/bin/python -m l2o_isac.suite --family scenarios --profile paper --output reports/scenarios
```

Append `--seeds 2026 2027 2028` for independent training seeds. Overrides such as `--epochs 2`
are forwarded to each run. These sweep commands are available; only the runs under `reports/`
have actually been executed. They are not claimed as pixel-identical reproductions of the figures.

## Use a trained model

```python
from l2o_isac.data import generate_channels
from l2o_isac.inference import load_checkpoint, design_beams

path = 'reports/paper_n16_l15/checkpoint.pt'
cfg, epsnet, model, metadata = load_checkpoint(path)
channels = generate_channels(cfg, count=8, seed=700)
state, trace, predicted_bounds = design_beams(path, channels, seed=42, extra_layers=10)
analog, digital, receive = state.a, state.d, state.w
```

For real data, construct `ChannelBatch(h, at, ar, alpha, eps)`. Shapes are `[B,M,Nt]`,
`[B,T+J,Nt]`, `[B,T+J,Nr]`, `[B,T+J]`, `[B,3]`. First T paths are targets; remaining J
paths are clutter. The presumed steering vectors and reflection coefficients are needed separately,
not only their product. Use dimension-normalized uncertainty radii: actual ball radii are
`sqrt(Nt)*eps_h`, `sqrt(Nt)*eps_t`, `sqrt(Nr)*eps_r`.

## Validation and interpretation

`tests/` checks all six analytical gradients against a separately written, literal signal model
with autograd and central finite differences in complex128. It also checks the derivative of the
training loss through the iterative solver, feasible projections at every iterate, zero uncertainty,
unequal transmit/receive dimensions, no clutter, recurrence equivalence, checkpoint round trips,
training, and independent feasible attacks.

Each run saves `report.json`, `summary.md`, `results.png`, `results.pdf`, `test_metrics.npz`,
and `checkpoint.pt`. The report separates:

1. **Internal objective:** evaluated at each solver’s own error iterates and estimated radii.
2. **Nominal objective:** evaluates those beamformers with zero errors.
3. **Independently attacked objective:** freezes beamformers and searches the true uncertainty
   balls, with common seeded starting points and per-sample acceptance/backtracking.

The third metric is a stress test, not a certificate of the global worst case. A feasible attack’s
objective is an **upper bound** on the true minimum; stronger attacks can lower it. Communication
and sensing errors are independent, so the best communication and sensing restarts can be combined
into one feasible error state. Comparisons include paired differences on the same channel samples.
The reported 95% SEM intervals describe channel sampling for one trained model, not variation over
training seeds. CPU timings include gradient/projection iterations and trace calculation but exclude
EpsNet prediction and independent attacks. These are batched timings, not deployment latency.

## Paper details that require explicit choices

- The paper does not specify angle distributions, random seeds, or the initialization distributions.
  We use half-wavelength ULAs, separated random angles in [-60°,60°], minimum separation 5°,
  random assignment of target/clutter roles, circular Gaussian beamformer proposals, and uniform
  complex-ball initial errors. Previous iterates initially equal current iterates (zero initial momentum).
  Changing these choices can change results.
- Radar uses `alpha * ar * at.T`, **not** `at.H`. Steering entries have magnitude one, without
  antenna normalization. The communication channel has CN(0,I) entries; 2/3 dB reflection variances
  are converted with `10**(dB/10)`. All logarithms are natural.
- The appendices use Wirtinger `df/dz*` gradients. For real-valued losses PyTorch’s complex
  autograd gives twice that value. The analytical implementation matches the paper’s convention,
  checked independently. In sensing sums, each target retains its own denominator from equation (6).
- The communication EpsNet branch sees real/imaginary nominal h; both sensing branches see
  real/imaginary flattened nominal radar matrices. Each branch normalizes inputs with BN and uses
  SoftPlus after every FC layer. The exact cross-branch feature wiring is not fully specified.
- The explicit staged procedure in Section IV-D is followed: train EpsNet, freeze it, train U-PGDA,
  freeze it, then select recurrence on validation data. This resolves the less precise “joint/end-to-end”
  wording elsewhere. Best validation checkpoints, including the initial U-PGDA, are retained.
- The default experiment labels all uncertainty bounds 0.05. A constant predictor therefore achieves
  zero RMSE. With `--variable-eps`, bounds are independently sampled in [0.001,epsilon]; because
  nominal CSI does not encode those independently sampled labels, prediction of an individual
  bound is statistically unidentifiable. No dynamic-bound generalization claim is made.
- The paper gives no positivity constraints on learned hyperparameters. Default `direct` uses
  literal unconstrained scalars. Reports save every learned value so negative step sizes can be
  detected. `--parameterization positive` is an optional, explicitly different softplus-step /
  sigmoid-momentum variant; its optimization conditioning differs from direct SGD.
  `--parameterization projected` instead applies direct SGD followed by nonnegative-step and
  [0,0.999] momentum projection. This is another explicit safeguard, not specified in the paper.
  Even nonnegative learned inner steps may collapse to zero under the negative-objective loss.
- Negative objective training, equation (22), does not independently enforce a strong inner
  adversary; learning can favor weaker minimization. Hence the separate attack evaluation.
- Projection and momentum do not by themselves guarantee monotone objectives or global min-max
  convergence in this nonconvex problem. The implementation tests feasibility and empirical behavior
  without asserting a theorem or reproducing all qualitative claims.
- The SCA/DNN/CNN baselines from other papers are not included: their adaptations are not specified
  sufficiently to reconstruct them faithfully. `Nonrobust-PGA` is the fixed-step PGDA solver with
  zero uncertainty, not a separately trained U-PGA baseline. The implemented core consists of all
  three proposed networks and the PGDA benchmark.

## Source map

| File | Purpose |
|---|---|
| `l2o_isac/data.py` | Channels, radar matrices, initialization |
| `l2o_isac/physics.py` | Objective and all six analytical gradients |
| `l2o_isac/solver.py` | PGDA, unrolling, momentum, projections, recurrence |
| `l2o_isac/networks.py` | EpsNet |
| `l2o_isac/training.py` | SGD, validation selection, recurrent grid search |
| `l2o_isac/evaluation.py` | Held-out comparison and multistart attacks |
| `l2o_isac/experiment.py` | Complete command-line experiment and artifacts |
| `l2o_isac/suite.py` | Core configurations, ablations, parameter sweeps |
| `l2o_isac/inference.py` | Checkpoint reload and inference |

The source paper is available through the DOI cited above.

To rerun stronger attacks on a trained model with fresh, held-out channels:

```bash
.venv/bin/python -m l2o_isac.audit --checkpoint reports/paper_n16_l15/checkpoint.pt
```

This compares 80-step/4-start attacks against 200-step/8-start attacks with nested starting seeds.
It asserts that stronger attacks cannot improve the attacked objective and that attacks do not
improve over zero error. Results are saved as `robustness_audit.json` next to the checkpoint.
