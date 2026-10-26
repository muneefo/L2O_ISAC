# Isolated single-user fluid transmit array

[REPORT.md](REPORT.md) gives measured findings and limitations. [PROTOCOL.md](PROTOCOL.md)
records the model, resource matching, validation choices and learning controls.
[STATUS.md](STATUS.md) records completed checks. All new code/configs/tests/outputs
are confined to this directory. The original reproduction files are imported, not
refactored or retrained. Their hashes are checked against the pre-fluid snapshot.

Current checkout: `/workspace/github-export/l2o-isac`. Run from that repository root (or the extracted archive root), using the existing CPU environment:

```bash
/workspace/.venv/bin/python -m pytest -q tests/test_physics.py experiments/fluid_tx_single_user/tests
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.run --stage tune
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.run --stage evaluate
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.run --stage support
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.static_random
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.checks
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.run --stage diagnostics
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.learning
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.tradeoff
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.run --stage timing
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.learning --retime-only
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.report
/workspace/.venv/bin/python -m experiments.fluid_tx_single_user.evidence
```

These stages have been executed. Completed training/evaluations are reused; the
input/config hash is checked. Learning starts only after classical_gate.json
passes. The small learning branch trains1 or20 scalar steps per regime with the
same eight updates/three paired seeds, frozen BF/MinNet step coefficients and
independent validation. It does not load an unrelated original checkpoint.

Quick classical integration smoke (4 scenarios/regime, not scientific evidence):

```bash
python -m experiments.fluid_tx_single_user.run --config experiments/fluid_tx_single_user/configs/smoke.json --output experiments/fluid_tx_single_user/outputs/smoke --stage all
```

`run --stage all` runs the classical phases, including stronger long-budget ULA
controls, diagnostics and timing. Static-random, learning and tradeoff studies have
separate commands to keep the primary geometry experiment distinguishable.

Clean CPU setup (provided, not reinstalled into a second environment):

```bash
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r experiments/fluid_tx_single_user/requirements.txt
.venv/bin/pip install --no-deps -e .
```

Then use `.venv/bin/python` in place of the execution-specific Python path.
SciPy is used only as an independent small convex-QP validation dependency; the
optimizer's projection is bounded PAV without a QP dependency.

Raw evidence: `outputs/pilot/scenarios.pt` contains fixed path gains/angles,
reflectivities,common radii and initializations. `A/`,`B/`,`C/` contain raw positions,
complex beam/error states, nominal/terminal/attack-budget/shared-union metrics and
per-scenario residuals. `learning/` contains selected/latest weights, optimizer/RNG
states and real training histories. `tradeoff/` contains independently attacked
rho sweeps. CSV/JSON summaries, plot data, timing samples, NumPy verification and
hash manifests are saved alongside. No paper result values generate these outputs.

The supplied λ=.01m merely converts normalized coordinates; CPU timings are not
physical antenna-movement latency. Spatial channels assume known path parameters
and exclude coupling,position error,actuation overhead and calibration dynamics.
Do not interpret this idealized pilot as a hardware demonstration or novelty claim.
