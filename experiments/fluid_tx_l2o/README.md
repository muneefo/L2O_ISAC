# Coupled fluid-transmit MinNet–MaxNet L2O

This isolated workflow extends the inherited unfolded PGDA to one user, one target, adjustable one-dimensional BS transmit coordinates, configurable clutter, and a fixed receive ULA. It imports the validated physics, six-block gradients, PGDA step, spatial response/position partial, exact PAV projector, and independent attacker without editing the reproduction or earlier pilot. No original-paper checkpoint is transferred or retrained. A new20-layer model learns all six inherited step/momentum schedules together with position step/momentum; this is not position-only training.

Start from the repository root with an environment containing torch,numpy,scipy,matplotlib,pytest. The observed environment uses `/workspace/.venv/bin/python`.

```bash
python -m pytest -q tests/test_physics.py experiments/fluid_tx_single_user/tests experiments/fluid_tx_l2o/tests
python -m experiments.fluid_tx_l2o.run --stage prepare
python -m experiments.fluid_tx_l2o.run --stage tune
python -m experiments.fluid_tx_l2o.run --stage train
python -m experiments.fluid_tx_l2o.run --stage evaluate
python -m experiments.fluid_tx_l2o.run --stage timing
python -m experiments.fluid_tx_l2o.report
python -m experiments.fluid_tx_l2o.evidence
```

`--stage all` executes prepare,tune,train,evaluate,timing. Each completed checkpoint/output is reused. A source/config mismatch prevents silently resuming incompatible training. Never remove outputs to repeat a test selection; use a new output path for a new protocol.

Smoke end-to-end run:

```bash
python -m experiments.fluid_tx_l2o.run --config experiments/fluid_tx_l2o/configs/smoke.json --output experiments/fluid_tx_l2o/outputs/smoke --stage all
```

Read FORMULATION.md for source equation mapping, ordering, derivatives, log/radius conventions, projection and source limitations. Configs and protocol lock define datasets/attacks before final tests. Validation chooses scalar initialization and classical settings; inherited Eq22 validation loss selects learned checkpoints, with epoch0 eligible. Independently attacked validation is a diagnostic, not a replacement training objective. Test evaluation freezes the final geometry and beamformers and searches the separate original uncertainty sets. “Lowest utility found” is not a certificate.

Raw outputs contain full complex states, scenario-wise utilities, errors, positions, constraints, nested attack budgets, shared-union candidates, timing and source/data hashes. Training stores initial/selected/latest models plus optimizer/scheduler/RNG and batch-order hashes. Three shuffle seeds are paired across methods; they are not extra independent test channels. The momentum ablation uses one training seed per regime and must be interpreted accordingly.

Online timing includes initial beamformer construction and the full design call: learned scalar access, spatial reconstruction, gradients, projections and layer metrics; classical backtracking is included. Independent attacks and offline fitting are recorded separately. Physical CSI acquisition, movement/settling and RF control are unmodeled. No hardware millisecond guarantee or count generalization is claimed.

Final validation: 78 tests,67 fully replayed designs,1041 baseline files unchanged. See outputs/study/training_cost.json for counted differentiated forward calls and recorded wall-time scope. Model/architecture diagrams are editable TikZ sources in diagrams/. The new letter is manuscripts/wcl_fluid_tx_l2o/main.pdf; prior letters remain untouched.
