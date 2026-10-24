# Executable audit commands

Run from repository root. Existing reference checkpoint is already committed.
CPU clean environment (requires the recorded wheel versions to remain available):

```bash
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-audit-lock.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m pytest -q
```

Executed pilot uses `/workspace/.venv/bin/python` with repository on Python path.
No reference training occurs in any architecture-audit command.

```bash
python -m l2o_isac.architecture_audit --stage prepare
python -m l2o_isac.architecture_audit --stage tune
python -m l2o_isac.architecture_audit --stage audit
python -m l2o_isac.architecture_audit --stage train
python -m l2o_isac.architecture_audit --stage evaluate
python -m l2o_isac.architecture_audit --stage figures
```

`--stage all` runs these in order. Existing splits are hash-checked; completed
attacks are cached, epochs resume from latest.pt. Resuming checks protocol, learning
rate and reference hash. Selected.pt evaluates without optimizer/training. Source
hashes record working files even when the worktree differs from source_commit.
Changing source after a partial run requires a new output directory; never mix
versions silently. A full-study configuration, explicitly **unexecuted**, is:

```bash
python -m l2o_isac.architecture_audit --config configs/full/architecture_audit.json --output results/full_architecture_audit --stage all
```

Historical reference-training command exists but is **not run by this audit**:

```bash
python -m l2o_isac.experiment --profile paper --seed 73129 --output reports/NEW_reference_run
```

That command trains a new independent reconstruction; use a fresh output folder,
never overwrite the completed reference. It is supplied only for reproducibility.

Executed additional checks (all use the saved reference/extension checkpoints):

```bash
python -m l2o_isac.checkpoint_regression
python -m l2o_isac.strong_attack_check
python -m l2o_isac.role_ablations
python -m l2o_isac.secondary_tradeoff
python -m l2o_isac.minnet_direction_audit
python -m l2o_isac.audit_artifacts
python -m l2o_isac.audit_report
python -m l2o_isac.architecture_audit --config configs/pilot/architecture_smoke.json --output results/architecture_smoke --stage all
```

The extra strength check uses 1,000 PGD steps and five random restarts on all 128
scenarios for U15, RU25, tuned PGDA180 and three role variants. It is still a
pilot, not the full-study 20-restart/1,000-scenario target. No model is reselected
using that check. Inner-only/outer-only controls retain the same selected learning
rates and three SGD shuffle seeds; they are exploratory mechanism ablations.
The rho/radius figure uses 32 channels and one extension seed, explicitly separate
from the primary three-seed study. No state controller/calibration was implemented.

The first tiny smoke failed while serializing an undefined one-seed sample SD.
This reporting bug was fixed to save null; the failed log is retained. Subsequent
smoke execution passed. Numerical pilot results with three seeds were unaffected.
The provided clean-environment installation commands were not independently
reinstalled in a second environment; all smoke/pilot commands ran in the recorded
existing Python 3.12 CPU environment.

Exact source blobs corresponding to recorded per-checkpoint hashes are retained in
`results/source_snapshots/`, indexed by `results/source_snapshot_index.json`.
This includes intermediate reporting revisions. Later reporting/cache changes do
not rewrite the trained weights or their provenance.
