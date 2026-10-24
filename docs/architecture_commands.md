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
