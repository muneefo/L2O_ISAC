# Controlled continuation: executable commands and provenance

Run from `/workspace/github-export/l2o-isac` (or a clean checkout root).
Use `/workspace/.venv/bin/python` in the current environment; a clean CPU install
is documented in architecture_commands.md with requirements-audit-lock.txt.
No numerical dependencies were added. Existing original checkpoint is mandatory.

```bash
python -m pytest -q
python -m l2o_isac.controlled_study --stage prepare
python -m l2o_isac.controlled_study --stage profile
python -m l2o_isac.controlled_study --stage pilot
python -m l2o_isac.controlled_study --stage balanced
python -m l2o_isac.controlled_study --stage pretrain
python -m l2o_isac.controlled_study --stage expanded
python -m l2o_isac.controlled_study --stage clean
python -m l2o_isac.controlled_study --stage pgda
python -m l2o_isac.controlled_study --stage timing
python -m l2o_isac.controlled_report
```

All are actual implemented stages. `--stage all` performs them in dependency order
and reuses completed checkpoints/results. `pilot`/`balanced`/`expanded` retain
terminal-robust-loss variants; `clean` runs the primary loss-matched B/C/D controls.
The original A model is loaded and frozen. The mapping-only condition is generated
from the exact reference before training. E_previous loads the existing historical
separated checkpoints without retraining.

Individual continuation conditions (default seed101/reference-index0):

```bash
python -m l2o_isac.controlled_study --stage condition --condition B --fine-seed 101 --reference-index 0
python -m l2o_isac.controlled_study --stage condition --condition C --fine-seed 101 --reference-index 0
python -m l2o_isac.controlled_study --stage condition --condition D --fine-seed 101 --reference-index 0
```

Those individual commands run the retained terminal-protocol condition family;
use `--stage clean` for all three primary loss-matched conditions together.
Primary labels in reports are B_clean/C_clean/D_clean. Raw names make this explicit.
The shared configuration is configs/pilot/controlled_ablation.json; separate frozen
compute profiles record the chosen repeat schedules without changing datasets or
attack settings. Calling any completed training stage resumes from latest.pt or
returns the existing selected checkpoint; it does not restart an original model.

Additional independent references use the established pipeline, not a reimplementation:

```bash
python -m l2o_isac.experiment --profile paper --seed 74129 --output reports/additional_reference_seed74129
python -m l2o_isac.experiment --profile paper --seed 75129 --output reports/additional_reference_seed75129
```

These exact commands were executed through the guarded pretrain stage, which
refuses to overwrite an existing incomplete reference directory and reuses a
complete matching checkpoint. Do not manually rerun them into an existing folder.
Their outputs include command/source hashes, software versions, full training logs
and new weights. Original pretraining seed73129 is preserved unchanged. Original
reference source/weights and prior audit remain archived/pinned at73159d4.

Controlled checkpoints include model and optimizer states, Torch and minibatch RNG
states, epoch, per-minibatch compute counts, configs, source hashes, reference hashes
and split hash. Writes are atomic. Additional reference training uses the unchanged
historical pipeline and its historical checkpoint format; it is not advertised as
mid-epoch resumable. Do not overwrite successful reference files to change metadata.

The first fractional-repeat attempt exposed a NumPy scalar metadata serialization
issue with `weights_only=True`. Failed originals are preserved in a separate folder;
repaired copies convert that scalar to a native float with identical weights/RNG.
A dedicated resume test covers this case. Failure/repair records are saved, not
silently excluded. Final scientific results use safely loadable checkpoints.
