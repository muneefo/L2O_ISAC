# Coupled fluid-transmit L2O letter draft

This revision follows the latest focused implementation brief. It retains inherited MinNet–MaxNet, learns all beamforming/error/position schedules, and uses the new spatial dataset and coupled-study outputs. It does not reuse the previous pilot's percentages as new L2O results. Earlier manuscript files remain untouched.

- `main.tex` / `main.pdf`: editable letter and compiled double-column draft.
- `main_single_column.pdf`: same-source double-spaced single-column companion.
- `initial_model_user.tex`: unchanged mathematical model supplied by the user.
- `make_assets.py`: regenerates tables and figures from `experiments/fluid_tx_l2o/outputs/study/summary.json` and the saved raw designs.
- System and architecture diagram sources/PDFs: `experiments/fluid_tx_l2o/diagrams/`.

From repository root:

```bash
python -m experiments.fluid_tx_l2o.report
python manuscripts/wcl_fluid_tx_l2o/make_assets.py
bash manuscripts/wcl_fluid_tx_l2o/build.sh
python -m experiments.fluid_tx_l2o.evidence
```

Author identities, affiliations, final novelty/full-literature assessment, broader baselines and physical validation remain open. This is a measured working draft, not a submitted paper. The main result is conditional: the coupled learner has a modest B gain and no C training gain under the inherited training/selection procedure. No sub-millisecond or general learning superiority is claimed. See the study REPORT.md, CLAIMS.md, FORMULATION.md and STATUS.md.
