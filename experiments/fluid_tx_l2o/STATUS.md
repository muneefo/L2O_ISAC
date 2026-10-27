# Coupled fluid-transmit L2O status

Completed focused workflow in experiments/fluid_tx_l2o/. Baseline4fc8a44: all1041 pre-existing tracked files remain byte-identical. Old reproduction/checkpoints, earlier fluid pilot and previous manuscript files were preserved.

## Completed execution

- Inspected repository source and original retained full-paper extraction; mapped Eqs8–22/appendices to code in FORMULATION.md. Official paper PDF could not be retrieved (HTTP403 / stalled alternate tool request); actual readable source is sources/paper_preexisting_extract.txt, copied from /workspace/reports/paper_extracted.txt. No official author-code claim.
- Implemented original MinNet→MaxNet plus spatial reconstruction, real position partial, exact PAV projection and optional learned position momentum. All inherited six step/momentum schedules train jointly with the new position schedules. Fixed20 layers and common radii; no EpsNet, RecNet or trust-region replacement.
- Passed78 tests: original regressions plus geometry/physics and coupled derivatives, frozen-variable, feasibility and recipient-pool tests. End-to-end smoke run completed.
- Saved fresh immutable128/32/100 train/validation/test splits per regime and protocol lock. All21 scheduled trainings completed:18 primary (fixed/fluid ×3 seeds×3 regimes),3 momentum ablations. SGD1e-4, decay.98,10epochs,40updates/model; no original checkpoint transfer or original training rerun.
- Tuned classical controls on validation only, including longer fixed budgets. Evaluated67 designs with100 paired scenarios/regime; common zero/terminal/random/Adam and50/200/1000 attacks; best retention and corrected shared union. Additional fixed-random and training-fitted frozen geometry controls included.
- Measured warm batch-one CPU initialize+full-design timing (20 scenarios×3 repeats;3warmups), approximate validation-selected runtime controls, raw timings, median/p95, offline fitting cost and attack cost separately. No physical movement/RF/CSI timing.
- Generated raw scenario CSV, component utilities, feasibility NPZ, paired confidence intervals, gradient/runtime curves, attack saturation, common-Pt beampatterns and position plots.
- Independent NumPy verification and full design replay passed for all67 outputs; max replay difference0.0; max feasibility residual1.6689301e-06.
- Editable TikZ system/architecture diagrams plus PDFs; updated letter in manuscripts/wcl_fluid_tx_l2o/ (old letter preserved). The letter has4 double-column pages, a94-word abstract, and a7-page single-column companion. All letter pages and both diagrams were rendered and visually inspected. No unresolved references or overfull boxes in the new letter formats. Export verification is recorded with the delivery archive.

## Measured interpretation

B fluid learning gain over identical untrained policy: +0.010223, paired95%[0.008499391078948974, 0.011952931344509126]. C selects epoch0 for all fluid models: no learning gain. Longer classical fluid solves attain higher utility than the short learner. Geometry gains, backtracking overhead, and actual learning effects are kept separate. Position momentum has no practical observed effect in the one-seed ablation. Direct coefficient signs are diagnostics, not robust certificates.

## Remaining scientific work

No broad L2O acceleration/generalization or publication-novelty claim is established. Full-source closest-literature comparison, competitive AO/stronger fixed-geometry optimization, wider channel/aperture/uncertainty/initialization coverage, more momentum seeds and physical imperfections are outside this completed focused pilot. No post-test rescue tuning was performed. Author names/affiliations and final submission positioning remain open. PDF recovery remains unresolved; available full text is explicitly identified.

Commands actually executed and reproducible entry points are in README.md; configs, outputs/events.log and stage logs retain execution details. Corrections are recorded in sources/WORK_HISTORY.md; the rejected initial union scores are archived and excluded from claims.
