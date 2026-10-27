# Initial IEEE Wireless Communications Letters draft

Working title: **Learning to Optimize Robust Hybrid ISAC with Fluid Transmit Antennas: A Controlled Study**.

- `main.pdf`: four-page, 10-point, double-column IEEE draft, with measured tables and figures.
- `main.tex`: editable manuscript; author names/affiliations are explicit placeholders.
- `main_single_column.pdf`: double-spaced single-column companion generated from the same source.
- `initial_model_user.tex` / `.pdf`: the supplied mathematical basis, preserved verbatim and compiled separately. This is source material, not a claim that a journal permits an extra appendix outside its page limit.
- `make_assets.py`: produces the three table fragments and two vector figures directly from saved experiment summaries. It performs no training, optimization, or test selection.
- `asset_provenance.json`: result-source hashes and pinned experiment commit.
- `DRAFT_NOTES.md`: claim boundaries, initial literature positioning, and remaining submission work.
- `verification.json`: manuscript/data checks and source hashes.

From this directory, run:

```bash
MPLCONFIGDIR=/tmp/wcl-mpl /workspace/.venv/bin/python make_assets.py
bash build.sh
/workspace/.venv/bin/python verify_draft.py
```

Python executable can be replaced with your local environment containing NumPy and Matplotlib. Compilation requires pdfLaTeX and standard AMS, graphicx, booktabs, hyperref, and balance packages. The unmodified IEEEtran 1.8b class is bundled in `vendor/IEEEtran.cls` from CTAN; its license is included in the class header. `build.sh` sets the class search path. No BibTeX run is needed because the initial bibliography is embedded.

The draft uses existing `experiments/fluid_tx_single_user/outputs/pilot/` results at experiment commit `dd842239b2afbda03d9b3b6d6742c4e07ef5dc81`. The reproduction, checkpoints, and experimental results are unchanged. Tables use `shared_union` independently attacked values, including the analytic case-A reference and separately validation-retuned long ULA baselines.

WCL requirements checked on 2026-10-26: at most five pages in the 10-point double-column IEEE format, including references; 75–100-word abstract; a double-spaced single-column version is also requested. Source: https://www.comsoc.org/publications/journals/ieee-wcl/policies-guidelines . Author information will consume some additional space; this draft has page-count headroom. This is a working research draft, not a submitted or accepted paper.

The preserved initial model retains one minor overfull text box from its original formatting; the two new manuscript PDFs have none. Source-only archives can compile the supplied tables/figures directly with `bash build.sh`; regenerating assets and running data-linkage verification additionally require the repository results at the relative location described above.
