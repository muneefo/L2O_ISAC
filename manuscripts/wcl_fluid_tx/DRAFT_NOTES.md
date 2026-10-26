# Draft scope and next revision

## What the draft currently says

The source supplied by the user is preserved without edits in `initial_model_user.tex`. The letter compresses its geometry, field-response channel, two-stream hybrid signal model, SINRs, separate uncertainty balls, and weighted robust problem. It retains the radar transpose convention, fixed receive ULA, sensing-stream interference, natural logarithms, and incoherent-clutter approximation. Position derivatives are expressed by the equivalent compact real/Wirtinger chain rule; the original supplied scalar-power derivations remain in the preserved source.

The method describes the implementation that generated the results: one inner error update, one outer beamformer update, then one fixed-error position partial; zero position momentum; exact PAV geometry projection; up to six Armijo trials for classical solves. Hardware normalization is described as the implemented phase mapping and rescaling, without claiming joint Euclidean projection onto the nonconvex hybrid feasible set.

The aligned-LOS proposition is derived in the draft, with a constructive hybrid-feasible solution and a matching adversarial upper bound. It is limited to aligned single-path, no-clutter, unit-magnitude communication gain, the stated unstructured balls, and radii below one. It does not certify arbitrary B/C solutions. The position projection is standard bounded isotonic regression, not asserted to be a new optimization primitive.

Main measured claims: 11.8% and 10.6% attacked-utility gains over validation-retuned fixed ULA at 360 full helper calls, conditional on the two tested multipath distributions. Three methods use the same physical resources. The static-random comparator is one fixed draw, so it does not establish performance relative to the best possible fixed nonuniform array. The learned schedule is a negative ablation, with equally trained scalar control; no useful learning advantage is claimed. The C/rho=0.2 counterexample explicitly shows that weighted gains need not improve both tasks.

## Initial literature positioning, checked 2026-10-26

1. Zhou et al., *Fluid Antenna-Assisted ISAC Systems*, WCL, DOI 10.1109/LWC.2024.3476148. Author-deposited record and abstract: https://discovery.ucl.ac.uk/id/eprint/10199665/ . Fluid BS/user antennas, rate and sensing-beampattern design already exist. Do not claim first fluid-antenna ISAC.
2. Hao et al., *Fluid-Antenna Enhanced ISAC: Joint Antenna Positioning and Dual-Functional Beamforming Design under Perfect and Imperfect CSI*, author preprint https://arxiv.org/abs/2407.18988 . Its abstract describes sensing-SNR optimization with user SINR constraints, antenna constraints, and both communication/sensing CSI uncertainty. Do not claim first robust fluid ISAC. Search also identifies published DOI 10.1109/TVT.2025.3574072; the publisher page was inaccessible through the web tool, so the draft deliberately cites the directly inspected author preprint rather than inventing final pagination.
3. Zhang et al., *Robust Beamforming and Antenna Position Optimization for MA-Assisted ISAC with Imperfectly Positioned MAs*, https://arxiv.org/abs/2609.23323 . Author abstract explicitly models position errors and CRB/SINR design. Position-error robustness is already studied and is not present in our pilot.
4. Wang, Vorobyov, and Ollila, *L2O Robust Hybrid Beamforming for ISAC*, TWC vol.25, pp.18778–18794, DOI 10.1109/TWC.2026.3702925. Metadata and objective checked at the institutional record: https://research.aalto.fi/fi/publications/l2o-robust-hybrid-beamforming-for-isac/ . This is the inherited robust hybrid formulation, not our own invention. Our code is an independent implementation.
5. A further close work surfaced: *Robust Beamforming and Antenna Position Optimization in Movable Antenna Assisted ISAC Systems*, DOI 10.1109/TCOMM.2026.3663183. Search summaries describe position-dependent uncertainty. Its publisher page could not be read through the tool. Full source comparison is pending; no unverified detailed claim about this work is placed in the manuscript.

This was an initial source/abstract screen, not an exhaustive full-text novelty audit. The source paper mathematical structure was supplied in this conversation and checked against existing implementation records. The original attachment PDF is not newly recovered here.

## What remains before a submission-ready version

- Establish a sharper technical distinction against the closest full-text robust fluid/movable-array papers. The combination of fluid positioning and robustness alone is insufficient. The current results support a focused experimental story, not an assured WCL novelty claim or acceptance prediction.
- Strengthen the experimental coverage: aperture/spacing and uncertainty sweeps; additional path/clutter laws; more initializations; a stronger fixed-array optimizer and a competitive robust antenna-positioning baseline under exactly the same objective. These runs have not been done during drafting and are not implied by any placeholder curve.
- The fixed random control is a single draw; test additional fixed geometries before making a general statement about optimized geometry versus fixed nonuniform arrays.
- Decide whether the small negative learning ablation deserves space in the final letter. It is retained now so the draft reflects the completed study faithfully. A learning-focused title is unsupported.
- If practical robustness/latency becomes the contribution, implement and evaluate physical position errors, coupling, CSI/field acquisition, movement cost/settling, and RF programming. Current timing is CPU design time, approximately 47–53 ms for the short fluid solves, not sub-millisecond hardware operation. No such physical assumptions are silently added here.
- Independent attacks yield feasible minima candidates and upper bounds on true robust utility; tiny observed budget changes cannot prove global attack optimality. Include stronger search if making stronger robustness claims.
- Add actual author names, affiliations, contact details, funding, and final bibliography metadata. Review the final wording, proofs, cited full texts, and applicable publication disclosures before submission. No submission, author consent, or copyright transfer has been performed.

## Evidence linkage

All draft tables and figures are regenerated from existing pilot summaries, whose hashes are recorded. Confidence intervals are the saved paired channel-bootstrap intervals, conditional on the tested models/scenario distribution; they are not independent replications of whole experimental pipelines. The 360-call main ULA is the separately retuned long reference. Figure 1 uses validation-retuned ULA for the 180/360 budgets. Figure 2 is separately retuned for each rho at 60 calls. Table III averages three paired training runs while retaining 100 independent test scenarios, including unfavorable selections.

## Build checks

The letter has four pages and a 95-word abstract; its same-source single-column double-spaced companion has eight pages. Neither new manuscript format has unresolved citations/references or overfull boxes. The verbatim initial model compiles to five pages and retains one minor 3.6-point overfull text box from the supplied source; this is recorded rather than silently editing the original. An independent NumPy calculation of the constructive LOS solution agrees with its closed form over 30 random draws to maximum absolute error below 1e-12. This checks the construction numerically and does not replace the proof or certify the non-LOS attacks.
