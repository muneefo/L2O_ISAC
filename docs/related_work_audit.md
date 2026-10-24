# Primary-source comparison and claim limits

Checked 2026-10-24. This is a targeted comparison, not an exhaustive novelty
search. No novelty claim is made for separating minimax roles or adding attacks.

| Verified source | Established idea | Difference / implication for this audit |
|---|---|---|
| Wang, Vorobyov, Ollila, [TWC 2026](https://research.aalto.fi/fi/publications/l2o-robust-hybrid-beamforming-for-isac/), DOI 10.1109/TWC.2026.3702925 | Robust hybrid ISAC; learned scalar inner/outer schedules, EpsNet and geometric continuation | We preserve its equations/sets and audit an independent reimplementation; no author weights available. Author portal confirms volume25,18778–18794. |
| Shen et al., [Learning a Minimax Optimizer: A Pilot Study, ICLR2021](https://github.com/VITA-Group/L2O-Minimax) | Twin L2O uses separate LSTMs for min/max; curriculum and safeguards under strong assumptions. Repository identifies MIT license. | Separate learned minimax roles are established. Our low-capacity scalar-schedule experiment is an application/ablation, not a new general minimax algorithm. No external code copied. |
| Chen, Vadori, Chen, Wang, [Learning to Optimize Differentiable Games, ICML2023](https://proceedings.mlr.press/v202/chen23ab.html) | L2PG predicts updates from trajectory using a gradient stability-aware sign-based loss, curriculum/self-learning. | State-dependent game optimizers are established. Optional state controller would need a specific ISAC advance and independent evidence, not a novelty claim for state inputs. |
| Grimmer, Lu, Worah, Mirrokni, [ALT2022](https://proceedings.mlr.press/v167/grimmer22a.html) | GDA, alternating GDA and extragradient can converge, cycle or diverge; analyzes limiting dynamics. | Projection or positive steps alone cannot be cited as a general minimax convergence proof. This does not refute a theorem under its actual assumptions. |
| Zheng, Zhu, So, Blanchet, Li, [Universal Gradient Descent Ascent Method for Nonconvex-Nonconcave Minimax Optimization, NeurIPS2023](https://papers.neurips.cc/paper_files/paper/2023/hash/a961dea42c23c3c0d01b79918701fb6e-Abstract-Conference.html) | Doubly smoothed GDA balances primal/dual updates. This title is source-paper reference[47]. | The paper's convergence discussion cites this work. Its algorithm is DS-GDA; automatic applicability to learned signed momentum/projected ISAC layers has not been established here. No convergence guarantee is claimed. |
| Su, Hou, He, Simeone, [Conformal Robust Beamforming via Generative Channel Models, arXiv2504.06934v2](https://arxiv.org/abs/2504.06934) | Separates generative/conformal uncertainty-set construction and robust min-max beamforming. | Conformal beamforming uncertainty is established. Optional calibration needs metadata, disjoint calibration and carefully stated coverage, rather than claiming RMSE implies guarantees. |

The 2025 ICASSP predecessor is **not** the 17-page TWC source:
https://aaltodoc.aalto.fi/items/d25e6a4f-a2df-490c-9467-d572988c29d5 .
Search found it prominently; it must not silently replace the journal formulation.
Current external checks establish metadata and comparisons, not absence of all
other code releases. An official TWC repository/checkpoint remains unverified.
