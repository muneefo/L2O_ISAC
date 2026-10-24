# Robust ISAC architecture audit — live status

Reference: independent corrected reimplementation at commit
83ba89148c8d89a7ab60105da0fa5cbfe6105d9e. Existing checkpoint and historical
reports remain immutable. No reference reproduction or training repeated.

- Gate A: equation map and graph/ambiguity audit written before extension training.
- Gate B: existing 35 tests available; new attacker/role tests pending.
- Gate C: existing raw-output checkpoint regression pending.
- Gate D: nested independent frozen-beamformer audit pending.
- Gate E: role-separated training gated on B–D; objective specified in docs.
- F/G: optional controller/calibration not started; no novelty claim.

Resources: CPU only, cgroup quota two CPUs; PyTorch one thread, complex64
experiments and complex128 reference tests. Pilot target 128 test channels,
5 random restarts plus zero/terminal candidates, 50/200 PGD steps and Adam.
Full-study commands will be provided separately; pilot is not publication ready.

The original PDF attachment path is absent; retained complete extracted text and
prior PDF reading evidence establish source identity without blocking reuse.
