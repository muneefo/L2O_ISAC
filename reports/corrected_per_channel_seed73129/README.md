# Fresh execution evidence

Executed command:

```bash
/workspace/.venv/bin/python -u -m l2o_isac.execution_evidence --profile paper --seed 73129 --output reports/corrected_per_channel_seed73129
```

Training ran from newly initialized weights and seeded simulated channels. A process-local file audit hook rejected reads of the paper PDF, extracted paper text, attachments, and previous result directories. It permitted writes of the newly computed figures and results. No reference result was loaded.

- `tests.log`: actual test output.
- `training.log`: actual training and evaluation stdout/stderr.
- `checkpoint.pt`: learned weights and recurrence parameters.
- `raw_beamformers_and_channels.pt`: eight newly generated channels, initial and final complex beamformers, uncertainty vectors, untrained beamformers, and layer objectives.
- `independent_numpy_metrics.npz`: objective values recomputed by literal NumPy signal equations.
- `raw_output_checks.json`: parameter changes, NumPy agreement, channel-perturbation checks, zero-signal checks, and constraint residuals.
- `execution_manifest.json`: command, timestamps, source and output hashes, and input-guard status.

These demonstrate executed computation. They do not establish exact reproduction of unspecified author implementation choices or general superiority over the benchmark. See `summary.md` for results.
