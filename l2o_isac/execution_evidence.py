"""Run fresh training with paper/old-result reads forbidden, and retain raw evidence.

This demonstrates execution of this reconstruction. It does not establish that
every unspecified detail matches the authors' unavailable implementation.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbid_reference_inputs(root, output, events):
    """A process-local audit hook: fresh training cannot load reference artifacts."""
    root, output = root.resolve(), output.resolve()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing:
            return
        old_result = path.is_relative_to(root / 'reports') and not path.is_relative_to(output)
        reference = not path.is_relative_to(output) and (
            path.name == 'paper_extracted.txt' or path.suffix.lower() == '.pdf')
        attachment = path.is_relative_to(Path('/tmp/codex-remote-attachments'))
        if old_result or reference or attachment:
            events.append(str(path))
            raise PermissionError(f'Fresh execution forbids reading reference artifacts: {path}')
    sys.addaudithook(audit)


def numpy_metrics(batch, state, cfg):
    """Literal signal equations, separate from physics.py and PyTorch gradients."""
    def arr(x):
        return x.detach().cpu().numpy().astype(np.complex128)
    h, at, ar = arr(batch.h + state.dh), arr(batch.at + state.dt), arr(batch.ar + state.dr)
    a, d, w, alpha = arr(state.a), arr(state.d), arr(state.w), arr(batch.alpha)
    comm, sense = [], []
    for b in range(len(batch)):
        p = a[b] @ d[b]
        gc = 0.
        for m in range(cfg.users):
            powers = np.abs(h[b, m].conj() @ p) ** 2
            noise_interference = cfg.noise_comm + sum(powers[k] for k in range(cfg.streams) if k != m)
            gc += np.log1p(powers[m] / noise_interference)
        radar = alpha[b, :, None, None] * ar[b, :, :, None] * at[b, :, None, :]
        gs = 0.
        for t in range(cfg.targets):
            powers = [float(np.sum(np.abs(w[b, t].conj() @ radar[l] @ p) ** 2)) for l in range(cfg.paths)]
            noise_interference = cfg.noise_sense * np.sum(np.abs(w[b, t]) ** 2) + sum(
                powers[l] for l in range(cfg.paths) if l != t)
            gs += np.log1p(powers[t] / noise_interference)
        comm.append(gc)
        sense.append(gs)
    comm, sense = np.array(comm), np.array(sense)
    return dict(comm=comm, sense=sense, objective=cfg.rho * comm + (1 - cfg.rho) * sense)


@torch.no_grad()
def verify_raw_outputs(output, seed):
    from .data import generate_channels, initialize_state
    from .inference import load_checkpoint
    from .physics import metrics
    from .solver import UnrolledPGDA, constraint_errors
    cfg, epsnet, model, saved = load_checkpoint(output / 'checkpoint.pt')
    batch = generate_channels(cfg, 8, seed=seed + 300)
    initial = initialize_state(batch, cfg, seed=seed + 301)
    eps = epsnet(batch)
    final, trace, _ = model(batch, initial, eps=eps)
    measured = metrics(batch, final, cfg)
    independent = numpy_metrics(batch, final, cfg)
    differences = {}
    for key in measured:
        differences[key] = float(np.max(np.abs(measured[key].cpu().numpy() - independent[key])))
        np.testing.assert_allclose(measured[key].cpu().numpy(), independent[key], rtol=1e-5, atol=1e-6)
    untrained = UnrolledPGDA(cfg, layers=model.layers, parameterization=model.parameterization)
    before, _, _ = untrained(batch, initial, eps=eps)
    before_objective = metrics(batch, before, cfg)['objective']
    changed_h = replace(batch, h=batch.h * (1.3 + 0.2j))
    changed = metrics(changed_h, final, cfg)['objective']
    assert not torch.allclose(changed, measured['objective'])
    zero_power = metrics(batch, replace(final, d=torch.zeros_like(final.d)), cfg)
    assert all(torch.count_nonzero(x) == 0 for x in zero_power.values())
    parameter_changes = {
        'mu_changed_entries': int(torch.count_nonzero(model.mu - untrained.mu)),
        'eta_changed_entries': int(torch.count_nonzero(model.eta - untrained.eta)),
        'mu_l2_change': (model.mu - untrained.mu).norm().item(),
        'eta_l2_change': (model.eta - untrained.eta).norm().item(),
    }
    assert parameter_changes['mu_changed_entries'] > 0
    raw = dict(channel={k: getattr(batch, k) for k in ('h', 'at', 'ar', 'alpha', 'eps')},
               initial=vars(initial), trained=vars(final), untrained=vars(before), predicted_eps=eps,
               objective_trace=torch.stack([x['objective'] for x in trace]),
               seed=seed, channel_seed=seed + 300, initialization_seed=seed + 301)
    torch.save(raw, output / 'raw_beamformers_and_channels.pt')
    np.savez_compressed(output / 'independent_numpy_metrics.npz', **independent)
    evidence = dict(
        independent_numpy_max_abs_error=differences,
        learned_parameter_changes=parameter_changes,
        untrained_objective_per_sample=before_objective.tolist(),
        trained_objective_per_sample=measured['objective'].tolist(),
        changed_channel_objective_per_sample=changed.tolist(),
        zero_transmit_signal_objectives={k: v.tolist() for k, v in zero_power.items()},
        constraint_residuals=constraint_errors(final, batch, cfg, eps),
    )
    (output / 'raw_output_checks.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=73129)
    parser.add_argument('--output', type=Path, default=Path('reports/fresh_execution_seed73129'))
    parser.add_argument('--profile', choices=['paper', 'smoke'], default='paper')
    args = parser.parse_args(argv)
    root, output = Path.cwd().resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory; this evidence run does not overwrite previous results')
    output.mkdir(parents=True)
    command = [sys.executable, '-u', '-m', 'l2o_isac.execution_evidence', '--profile', args.profile,
               '--seed', str(args.seed), '--output', str(args.output)]
    test_command = [sys.executable, '-m', 'pytest', '-q']
    with (output / 'tests.log').open('w') as log:
        completed = subprocess.run(test_command, stdout=log, stderr=subprocess.STDOUT, cwd=root)
    if completed.returncode:
        raise RuntimeError(f'Tests failed; inspect {output / "tests.log"}')
    sources = [*sorted((root / 'l2o_isac').glob('*.py')), *sorted((root / 'tests').glob('*.py')), root / 'pyproject.toml']
    source_hashes = {str(p.relative_to(root)): digest(p) for p in sources}
    forbidden_reads = []
    forbid_reference_inputs(root, output, forbidden_reads)
    from .experiment import main as experiment
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    # Redirect actual training stdout/stderr at the OS-descriptor level, including library output.
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    try:
        with (output / 'training.log').open('w') as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            experiment(['--profile', args.profile, '--seed', str(args.seed), '--output', str(output)])
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
    checks = verify_raw_outputs(output, args.seed)
    manifest = dict(command=shlex.join(command), test_command=shlex.join(test_command),
                    started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - clock,
                    seed=args.seed, paper_and_previous_result_reads_forbidden=True,
                    attempted_forbidden_reads=forbidden_reads, source_sha256=source_hashes,
                    artifacts_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
                    scope='Fresh execution of an independent reconstruction; not exact-author-code certification')
    (output / 'execution_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Fresh execution evidence\n\n'
        f'Executed command:\n\n```bash\n{shlex.join(command)}\n```\n\n'
        'Training ran from newly initialized weights and seeded simulated channels. A process-local file audit '
        'hook rejected reads of the paper PDF, extracted paper text, attachments, and previous result directories. '
        'It permitted writes of the newly computed figures and results. No reference result was loaded.\n\n'
        '- `tests.log`: actual test output.\n'
        '- `training.log`: actual training and evaluation stdout/stderr.\n'
        '- `checkpoint.pt`: learned weights and recurrence parameters.\n'
        '- `raw_beamformers_and_channels.pt`: eight newly generated channels, initial and final complex '
        'beamformers, uncertainty vectors, untrained beamformers, and layer objectives.\n'
        '- `independent_numpy_metrics.npz`: objective values recomputed by literal NumPy signal equations.\n'
        '- `raw_output_checks.json`: parameter changes, NumPy agreement, channel-perturbation checks, '
        'zero-signal checks, and constraint residuals.\n'
        '- `execution_manifest.json`: command, timestamps, source and output hashes, and input-guard status.\n\n'
        'These demonstrate executed computation. They do not establish exact reproduction of unspecified '
        'author implementation choices or general superiority over the benchmark. See `summary.md` for results.\n')
    print(json.dumps({'output': str(output), 'elapsed_seconds': manifest['elapsed_seconds'],
                      'parameter_changes': checks['learned_parameter_changes'],
                      'numpy_agreement': checks['independent_numpy_max_abs_error'],
                      'forbidden_read_attempts': forbidden_reads}, indent=2))


if __name__ == '__main__':
    main()
