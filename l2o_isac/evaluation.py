"""Held-out evaluation and independent multistart adversarial stress testing.

The adversarial search finds feasible errors; it is not a global worst-case certificate.
In particular, the alternating solver's own error variables are NOT used as proof of robustness.
"""
from dataclasses import replace
import math
import time
import torch
from .data import initialize_state
from .physics import metrics, gradients
from .solver import PGDA, project_ball, constraint_errors


@torch.no_grad()
def adversarial_evaluate(batch, design, cfg, steps=80, restarts=4, seed=1000, normalize_gradients=False):
    if steps < 1 or restarts < 1:
        raise ValueError("Need positive attack steps and restarts")
    best_comm, best_sense = None, None
    best_state = None
    for restart in range(restarts):
        random = initialize_state(batch, cfg, seed=seed + restart, zero_errors=restart == 0)
        current = replace(design, dh=random.dh, dt=random.dt, dr=random.dr)
        current_metrics = metrics(batch, current, cfg)
        lr_c = torch.full((len(batch), 1, 1), 0.05, device=batch.h.device, dtype=batch.eps.dtype)
        lr_s = lr_c.clone()
        for _ in range(steps):
            g = gradients(batch, current, cfg)
            proposals = {}
            for i, name in enumerate(('dh', 'dt', 'dr')):
                n = cfg.nr if name == 'dr' else cfg.nt
                lr = lr_c if name == 'dh' else lr_s
                gradient = g[name]
                if normalize_gradients:
                    gradient = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(gradient.real.dtype).tiny)
                proposals[name] = project_ball(getattr(current, name) - lr * gradient, batch.eps[:, i, None, None] * math.sqrt(n))
            candidate = replace(current, **proposals)
            candidate_metrics = metrics(batch, candidate, cfg)
            accept_c = candidate_metrics['comm'] <= current_metrics['comm']
            accept_s = candidate_metrics['sense'] <= current_metrics['sense']
            updates = {name: torch.where((accept_c if name == 'dh' else accept_s)[:, None, None],
                                          proposals[name], getattr(current, name)) for name in proposals}
            current = replace(current, **updates)
            current_metrics = metrics(batch, current, cfg)
            lr_c = torch.where(accept_c[:, None, None], lr_c * 1.05, lr_c * 0.5).clamp(max=1.)
            lr_s = torch.where(accept_s[:, None, None], lr_s * 1.05, lr_s * 0.5).clamp(max=1.)
        if best_state is None:
            best_state = current
            best_comm, best_sense = current_metrics['comm'], current_metrics['sense']
        else:
            take_c, take_s = current_metrics['comm'] < best_comm, current_metrics['sense'] < best_sense
            best_state = replace(best_state,
                                 dh=torch.where(take_c[:, None, None], current.dh, best_state.dh),
                                 dt=torch.where(take_s[:, None, None], current.dt, best_state.dt),
                                 dr=torch.where(take_s[:, None, None], current.dr, best_state.dr))
            best_comm = torch.minimum(best_comm, current_metrics['comm'])
            best_sense = torch.minimum(best_sense, current_metrics['sense'])
    return metrics(batch, best_state, cfg), best_state


def statistics(x):
    return {"mean": x.mean().item(), "std": x.std(unbiased=len(x) > 1).item(),
            "sem": (x.std(unbiased=len(x) > 1) / math.sqrt(len(x))).item(),
            "min": x.min().item(), "max": x.max().item()}


@torch.no_grad()
def evaluate_methods(cfg, test, model, epsnet, recurrence, batch_size=250,
                     benchmark_iterations=180, attack_steps=80, attack_restarts=4,
                     attack_samples=128, seed=999):
    methods = {
        'B-PGDA': (PGDA(cfg, iterations=benchmark_iterations).to(test.h.device), {}, False),
        'U-PGDA': (model, {}, True),
        'RU-PGDA+5': (model, dict(extra_layers=5, **recurrence), True),
        'RU-PGDA+10': (model, dict(extra_layers=10, **recurrence), True),
        'Nonrobust-PGA': (PGDA(cfg, iterations=benchmark_iterations).to(test.h.device), {}, False),
    }
    report, arrays = {}, {}
    for name, (solver, kwargs, predict_eps) in methods.items():
        values = {key: [] for key in ('comm', 'sense', 'objective')}
        nominals = {key: [] for key in values}
        attacked = {key: [] for key in values}
        curves, violations = [], []
        total_seconds = 0.
        for i, start in enumerate(range(0, len(test), batch_size)):
            batch = test[start:start + batch_size]
            initial = initialize_state(batch, cfg, seed=seed + i)
            if name == 'Nonrobust-PGA':
                eps = torch.zeros_like(batch.eps)
                initial = replace(initial, dh=initial.dh * 0, dt=initial.dt * 0, dr=initial.dr * 0)
            else:
                eps = epsnet(batch) if predict_eps else batch.eps
            started = time.perf_counter()
            state, trace, _ = solver(batch, initial, eps=eps, **kwargs)
            total_seconds += time.perf_counter() - started
            for key in values:
                values[key].append(trace[-1][key].cpu())
            curves.append(torch.stack([t['objective'] for t in trace], dim=-1).cpu())
            violations.append(constraint_errors(state, batch, cfg, eps=eps))
            nominal = replace(state, dh=state.dh * 0, dt=state.dt * 0, dr=state.dr * 0)
            for key, v in metrics(batch, nominal, cfg).items():
                nominals[key].append(v.cpu())
            if start < attack_samples:
                n = min(len(batch), attack_samples - start)
                substate = type(state)(**{k: getattr(state, k)[:n] for k in ('dh', 'dt', 'dr', 'a', 'd', 'w')})
                attack, attack_state = adversarial_evaluate(batch[:n], substate, cfg, steps=attack_steps,
                                                           restarts=attack_restarts, seed=seed + 100 + i)
                for key, v in attack.items():
                    attacked[key].append(v.cpu())
                violations.append(constraint_errors(attack_state, batch[:n], cfg))
        values = {k: torch.cat(v) for k, v in values.items()}
        nominals = {k: torch.cat(v) for k, v in nominals.items()}
        attacked = {k: torch.cat(v) for k, v in attacked.items()}
        if not all(torch.isfinite(v).all() for d in (values, nominals, attacked) for v in d.values()):
            raise FloatingPointError(f"Nonfinite evaluation for {name}")
        report[name] = {"internal": {k: statistics(v) for k, v in values.items()},
                        "nominal": {k: statistics(v) for k, v in nominals.items()},
                        "adversarial": {k: statistics(v) for k, v in attacked.items()},
                        "attack_samples": len(attacked['objective']),
                        "max_constraint_errors": {k: max(v[k] for v in violations) for k in violations[0]},
                        "solver_seconds_total": total_seconds,
                        "solver_ms_per_sample": total_seconds / len(test) * 1000,
                        "objective_curve": torch.cat(curves).mean(0).tolist()}
        arrays[name] = {"internal": values, "nominal": nominals, "adversarial": attacked}
        print(f"{name}: internal={values['objective'].mean():.4f}, independently attacked={attacked['objective'].mean():.4f}", flush=True)
    return report, arrays
