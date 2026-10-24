"""The three-stage procedure of Figure 2 / Algorithms 2 and 3."""
import copy
import time
import torch
from .data import initialize_state
from .networks import eps_rmse
from .solver import unrolled_loss, pgda_step
from .physics import metrics


def minibatches(data, batch_size, generator, shuffle=True):
    indices = torch.randperm(len(data), generator=generator) if shuffle else torch.arange(len(data))
    for ids in indices.split(batch_size):
        yield data[ids.to(data.h.device)]


@torch.no_grad()
def bound_metrics(model, data):
    model.eval()
    predicted = model(data)
    return {"rmse": eps_rmse(predicted, data.eps).item(),
            "rmse_by_bound": (predicted - data.eps).square().mean(0).sqrt().tolist(),
            "mean_predicted": predicted.mean(0).tolist(), "mean_true": data.eps.mean(0).tolist()}


def train_epsnet(model, train, valid, epochs=10, batch_size=250, lr=0.05, seed=0):
    if batch_size < 2:
        raise ValueError("EpsNet BatchNorm requires batch_size >= 2")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history, best, best_loss = [], None, float('inf')
    for epoch in range(epochs):
        model.train()
        total = 0.
        count = 0
        for batch in minibatches(train, batch_size, generator):
            if len(batch) == 1:
                # Keep singleton batch while using the running BN statistics.
                model.eval()
            optimizer.zero_grad()
            loss = eps_rmse(model(batch), batch.eps)
            if not torch.isfinite(loss):
                raise FloatingPointError("EpsNet loss is nonfinite")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            count += len(batch)
            model.train()
        evaluation = bound_metrics(model, valid)
        history.append(dict(epoch=epoch + 1, train_rmse=total / count, validation_rmse=evaluation['rmse']))
        if evaluation['rmse'] < best_loss:
            best_loss, best = evaluation['rmse'], copy.deepcopy(model.state_dict())
        print(f"EpsNet {epoch+1}/{epochs}: train RMSE={total/count:.5f}, validation={evaluation['rmse']:.5f}", flush=True)
    model.load_state_dict(best)
    model.eval()
    return history


@torch.no_grad()
def average_network_loss(model, data, epsnet, batch_size=250, seed=0, **forward_kwargs):
    model.eval()
    total = 0.
    generator = torch.Generator().manual_seed(0)
    for i, batch in enumerate(minibatches(data, batch_size, generator, shuffle=False)):
        initial = initialize_state(batch, model.cfg, seed=seed + i)
        eps = epsnet(batch) if epsnet is not None else batch.eps
        _, trace, _ = model(batch, initial, eps=eps, **forward_kwargs)
        total += unrolled_loss(trace).item() * len(batch)
    return total / len(data)


def train_unrolled(model, train, valid, epsnet=None, epochs=10, batch_size=250,
                   lr=0.0001, decay=0.98, seed=0):
    generator = torch.Generator().manual_seed(seed)
    if epsnet is not None:
        epsnet.eval()
        for parameter in epsnet.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    initial_loss = average_network_loss(model, valid, epsnet, batch_size, seed=seed + 10000)
    # Model selection on validation, including initialization, prevents reporting a worse model as learned.
    best, best_loss = copy.deepcopy(model.state_dict()), initial_loss
    history = []
    for epoch in range(epochs):
        model.train()
        total, largest_gradient = 0., 0.
        started = time.perf_counter()
        for i, batch in enumerate(minibatches(train, batch_size, generator)):
            initial = initialize_state(batch, model.cfg, seed=seed + epoch * 10000 + i)
            with torch.no_grad():
                eps = epsnet(batch) if epsnet is not None else batch.eps
            optimizer.zero_grad()
            _, trace, _ = model(batch, initial, eps=eps)
            loss = unrolled_loss(trace)
            if not torch.isfinite(loss):
                raise FloatingPointError("Unrolled loss is nonfinite")
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError("Nonfinite parameter gradient")
                    largest_gradient = max(largest_gradient, p.grad.abs().max().item())
            optimizer.step()
            model.project_hyperparameters()
            total += loss.item() * len(batch)
        val = average_network_loss(model, valid, epsnet, batch_size, seed=seed + 10000)
        if val < best_loss:
            best, best_loss = copy.deepcopy(model.state_dict()), val
        history.append(dict(epoch=epoch + 1, train_loss=total / len(train), validation_loss=val,
                            max_abs_gradient=largest_gradient, lr=scheduler.get_last_lr()[0],
                            seconds=time.perf_counter() - started))
        scheduler.step()
        print(f"U-PGDA {epoch+1}/{epochs}: train objective={-total/len(train):.5f}, validation={-val:.5f}", flush=True)
    model.load_state_dict(best)
    model.eval()
    return {"initial_validation_loss": initial_loss, "best_validation_loss": best_loss, "epochs": history}


@torch.no_grad()
def recurrent_grid_search(model, valid, epsnet=None, grids=10, extra_layers=5,
                          batch_size=250, seed=0):
    model.eval()
    # The frozen unrolled prefix is identical for every grid point. Cache its two
    # final states (momentum needs both) and trace instead of recomputing it G² times.
    cached = []
    generator = torch.Generator().manual_seed(0)
    for i, batch in enumerate(minibatches(valid, batch_size, generator, shuffle=False)):
        initial = initialize_state(batch, model.cfg, seed=seed + i)
        eps = epsnet(batch) if epsnet is not None else batch.eps
        current, trace, states = model(batch, initial, eps=eps, return_states=True)
        previous = states[-2] if len(states) > 1 else initial
        cached.append((batch, eps, current, previous, trace))
    values = torch.linspace(0.01, 0.99, grids).tolist()
    scores = []
    final_mu, final_eta = model.mu[-1], model.eta[-1]
    for beta in values:
        for phi in values:
            total = 0.
            for batch, eps, initial_current, initial_previous, prefix_trace in cached:
                current, previous = initial_current, initial_previous
                trace = list(prefix_trace)
                for depth in range(1, extra_layers + 1):
                    next_state = pgda_step(batch, current, previous, model.cfg,
                                           final_mu * beta ** depth, final_eta * phi ** depth, eps)
                    previous, current = current, next_state
                    trace.append(metrics(batch, current, model.cfg))
                total += unrolled_loss(trace).item() * len(batch)
            loss = total / len(valid)
            scores.append(dict(beta=beta, phi=phi, loss=loss))
    best = min(scores, key=lambda r: r['loss'])
    return {"best": best, "extra_training_layers": extra_layers, "scores": scores}
