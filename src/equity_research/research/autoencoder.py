"""A conditional autoencoder of returns (Gu, Kelly and Xiu, 2021): latent factors whose loadings a network learns.

    model = fit(train_months, k=3)                   # torch (the CPU is faster here); the optional group `deep`
    beta, f = predict(model, month)                   # each stock's loadings and the month's factors
    out = walk_forward(months, k=3, first_test=2015, last_test=2024)   # latent.evaluate's measures, out of sample

IPCA (``latent.py``) makes each stock's factor loadings a *linear* function of its characteristics. The conditional
autoencoder lets a small network learn that function instead, and keeps everything else:

1. **Beta network** - each stock's characteristics ``z_i,t`` (with a constant) pass through one hidden layer of 32
   ReLU units to ``K`` loadings ``beta_i,t`` (their "CA1" architecture).
2. **Factor network** - the month's managed portfolios ``x_t = Z_t' r_t+1 / N_t`` map linearly to ``K`` factors
   ``f_t``: the factors are portfolios of the stocks.
3. **Output** - ``r_hat_i,t+1 = beta_i,t' f_t``, fitted by least squares over every stock-month with an L1 penalty on
   the weights, by Adam, one month a step; training stops when the error on its last ``validation_months`` stops
   falling (early stopping), and the best epoch's weights are kept.
4. **Out of sample** - refitted each January on every earlier month; a test month's factors come from its own managed
   portfolios, its fitted managed returns are ``Z' r_hat / N`` (the total R-squared of the paper), and the tangency
   portfolio of the factors uses weights from the training months. Gu, Kelly and Xiu average several random starts;
   one seed is used here, so a rerun repeats it exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from equity_research.research.latent import Month, evaluate, tangency_weights

SEED = 42


def fit(
    months: Sequence[Month],
    k: int,
    *,
    hidden: int = 32,
    l1: float = 1e-5,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 5,
    validation_months: int = 12,
    seed: int = SEED,
    device: str = "cpu",
) -> Any:
    """The fitted (beta network, factor network) pair, trained on ``months`` with early stopping on the last ones.

    ``device``: where it trains (``"cuda"`` for a GPU; results then differ from the CPU's by rounding).
    """
    import torch  # noqa: PLC0415 - the optional deep group

    torch.manual_seed(seed)
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    lsize = months[0].z.shape[1]
    beta_net = torch.nn.Sequential(torch.nn.Linear(lsize, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, k))
    factor_net = torch.nn.Linear(lsize, k, bias=False)
    beta_net.to(device)
    factor_net.to(device)
    params = [*beta_net.parameters(), *factor_net.parameters()]
    optimiser = torch.optim.Adam(params, lr=lr)
    data = [
        (torch.tensor(m.z, dtype=torch.float32).to(device), torch.tensor(m.r, dtype=torch.float32).to(device))
        for m in months
    ]
    train, valid = data[:-validation_months], data[-validation_months:]

    def loss_of(z: Any, r: Any) -> Any:
        x = z.T @ r / len(r)
        return torch.mean((r - beta_net(z) @ factor_net(x)) ** 2)

    best, best_state, waited = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(train)):
            z, r = train[i]
            optimiser.zero_grad()
            penalty = sum(p.abs().sum() for p in params)
            (loss_of(z, r) + l1 * penalty).backward()
            optimiser.step()
        with torch.no_grad():
            error = float(sum(loss_of(z, r) for z, r in valid) / len(valid))
        if error < best - 1e-9:
            best, waited = error, 0
            best_state = ([p.detach().clone() for p in beta_net.parameters()],
                          [p.detach().clone() for p in factor_net.parameters()])  # fmt: skip
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        with torch.no_grad():
            for p, saved in zip(beta_net.parameters(), best_state[0], strict=True):
                p.copy_(saved)
            for p, saved in zip(factor_net.parameters(), best_state[1], strict=True):
                p.copy_(saved)
    return beta_net, factor_net


def predict(model: Any, month: Month) -> tuple[np.ndarray, np.ndarray]:
    """``(beta, f)``: the month's stocks' loadings (``N x K``) and its factors (``K``)."""
    import torch  # noqa: PLC0415 - the optional deep group

    beta_net, factor_net = model
    device = next(factor_net.parameters()).device  # where the model was trained
    with torch.no_grad():
        z = torch.tensor(month.z, dtype=torch.float32).to(device)
        r = torch.tensor(month.r, dtype=torch.float32).to(device)
        beta = beta_net(z).cpu().numpy().astype(float)
        f = factor_net(z.T @ r / len(r)).cpu().numpy().astype(float)
    return beta, f


def walk_forward(
    months: Sequence[Month], k: int, *, first_test: int, last_test: int, **kwargs: Any
) -> dict[str, float]:
    """The autoencoder refitted each January on all earlier months; its out-of-sample measures (latent.evaluate)."""
    fitted, actual, sdf = [], [], []
    for year in range(first_test, last_test + 1):
        train = [m for m in months if m.date.year < year]
        test = [m for m in months if m.date.year == year]
        if not test:
            continue
        model = fit(train, k, **kwargs)
        weights = tangency_weights(np.array([predict(model, m)[1] for m in train]))
        for m in test:
            beta, f = predict(model, m)
            fitted.append(m.z.T @ (beta @ f) / len(m.r))
            actual.append(m.z.T @ m.r / len(m.r))
            sdf.append(float(f @ weights))
    return evaluate(np.array(fitted), np.array(actual), np.array(sdf))
