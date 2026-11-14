"""Deep learning (docs/ML4T_PLAN.md, P6): sequence models of weekly returns and an MLP ensemble on the features.

    weeks = week_ends(data.returns.index)                              # the last trading day of each week
    past = weekly_ranks(data.returns, data.members, weeks)             # each week's return, centred rank among members
    y = labels.trimmed_excess(labels.period_returns(data.returns, weeks), data.members)   # next week, over the average
    run = walk_forward_sequences("lstm", past, y, data.members, first_test=2015, last_test=2024)
    mlp = walk_forward_mlp(x, y_monthly, folds, ends)                  # the MLP ensemble on 6.2's features

Everything runs on a CPU, seeded and deterministic (``torch.use_deterministic_algorithms``), so a rerun repeats it.

1. **Sequence models** (*Machine Learning for Trading*, chapter 13) - each stock's last :data:`LOOKBACK` weekly
   returns, each week as its centred percentile among the members (the market's move is taken out, a missing week is
   0), predict next week's return over the average member (each date's extreme 1 % clipped):

   * **NLinear** (Zeng et al., 2023) - one linear layer over the sequence, after subtracting its last value (and
     adding it back): the "embarrassingly simple" baseline that matches many deep forecasters;
   * **LSTM** (Hochreiter and Schmidhuber, 1997) - a recurrent network of :data:`HIDDEN` units reading the weeks in
     order and keeping a memory of what matters;
   * **TSMixer** (Chen et al., 2023) - blocks of multilayer perceptrons that mix information across the weeks, with
     residual connections and layer normalisation.

   Each is refitted every January on up to :data:`MAX_ROWS` sequences sampled from every earlier year whose target
   week had ended, with early stopping on the training window's last year.
2. **MLP ensemble** - in the spirit of TabM (Gorishniy et al., 2024), which found an ensemble of multilayer
   perceptrons the strongest deep model on tabular data: :data:`MEMBERS` networks of two hidden layers (64 units, ReLU,
   dropout 0.1) with different seeds, averaged, on the monthly features, target and folds of notebooks 6.3-6.4.
   (TabM shares most weights between its members; separate networks are the simpler, costlier form of the idea.)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from equity_research.research.linear import design_rows, to_wide
from equity_research.research.validation import Fold

SEED = 42
#: Weeks of history each sequence model reads.
LOOKBACK = 52
#: Hidden units of the LSTM and the TSMixer.
HIDDEN = 32
#: The most training sequences one fit samples (CPU-sized).
MAX_ROWS = 150_000
#: Networks in the MLP ensemble.
MEMBERS = 8
#: The registry family of P6's architectures.
FAMILY = "ml4t_deep"
#: The sequence models.
SEQUENCE_MODELS = ("nlinear", "lstm", "tsmixer")


def _torch() -> Any:
    # cuBLAS repeats its results only with a fixed workspace; set before CUDA starts (no effect on the CPU)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch  # noqa: PLC0415 - the optional deep group

    torch.use_deterministic_algorithms(True)
    return torch


def best_device() -> str:
    """``"cuda"`` when torch has a GPU to use, else ``"cpu"``: a notebook's ``device=``.

    A GPU's results differ from the CPU's by rounding, so a cached artifact names its device.
    """
    return "cuda" if _torch().cuda.is_available() else "cpu"


def week_ends(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each calendar week in ``days``."""
    d = pd.DatetimeIndex(days)
    frame = pd.Series(d, index=d)
    return pd.DatetimeIndex(frame.groupby(d.to_period("W-FRI")).max().to_numpy())


def weekly_ranks(returns: pd.DataFrame, members: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """``week x listing_id``: each week's return to its end, as a centred percentile among the members (missing: 0)."""
    days = pd.DatetimeIndex(returns.index)
    growth = pd.DataFrame(np.log1p(returns.fillna(0.0).to_numpy()), index=days, columns=returns.columns).cumsum()
    level = growth.reindex(weeks)
    weekly = pd.DataFrame(np.expm1((level - level.shift(1)).to_numpy()), index=level.index, columns=level.columns)
    on = members.reindex(index=weeks, columns=returns.columns, fill_value=False).astype(bool)
    ranks = weekly.where(on).rank(axis=1, pct=True) - 0.5
    return ranks.fillna(0.0)


def sequences(
    past: pd.DataFrame, dates: pd.DatetimeIndex, members: pd.DataFrame, target: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    """``(X, y, index)``: for the members on ``dates``, their last :data:`LOOKBACK` weeks of ``past``.

    Oldest week first; ``y`` is their ``target`` (NaN without one), ``index`` their ``(date, listing_id)``. Dates
    without a full lookback are skipped.
    """
    values = past.to_numpy(dtype=float)
    position = {d: i for i, d in enumerate(past.index)}
    on = members.reindex(index=dates, columns=past.columns, fill_value=False).astype(bool).to_numpy()
    tgt = None if target is None else target.reindex(index=dates, columns=past.columns).to_numpy(dtype=float)
    xs, ys, rows = [], [], []
    for k, d in enumerate(dates):
        i = position.get(d)
        if i is None or i + 1 < LOOKBACK:
            continue
        cols = np.nonzero(on[k])[0]
        if tgt is not None:
            cols = cols[np.isfinite(tgt[k, cols])]
        if len(cols) == 0:
            continue
        xs.append(values[i + 1 - LOOKBACK : i + 1, cols].T)
        ys.append(tgt[k, cols] if tgt is not None else np.full(len(cols), np.nan))
        rows += [(d, past.columns[c]) for c in cols]
    if not xs:
        return np.empty((0, LOOKBACK)), np.empty(0), pd.MultiIndex.from_tuples([], names=["date", "listing_id"])
    index = pd.MultiIndex.from_tuples(rows, names=["date", "listing_id"])
    return np.vstack(xs), np.concatenate(ys), index


def build(name: str, seed: int = SEED) -> Any:
    """A fresh, seeded network: ``nlinear``, ``lstm`` or ``tsmixer`` (input ``batch x LOOKBACK``, output ``batch``)."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    module: Any = nn.Module

    class NLinear(module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(LOOKBACK, 1)

        def forward(self, x: Any) -> Any:
            last = x[:, -1:]
            return (self.linear(x - last) + last).squeeze(-1)

    class LSTMNet(module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x.unsqueeze(-1))
            return self.head(out[:, -1]).squeeze(-1)

    class TSMixer(module):  # type: ignore[misc]
        def __init__(self, blocks: int = 2) -> None:
            super().__init__()
            self.norms = nn.ModuleList([nn.LayerNorm(LOOKBACK) for _ in range(blocks)])
            self.mixers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(LOOKBACK, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LOOKBACK))
                    for _ in range(blocks)
                ]
            )
            self.head = nn.Linear(LOOKBACK, 1)

        def forward(self, x: Any) -> Any:
            for norm, mix in zip(self.norms, self.mixers, strict=True):
                x = x + mix(norm(x))
            return self.head(x).squeeze(-1)

    models = {"nlinear": NLinear, "lstm": LSTMNet, "tsmixer": TSMixer}
    if name not in models:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(models)}")
    return models[name]()


def mlp(inputs: int, seed: int) -> Any:
    """One member of the MLP ensemble: two hidden layers of 64 ReLU units with dropout 0.1."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(inputs, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 1), nn.Flatten(0),
    )  # fmt: skip


def train(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    epochs: int = 20,
    patience: int = 3,
    batch: int = 2048,
    lr: float = 1e-3,
    seed: int = SEED,
    device: str = "cpu",
) -> Any:
    """Fit ``model`` by Adam on squared error in shuffled mini-batches, with early stopping.

    The epoch with the lowest validation error is kept; training stops after ``patience`` epochs without improvement.
    The target is divided by its training standard deviation first: returns are a few thousandths, and a network
    learns badly at that scale. Predictions come out in those units; a ranking does not care.

    ``device`` is where the model trains; the data move there a batch at a time and the shuffling stays on the CPU,
    so a GPU sees the batches the CPU would.
    """
    torch = _torch()
    torch.manual_seed(seed)
    model.to(device)
    scale = float(np.std(y)) or 1.0
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y / scale, dtype=torch.float32)
    xv = torch.tensor(x_valid, dtype=torch.float32).to(device)
    yv = torch.tensor(y_valid / scale, dtype=torch.float32).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, waited = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(xt), batch):
            idx = order[start : start + batch]
            optimiser.zero_grad()
            torch.mean((model(xt[idx].to(device)) - yt[idx].to(device)) ** 2).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            error = float(torch.mean((model(xv) - yv) ** 2))
        if error < best - 1e-12:
            best, waited = error, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: Any, x: np.ndarray, *, batch: int = 50_000, device: str = "cpu") -> np.ndarray:
    """The model's predictions for ``x``, in batches (on ``device``)."""
    torch = _torch()
    model.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.tensor(x[start : start + batch], dtype=torch.float32).to(device)
            out.append(model(chunk).cpu().numpy())
    return np.concatenate(out).astype(float) if out else np.empty(0)


@dataclass(frozen=True)
class DeepRun:
    """Out-of-sample predictions (``date x listing_id``) and the number of training rows of each yearly fit."""

    scores: pd.DataFrame
    rows: pd.Series


def walk_forward_sequences(
    name: str,
    past: pd.DataFrame,
    target: pd.DataFrame,
    members: pd.DataFrame,
    *,
    first_test: int,
    last_test: int,
    max_rows: int = MAX_ROWS,
    device: str = "cpu",
    **kwargs: Any,
) -> DeepRun:
    """``name`` refitted each January on sampled sequences of earlier years (targets ended), scoring the year."""
    weeks = pd.DatetimeIndex(past.index)
    scores, rows = [], {}
    rng = np.random.default_rng(SEED)
    for year in range(first_test, last_test + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        cutoff = pd.Timestamp(start - pd.Timedelta(days=7))
        known = weeks[weeks < cutoff]  # the target week ended before the year began
        valid_from = pd.Timestamp(cutoff - pd.DateOffset(years=1))
        pool = known[(known < valid_from) & (np.arange(len(known)) + 1 >= LOOKBACK)]
        per_week = float(members.reindex(index=pool, fill_value=False).sum(axis=1).median()) if len(pool) else 1.0
        wanted = min(len(pool), int(np.ceil(max_rows / max(per_week, 1.0))) + 1)  # sample weeks, not every sequence
        chosen = pd.DatetimeIndex(np.sort(rng.choice(pool.to_numpy(), wanted, replace=False))) if wanted else pool
        xt, yt, _ = sequences(past, chosen, members, target)
        xv, yv, _ = sequences(past, known[known >= valid_from], members, target)
        if len(xt) > max_rows:
            keep = rng.choice(len(xt), max_rows, replace=False)
            xt, yt = xt[keep], yt[keep]
        model = train(build(name), xt, yt, xv, yv, device=device, **kwargs)
        xs, _, index = sequences(past, weeks[weeks.year == year], members)
        scores.append(to_wide(predict(model, xs, device=device), index))
        rows[year] = len(xt)
    return DeepRun(pd.concat(scores).sort_index(), pd.Series(rows, name="training rows"))


def walk_forward_mlp(
    x: pd.DataFrame,
    y: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    members_count: int = MEMBERS,
    validation_months: int = 12,
    device: str = "cpu",
    **kwargs: Any,
) -> DeepRun:
    """The MLP ensemble on each walk-forward fold, its members' predictions averaged on the test year.

    Each member trains on the fold's dates but the last ``validation_months``, which decide when it stops.
    """
    scores, rows = [], {}
    for fold in folds:
        year = int(fold.test.min().year)
        cut = fold.train[-validation_months]
        xt, yt, _ = design_rows(x, y, fold.train[fold.train < cut])
        xv, yv, _ = design_rows(x, y, fold.train[fold.train >= cut])
        part = x[x.index.get_level_values("date").isin(fold.test)]
        index = pd.MultiIndex.from_frame(part.index.to_frame(index=False))
        xs = part.to_numpy(dtype=float)
        preds = []
        for m in range(members_count):
            model = train(mlp(x.shape[1], SEED + m), xt, yt, xv, yv, seed=SEED + m, device=device, **kwargs)
            preds.append(predict(model, xs, device=device))
        scores.append(to_wide(np.mean(preds, axis=0), index))
        rows[year] = len(xt)
    return DeepRun(pd.concat(scores).sort_index(), pd.Series(rows, name="training rows"))
