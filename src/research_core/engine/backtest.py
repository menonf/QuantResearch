"""Daily backtest of rebalanced books, with trading costs and short-borrow fees.

    result = run_backtest(book, returns, cost_bps=costs, borrow_bps=borrow)
    result.metrics()

Timing: a book set at the close of a rebalance date earns from the **next** trading day. Between rebalances the
weights drift with returns (no daily re-weighting); each rebalance trades from the drifted weights to the targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_core.analytics.performance.metrics import performance
from research_core.data.calendars import TRADING_DAYS


@dataclass(frozen=True)
class BacktestResult:
    """Daily gross and net returns, turnover and costs."""

    gross: pd.Series
    net: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame

    def metrics(self, *, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
        """CAGR, volatility, Sharpe, max drawdown and annual turnover of the net returns."""
        return performance(self.net, periods_per_year=periods_per_year) | {
            "turnover": float(self.turnover.sum() / max(len(self.net) / periods_per_year, 1e-9))
        }


def run_backtest(
    book: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: pd.DataFrame | float = 10.0,
    borrow_bps: pd.DataFrame | float = 0.0,
    buy_tax_bps: pd.DataFrame | float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestResult:
    """Simulate ``book`` (target weights on rebalance dates) over ``returns`` (daily, ``date x instrument``).

    ``cost_bps`` is the one-way cost per unit traded; ``borrow_bps`` the annual fee on short positions; ``buy_tax_bps``
    a tax on buying a long holding only (UK stamp duty: not on sales, nor on shorts, taken as contracts for
    difference). Each can be a scalar or a ``date x instrument`` panel (size-segment costs). A missing return counts
    as 0 for a held name. ``periods_per_year`` is how many of ``returns``' periods make a year: the borrow fee is
    charged a period at a time.
    """
    names = book.columns
    returns = returns.reindex(columns=names).fillna(0.0)
    days = returns.index[returns.index > book.index.min()]
    cost = _panel(cost_bps, returns.index, names)
    borrow = _panel(borrow_bps, returns.index, names)
    tax = _panel(buy_tax_bps, returns.index, names)
    targets = book.reindex(columns=names).fillna(0.0)
    position = {d: i for i, d in enumerate(returns.index)}

    r_all, cost_all, borrow_all, tax_all = returns.to_numpy(), cost.to_numpy(), borrow.to_numpy(), tax.to_numpy()
    w: np.ndarray = np.zeros(len(names))
    n = len(days)
    gross, net, turnover, costs = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    held = np.zeros((n, len(names)))
    # the first book is traded at the close of its rebalance date and earns from the next day
    started = returns.index[returns.index <= book.index.min()]
    previous: pd.Timestamp | None = started[-1] if len(started) else None
    for i, d in enumerate(days):
        if previous is not None and previous in targets.index:  # trade at the previous close, earn from today
            target = targets.loc[previous].to_numpy(dtype=float)
            traded = np.abs(target - w)
            turnover[i] = traded.sum()
            bought = np.maximum(np.maximum(target, 0.0) - np.maximum(w, 0.0), 0.0)  # long holdings added to
            costs[i] = float(traded @ cost_all[position[previous]] + bought @ tax_all[position[previous]]) / 1e4
            w = target
        r = r_all[position[d]]
        gross[i] = float(w @ r)
        fee = float(np.abs(np.minimum(w, 0.0)) @ borrow_all[position[d]]) / 1e4 / periods_per_year
        net[i] = gross[i] - fee - costs[i]
        held[i] = w
        # drift: each position grows with its return, the book with the book's return
        if gross[i] > -1:
            w = w * (1 + r) / (1 + gross[i])
        previous = d
    index = pd.DatetimeIndex(days)
    return BacktestResult(
        gross=pd.Series(gross, index=index),
        net=pd.Series(net, index=index),
        turnover=pd.Series(turnover, index=index),
        costs=pd.Series(costs, index=index),
        weights=pd.DataFrame(held, index=index, columns=names),
    )


def _panel(value: pd.DataFrame | float, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """A scalar or a panel as a full ``date x listing`` panel (forward-filled, missing as 0)."""
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index, columns=columns).ffill().fillna(0.0)
    return pd.DataFrame(float(value), index=index, columns=columns)
