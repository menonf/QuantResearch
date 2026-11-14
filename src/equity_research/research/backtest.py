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

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestResult:
    """Daily gross and net returns, turnover and costs."""

    gross: pd.Series
    net: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame

    def metrics(self) -> dict[str, float]:
        """CAGR, volatility, Sharpe, max drawdown and annual turnover of the net returns."""
        return performance(self.net) | {
            "turnover": float(self.turnover.sum() / max(len(self.net) / TRADING_DAYS, 1e-9))
        }


def performance(returns: pd.Series) -> dict[str, float]:
    """CAGR, annualised volatility, Sharpe (no risk-free rate), max drawdown."""
    r = returns.dropna()
    if r.empty:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    wealth = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    return {
        "cagr": float(wealth.iloc[-1] ** (1 / years) - 1),
        "vol": vol,
        "sharpe": float(r.mean() * TRADING_DAYS / vol) if vol > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
    }


MONTHS = 12


def decile_returns(score: pd.DataFrame, returns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Per decile of ``score`` (1 worst, 10 best): the next period's return over the average member, a year.

    ``score`` is on decision dates; each date's members are ranked and the equal-weighted return of each tenth to the
    next decision date is compared with all scored members' (before costs). Columns ``excess`` (a year) and ``t`` -
    a long-only book can only use the top; a score that works by spotting losers shows in the bottom.
    """
    dates = pd.DatetimeIndex(score.index)
    daily = returns.reindex(columns=score.columns).fillna(0.0)
    grown = pd.DataFrame(np.log1p(daily.to_numpy(dtype=float)), index=daily.index, columns=daily.columns).cumsum()
    at = grown.reindex(dates)
    nxt = (at.shift(-1) - at).apply(np.expm1).where(members.reindex(index=dates, columns=score.columns).astype(bool))
    scored = score.where(nxt.notna())
    decile = np.ceil(scored.rank(axis=1, pct=True) * 10).clip(1, 10)
    average = nxt.where(scored.notna()).mean(axis=1)
    rows = {}
    per_year = len(dates) / max((dates.max() - dates.min()).days / 365.25, 1e-9)
    for k in range(1, 11):
        x = (nxt.where(decile == k).mean(axis=1) - average).dropna()
        rows[k] = {
            "excess": float(x.mean() * per_year) if len(x) else float("nan"),
            "t": float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else float("nan"),
        }
    return pd.DataFrame(rows).T.rename_axis("decile")


def monthly_pair(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """``[r, b]``: both daily series compounded to calendar months, over the days both have a return."""
    both = pd.concat({"r": returns, "b": benchmark}, axis=1, sort=True).dropna()
    months = pd.DatetimeIndex(both.index).to_period("M")
    return (1.0 + both).groupby(months).prod() - 1.0


def periods_against(returns: pd.Series, benchmark: pd.Series, boundary: str | pd.Timestamp) -> pd.DataFrame:
    """The book against ``benchmark`` before and from ``boundary`` (the hold-out's first day), one row each.

    Columns: ``months``, the book's ``cagr`` / ``vol`` / ``max_drawdown``, the ``benchmark_cagr``, and
    :func:`active_performance`'s measures, plus ``active_t`` - the t-statistic of the mean monthly difference (mean over
    its standard error): with few months a large active return can still be noise.
    """
    start = pd.Timestamp(boundary)
    rows = {}
    for label, inside in (("before", returns.index < start), ("from", returns.index >= start)):
        r, b = returns[inside], benchmark.reindex(returns.index)[inside]
        pair = monthly_pair(r, b)
        diff = pair["r"] - pair["b"]
        t = float(diff.mean() / (diff.std() / np.sqrt(len(diff)))) if len(diff) > 1 and diff.std() > 0 else float("nan")
        own = performance(r)
        rows[f"{label} {start:%Y-%m-%d}"] = {
            "months": len(diff),
            "cagr": own["cagr"],
            "vol": own["vol"],
            "max_drawdown": own["max_drawdown"],
            "benchmark_cagr": performance(b)["cagr"],
            **active_performance(r, b),
            "active_t": t,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def active_performance(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Against ``benchmark``: what the choice of names added, measured on monthly returns.

    A long-only book's own Sharpe ratio mostly measures the market it holds; these measure the difference:

    * ``beta`` and ``alpha`` (the yearly return left after ``beta`` times the benchmark's);
    * ``active_return`` - how much faster the book grew a year than the benchmark (from the two growth rates);
    * ``tracking_error`` - the yearly volatility of the monthly difference;
    * ``information_ratio`` - the yearly mean difference divided by the tracking error (a Sharpe ratio of the bet
      against the benchmark).

    Monthly, not daily: a thinly traded benchmark fund's closes are often stale (SPTM's daily correlation with SPY
    was 0.63 in 2008, its monthly 0.98), and daily differences would count that noise as tracking error and drag beta
    towards zero.
    """
    m = monthly_pair(returns, benchmark)
    keys = ("beta", "alpha", "active_return", "tracking_error", "information_ratio")
    if len(m) < 2:  # a variance needs two months
        return dict.fromkeys(keys, float("nan"))
    r, b = m["r"], m["b"]
    beta = float(r.cov(b) / b.var()) if b.var() > 0 else float("nan")
    diff = r - b
    te = float(diff.std() * np.sqrt(MONTHS))
    growth = (1.0 + m).prod() ** (MONTHS / len(m))
    return {
        "beta": beta,
        "alpha": float((r - beta * b).mean() * MONTHS),
        "active_return": float(growth["r"] / growth["b"] - 1.0),
        "tracking_error": te,
        "information_ratio": float(diff.mean() * MONTHS / te) if te > 0 else float("nan"),
    }


def run_backtest(
    book: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: pd.DataFrame | float = 10.0,
    borrow_bps: pd.DataFrame | float = 0.0,
    buy_tax_bps: pd.DataFrame | float = 0.0,
) -> BacktestResult:
    """Simulate ``book`` (target weights on rebalance dates) over ``returns`` (daily, ``date x listing_id``).

    ``cost_bps`` is the one-way cost per unit traded; ``borrow_bps`` the annual fee on short positions; ``buy_tax_bps``
    a tax on buying a long holding only (UK stamp duty: not on sales, nor on shorts, taken as contracts for
    difference). Each can be a scalar or a ``date x listing_id`` panel (size-segment costs). A missing return counts
    as 0 for a held name.
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
        fee = float(np.abs(np.minimum(w, 0.0)) @ borrow_all[position[d]]) / 1e4 / TRADING_DAYS
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
