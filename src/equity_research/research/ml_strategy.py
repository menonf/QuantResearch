"""The ML strategies of P4: every (signal, entry, weighting, overlay) run, measured and comparable.

    result = backtest(book, data, end)                          # the book over the data's returns, with its costs
    numbers = measure(result.net, data.benchmark, long_only=True)   # return, risk, active measures
    active = monthly_active(result.net, data.benchmark, long_only=True)   # for White's Reality Check
    deflated = deflated_sharpe_of(result.net, trials)           # after this many strategies were tried

A long-only book is judged against the index (its benchmark); a long-short book holds as much short as long, so it is
judged against cash - its own return is its active return. Every strategy of P4 is registered in :data:`FAMILY`, so
each deflated Sharpe ratio counts all of them.
"""

from __future__ import annotations

import pandas as pd

from equity_research.research.backtest import (
    MONTHS,
    TRADING_DAYS,
    BacktestResult,
    active_performance,
    monthly_pair,
    performance,
    run_backtest,
)
from equity_research.research.data import UniverseData
from equity_research.research.registry import deflated_sharpe

#: The registry family of P4's strategies.
FAMILY = "ml4t_strategy_monthly"


def backtest(book: pd.DataFrame, data: UniverseData, end: pd.Timestamp | str) -> BacktestResult:
    """``book`` over ``data``'s daily returns up to ``end``, with the universe's trading, borrowing and buying costs."""
    returns = data.returns.loc[: pd.Timestamp(end)]
    return run_backtest(book, returns, cost_bps=data.cost_bps, borrow_bps=data.borrow_bps, buy_tax_bps=data.buy_tax_bps)


def monthly_active(net: pd.Series, benchmark: pd.Series, *, long_only: bool) -> pd.Series:
    """Monthly returns over the benchmark (long-only) or over cash (long-short, whose own return is its edge)."""
    if long_only:
        pair = monthly_pair(net, benchmark)
        return (pair["r"] - pair["b"]).rename("active")
    months = pd.DatetimeIndex(net.index).to_period("M")
    return ((1 + net).groupby(months).prod() - 1).rename("active")


def measure(net: pd.Series, benchmark: pd.Series, *, long_only: bool) -> dict[str, float]:
    """Return, volatility, Sharpe, drawdown, and the active return, tracking error and information ratio.

    For a long-short book the active measures are against cash: its mean return, volatility and Sharpe ratio.
    """
    own = performance(net)
    if long_only:
        active = active_performance(net, benchmark)
        return own | {k: active[k] for k in ("active_return", "tracking_error", "information_ratio")}
    return own | {"active_return": own["cagr"], "tracking_error": own["vol"], "information_ratio": own["sharpe"]}


def deflated_sharpe_of(series: pd.Series, sharpe: float, trials: int, *, periods: int = TRADING_DAYS) -> float:
    """The deflated Sharpe ratio of ``series`` (``periods`` a year, annualised ``sharpe``) after ``trials`` tries."""
    r = series.dropna()
    z = (r - r.mean()) / r.std() if r.std() > 0 else r * 0
    return deflated_sharpe(
        sharpe=sharpe,
        trials=trials,
        observations=len(r),
        skew=float((z**3).mean()),
        kurtosis=float((z**4).mean()),
        periods=periods,
    )


__all__ = ["FAMILY", "MONTHS", "backtest", "deflated_sharpe_of", "measure", "monthly_active"]
