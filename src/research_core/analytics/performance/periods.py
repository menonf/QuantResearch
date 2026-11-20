"""Returns and risk over the periods a dashboard shows: a month, a quarter, the year to date, one to fifteen years.

    returns = period_returns(daily, asof)                 # name x 1M, 3M, YTD, 12M, 3Y, 5Y, 10Y, 15Y
    excess, te = relative_to(daily, benchmark, asof)      # the same periods against the benchmark; tracking errors
    excess_correlations(daily, benchmark, "5Y", asof)     # do the books beat the benchmark at the same times?

After S&P DJI's *Dashboard: S&P 500 Factor Indices*, whose conventions these keep: a period longer than a year is a
yearly rate (compound); "relative" is the difference of the two returns over the period; volatility and tracking
error are yearly, from monthly returns. Pure functions over daily return series: nothing here knows what they are
returns of.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

#: The dashboard's periods: a label and how far back it reaches (``YTD`` from the year's start).
PERIODS: dict[str, pd.DateOffset | None] = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "YTD": None,
    "12M": pd.DateOffset(years=1),
    "3Y": pd.DateOffset(years=3),
    "5Y": pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10),
    "15Y": pd.DateOffset(years=15),
}
#: The periods volatility and tracking error are shown for (a month of a few days says nothing about risk).
RISK_PERIODS = ("12M", "3Y", "5Y", "10Y", "15Y")
#: A series that starts more than this after a period's start does not cover it (a gap of a holiday weekend does).
LATE_START = pd.Timedelta("7D")


def period_start(label: str, asof: pd.Timestamp) -> pd.Timestamp:
    """When the period ``label`` ending ``asof`` starts (``YTD``: the last day of the year before)."""
    offset = PERIODS[label]
    return pd.Timestamp(year=asof.year - 1, month=12, day=31) if offset is None else asof - offset


def _years(label: str, asof: pd.Timestamp) -> float:
    return float((asof - period_start(label, asof)).days / 365.25)


def _window(daily: pd.Series, label: str, asof: pd.Timestamp) -> pd.Series | None:
    """The returns after the period's start up to ``asof``; None when the series does not cover the period."""
    start = period_start(label, asof)
    s = daily.dropna()
    s = s[(s.index > start) & (s.index <= asof)]
    if s.empty or s.index.min() > start + LATE_START:
        return None
    return s


def period_return(daily: pd.Series, label: str, asof: pd.Timestamp) -> float:
    """The return over the period (a yearly rate beyond a year); NaN when the series does not cover it."""
    s = _window(daily, label, asof)
    if s is None:
        return float("nan")
    total = float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)
    years = _years(label, asof)
    return (1 + total) ** (1 / years) - 1 if years > 1.01 else total


def period_returns(daily: Mapping[str, pd.Series], asof: pd.Timestamp) -> pd.DataFrame:
    """``name x period``: :func:`period_return` for every series and period."""
    return pd.DataFrame({p: {n: period_return(s, p, asof) for n, s in daily.items()} for p in PERIODS})


def _monthly(s: pd.Series) -> pd.Series:
    return (1 + s).groupby(pd.DatetimeIndex(s.index).to_period("M")).prod() - 1


def period_volatility(daily: Mapping[str, pd.Series], asof: pd.Timestamp) -> pd.DataFrame:
    """``name x period``: yearly volatility of the monthly returns over each of :data:`RISK_PERIODS`."""
    out: dict[str, dict[str, float]] = {}
    for p in RISK_PERIODS:
        out[p] = {}
        for name, s in daily.items():
            w = _window(s, p, asof)
            out[p][name] = float(_monthly(w).std() * np.sqrt(12)) if w is not None else float("nan")
    return pd.DataFrame(out)


def relative_to(
    daily: Mapping[str, pd.Series], benchmark: pd.Series, asof: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(excess, tracking error)``: each period's return less the benchmark's; yearly tracking error per period."""
    mine = period_returns(daily, asof)
    theirs = pd.Series({p: period_return(benchmark, p, asof) for p in PERIODS})
    excess = mine - theirs
    te: dict[str, dict[str, float]] = {}
    for p in RISK_PERIODS:
        te[p] = {}
        for name, s in daily.items():
            w = _window(s, p, asof)
            if w is None:
                te[p][name] = float("nan")
                continue
            both = pd.concat({"r": _monthly(w), "b": _monthly(benchmark.reindex(w.index))}, axis=1).dropna()
            te[p][name] = float((both["r"] - both["b"]).std() * np.sqrt(12))
    return excess, pd.DataFrame(te)


def longest_period(table: pd.DataFrame) -> str:
    """The longest of :data:`PERIODS` every row of ``table`` (``name x period``) has a value for.

    Books compared over different spans are compared only over the span they share.
    """
    covered = [p for p in PERIODS if p in table.columns and table[p].notna().all()]  # PERIODS run short to long
    if not covered:
        raise ValueError("no period is covered by every series")
    return covered[-1]


def excess_correlations(
    daily: Mapping[str, pd.Series], benchmark: pd.Series, label: str, asof: pd.Timestamp
) -> pd.DataFrame:
    """``name x name``: the correlation of monthly returns over ``benchmark`` across the period ``label``.

    Two books whose excess returns move together are largely one active bet, whatever their names.
    """
    start = period_start(label, asof)
    excess = {}
    for name, s in daily.items():
        mine = s[(s.index > start) & (s.index <= asof)].dropna()
        excess[name] = _monthly(mine) - _monthly(benchmark.reindex(mine.index))
    return pd.DataFrame(excess).corr()


def trailing_excess(daily: pd.Series, benchmark: pd.Series, days: int = 63) -> pd.Series:
    """The return over the trailing ``days`` trading days less the benchmark's, day by day."""
    both = pd.concat({"r": daily, "b": benchmark}, axis=1, sort=True).dropna()
    logs = pd.DataFrame(np.log1p(both.to_numpy(dtype=float)), index=both.index, columns=both.columns)
    grown = np.expm1(logs.rolling(days).sum())
    out: pd.Series = (grown["r"] - grown["b"]).dropna()
    return out


# ---------------------------------------------------------------- what the books hold
