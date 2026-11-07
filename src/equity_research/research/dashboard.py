"""The numbers of a factor dashboard, after S&P DJI's *Dashboard: S&P 500 Factor Indices*. Pure functions.

    returns = period_returns(daily, asof)                 # name x 1M, 3M, YTD, 12M, 3Y, 5Y, 10Y, 15Y
    excess, te = relative_to(daily, benchmark, asof)      # the same periods against the benchmark; tracking errors
    overlap = portfolio_overlap(weights)                  # how much weight two books hold in common
    traits = stock_characteristics(data, day, market)     # per stock: volatility, momentum, yields, beta, ...
    tilt = exposures(book, benchmark_weights, factor_z(traits, members))

Conventions, as in the dashboard: a period longer than a year is a yearly rate (compound); "relative" is the
difference of the two returns over the period; volatility and tracking error are yearly, from monthly returns.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from equity_research.research.data import UniverseData
from equity_research.research.factors import dividend_yield, momentum
from equity_research.research.garp import standard_z
from equity_research.research.portfolio import trailing_beta

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


def portfolio_overlap(weights: Mapping[str, pd.Series]) -> pd.DataFrame:
    """``book x book``: the weight two books hold in common (the sum of the smaller of each stock's two weights)."""
    names = list(weights)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            both = pd.concat({"a": weights[a], "b": weights[b]}, axis=1).fillna(0.0)
            out.loc[a, b] = float(both.min(axis=1).sum())
    return out


def active_share(weights: pd.Series, benchmark: pd.Series) -> float:
    """Half the sum of the absolute weight differences: 0 for the benchmark itself, 1 for nothing in common."""
    both = pd.concat({"w": weights, "b": benchmark}, axis=1).fillna(0.0)
    return float((both["w"] - both["b"]).abs().sum() / 2)


def sector_weights(weights: pd.Series, sectors: pd.Series) -> pd.Series:
    """Weight per sector (a stock without one counts as ``unknown``)."""
    held = weights[weights > 0]
    return held.groupby(sectors.reindex(held.index).fillna("unknown").to_numpy()).sum()


def concentration(weights: pd.Series) -> float:
    """The Herfindahl-Hirschman index of the weights in percent: 10,000 for one stock, 100 for a hundred equal ones."""
    w = weights[weights > 0]
    return float(((100 * w / w.sum()) ** 2).sum())


def weighted_average(weights: pd.Series, values: pd.Series) -> float:
    """The weighted average of ``values``, over the held stocks that have a value."""
    both = pd.concat({"w": weights[weights > 0], "v": values}, axis=1, join="inner").dropna()
    return float((both["w"] * both["v"]).sum() / both["w"].sum()) if both["w"].sum() > 0 else float("nan")


def stock_characteristics(data: UniverseData, day: pd.Timestamp, market: pd.Series) -> pd.DataFrame:
    """``listing_id x trait`` on ``day``: what the dashboard averages over a book.

    Volatility and beta over the trailing year (beta to ``market``), the 12-1 month return, earnings, book and sales
    yields, the trailing year's dividend yield, return on equity, accruals, leverage, and market value in dollars.
    """
    ratios = data.ratios.xs(day, level="date") if day in data.ratios.index.get_level_values("date") else None
    returns = data.returns.loc[:day]

    def ratio(name: str) -> pd.Series:
        return ratios[name] if ratios is not None and name in ratios else pd.Series(dtype=float)

    def inverse(name: str) -> pd.Series:
        r = ratio(name)
        return (1.0 / r).where(r != 0)

    traits = {
        "volatility": returns.tail(252).std() * np.sqrt(252),
        "return_12_1": momentum(data.adjusted.loc[:day]).iloc[-1],
        "beta": trailing_beta(returns.tail(400), market.reindex(returns.tail(400).index)).iloc[-1],
        "earnings_to_price": inverse("P/E"),
        "book_to_price": inverse("P/B"),
        "sales_to_price": inverse("P/S"),
        "dividend_yield": dividend_yield(returns.tail(300), data.adjusted.loc[:day].tail(300)).iloc[-1],
        "roe": ratio("RoE"),
        "accruals": ratio("Accruals/TA"),
        "leverage": ratio("Debt/Equity"),
        "market_cap": data.market_cap.loc[day] if day in data.market_cap.index else pd.Series(dtype=float),
    }
    return pd.DataFrame(traits).reindex(data.members.columns)


def factor_z(traits: pd.DataFrame, members: pd.Series) -> pd.DataFrame:
    """``listing_id x factor``: the dashboard's seven factor scores, standardised among the members.

    Low volatility (minus volatility), momentum (the 12-1 return per unit of volatility), value (earnings, book and
    sales yields), high beta, dividend (the yield), quality (return on equity, minus accruals, minus leverage) and
    small size (minus the log of market value) - each winsorized and standardised like the GARP scores.
    """

    def z(values: pd.Series, *, reverse: bool = False) -> pd.Series:
        frame = values.to_frame().T
        # one row of membership under the same label as the values', or nothing would line up
        on = members.reindex(frame.columns).eq(True).to_numpy()
        member_frame = pd.DataFrame([on], index=frame.index, columns=frame.columns)
        return standard_z(frame, member_frame, reverse=reverse).iloc[0]

    def mean(*parts: pd.Series) -> pd.Series:
        return pd.concat(parts, axis=1).mean(axis=1)

    return pd.DataFrame(
        {
            "Low Volatility": z(traits["volatility"], reverse=True),
            "Momentum": z(traits["return_12_1"] / traits["volatility"]),
            "Value": mean(z(traits["earnings_to_price"]), z(traits["book_to_price"]), z(traits["sales_to_price"])),
            "High Beta": z(traits["beta"]),
            "Dividend": z(traits["dividend_yield"]),
            "Quality": mean(
                z(traits["roe"]),
                z(traits["accruals"], reverse=True),
                z(traits["leverage"].where(traits["leverage"] >= 0), reverse=True),
            ),
            "Small Size": z(np.log(traits["market_cap"].where(traits["market_cap"] > 0)), reverse=True),
        }
    )


def exposures(weights: pd.Series, benchmark: pd.Series, scores: pd.DataFrame) -> pd.Series:
    """Per factor: the book's weighted average score less the benchmark's (0 = the benchmark's own exposure)."""
    return pd.Series(
        {f: weighted_average(weights, scores[f]) - weighted_average(benchmark, scores[f]) for f in scores.columns}
    )
