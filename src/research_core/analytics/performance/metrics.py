"""Performance measures of a daily return series, on its own and against a benchmark.

    performance(net)                                  # CAGR, volatility, Sharpe, worst drawdown
    active_performance(net, benchmark)                # beta, alpha, active return, tracking error, information ratio
    capture(net, benchmark)                           # how much of the benchmark's rising and falling months it took
    periods_against(net, benchmark, "2025-01-01")     # the same before and from a boundary (a hold-out's first day)
    decile_returns(score, returns, members)           # what each tenth of a score went on to earn

Pure functions over series and ``date x instrument`` panels: nothing here knows what the instruments are. A year is
:data:`~research_core.data.calendars.TRADING_DAYS` daily returns unless the caller says otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research_core.data.calendars import MONTHS, TRADING_DAYS


def performance(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """CAGR, annualised volatility, Sharpe (no risk-free rate), max drawdown; ``periods_per_year`` returns a year."""
    r = returns.dropna()
    if r.empty:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    wealth = (1 + r).cumprod()
    years = len(r) / periods_per_year
    vol = float(r.std() * np.sqrt(periods_per_year))
    return {
        "cagr": float(wealth.iloc[-1] ** (1 / years) - 1),
        "vol": vol,
        "sharpe": float(r.mean() * periods_per_year / vol) if vol > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
    }


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


def capture(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """``up_capture`` and ``down_capture``: how much of the benchmark's rises, and of its falls, the book took part in.

    On monthly returns (:func:`monthly_pair`): the book's growth rate a year over the months the benchmark rose, divided
    by the benchmark's over the same months - and the same over the months it fell. A book with up capture above its
    down capture gained more than it gave back; a down capture above 1 fell further than the benchmark in its bad
    months. Missing with fewer than two such months.
    """
    m = monthly_pair(returns, benchmark)
    out = {}
    for key, months in (("up_capture", m["b"] > 0), ("down_capture", m["b"] < 0)):
        r, b = m.loc[months, "r"].to_numpy(dtype=float), m.loc[months, "b"].to_numpy(dtype=float)
        if len(b) < 2:
            out[key] = float("nan")
            continue
        rate_r = float(np.prod(1.0 + r) ** (MONTHS / len(r)) - 1.0)
        rate_b = float(np.prod(1.0 + b) ** (MONTHS / len(b)) - 1.0)
        out[key] = rate_r / rate_b if rate_b != 0 else float("nan")
    return out


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
