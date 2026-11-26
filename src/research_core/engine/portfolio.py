"""From scores to target weights on rebalance dates. Pure functions over ``date x instrument`` panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def buffered_selection(score: pd.Series, held: set[int], *, enter: float, exit_: float, top: bool = True) -> set[int]:
    """Names in the top (or bottom) ``enter`` fraction, plus held names still inside the ``exit_`` fraction.

    The buffer cuts turnover: a holding is not sold the moment it slips just below the entry line.
    """
    s = score.dropna()
    if s.empty:
        return set()
    pct = s.rank(pct=True, ascending=top)  # 1.0 = the best name for the requested side
    entering = set(pct[pct > 1 - enter].index.astype(int))
    staying = {n for n in held if n in pct.index and pct[n] > 1 - exit_}
    return entering | staying


def quantile_book(
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    enter: float = 0.1,
    exit_: float = 0.2,
    long_only: bool = False,
) -> pd.DataFrame:
    """Equal-weighted books on ``dates``: long the top names (sum +1), short the bottom (sum -1) unless long-only."""
    rows = {}
    longs: set[int] = set()
    shorts: set[int] = set()
    for d in dates:
        if d not in score.index:
            continue
        s = pd.Series(score.loc[d], index=score.columns)
        longs = buffered_selection(s, longs, enter=enter, exit_=exit_, top=True)
        w = pd.Series(0.0, index=score.columns)
        if longs:
            w[list(longs)] = 1.0 / len(longs)
        if not long_only:
            shorts = buffered_selection(s, shorts, enter=enter, exit_=exit_, top=False) - longs
            if shorts:
                w[list(shorts)] = -1.0 / len(shorts)
        rows[d] = w
    return pd.DataFrame(rows).T.reindex(columns=score.columns).fillna(0.0)


def quantile_books(score: pd.DataFrame, dates: pd.DatetimeIndex, *, quantiles: int = 5) -> dict[int, pd.DataFrame]:
    """``{quantile: book}``: each quantile's names of ``score`` on ``dates``, equally weighted (each row sums to 1).

    Quantile 1 holds the lowest scores and ``quantiles`` the highest, cut as
    :func:`~research_core.analytics.signals.evaluation.quantile_returns` cuts them (the rank percentile, rounded up),
    so a book holds the names whose outcomes that function averages. Every quantile of a score is a book: whether a
    signal sorts its universe shows in the whole fan, not in the best slice alone. A date on which fewer than
    ``quantiles`` names have a score gets no row, so a backtest holds the previous book on through it.
    """
    on = score.reindex(pd.DatetimeIndex([d for d in dates if d in score.index]))
    on = on[on.notna().sum(axis=1) >= quantiles]
    bucket = np.ceil(on.rank(axis=1, pct=True) * quantiles).clip(1, quantiles)
    books = {}
    for q in range(1, quantiles + 1):
        inside = (bucket == q).astype(float)
        books[q] = inside.div(inside.sum(axis=1), axis=0).fillna(0.0)
    return books


def weighted_by(book: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    """Each leg's names re-weighted in proportion to ``size`` (market value), each leg keeping its total.

    A long-only book weighted like the market it is chosen from keeps the market's mix of large and small companies,
    so what it adds is the choice of names, not a tilt to small ones. A chosen name without a positive size takes the
    median of its leg, so a gap in the data does not quietly sell it (a leg with no sizes at all stays equal-weighted).
    """
    s = size.reindex(index=book.index, columns=book.columns)
    s = s.where(s > 0)
    out = pd.DataFrame(0.0, index=book.index, columns=book.columns)
    for side in (book > 0, book < 0):
        leg = s.where(side)
        leg = leg.T.fillna(leg.median(axis=1)).T.fillna(1.0).where(side)
        total = book.where(side).sum(axis=1)
        out += leg.div(leg.sum(axis=1), axis=0).mul(total, axis=0).fillna(0.0)
    return out


def capped(book: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Each leg's weights held to at most ``cap`` of the book each, the excess shared among the leg's other names.

    A book weighted by market value scales up its largest holdings (the top 30 % of a universe holds the largest
    companies at three times their index weight), and one bad share count can hand a name a third of the book: a cap
    bounds what any single name can do. The excess goes to the uncapped names in proportion to their weights, again
    until none is over the cap; a leg too small to hold its total under the cap (fewer than ``total / cap`` names)
    ends equally weighted.
    """
    out = book.copy()
    for d in out.index:
        w = out.loc[d].to_numpy(dtype=float)
        for side in (w > 0, w < 0):
            leg = np.abs(w[side])
            total = leg.sum()
            if total == 0:
                continue
            limit = cap
            if len(leg) * limit < total:  # cannot fit: equal weights are as flat as it gets
                leg = np.full(len(leg), total / len(leg))
            else:
                for _ in range(len(leg)):
                    over = leg > limit + 1e-15
                    if not over.any():
                        break
                    excess = float((leg[over] - limit).sum())
                    leg[over] = limit
                    room = ~over & (leg < limit)
                    leg[room] += excess * leg[room] / leg[room].sum()
            w[side] = np.sign(w[side]) * leg
        out.loc[d] = w
    return out


def trailing_beta(returns: pd.DataFrame, market: pd.Series, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    """Rolling beta of each name to ``market`` (covariance / variance)."""
    cov = returns.rolling(window, min_periods=min_periods).cov(market)
    var = market.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)


def beta_neutral(book: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """Scale each leg so its weighted beta is 1 (long) and -1 (short): the book's net beta is ~0."""
    out = book.copy()
    b = beta.reindex(index=book.index, columns=book.columns)
    for d in book.index:
        w, bd = book.loc[d], b.loc[d].fillna(1.0).clip(lower=0.1)
        for side in (w > 0, w < 0):
            leg_beta = float((w[side] * bd[side]).abs().sum())
            if leg_beta > 0:
                out.loc[d, side] = w[side] / leg_beta
    return out.replace([np.inf, -np.inf], 0.0)
