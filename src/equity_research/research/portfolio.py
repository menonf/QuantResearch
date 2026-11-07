"""From scores to target weights on rebalance dates. Pure functions over ``date x listing_id`` panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> pd.DatetimeIndex:
    """The last trading day of each period (``ME`` month, ``QE`` quarter) in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(frequency[0])).max().to_numpy())


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
