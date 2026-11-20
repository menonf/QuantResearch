"""Which days count and how many make a year - read off the data, never from one exchange's calendar.

    days = trading_days(close)                      # the days the instruments actually traded
    month_ends = rebalance_dates(days, "ME")        # the last trading day of each month

Nothing here knows a market's holidays: trading days are taken from the prices held, so a universe on another
calendar (London, a market priced weekly or monthly) brings its own. The number of periods in a year is a default
(:data:`TRADING_DAYS`: the daily closes of an equity market) that every function annualising a figure takes as the
keyword ``periods_per_year``.
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS = 252


MONTHS = 12


def trading_days(close: pd.DataFrame, min_share: float = 0.5) -> pd.DatetimeIndex:
    """The days on which at least ``min_share`` of the listings alive (between their first and last price) traded.

    A vendor may print index levels on an exchange holiday; kept, that day would leave every stock without a price and
    cost each one its next day's return.
    """
    priced = close.notna()
    alive = priced.cummax() & priced[::-1].cummax()[::-1]
    share = priced.sum(axis=1) / alive.sum(axis=1).replace(0, float("nan"))
    return pd.DatetimeIndex(close.index[share >= min_share])


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> pd.DatetimeIndex:
    """The last trading day of each period (``ME`` month, ``QE`` quarter) in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(frequency[0])).max().to_numpy())
