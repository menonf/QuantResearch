"""Norgate's answers -> canonical rows. Pure functions.

Prices arrive split-adjusted with the real close beside them. On each day the factor ``Unadjusted Close / Close`` is
how many of today's shares one share of that day was; it changes on the day of a split. So:

* real open / high / low = adjusted x factor, real volume = adjusted / factor;
* a split's ratio (new shares per old share) = yesterday's factor / today's factor, on the day it changes;
* a dividend in that day's shares = the adjusted dividend x factor.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: A capital change this far from 1 is a split (or a consolidation); nearer, a spin-off or rights issue.
SPLIT_BAND = (0.85, 1.2)
DELISTED = re.compile(r"^(?P<ticker>.+)-(?P<year>\d{4})(?P<month>\d{2})$")
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Unadjusted Close")


def parse_symbol(symbol: str) -> tuple[str, pd.Timestamp | None]:
    """``(ticker, month delisted)``: ``LEH-200809`` -> ``("LEH", 2008-09-01)``; ``BRK.B`` -> ``("BRK.B", None)``."""
    m = DELISTED.match(symbol)
    if m is None:
        return symbol, None
    return m["ticker"], pd.Timestamp(year=int(m["year"]), month=int(m["month"]), day=1)


def as_level(frame: pd.DataFrame) -> pd.DataFrame:
    """An index level's bars in the shape of a stock's.

    An index is never split-adjusted, so Norgate sends no ``Unadjusted Close``: its close is the real one already.
    """
    return frame.assign(**{"Unadjusted Close": frame["Close"]}) if "Close" in frame.columns else frame


def _dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()


def bars(frame: pd.DataFrame) -> pd.DataFrame:
    """``[trade_date, open, high, low, close, volume]`` in each day's own shares (unadjusted)."""
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "volume"])
    missing = [c for c in PRICE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"Norgate prices: no column(s) {missing}; it sent {list(frame.columns)}")
    adjusted = frame["Close"].astype(float)
    real = frame["Unadjusted Close"].astype(float)
    factor = (real / adjusted).where(adjusted > 0)
    out = pd.DataFrame(
        {
            "trade_date": _dates(frame),
            "open": (frame["Open"].astype(float) * factor).to_numpy(),
            "high": (frame["High"].astype(float) * factor).to_numpy(),
            "low": (frame["Low"].astype(float) * factor).to_numpy(),
            "close": real.to_numpy(),
            "volume": (frame["Volume"].astype(float) / factor).round().to_numpy(),
        }
    )
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    return out.assign(volume=out["volume"].astype("Int64")).reset_index(drop=True)


def actions(frame: pd.DataFrame) -> pd.DataFrame:
    """``[action_type, ex_date, amount, ratio]``: splits and spin-offs from the factor's changes, cash dividends."""
    cols = ["action_type", "ex_date", "amount", "ratio"]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    days = _dates(frame)
    factor = pd.Series(
        (frame["Unadjusted Close"].astype(float) / frame["Close"].astype(float)).to_numpy(), index=days
    ).replace([np.inf, -np.inf], np.nan)
    change = (factor.shift(1) / factor).dropna()
    change = change[(change - 1).abs() > 1e-6]
    rows = [
        ("split" if (r <= SPLIT_BAND[0] or r >= SPLIT_BAND[1]) else "spinoff", day, np.nan, float(r))
        for day, r in change.items()
    ]
    if "Dividend" in frame.columns:
        div = pd.Series(frame["Dividend"].astype(float).to_numpy(), index=days) * factor
        rows += [("cash_dividend", day, float(a), np.nan) for day, a in div[div > 0].items()]
    return pd.DataFrame(rows, columns=cols).sort_values(["ex_date", "action_type"]).reset_index(drop=True)


def memberships(frame: pd.DataFrame) -> pd.DataFrame:
    """``[start_date, end_date]`` from a daily 1 / 0 ``Index Constituent`` series (``end_date`` = first day out)."""
    if frame.empty or "Index Constituent" not in frame.columns:
        return pd.DataFrame(columns=["start_date", "end_date"])
    member = pd.Series(frame["Index Constituent"].astype(float).to_numpy() > 0, index=_dates(frame))
    starts = member & ~member.shift(1, fill_value=False)
    ends = ~member & member.shift(1, fill_value=False)
    start_days = list(member.index[starts.to_numpy()])
    end_days = list(member.index[ends.to_numpy()])
    end_days += [None] * (len(start_days) - len(end_days))  # still a member on the last day
    return pd.DataFrame({"start_date": start_days, "end_date": pd.to_datetime(pd.Series(end_days, dtype="object"))})
