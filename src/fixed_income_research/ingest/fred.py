"""FRED (Federal Reserve Bank of St. Louis): the official total-return level of ICE's headline bond indices.

    payload = fetch("BAMLHYH0A0HYM2TRIV")              # the series as FRED's CSV: a date and a level per business day
    levels = month_end_levels(payload, today)          # [as_of_date, total_return_level], one row per finished month

A rebuilt index is only as good as its distance from the real one; this is the real one. FRED carries the last three
years of an ICE series and no more, so it is fetched every month and what was fetched is kept.
"""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import requests

SERIES_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
TIMEOUT_SECONDS = 60
#: A month counts only when its last level is from its last few days (a series that stops mid-month has no month end).
LAST_DAYS = 5


def fetch(series: str) -> bytes:
    """The whole series as FRED serves it (no key is needed for the CSV download)."""
    response = requests.get(SERIES_URL.format(series=series), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def month_end_levels(payload: bytes, today: dt.date) -> pd.DataFrame:
    """``[as_of_date, total_return_level]``: each finished month's last level, dated by the month's last day.

    The month ``today`` is in is not finished, and a month whose last level is more than :data:`LAST_DAYS` before
    its end is left out. A day FRED has no level for is a ``.`` in its file.
    """
    table = pd.read_csv(io.BytesIO(payload), na_values=["."])
    table.columns = ["date", "level"]
    table["date"] = pd.to_datetime(table["date"])
    table = table.dropna(subset=["level"]).sort_values("date")
    last = table.groupby(table["date"].dt.to_period("M")).tail(1)
    month_end = last["date"] + pd.offsets.MonthEnd(0)
    keep = (month_end.dt.date < today) & ((month_end - last["date"]).dt.days <= LAST_DAYS)
    return pd.DataFrame(
        {"as_of_date": month_end[keep].dt.date.to_numpy(), "total_return_level": last.loc[keep, "level"].astype(float)}
    ).reset_index(drop=True)
