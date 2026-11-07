"""Yahoo chart responses -> canonical daily bars and corporate actions.

Yahoo expresses every price, volume and dividend in today's shares (adjusted for later splits); here they are put
back in the shares of their own day: before a 4-for-1 split, a close is multiplied by 4, a volume divided by 4.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

BAR_COLUMNS = ["trade_date", "open", "high", "low", "close", "volume", "currency"]
ACTION_COLUMNS = ["action_type", "ex_date", "amount", "ratio", "currency"]


def _result(body: bytes | str) -> dict[str, Any]:
    payload = json.loads(body)
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"Yahoo error: {chart['error']}")
    result: dict[str, Any] = chart["result"][0]
    return result


def _local_dates(stamps: list[int], timezone: str) -> pd.Series:
    """Exchange-local calendar dates of Unix timestamps (a bar is stamped at the session open)."""
    return pd.Series(pd.to_datetime(stamps, unit="s", utc=True).tz_convert(timezone).normalize().tz_localize(None))


def _later_splits(dates: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """Per date: the product of the ratios of splits with ex-dates after it (1.0 when none)."""
    factor = np.ones(len(dates))
    for ex_date, ratio in zip(splits["ex_date"].tolist(), splits["ratio"].tolist(), strict=True):
        factor[(dates < ex_date).to_numpy()] *= ratio
    return factor


def parse_chart(body: bytes | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(bars, actions)`` from one chart response, unadjusted (see the module docstring)."""
    r = _result(body)
    meta = r.get("meta", {})
    timezone = str(meta.get("exchangeTimezoneName") or "America/New_York")
    currency = meta.get("currency")
    events = r.get("events") or {}

    raw_splits = list((events.get("splits") or {}).values())
    splits = pd.DataFrame(
        {
            "action_type": "split",
            "ex_date": _local_dates([int(s["date"]) for s in raw_splits], timezone),
            "amount": np.nan,
            "ratio": [float(s["numerator"]) / float(s["denominator"]) for s in raw_splits],
            "currency": None,
        },
        columns=ACTION_COLUMNS,
    )
    raw_divs = list((events.get("dividends") or {}).values())
    div_dates = _local_dates([int(d["date"]) for d in raw_divs], timezone)
    dividends = pd.DataFrame(
        {
            "action_type": "cash_dividend",
            "ex_date": div_dates,
            "amount": np.array([float(d["amount"]) for d in raw_divs]) * _later_splits(div_dates, splits),
            "ratio": np.nan,
            "currency": currency,
        },
        columns=ACTION_COLUMNS,
    )
    actions = pd.concat([splits, dividends], ignore_index=True) if raw_splits or raw_divs else splits

    stamps = r.get("timestamp") or []
    quote = (r.get("indicators", {}).get("quote") or [{}])[0]
    bars = pd.DataFrame(
        {
            "trade_date": _local_dates(stamps, timezone),
            **{
                c: pd.to_numeric(pd.Series(quote.get(c, [None] * len(stamps))), errors="coerce")
                for c in ("open", "high", "low", "close", "volume")
            },
            "currency": currency,
        }
    )
    bars = bars.dropna(subset=["close"]).drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    factor = _later_splits(bars["trade_date"], splits)
    for col in ("open", "high", "low", "close"):
        bars[col] = bars[col] * factor
    bars["volume"] = (bars["volume"] / factor).round().astype("Int64")
    return bars[BAR_COLUMNS], actions.reset_index(drop=True)
