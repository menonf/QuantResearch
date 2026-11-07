"""Tiingo's answers -> canonical rows. Pure functions.

A daily bar from Tiingo holds the day's real prices (``open``... ``close``, ``volume``, in that day's shares) beside
adjusted ones (``adjClose``...), with ``divCash`` (the cash dividend going ex that day, per share of that day) and
``splitFactor`` (new shares per old share on the day a split takes effect, else 1). The real prices and those two
columns are what we store; the adjusted ones are recomputed from them when read, like every vendor's.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

BAR_COLUMNS = ["trade_date", "open", "high", "low", "close", "volume"]
ACTION_COLUMNS = ["action_type", "ex_date", "amount", "ratio"]
#: Closes on the days both vendors priced must agree this closely (median relative difference) to be the same stock.
AGREEMENT = 0.02
MIN_COMMON_DAYS = 5


def tiingo_ticker(ticker: str) -> str:
    """Our ticker as Tiingo writes it: a share class after a hyphen (``BRK.B`` / ``BRK/B`` -> ``BRK-B``)."""
    return ticker.strip().upper().replace(".", "-").replace("/", "-")


def frame(body: bytes) -> pd.DataFrame:
    """The JSON body as a table, one row per day (empty when Tiingo has no bars in the window)."""
    rows = json.loads(body)
    table = pd.DataFrame(rows if isinstance(rows, list) else [])
    if table.empty:
        return table
    return table.assign(date=pd.to_datetime(table["date"], utc=True).dt.tz_localize(None).dt.normalize())


def bars(table: pd.DataFrame) -> pd.DataFrame:
    """``[trade_date, open, high, low, close, volume]``, unadjusted."""
    if table.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in table.columns]
    if missing:
        raise KeyError(f"Tiingo prices: no column(s) {missing}; it sent {list(table.columns)}")
    out = pd.DataFrame(
        {
            "trade_date": table["date"],
            "open": table["open"].astype(float),
            "high": table["high"].astype(float),
            "low": table["low"].astype(float),
            "close": table["close"].astype(float),
            "volume": table["volume"].astype(float).round(),
        }
    )
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    return out.assign(volume=out["volume"].astype("Int64")).reset_index(drop=True)


def actions(table: pd.DataFrame) -> pd.DataFrame:
    """``[action_type, ex_date, amount, ratio]``: cash dividends and splits on the days they go ex."""
    if table.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    div = table.get("divCash", pd.Series(0.0, index=table.index)).astype(float).fillna(0.0)
    split = table.get("splitFactor", pd.Series(1.0, index=table.index)).astype(float).fillna(1.0)
    rows = [("cash_dividend", d, float(a), np.nan) for d, a in zip(table["date"], div, strict=True) if a > 0]
    rows += [
        ("split", d, np.nan, float(r)) for d, r in zip(table["date"], split, strict=True) if r > 0 and abs(r - 1) > 1e-9
    ]
    return pd.DataFrame(rows, columns=ACTION_COLUMNS).sort_values(["ex_date", "action_type"]).reset_index(drop=True)


def supported(zipped: bytes) -> pd.DataFrame:
    """The supported-tickers file: ``[ticker, exchange, asset_type, currency, first_day, last_day]``."""
    with zipfile.ZipFile(io.BytesIO(zipped)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), dtype=str, keep_default_na=False)

    def col(name: str) -> pd.Series:
        return raw[name] if name in raw.columns else pd.Series("", index=raw.index)

    out = pd.DataFrame(
        {
            "ticker": raw["ticker"].str.upper(),
            "exchange": col("exchange"),
            "asset_type": col("assetType"),
            "currency": col("priceCurrency"),
            "first_day": pd.to_datetime(col("startDate"), errors="coerce"),
            "last_day": pd.to_datetime(col("endDate"), errors="coerce"),
        }
    )
    # a ticker listed twice (it passed to another company): the most recent holder is the one Tiingo answers for
    return out.sort_values("last_day").drop_duplicates("ticker", keep="last").reset_index(drop=True)


def agreement(tiingo: pd.DataFrame, other: pd.DataFrame) -> tuple[int, float | None]:
    """``(common days, median relative difference of the closes)`` of two vendors' bars of one listing."""
    both = tiingo[["trade_date", "close"]].merge(other[["trade_date", "close"]], on="trade_date", suffixes=("", "_o"))
    both = both[both["close_o"] > 0]
    if both.empty:
        return 0, None
    return len(both), float((both["close"] / both["close_o"] - 1).abs().median())


def same_stock(tiingo: pd.DataFrame, other: pd.DataFrame) -> bool | None:
    """True / False when the two vendors share enough days to tell; ``None`` (unverified) otherwise."""
    days, diff = agreement(tiingo, other)
    if days < MIN_COMMON_DAYS or diff is None:
        return None
    return diff <= AGREEMENT
