"""FinanceDatabase's equities table -> the rows our security master takes. Pure functions.

codes = us_equities(raw)          # one row per US-listed share, with the codes it is matched on
"""

from __future__ import annotations

import pandas as pd

from equity_research.ingest.countries import country_code

#: Exchanges of US listings (ISO 10383 market codes as FinanceDatabase gives them).
US_MICS = ("XNAS", "XNYS", "XASE", "ARCX", "BATS")
COLUMNS = ["name", "currency", "sector", "industry_group", "industry", "mic", "country", "isin", "cusip", "figi"]


def ticker(symbol: str) -> str:
    """Our ticker spelling: FinanceDatabase writes Yahoo's ``BRK-B``, we write the class with a dot: ``BRK.B``."""
    return symbol.replace("-", ".")


def us_equities(raw: pd.DataFrame) -> pd.DataFrame:
    """``[symbol, ticker, name, currency, sector, industry_group, industry, mic, domicile, ISIN, CUSIP, FIGI]``.

    US listings only; the code columns are named as our identifier types so they can be matched directly.
    """
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"FinanceDatabase: no column(s) {missing}; it has {list(raw.columns)}")
    frame = raw.reset_index()
    frame = frame.rename(columns={frame.columns[0]: "symbol"})
    us = frame[frame["mic"].isin(US_MICS)].copy()
    for col in COLUMNS:
        us[col] = us[col].where(us[col].notna() & (us[col].astype(str).str.strip() != ""), None)
    us["ticker"] = us["symbol"].astype(str).map(ticker)
    us["domicile"] = us["country"].map(country_code)
    out = us.rename(columns={"isin": "ISIN", "cusip": "CUSIP", "figi": "FIGI"})
    cols = ["symbol", "ticker", "name", "currency", "sector", "industry_group", "industry", "mic", "domicile"]
    return out[[*cols, "ISIN", "CUSIP", "FIGI"]].drop_duplicates("symbol").reset_index(drop=True)
