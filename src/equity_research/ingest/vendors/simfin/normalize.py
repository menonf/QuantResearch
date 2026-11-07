"""SimFin's bulk files -> company codes and canonical financial facts. Pure functions.

A quarterly statement file has one row per company and fiscal quarter: ``SimFinId``, ``Ticker``, ``Currency``,
``Fiscal Year``, ``Fiscal Period``, ``Report Date`` (the period end), ``Publish Date`` (first published),
``Restated Date`` (last changed) and one column per line item. Flows (revenue...) are the quarter's own amount;
balances (assets...) the figure on the period end. SimFin writes costs as negative numbers; ours are positive.

The file holds each figure's latest version only, and SimFin re-dates a quarter whenever a later report repeats it
(next year's comparatives: ``Restated Date`` is typically a year after ``Publish Date``, the value unchanged). So a
figure is dated by its ``Restated Date`` - when this version was surely known - unless :func:`verified_dates` finds
the SEC's first report of it equal: then it was public from its first publication.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

#: SimFin column -> (our item, ``flow`` / ``balance`` / ``average``, sign). Columns a file lacks are skipped - the
#: banks' and insurers' files share most names with the others'. ``Interest Expense, Net`` is net of interest income.
INCOME_ITEMS: dict[str, tuple[str, str, int]] = {
    "Revenue": ("revenue", "flow", 1),
    "Cost of Revenue": ("cost_of_revenue", "flow", -1),
    "Gross Profit": ("gross_profit", "flow", 1),
    "Operating Income (Loss)": ("operating_income", "flow", 1),
    "Pretax Income (Loss)": ("pretax_income", "flow", 1),
    "Net Income": ("net_income", "flow", 1),
    "Interest Expense, Net": ("interest_expense", "flow", -1),
    "Shares (Basic)": ("shares_weighted_basic", "average", 1),
    "Shares (Diluted)": ("shares_weighted_diluted", "average", 1),
}
BALANCE_ITEMS: dict[str, tuple[str, str, int]] = {
    "Total Assets": ("total_assets", "balance", 1),
    "Total Current Assets": ("current_assets", "balance", 1),
    "Total Current Liabilities": ("current_liabilities", "balance", 1),
    "Total Liabilities": ("total_liabilities", "balance", 1),
    "Total Liabilities & Equity": ("liabilities_and_equity", "balance", 1),
    "Total Equity": ("equity_total", "balance", 1),
    "Equity Before Minority Interest": ("equity_parent", "balance", 1),
    "Retained Earnings": ("retained_earnings", "balance", 1),
    "Cash, Cash Equivalents & Short Term Investments": ("cash_and_st_investments", "balance", 1),
    "Property, Plant & Equipment, Net": ("ppe_net", "balance", 1),
    "Long Term Debt": ("lt_debt_noncurrent", "balance", 1),
    "Short Term Debt": ("debt_current", "balance", 1),
}
FACT_COLUMNS = ["simfin_id", "item", "period_type", "period_start", "period_end", "value", "unit", "published_at"]
#: Two vendors' figures for one item agree within this relative difference; their period ends within this many days.
SAME_VALUE = 0.005
SAME_PERIOD_DAYS = 7


def read_table(path: Path) -> pd.DataFrame:
    """A landed bulk file (a zip holding one ``;``-separated CSV, or the CSV itself)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            return pd.read_csv(io.BytesIO(z.read(name)), sep=";", low_memory=False)
    return pd.read_csv(path, sep=";", low_memory=False)


def _col(raw: pd.DataFrame, name: str) -> pd.Series:
    """A column, or an all-missing one when the file has none."""
    return raw[name] if name in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="object")


def _ids(series: pd.Series) -> pd.Series:
    """Integer-valued codes as text (``320193.0`` -> ``320193``); missing stays missing."""
    numbers = pd.to_numeric(series, errors="coerce")
    return numbers.astype("Int64").astype(str).where(numbers.notna())


def companies(raw: pd.DataFrame) -> pd.DataFrame:
    """``[simfin_id, ticker, name, industry_id, isin, cik]`` (``isin`` / ``cik`` missing when the file has none)."""
    out = pd.DataFrame(
        {
            "simfin_id": _ids(raw["SimFinId"]),
            "ticker": _col(raw, "Ticker"),
            "name": _col(raw, "Company Name"),
            "industry_id": pd.to_numeric(_col(raw, "IndustryId"), errors="coerce").astype("Int64"),
            "isin": _col(raw, "ISIN").astype("string").str.strip(),
            "cik": _ids(_col(raw, "CIK")),
        }
    )
    return out.dropna(subset=["simfin_id"]).drop_duplicates("simfin_id").reset_index(drop=True)


def industries(raw: pd.DataFrame) -> pd.DataFrame:
    """``[industry_id, sector, industry]``."""
    return pd.DataFrame(
        {
            "industry_id": pd.to_numeric(raw["IndustryId"], errors="coerce").astype("Int64"),
            "sector": _col(raw, "Sector"),
            "industry": _col(raw, "Industry"),
        }
    ).dropna(subset=["industry_id"])


def facts(raw: pd.DataFrame, items: dict[str, tuple[str, str, int]]) -> pd.DataFrame:
    """Financial facts: :data:`FACT_COLUMNS` and ``first_published``.

    ``published_at`` is the later of the restated and publish dates, ``first_published`` the publish date.

    US-dollar statements only (a row in another currency is dropped - see :func:`currencies`).
    """
    present = {c: v for c, v in items.items() if c in raw.columns}
    if raw.empty or not present:
        return pd.DataFrame(columns=[*FACT_COLUMNS, "first_published"])
    usd = raw[_col(raw, "Currency").fillna("USD") == "USD"]
    first = pd.to_datetime(usd["Publish Date"], errors="coerce")
    restated = pd.to_datetime(_col(usd, "Restated Date"), errors="coerce")
    wide = pd.DataFrame(
        {
            "simfin_id": _ids(usd["SimFinId"]),
            "period_end": pd.to_datetime(usd["Report Date"], errors="coerce"),
            "published_at": restated.where(restated >= first, first),  # the later of the two
            "first_published": first,
            **{c: pd.to_numeric(usd[c], errors="coerce") for c in present},
        }
    ).dropna(subset=["simfin_id", "period_end", "published_at"])
    ids = ["simfin_id", "period_end", "published_at", "first_published"]
    long = wide.melt(id_vars=ids, var_name="column", value_name="value")
    long = long.dropna(subset=["value"])
    item, kind, sign = (long["column"].map({c: v[i] for c, v in present.items()}) for i in range(3))
    long = long.assign(item=item, value=long["value"] * sign.astype(float))
    duration = kind != "balance"
    long["period_type"] = duration.map({True: "Q", False: "I"})
    long["period_start"] = (long["period_end"] - pd.DateOffset(months=3) + pd.Timedelta(days=1)).where(duration)
    long["unit"] = kind.map({"average": "shares"}).fillna("USD")
    long = long.drop_duplicates(["simfin_id", "item", "period_type", "period_end", "published_at"])
    return long[[*FACT_COLUMNS, "first_published"]].reset_index(drop=True)


def verified_dates(facts: pd.DataFrame, sec_first: pd.DataFrame) -> pd.DataFrame:
    """``facts`` with ``published_at`` moved back to the first publication where the SEC confirms the value.

    ``facts``: :func:`facts` rows with an ``issuer_id``. ``sec_first``: the SEC's first-reported facts
    ``[issuer_id, item, period_type, period_end, value, published_at]``. A figure whose SEC counterpart (same issuer,
    item and period type, period end within :data:`SAME_PERIOD_DAYS`) has the same value within :data:`SAME_VALUE` is
    dated by the later of SimFin's first publication and the SEC filing; any other keeps its restated date. Adds
    ``verified`` (True / False, missing when the SEC has no such figure).
    """
    keys = ["issuer_id", "item", "period_type"]
    left = facts.assign(_row=range(len(facts)), period_end=pd.to_datetime(facts["period_end"]).astype("datetime64[ns]"))
    right = sec_first.rename(columns={"value": "sec_value", "published_at": "sec_published"})[
        [*keys, "period_end", "sec_value", "sec_published"]
    ]
    right = right.assign(period_end=pd.to_datetime(right["period_end"]).astype("datetime64[ns]"))
    left = left.astype({"issuer_id": "int64"})
    right = right.astype({"issuer_id": "int64"})
    both = pd.merge_asof(
        left.sort_values("period_end"),
        right.sort_values("period_end"),
        on="period_end",
        by=keys,
        tolerance=pd.Timedelta(days=SAME_PERIOD_DAYS),
        direction="nearest",
    ).sort_values("_row")
    has = both["sec_value"].notna()
    scale = both[["value", "sec_value"]].abs().max(axis=1)
    same = has & ((both["value"] - both["sec_value"]).abs() <= SAME_VALUE * scale)
    first = pd.concat([both["first_published"], pd.to_datetime(both["sec_published"])], axis=1).max(axis=1)
    out = facts.copy()
    out["published_at"] = first.where(same, both["published_at"]).to_numpy()
    out["verified"] = same.where(has).to_numpy()
    return out


def currencies(raw: pd.DataFrame) -> dict[str, int]:
    """Statements per reporting currency - what :func:`facts` kept (USD) and dropped."""
    if "Currency" not in raw.columns:
        return {}
    return {str(k): int(v) for k, v in raw["Currency"].fillna("?").value_counts().items()}


def last_published(raw: pd.DataFrame) -> pd.Series:
    """Each company's latest ``Publish Date`` (``simfin_id`` -> day): the day its ticker is matched on."""
    days = pd.to_datetime(raw["Publish Date"], errors="coerce")
    return days.groupby(_ids(raw["SimFinId"])).max()
