"""ICE Data Indices' constituent files -> canonical rows. Pure functions, but for reading the workbook.

    code, as_of = parse_name("H0A0-01312019.xlsx")       # the index and the month end are in the file's name
    table = read_constituents(path)                       # the sheet as ICE wrote it, title and footer dropped
    month = normalize(table)                              # month.bonds: one row per bond; month.issuers: per issuer

What the files are (Index Viewer downloads, one per index and month end):

1. **One row per bond.** A title row may come first (``H0A0 for 01/31/2019 ( February Universe )``), then the
   header, the bonds, and a few lines of small print. The header is found by its first cell, so a file with or
   without the title reads the same.
2. **The universe is next month's.** ICE rebalances on the last calendar day: the file dated 31 January lists the
   index's members and weights for February (``% Mkt Value``), with each bond's price, analytics and January return
   as of that day. ``% MktVal-PrevMend`` is the weight the bond had during January - missing for a bond that is new.
   A bond that left the index at that rebalance is not in the file, so its last month's return is in no file.
3. **An issuer is ICE's ticker.** ICE files a company and its financing subsidiaries under one ticker
   (``Alcoa Nederland Holding B.V.`` under ``AA``), so the ticker is the issuer and each bond keeps its own legal
   name. The four industry levels and the country are the issuer's: where its bonds disagree in a month (a few in
   a thousand), the issuer takes those of its largest weight.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: ``H0A0-01312019.xlsx``, or the downloader's ``H0A0_20190131_H0A0-01312019.xlsx``: index code, month, day, year.
NAME = re.compile(r"^(?:.*_)?([A-Z0-9]{2,16})-(\d{2})(\d{2})(\d{4})\.xlsx$")
#: The universe a title names: ``H0A0 for 01/31/2019  ( February Universe )``.
TITLE = re.compile(r"^(\S+) for (\d{2})/(\d{2})/(\d{4})\s*\(\s*(\w+) Universe\s*\)")

#: ICE's column -> ours, per bond (:data:`~fixed_income_research.data.schema.bond_analytics`).
ANALYTICS = {
    "Par Wtd Coupon": "coupon_pct",
    "Maturity Date": "maturity_date",
    "Rating": "rating",
    "Subordination Type": "subordination",
    "Price": "price",
    "Accrued Interest": "accrued_interest",
    "OAS": "oas_bp",
    "Asset Swap Spread": "asset_swap_spread_bp",
    "Effective Yield": "effective_yield_pct",
    "Yield to Worst (s.a.)": "yield_to_worst_pct",
    "Yield to Maturity (s.a.)": "yield_to_maturity_pct",
    "Total Return % MTD_LOC": "total_return_mtd_pct",
    "Excess Return % MTD": "excess_return_mtd_pct",
    "Effective Duration": "effective_duration",
    "Mod. Dur to Worst (s.a.)": "modified_duration_to_worst",
    "Modified Duration (s.a.)": "modified_duration",
    "Effective Convexity": "effective_convexity",
    "Convexity to Worst (s.a.)": "convexity_to_worst",
    "Convexity (s.a.)": "convexity",
    "Spread Duration": "spread_duration",
}
#: ICE's column -> ours, per bond and index (:data:`~fixed_income_research.data.schema.index_constituent`).
CONSTITUENT = {
    "% Mkt Value": "weight_pct",
    "% MktVal-PrevMend": "prior_weight_pct",
    "Face Value_LOC": "face_value",
    "Cash": "cash",
}
#: ICE's column -> ours, per issuer (:data:`~fixed_income_research.data.schema.issuer_classification`).
ISSUER = {
    "Industry Lvl 1 Desc": "sector",
    "Industry Lvl 2 Desc": "industry_group",
    "Industry Lvl 3 Desc": "industry",
    "Industry Lvl 4 Desc": "sub_industry",
    "ISO Country Code": "country",
    "Country-Region": "region",
}
IDENTITY = {
    "Cusip": "cusip",
    "ISIN number": "isin",
    "Description": "name",
    "Ticker": "ticker",
    "ISO Currency Code": "currency",
}
#: Not kept: derivable from two months (the MTD changes) or from the maturity (the years to it, empty in most files).
DROPPED = ("OAS MTD Change", "Asset Swap MTD Chg", "Years to Final Maturity")
REQUIRED = (*IDENTITY, *ANALYTICS, *CONSTITUENT, *ISSUER)


class LayoutError(ValueError):
    """A file is not an ICE constituent file as we know it."""


def parse_name(name: str) -> tuple[str, dt.date]:
    """``(index code, month end)`` from a file's name; refuses a name it cannot read or a date inside a month."""
    match = NAME.match(name)
    if match is None:
        raise LayoutError(f"{name}: expected <index>-<MMDDYYYY>.xlsx")
    code, month, day, year = match.groups()
    as_of = dt.date(int(year), int(month), int(day))
    if (as_of + dt.timedelta(days=1)).month == as_of.month:
        raise LayoutError(f"{name}: {as_of} is not a month end - the constituent files are month-end files")
    return code, as_of


def parse_title(title: str) -> tuple[str, dt.date, str] | None:
    """``(index code, date, universe month)`` a title row states, ``None`` when it is not a title."""
    match = TITLE.match(title.strip())
    if match is None:
        return None
    code, month, day, year, universe = match.groups()
    return code, dt.date(int(year), int(month), int(day)), universe


def read_constituents(path: Path) -> pd.DataFrame:
    """The bonds of one file under ICE's own column names; checks the title, where there is one, against the name."""
    sheet = pd.read_excel(path, header=None, dtype=object)
    first = sheet.iloc[:, 0].astype("string").str.strip()
    header_rows = first.index[first == "Cusip"]
    if len(header_rows) == 0:
        raise LayoutError(f"{path.name}: no header row (a first cell reading 'Cusip')")
    header = int(header_rows[0])
    stated = parse_title(str(sheet.iloc[0, 0])) if header > 0 else None
    code, as_of = parse_name(path.name)
    if stated is not None and (stated[0], stated[1]) != (code, as_of):
        raise LayoutError(f"{path.name}: its title says {stated[0]} for {stated[1]}")
    table = sheet.iloc[header + 1 :].copy()
    table.columns = [str(c).strip() for c in sheet.iloc[header]]
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise LayoutError(f"{path.name}: columns missing: {missing}")
    table = table[table["ISIN number"].notna()].reset_index(drop=True)  # the small print has no ISIN
    if table.empty:
        raise LayoutError(f"{path.name}: no bonds")
    return table


def cusip9(cusip8: str) -> str:
    """A CUSIP with its check digit (ICE prints the first eight characters; identifiers are kept in full)."""
    total = 0
    for position, char in enumerate(cusip8.upper()):
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char) - ord("A") + 10
        else:
            value = {"*": 36, "@": 37, "#": 38}[char]
        if position % 2 == 1:
            value *= 2
        total += value // 10 + value % 10
    return f"{cusip8.upper()}{(10 - total % 10) % 10}"


@dataclass(frozen=True)
class IndexMonth:
    """One file, normalised: its bonds, and its issuers with what ICE says of each."""

    bonds: pd.DataFrame
    issuers: pd.DataFrame
    #: issuers whose bonds named more than one sub-industry or country this month (the largest weight's was kept)
    issuers_disagreeing: int


def normalize(table: pd.DataFrame) -> IndexMonth:
    """One row per bond (identity, analytics, index weights) and one per issuer, from :func:`read_constituents`."""
    columns = {**IDENTITY, **ANALYTICS, **CONSTITUENT, **ISSUER}
    bonds = table[list(columns)].rename(columns=columns).copy()
    for text in (*IDENTITY.values(), "rating", "subordination", *ISSUER.values()):
        bonds[text] = bonds[text].astype("string").str.strip()
    numbers = [
        c for c in (*ANALYTICS.values(), *CONSTITUENT.values()) if c not in ("maturity_date", "rating", "subordination")
    ]
    bonds[numbers] = bonds[numbers].apply(pd.to_numeric, errors="coerce").astype(float)
    bonds["maturity_date"] = pd.to_datetime(bonds["maturity_date"], errors="coerce").dt.date
    twice = bonds["isin"][bonds["isin"].duplicated()].tolist()
    if twice:
        raise LayoutError(f"a bond is listed twice: {twice[:5]}")
    if bonds["weight_pct"].isna().any():
        raise LayoutError("a bond has no index weight")
    bonds["cusip"] = bonds["cusip"].map(lambda c: cusip9(c) if isinstance(c, str) and len(c) == 8 else c)
    # the issuer's attributes are those of its largest weight (a stable choice: by weight, then ISIN)
    ordered = bonds.sort_values(["ticker", "weight_pct", "isin"], ascending=[True, False, True])
    issuers = ordered.drop_duplicates("ticker")[["ticker", "name", *ISSUER.values()]].reset_index(drop=True)
    varied = bonds.groupby("ticker")[["sub_industry", "country"]].nunique().gt(1).any(axis=1)
    return IndexMonth(
        bonds=bonds.drop(columns=list(ISSUER.values())), issuers=issuers, issuers_disagreeing=int(varied.sum())
    )
