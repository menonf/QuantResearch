"""A bond index constituent workbook as ICE's Index Viewer writes one, for the tests of its loader."""

import datetime as dt
from pathlib import Path

import pandas as pd

BOND: dict[str, object] = {
    "Cusip": "013822AC",
    "ISIN number": "US013822AC54",
    "Description": "Alcoa Nederland Holding B.V.",
    "Ticker": "AA",
    "ISO Currency Code": "USD",
    "Par Wtd Coupon": 6.125,
    "Maturity Date": dt.datetime(2028, 5, 15),
    "Rating": "BB1",
    "Subordination Type": "Senior",
    "Price": 101.25,
    "Accrued Interest": 1.29,
    "OAS": 312.0,
    "Asset Swap Spread": 298.0,
    "Effective Yield": 5.71,
    "Yield to Worst (s.a.)": 5.91,
    "Yield to Maturity (s.a.)": 5.95,
    "Total Return % MTD_LOC": 6.226,
    "Excess Return % MTD": 5.554,
    "Effective Duration": 6.11,
    "Mod. Dur to Worst (s.a.)": 5.75,
    "Modified Duration (s.a.)": 6.94,
    "Effective Convexity": -0.05,
    "Convexity to Worst (s.a.)": 0.41,
    "Convexity (s.a.)": 0.6,
    "Spread Duration": 5.355,
    "% Mkt Value": 60.0,
    "% MktVal-PrevMend": 58.0,
    "Face Value_LOC": 500.0,
    "Cash": 0.0,
    "Industry Lvl 1 Desc": "Corporate",
    "Industry Lvl 2 Desc": "Industrials",
    "Industry Lvl 3 Desc": "Basic Industry",
    "Industry Lvl 4 Desc": "Metals/Mining Excluding Steel",
    "ISO Country Code": "US",
    "Country-Region": "Americas",
    "OAS MTD Change": -88.0,
    "Asset Swap MTD Chg": -78.0,
    "Years to Final Maturity": None,
}
SECOND = BOND | {"Cusip": "013822AA", "ISIN number": "US013822AA98", "% Mkt Value": 40.0, "Par Wtd Coupon": 6.75}


def write_file(path: Path, bonds: list[dict[str, object]], title: str | None = None) -> Path:
    """A workbook as ICE's Index Viewer writes one: a title (sometimes), the header, the bonds, the small print."""
    header = list(BOND)
    rows: list[list[object]] = [] if title is None else [[title, *[None] * (len(header) - 1)]]
    rows.append([*header])
    rows += [[bond[c] for c in header] for bond in bonds]
    rows += [[None] * len(header), ["Please see indices.ice.com for more", *[None] * (len(header) - 1)]]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    return path
