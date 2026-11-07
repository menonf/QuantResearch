"""Match issuers to SEC CIKs. Pure functions.

An issuer gets a CIK from, in order:

1. **ticker** - one of its live listings' tickers in the SEC's current ticker file (``BRKb.N`` -> ``BRK-B``); a retired
   (``^``) RIC is never used: tickers are reused by other companies;
2. **CUSIP** - its 6-character CUSIP issuer code in the public CIK-CUSIP map (13D / 13G filings): a code that points to
   one CIK in at least 80 % of its rows, else - for ties - the candidate that files XBRL statements (the other
   candidates are usually the filers of the 13D, not the issuer).

An issuer whose ticker and CUSIP disagree keeps the ticker's CIK and is flagged (``conflict``).

3. **CUSIP and name** (``cusip+name``) - every other CIK of the map's rows for its CUSIP that files XBRL statements
   under the company's own name. A tie the 80 % rule cannot settle (Stericycle: the company and one holder that filed
   about it) is settled this way, and a company that re-registered (Disney in 2019, Cigna in 2018: a new holding
   company, a new CIK) finds its old CIK - its accounts before the change are filed there. Such a CIK can also be a
   subsidiary that files its own accounts (Duke Energy Carolinas) or a number that changed hands; the SEC load uses it
   only for periods before the company's main CIK begins, and never merges companies over it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd

#: Words a registrant's name carries that do not tell one company from another.
NAME_NOISE = frozenset(
    {"the", "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited", "plc", "llc"}
    | {"lp", "l", "p", "sa", "nv", "ag", "se", "holdings", "holding", "group", "new", "class", "common", "de", "del"}
)
_STATE = re.compile(r"/[a-z]{2,3}/?\s*$")  # EDGAR's state of incorporation: "DUN & BRADSTREET CORP/NW/"


def name_words(name: object) -> frozenset[str]:
    """The words of a company name that identify it (``"The Walt Disney Company"`` -> ``{walt, disney}``)."""
    if not isinstance(name, str):
        return frozenset()
    text = _STATE.sub(" ", name.lower()).replace("&", " and ")
    return frozenset(w for w in re.findall(r"[a-z0-9]+", text) if w not in NAME_NOISE)


def same_name(a: object, b: object) -> bool:
    """One name's identifying words include the other's (``CIGNA CORP`` and ``The Cigna Group``)."""
    wa, wb = name_words(a), name_words(b)
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


def sec_ticker(ticker: object) -> str | None:
    """The SEC's spelling of one of our tickers: a share class with a hyphen (``BRK.B`` -> ``BRK-B``)."""
    if not isinstance(ticker, str) or not ticker.strip():
        return None
    return ticker.strip().upper().replace(".", "-")


def unique_cik(cusip_map: pd.DataFrame, key: str, *, share: float = 0.8) -> pd.Series:
    """``code -> CIK`` where the code points to one CIK in at least ``share`` of its rows."""
    counts = cusip_map.groupby([key, "cik"]).size().rename("n").reset_index()
    total = counts.groupby(key)["n"].transform("sum")
    best = counts[counts["n"] / total >= share].drop_duplicates(key)
    return best.set_index(key)["cik"]


def match_issuers(
    issuers: pd.DataFrame,
    tickers: pd.DataFrame,
    cusip_map: pd.DataFrame,
    filer_ciks: Iterable[int] | None = None,
    names: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """``[issuer_id, cik, method, conflict]`` - one row per issuer and CIK (a re-registered company has several).

    Args:
        issuers: ``[issuer_id, cusip6, ticker, name]`` - one row per issuer and current ticker of one of its listings
            (our spelling, whichever vendor recorded it); ``name`` is optional.
        tickers: the SEC ticker file ``[cik, ticker]``.
        cusip_map: ``[cik, cusip6]`` from the 13D / 13G map.
        filer_ciks: CIKs with XBRL statements (settles CUSIP ties).
        names: ``{CIK: registrant name}`` of XBRL filers - with ``issuers.name``, adds the CUSIP's CIKs filed under
            the company's own name (step 3 of the module docstring).
    """
    df = issuers.copy()
    df["ticker"] = df["ticker"].map(sec_ticker)
    by_ticker = tickers.dropna(subset=["ticker"]).drop_duplicates("ticker").set_index("ticker")["cik"]
    df["ticker_cik"] = df["ticker"].map(lambda t: by_ticker.get(t) if isinstance(t, str) else None).astype(float)
    t_cik = df.dropna(subset=["ticker_cik"]).groupby("issuer_id")["ticker_cik"].agg(lambda s: s.mode().iloc[0])

    codes = df.drop_duplicates("issuer_id").set_index("issuer_id")["cusip6"]
    c_cik = codes.map(unique_cik(cusip_map, "cusip6")).astype(float)
    if filer_ciks is not None:
        filers = cusip_map[cusip_map["cik"].isin({int(c) for c in filer_ciks})]
        c_cik = c_cik.where(c_cik.notna(), codes.map(unique_cik(filers, "cusip6")).astype(float))

    out = pd.DataFrame(index=pd.Index(sorted(df["issuer_id"].unique()), name="issuer_id"))
    out["ticker_cik"] = t_cik
    out["cusip_cik"] = c_cik
    out["cik"] = out["ticker_cik"].where(out["ticker_cik"].notna(), out["cusip_cik"])
    out["method"] = None
    out.loc[out["cusip_cik"].notna(), "method"] = "cusip"
    out.loc[out["ticker_cik"].notna(), "method"] = "ticker"
    out["conflict"] = out["ticker_cik"].notna() & out["cusip_cik"].notna() & (out["ticker_cik"] != out["cusip_cik"])
    out["cik"] = out["cik"].astype("Int64")
    found = out.reset_index()[["issuer_id", "cik", "method", "conflict"]]
    if names is None or "name" not in issuers.columns:
        return found
    own = issuers.drop_duplicates("issuer_id")[["issuer_id", "cusip6", "name"]].dropna(subset=["cusip6"])
    candidates = own.merge(cusip_map[["cik", "cusip6"]].drop_duplicates(), on="cusip6")
    named = candidates[
        [c in names and same_name(n, names[c]) for c, n in zip(candidates["cik"], candidates["name"], strict=True)]
    ]
    known = set(zip(found["issuer_id"], found["cik"].astype("float"), strict=True))
    extra = named[[(i, float(c)) not in known for i, c in zip(named["issuer_id"], named["cik"], strict=True)]]
    extra = extra.drop_duplicates(["issuer_id", "cik"]).assign(method="cusip+name", conflict=False)
    extra = extra.astype({"cik": "Int64"})[["issuer_id", "cik", "method", "conflict"]]
    out_rows = pd.concat([found, extra], ignore_index=True)
    # an issuer whose only row has no CIK keeps it out; one with a CIK found by name loses its empty row
    has = out_rows.dropna(subset=["cik"])["issuer_id"].unique()
    out_rows = out_rows[out_rows["cik"].notna() | ~out_rows["issuer_id"].isin(has)]
    return out_rows.sort_values(["issuer_id", "method"]).reset_index(drop=True)
