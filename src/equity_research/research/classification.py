"""Which sector a company was in on a given day, from the dated ``ref.classification`` history.

    on_day = classification_on(cache)                       # today
    then = classification_on(cache, pd.Timestamp("2015-06-30"))

History starts the day a classification was first recorded. Asked about an earlier day, the earliest known one is
returned with ``known_from`` after the day - the caller sees that it is a stand-in (GICS has changed: Alphabet and
Meta moved to Communication Services in 2018, Visa and Mastercard to Financials in 2023).
"""

from __future__ import annotations

import pandas as pd

from equity_research.research.cache import ResearchCache
from equity_research.research.reads import listing_issuers
from equity_research.settings import vendor_preferences

LEVELS = ["sector", "industry_group", "industry", "sub_industry"]


def classification_on(
    cache: ResearchCache,
    day: pd.Timestamp | None = None,
    *,
    vendors: tuple[str, ...] | None = None,
    scheme: str = "GICS",
) -> pd.DataFrame:
    """``issuer_id`` x ``[sector, industry_group, industry, sub_industry, vendor, known_from]`` on ``day``.

    From the first of ``vendors`` (default: the preferred classification vendors) that classifies the company.
    ``known_from`` is when that classification was first recorded: later than ``day`` means a stand-in.
    """
    when = pd.Timestamp(day or pd.Timestamp.today().normalize())
    rows = cache.query(
        "SELECT issuer_id, vendor, valid_from, valid_to, sector, industry_group, industry, sub_industry "
        "FROM ref.classification WHERE scheme = ?",
        [scheme],
    )
    order = {v: i for i, v in enumerate(vendors or vendor_preferences().classification)}
    rows = rows[rows["vendor"].isin(order)].assign(
        _rank=lambda d: d["vendor"].map(order),
        valid_from=lambda d: pd.to_datetime(d["valid_from"]),
        valid_to=lambda d: pd.to_datetime(d["valid_to"]),
    )
    held = (rows["valid_from"] <= when) & (rows["valid_to"].isna() | (rows["valid_to"] > when))
    first_known = rows.sort_values("valid_from").drop_duplicates(["issuer_id", "vendor"])
    stand_in = first_known[
        ~first_known.set_index(["issuer_id", "vendor"]).index.isin(rows[held].set_index(["issuer_id", "vendor"]).index)
    ]
    stand_in = stand_in[stand_in["valid_from"] > when]
    candidates = pd.concat([rows[held].assign(_stand_in=False), stand_in.assign(_stand_in=True)])
    # a classification that held on the day beats a stand-in, then the preferred vendor
    picked = candidates.sort_values(["_stand_in", "_rank"]).drop_duplicates("issuer_id")
    picked = picked.assign(known_from=picked["valid_from"])
    return picked.set_index("issuer_id")[[*LEVELS, "vendor", "known_from"]].sort_index()


def listing_sectors(
    cache: ResearchCache, listing_ids: pd.Index | list[int], day: pd.Timestamp | None = None
) -> pd.Series:
    """``listing_id -> sector`` on ``day`` (each listing takes its company's); "unknown" where none is recorded."""
    sector = listing_issuers(cache, listing_ids).map(classification_on(cache, day)["sector"])
    return sector.reindex(list(listing_ids)).fillna("unknown")
