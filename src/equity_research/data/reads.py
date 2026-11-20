"""The research cache reads several research modules share, each written once.

    issuer = listing_issuers(cache, listings)                     # listing_id -> issuer_id
    facts = financial_facts(cache, issuer, ("sec", "refinitiv"))  # every fact of those vendors, with its unit
    counts = first_preferred(outstanding_counts(cache, listings), price_vendors, ["listing_id", "as_of_date"])

A universe, a risk model and a strategy's own inputs all read the same companies, accounts, corporate actions and
share counts. Reading them through one function each means a fix (a column added, a filter corrected) reaches every
reader at once instead of leaving copies that drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from equity_research.data.cache import ResearchCache


def id_list(values: Iterable[object]) -> str:
    """Integer ids as the inside of a SQL ``IN (...)``, each once; ``NULL`` when there are none (it matches nothing).

    Ids are written into the SQL rather than passed as parameters because a universe has thousands of them. Each
    goes through ``int``, so only numbers ever reach the query text.
    """
    ids = pd.Series(list(values), dtype=object).dropna().unique()
    return ", ".join(str(int(v)) for v in ids) or "NULL"


def as_vendors(vendors: str | Sequence[str]) -> tuple[str, ...]:
    """One vendor or several, in preference order, as a tuple (settings and callers name either)."""
    return (vendors,) if isinstance(vendors, str) else tuple(vendors)


def listing_issuers(cache: ResearchCache, listing_ids: Iterable[object]) -> pd.Series:
    """``listing_id -> issuer_id``: the company each listing belongs to (its accounts are the company's)."""
    rows = cache.query(f"SELECT listing_id, issuer_id FROM core.listing WHERE listing_id IN ({id_list(listing_ids)})")
    return rows.set_index("listing_id")["issuer_id"]


def financial_facts(cache: ResearchCache, issuer_ids: Iterable[object], vendors: Sequence[str]) -> pd.DataFrame:
    """Every accounting fact ``vendors`` report for these companies, with its ``unit`` and ``vendor``.

    The unit stays so the reader keeps only facts in its universe's currency
    (:func:`~equity_research.data.universe.vendor_items`); the vendor so each vendor's figures are built from its own
    facts alone.
    """
    marks = ", ".join("?" for _ in vendors)
    return cache.query(
        "SELECT issuer_id, item, period_type, period_start, period_end, value, published_at, unit, vendor "
        f"FROM equity.financial_fact WHERE vendor IN ({marks}) AND issuer_id IN ({id_list(issuer_ids)})",
        list(vendors),
    )


def corporate_actions(cache: ResearchCache, listing_ids: Iterable[object]) -> pd.DataFrame:
    """Every vendor's corporate actions for these listings: dividends with their currency, splits, spin-offs."""
    return cache.query(
        "SELECT listing_id, vendor, action_type, ex_date, amount, ratio, currency FROM equity.corporate_action "
        f"WHERE listing_id IN ({id_list(listing_ids)})"
    )


def outstanding_counts(cache: ResearchCache, listing_ids: Iterable[object]) -> pd.DataFrame:
    """Every vendor's dated outstanding share counts for these listings (each share line's own, never restated)."""
    return cache.query(
        "SELECT listing_id, vendor, as_of_date, shares FROM equity.share_count "
        f"WHERE kind = 'outstanding' AND listing_id IN ({id_list(listing_ids)})"
    )


def first_preferred(rows: pd.DataFrame, vendors: Sequence[str], key: Sequence[str]) -> pd.DataFrame:
    """One row per ``key``: the first of ``vendors`` that has it; a vendor not named ranks after all named ones.

    The order of ``rows`` is kept among equals (a stable sort), so a caller that already sorted them keeps that
    order as the tie-break.
    """
    rank = {v: i for i, v in enumerate(vendors)}
    ranked = rows.assign(_rank=rows["vendor"].map(rank).fillna(len(rank)))
    best = ranked.sort_values("_rank", kind="stable").drop_duplicates(list(key), keep="first")
    return best.drop(columns="_rank")
