"""The bond indices we know: what each is called, who publishes it, and where its official level is published.

    ensure_indices(conn)                 # every index below in ``fixed_income.index_definition``; {code: index_id}

An index's constituent files name it by the provider's code (``H0A0``); a file of an index not listed here is
refused, so a typo in a file name cannot start a new index.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from fixed_income_research.data import schema


@dataclass(frozen=True)
class BondIndex:
    """One index: its name, provider and currency, and the FRED series of its official total-return level."""

    code: str
    name: str
    provider: str = "ICE Data Indices"
    currency: str = "USD"
    fred_series: str | None = None


#: FRED publishes the total-return level of ICE's headline indices (the last three years only, so it is loaded
#: every month and kept).
INDICES: dict[str, BondIndex] = {
    i.code: i
    for i in (
        BondIndex("H0A0", "ICE BofA US High Yield Index", fred_series="BAMLHYH0A0HYM2TRIV"),
        BondIndex("C0A0", "ICE BofA US Corporate Index", fred_series="BAMLCC0A0CMTRIV"),
    )
}


def ensure_indices(conn: Connection) -> dict[str, int]:
    """``{code: index_id}`` of every index in :data:`INDICES`, created where the database lacks it."""
    t = schema.index_definition
    known = {str(code): int(index_id) for code, index_id in conn.execute(sa.select(t.c.code, t.c.index_id))}
    for index in INDICES.values():
        if index.code not in known:
            known[index.code] = int(
                conn.execute(
                    sa.insert(t)
                    .values(code=index.code, name=index.name, provider=index.provider, currency=index.currency)
                    .returning(t.c.index_id)
                ).scalar_one()
            )
    return known
