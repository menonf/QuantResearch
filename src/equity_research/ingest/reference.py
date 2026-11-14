"""Our indices and universes: what they are called, what trading in them costs, and their benchmark.

These are our own definitions, not any vendor's: the S&P 1500 universe is the S&P 500 + MidCap 400 + SmallCap 600,
the FTSE All-Share the FTSE 100 + 250 + SmallCap, and a stock is charged the costs of the segment it belongs to.
Vendors only supply the members (LSEG) and the levels. An index's level and a universe's benchmark are named by an
identifier, so whichever vendor's listing carries that code is used.

    index_ids = write_reference(conn)        # idempotent; returns {index code: index_id}
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection

from equity_research.db import schema
from equity_research.db.upsert import upsert
from equity_research.ingest.identity import anchor_map


@dataclass(frozen=True)
class IndexSpec:
    """An index whose members we track."""

    code: str
    name: str
    levels: tuple[tuple[str, str], ...]  # (id_type, value) codes of the listing carrying its level, any vendor's
    provider: str = "S&P Dow Jones Indices"
    currency: str = "USD"
    country: str = "US"


@dataclass(frozen=True)
class Component:
    """One index inside a universe: its segment and what trading it costs (basis points)."""

    index: str
    segment: str
    cost_bps: float  # one way, per unit traded
    borrow_bps: float  # a year, on short positions
    priority: int = 0  # a stock in two indices takes the segment with the lowest priority number


@dataclass(frozen=True)
class UniverseSpec:
    """A pool strategies pick from."""

    code: str
    name: str
    benchmarks: tuple[tuple[str, str], ...]  # the same, for the benchmark
    components: tuple[Component, ...]
    country: str = "US"


#: A trade's cost, one way, in basis points of the trade: an all-in assumption by segment (no vendor measures it) for
#: the broker's commission, half the quoted spread and the price our own order moves - the wider the smaller the
#: company. The commission is the small part: Interactive Brokers' fixed schedule (IBKR Pro, read 2026-11-12) is
#: $0.005 a share in New York (at least $1 an order) - about 0.5, 0.75 and 1.5 basis points at the segments' median
#: share prices of 2024 ($120, $67, $33) - and 0.05 % of the trade in London (5 basis points, at least GBP 3 an
#: order), for orders of $10,000 / GBP 6,000 or more. These costs cover it (the owner's decision of 2026-11-12).
LARGE = Component("SP500", "large", 10.0, 25.0, 0)
MID = Component("SP400", "mid", 25.0, 50.0, 1)
SMALL = Component("SP600", "small", 50.0, 100.0, 2)

#: LSEG (RIC), Norgate (its index symbols are tickers) and Yahoo name the same levels differently; the first code
#: some vendor's listing carries is used.
SP500_LEVEL = (("RIC", ".SPX"), ("TICKER", "$SPX"), ("YAHOO", "^GSPC"))
SP400_LEVEL = (("RIC", ".MID"), ("TICKER", "$MID"), ("YAHOO", "^SP400"))
SP600_LEVEL = (("RIC", ".SP600"), ("TICKER", "$SML"), ("YAHOO", "^SP600"))
#: The S&P 1500 is measured by the fund that holds it (SPTM - its fees and dividends make it a total-return
#: benchmark): LSEG's level (.SPCOMP) starts only in May 2011, Yahoo keeps no history of ^SP1500, and Norgate's index
#: symbols do not include it, while SPTM's prices reach back to 2000 from a free source - the same in every build
SP1500_LEVEL = (("YAHOO", "SPTM"), ("RIC", ".SPCOMP"), ("YAHOO", "^SP1500"))

#: London: the FTSE UK indices. Small caps trade wider in London than in New York, and shorts are borrowed dearer;
#: stamp duty on purchases is charged apart (it depends on where a company is incorporated, not on its segment).
UK_LARGE = Component("FTSE100", "large", 10.0, 25.0, 0)
UK_MID = Component("FTSE250", "mid", 30.0, 75.0, 1)
UK_SMALL = Component("FTSESC", "small", 75.0, 150.0, 2)
FTSE100_LEVEL = (("RIC", ".FTSE"),)
FTSE250_LEVEL = (("RIC", ".FTMC"),)
FTSESC_LEVEL = (("RIC", ".FTSC"),)
#: The All-Share universe is measured against the All-Share with dividends (from 2003). Research leaves investment
#: trusts out of the universe, but LSEG's All-Share without them (.FTASXXMGBPT) starts only in September 2024 and its
#: daily closes do not line up with the All-Share's (a correlation of 0.73); with or without the trusts our own
#: reconstruction differs by 0.04 points a year (notebook 13), so the whole All-Share measures the universe well.
FTSEAS_BENCHMARK = (("RIC", ".FTASXGBPT"),)
UK = {"provider": "FTSE Russell", "currency": "GBP", "country": "GB"}

INDEXES: tuple[IndexSpec, ...] = (
    IndexSpec("SP500", "S&P 500", SP500_LEVEL),
    IndexSpec("SP400", "S&P MidCap 400", SP400_LEVEL),
    IndexSpec("SP600", "S&P SmallCap 600", SP600_LEVEL),
    IndexSpec("FTSE100", "FTSE 100", FTSE100_LEVEL, **UK),
    IndexSpec("FTSE250", "FTSE 250", FTSE250_LEVEL, **UK),
    IndexSpec("FTSESC", "FTSE SmallCap", FTSESC_LEVEL, **UK),
)
UNIVERSES: tuple[UniverseSpec, ...] = (
    UniverseSpec("SP500", "S&P 500", SP500_LEVEL, (LARGE,)),
    UniverseSpec("SP400", "S&P MidCap 400", SP400_LEVEL, (MID,)),
    UniverseSpec("SP600", "S&P SmallCap 600", SP600_LEVEL, (SMALL,)),
    UniverseSpec("SP1500", "S&P Composite 1500", SP1500_LEVEL, (LARGE, MID, SMALL)),
    UniverseSpec("FTSE100", "FTSE 100", FTSE100_LEVEL, (UK_LARGE,), country="GB"),
    UniverseSpec("FTSE250", "FTSE 250", FTSE250_LEVEL, (UK_MID,), country="GB"),
    UniverseSpec("FTSESC", "FTSE SmallCap", FTSESC_LEVEL, (UK_SMALL,), country="GB"),
    UniverseSpec("FTSEAS", "FTSE All-Share", FTSEAS_BENCHMARK, (UK_LARGE, UK_MID, UK_SMALL), country="GB"),
)


def level_codes(country: str | None = None) -> list[tuple[str, str]]:
    """Every identifier an index level or a benchmark is named by - of one country's indices and universes if given."""
    indexes = [i for i in INDEXES if country in (None, i.country)]
    universes = [u for u in UNIVERSES if country in (None, u.country)]
    return sorted({c for i in indexes for c in i.levels} | {c for u in universes for c in u.benchmarks})


def _listing(conn: Connection, codes: tuple[tuple[str, str], ...]) -> int | None:
    """The listing the first of ``codes`` names, if any vendor's listing carries one of them."""
    for id_type, value in codes:
        found = anchor_map(conn, "listing", id_type).get(value)
        if found is not None:
            return found
    return None


def write_reference(conn: Connection) -> dict[str, int]:
    """Upsert :data:`INDEXES` and :data:`UNIVERSES`; ``{index code: index_id}``."""
    upsert(
        conn,
        schema.index_definition,
        pd.DataFrame(
            {
                "code": [i.code for i in INDEXES],
                "name": [i.name for i in INDEXES],
                "provider": [i.provider for i in INDEXES],
                "currency": [i.currency for i in INDEXES],
                "country": [i.country for i in INDEXES],
                "level_listing_id": pd.array([_listing(conn, i.levels) for i in INDEXES], dtype="Int64"),
            }
        ),
        key=["code"],
    )
    index_ids = {
        str(c): int(i)
        for c, i in conn.execute(sa.select(schema.index_definition.c.code, schema.index_definition.c.index_id))
    }
    upsert(
        conn,
        schema.universe,
        pd.DataFrame(
            {
                "code": [u.code for u in UNIVERSES],
                "name": [u.name for u in UNIVERSES],
                "benchmark_listing_id": pd.array([_listing(conn, u.benchmarks) for u in UNIVERSES], dtype="Int64"),
            }
        ),
        key=["code"],
    )
    universe_ids = {
        str(c): int(i) for c, i in conn.execute(sa.select(schema.universe.c.code, schema.universe.c.universe_id))
    }
    upsert(
        conn,
        schema.universe_component,
        pd.DataFrame(
            [
                {
                    "universe_id": universe_ids[u.code],
                    "index_id": index_ids[c.index],
                    "segment": c.segment,
                    "cost_bps": c.cost_bps,
                    "borrow_bps": c.borrow_bps,
                    "priority": c.priority,
                }
                for u in UNIVERSES
                for c in u.components
            ]
        ),
        key=["universe_id", "index_id"],
    )
    return index_ids
