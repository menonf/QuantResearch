"""The fixed income research cache: the tables it mirrors (see :mod:`research_core.data.cache` for how).

    cache = ResearchCache()                 # under <data folder>/cache/fixed_income
    cache.sync(engine)                      # pull what changed since the last sync
    cache.query("SELECT * FROM fixed_income.bond_analytics WHERE listing_id = 7")

Each asset class's cache has a folder of its own (``cache/<asset class>``), as its tables have a schema of their own:
loading bonds must not move the equity research's data version (the newest row its cache has synced), which would make
every equity artifact stale.
"""

from __future__ import annotations

from pathlib import Path

from fixed_income_research import ASSET_CLASS
from fixed_income_research.data import schema
from research_core.data import cache as core
from research_core.data.cache import CachedTable
from research_core.data.db import schema as core_schema
from research_core.settings import get_settings

TABLES: tuple[CachedTable, ...] = (
    CachedTable(core_schema.vendor, incremental=False),
    CachedTable(core_schema.issuer, incremental=False),
    CachedTable(core_schema.listing, incremental=False),
    CachedTable(core_schema.identifier, incremental=False),
    CachedTable(schema.index_definition, incremental=False),
    CachedTable(schema.index_level, incremental=True),
    CachedTable(schema.index_constituent, incremental=True),
    CachedTable(schema.bond_analytics, incremental=True),
    CachedTable(schema.issuer_classification, incremental=True),
)


class ResearchCache(core.ResearchCache):
    """A per-researcher mirror of the fixed income tables and of the tables every asset class shares."""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(TABLES, path or get_settings().cache_dir / ASSET_CLASS)
