"""The equity research cache: the tables it mirrors (see :mod:`research_core.data.cache` for how).

    cache = ResearchCache()                 # under <data folder>/cache/equity
    cache.sync(engine)                      # pull what changed since the last sync
    cache.query("SELECT * FROM equity.price_bar WHERE listing_id = 7")

Reference tables are replaced whole at each sync; fact tables gain the rows loaded since the last one.
"""

from __future__ import annotations

from pathlib import Path

from equity_research import ASSET_CLASS
from equity_research.data import schema
from research_core.data import cache as core
from research_core.data.cache import CachedTable
from research_core.data.db import schema as core_schema
from research_core.settings import get_settings

TABLES: tuple[CachedTable, ...] = (
    CachedTable(core_schema.vendor, incremental=False),
    CachedTable(core_schema.issuer, incremental=False),
    CachedTable(schema.classification, incremental=False),
    CachedTable(core_schema.listing, incremental=False),
    CachedTable(core_schema.identifier, incremental=False),
    CachedTable(schema.index_definition, incremental=False),
    CachedTable(schema.index_membership, incremental=False),
    CachedTable(schema.universe, incremental=False),
    CachedTable(schema.universe_component, incremental=False),
    CachedTable(schema.price_bar, incremental=True),
    CachedTable(schema.corporate_action, incremental=True),
    CachedTable(schema.share_count, incremental=True),
    CachedTable(schema.filing, incremental=True),
    CachedTable(schema.financial_fact, incremental=True),
    CachedTable(schema.estimate, incremental=True),
    CachedTable(schema.estimate_revision, incremental=True),
    CachedTable(schema.estimate_actual, incremental=True),
    CachedTable(schema.document, incremental=True),
    CachedTable(schema.text_measure, incremental=True),
    CachedTable(core_schema.rule_set, incremental=False),
    CachedTable(core_schema.strategy_rule, incremental=False),
)


class ResearchCache(core.ResearchCache):
    """A per-researcher mirror of the equity tables and of the tables every asset class shares."""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__(TABLES, path or get_settings().cache_dir / ASSET_CLASS)
