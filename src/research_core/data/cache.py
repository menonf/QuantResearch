"""The local research cache: canonical tables as Parquet, queried with DuckDB.

    cache = ResearchCache(tables)           # under settings.cache_dir; each asset class names its tables
    cache.sync(engine)                      # pull what changed since the last sync
    cache.query("SELECT * FROM equity.price_bar WHERE listing_id = 7")

Backtests read millions of rows; reading them from the local cache is fast and costs the shared database nothing.

* **Fact tables** (prices, actions, share counts, filings, financial facts) sync incrementally: rows
  with ``loaded_at`` after the last sync are appended as a new part file. A changed row arrives again with a later
  ``loaded_at``; the views keep the latest version of each primary key.
* **Reference tables** (``ref.*``) are small and replaced whole on every sync, so deletions (issuer merges) arrive too.
* :meth:`ResearchCache.compact` rewrites a table's parts into one file with one version per key.
* Tables store a vendor as its ``vendor_id``; every view also has the readable ``vendor`` code (``refinitiv``, ``sec``,
  ``yahoo``), so research asks for ``vendor = 'sec'``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from research_core.data.db import schema
from research_core.settings import get_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedTable:
    """A canonical table mirrored in the cache."""

    table: sa.Table
    incremental: bool

    @property
    def name(self) -> str:
        """``schema.table``."""
        return f"{self.table.schema}.{self.table.name}"

    @property
    def key(self) -> list[str]:
        """Primary-key columns."""
        return [c.name for c in self.table.primary_key.columns]


CHUNK_ROWS = 500_000


def _duckdb_type(sql_type: sa.types.TypeEngine[object]) -> str:
    """The DuckDB type an empty cached table gives a column (what its Parquet files would hold)."""
    if isinstance(sql_type, sa.Integer):  # SmallInteger and BigInteger too
        return "BIGINT"
    if isinstance(sql_type, sa.Float | sa.Numeric):
        return "DOUBLE"
    if isinstance(sql_type, sa.DateTime):
        return "TIMESTAMP"
    if isinstance(sql_type, sa.Date):
        return "DATE"
    if isinstance(sql_type, sa.Boolean):
        return "BOOLEAN"
    return "VARCHAR"


class ResearchCache:
    """A per-researcher mirror of ``tables`` (see the module docstring).

    ``tables`` are mirrored in the order given: the vendors' table first, so every view can name its vendor.
    """

    def __init__(self, tables: Sequence[CachedTable], path: Path | None = None) -> None:
        self.tables = tuple(tables)
        self.path = path or get_settings().cache_dir
        self.path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.path / "manifest.json"

    # ------------------------------------------------------------ manifest

    def manifest(self) -> dict[str, str]:
        """``{table: latest loaded_at synced (ISO)}``."""
        if not self._manifest_path.exists():
            return {}
        data: dict[str, str] = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return data

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        self._manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _dir(self, t: CachedTable) -> Path:
        d = self.path / t.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ sync

    def sync(
        self, engine: Engine, tables: Sequence[CachedTable] | None = None, *, rebuild: Sequence[str] = ()
    ) -> dict[str, int]:
        """Pull new and changed rows of ``tables`` (default: all of the cache's); returns ``{table: rows written}``.

        An incremental table only gains rows: one deleted in the database (a repair re-deriving facts from the raw
        zone) stays in the cache until the table is named in ``rebuild``, which copies it afresh.
        """
        tables = self.tables if tables is None else tables
        manifest = self.manifest()
        unknown = set(rebuild) - {t.name for t in tables}
        if unknown:
            raise KeyError(f"no cached table {sorted(unknown)}; known: {sorted(t.name for t in tables)}")
        for t in tables:
            if t.name in rebuild:
                for old in self._dir(t).glob("*.parquet"):
                    old.unlink()
                manifest.pop(t.name, None)
        written: dict[str, int] = {}
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
        for t in tables:
            folder = self._dir(t)
            select = sa.select(t.table)
            since = manifest.get(t.name) if t.incremental else None
            if since is not None:
                select = select.where(t.table.c.loaded_at > dt.datetime.fromisoformat(since))
            if not t.incremental:
                for old in folder.glob("*.parquet"):
                    old.unlink()
            rows, high = 0, since
            with engine.connect() as conn:
                for i, chunk in enumerate(pd.read_sql_query(select, conn, chunksize=CHUNK_ROWS)):
                    if chunk.empty:
                        continue
                    chunk.to_parquet(folder / f"part-{stamp}-{i:04d}.parquet", index=False, compression="zstd")
                    rows += len(chunk)
                    if "loaded_at" in chunk:
                        latest = pd.Timestamp(chunk["loaded_at"].max()).isoformat()
                        high = max(high, latest) if high else latest
            if t.incremental and high:
                manifest[t.name] = high
            written[t.name] = rows
            log.info("cache %s: %d rows", t.name, rows)
        self._save_manifest(manifest)
        return written

    def compact(self, table: CachedTable) -> int:
        """Rewrite a table's parts as one file with the latest version per key; returns the rows kept."""
        folder = self._dir(table)
        parts = sorted(folder.glob("*.parquet"))
        if len(parts) <= 1:
            return 0
        frame = self.query(f"SELECT * FROM {table.name}")
        target = folder / f"part-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-compact.parquet"
        frame.to_parquet(target, index=False, compression="zstd")
        for p in parts:
            p.unlink()
        return len(frame)

    # ------------------------------------------------------------ reads

    def connect(self) -> duckdb.DuckDBPyConnection:
        """An in-memory DuckDB with one view per cached table (latest version of each key)."""
        con = duckdb.connect()
        has_vendors = any((self.path / f"{schema.vendor.schema}.{schema.vendor.name}").glob("*.parquet"))
        for t in self.tables:
            folder = self.path / t.name
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {t.table.schema}")
            if not any(folder.glob("*.parquet")):
                # nothing loaded yet (share counts before their step has run): an empty table, so a query finds no
                # rows instead of failing
                columns = [f"{c.name} {_duckdb_type(c.type)}" for c in t.table.c]
                if has_vendors and "vendor_id" in t.table.c and t.table is not schema.vendor:
                    columns.append("vendor VARCHAR")
                con.execute(f"CREATE TABLE {t.name} ({', '.join(columns)})")
                continue
            source = f"read_parquet('{(folder / '*.parquet').as_posix()}', union_by_name = true)"
            # a column added by a migration after these files were written (an issuer's incorporation country) reads
            # as empty until the next sync brings it, instead of failing every query that names it
            present = {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
            added = [f"CAST(NULL AS {_duckdb_type(c.type)}) AS {c.name}" for c in t.table.c if c.name not in present]
            if added:
                source = f"(SELECT *, {', '.join(added)} FROM {source})"
            if t.incremental:
                keys = ", ".join(t.key)
                latest = f"row_number() OVER (PARTITION BY {keys} ORDER BY loaded_at DESC) = 1"
                source = f"(SELECT * FROM {source} QUALIFY {latest})"
            if has_vendors and "vendor_id" in t.table.c and t.table is not schema.vendor:
                source = (
                    f"(SELECT s.*, v.vendor_code AS vendor FROM {source} s "
                    "LEFT JOIN core.vendor v ON v.vendor_id = s.vendor_id)"
                )
            con.execute(f"CREATE VIEW {t.name} AS SELECT * FROM {source}")
        return con

    def query(self, sql: str, params: Sequence[object] | None = None) -> pd.DataFrame:
        """Run ``sql`` against the cached tables."""
        with self.connect() as con:
            result: pd.DataFrame = con.execute(sql, params or []).df()
        return result
