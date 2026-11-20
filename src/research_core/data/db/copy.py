"""Copy a verified database to another one - the local build to the shared Azure database - and prove the copy.

    report = copy_database(source_engine, target_engine, source_raw=..., target_raw=...)

1. the target must be migrated to the same revision and hold no data yet (except the reference rows the migrations
   write - :data:`~research_core.data.db.schema.SEEDED` - which must match);
2. every table is copied in foreign-key order with its ids unchanged (``IDENTITY_INSERT``), so every reference
   (``listing_id``, ``batch_id``...) stays valid;
3. the raw zone is copied too: ``ingest.batch`` rows point at its files;
4. each table is compared on both sides - rows and a checksum of every column - and a difference is reported.

Every table of every asset class is copied, whichever command asks: they are loaded first
(:func:`~research_core.data.db.schema.load_asset_classes`). Nothing is changed in the source. A copy that stopped half
way is restarted on an emptied target.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from alembic.runtime.migration import MigrationContext
from sqlalchemy.engine import Connection, Engine

from research_core.data.db import schema
from research_core.data.db.upsert import records

log = logging.getLogger(__name__)

CHUNK_ROWS = 100_000


class CopyRefusedError(RuntimeError):
    """The target is not an empty database at the source's revision."""


@dataclass(frozen=True)
class TableCopy:
    """One table's copy and its check."""

    table: str
    source_rows: int
    target_rows: int
    same_checksum: bool

    @property
    def ok(self) -> bool:
        """Rows and checksum agree."""
        return self.source_rows == self.target_rows and self.same_checksum


def _name(t: sa.Table) -> str:
    return f"[{t.schema}].[{t.name}]"


def _revision(conn: Connection) -> str | None:
    return MigrationContext.configure(conn).get_current_revision()


def _rows(conn: Connection, t: sa.Table) -> int:
    return int(conn.execute(sa.text(f"SELECT COUNT_BIG(*) FROM {_name(t)}")).scalar_one())


def fingerprint(conn: Connection, t: sa.Table) -> tuple[int, int]:
    """``(rows, checksum)`` of a table over every column - equal on two servers when the contents are equal."""
    row = conn.execute(sa.text(f"SELECT COUNT_BIG(*), CHECKSUM_AGG(BINARY_CHECKSUM(*)) FROM {_name(t)}")).one()
    return int(row[0]), int(row[1] or 0)


def _seeded() -> tuple[sa.Table, ...]:
    """The reference tables the migrations fill: the same in every database, so compared, never copied."""
    return schema.SEEDED


def _tables() -> list[sa.Table]:
    """Every table of the database, parents before children."""
    schema.load_asset_classes()
    return list(schema.metadata.sorted_tables)


def _check_target(source: Engine, target: Engine) -> None:
    with source.connect() as src, target.connect() as dst:
        if _revision(src) != _revision(dst):
            raise CopyRefusedError(
                f"target is at revision {_revision(dst)}, the source at {_revision(src)}: migrate first"
            )
        busy = [x.fullname for x in _tables() if x not in _seeded() and _rows(dst, x) > 0]
        if busy:
            raise CopyRefusedError(f"the target already holds data in {busy[:5]}...: copy only into an empty database")
        for seeded in _seeded():
            if fingerprint(src, seeded) != fingerprint(dst, seeded):
                raise CopyRefusedError(f"the two databases differ in {seeded.fullname}")


def _copy_table(source: Engine, target: Engine, t: sa.Table) -> None:
    cols = [c.name for c in t.columns]
    col_list = ", ".join(f"[{c}]" for c in cols)
    insert = f"INSERT INTO {_name(t)} ({col_list}) VALUES ({', '.join('?' for _ in cols)})"
    identity = any(c.identity is not None for c in t.columns)
    with source.connect() as s:
        for chunk in pd.read_sql_query(sa.select(t), s, chunksize=CHUNK_ROWS):
            if chunk.empty:
                continue
            with target.begin() as conn:
                raw: Any = conn.connection.dbapi_connection
                cursor = raw.cursor()
                cursor.fast_executemany = True
                if identity:
                    cursor.execute(f"SET IDENTITY_INSERT {_name(t)} ON")
                cursor.executemany(insert, records(chunk, cols))
                if identity:
                    cursor.execute(f"SET IDENTITY_INSERT {_name(t)} OFF")
                cursor.close()
            log.info("copied %s: %d rows", t.fullname, len(chunk))


def copy_database(
    source: Engine, target: Engine, *, source_raw: Path | None = None, target_raw: Path | None = None
) -> list[TableCopy]:
    """Copy every table (and the raw zone when both folders are given); returns the per-table check."""
    _check_target(source, target)
    tables = _tables()
    for t in tables:  # parents before children
        if t not in _seeded():
            _copy_table(source, target, t)
    if source_raw is not None and target_raw is not None and source_raw.exists():
        shutil.copytree(source_raw, target_raw, dirs_exist_ok=True)
    report = []
    with source.connect() as src, target.connect() as dst:
        for table in tables:
            (sr, sc), (tr, tc) = fingerprint(src, table), fingerprint(dst, table)
            report.append(TableCopy(table.fullname, sr, tr, sc == tc))
    return report
