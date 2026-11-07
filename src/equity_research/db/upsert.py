"""Idempotent, change-only writes of DataFrames into canonical tables.

    result = upsert(conn, schema.price_bar, frame, key=["listing_id", "vendor", "trade_date"])

1. the frame is bulk-inserted into a session temp table copied from the target (same types and collations);
2. one ``MERGE`` inserts new keys and updates existing ones **only where a value changed** (``EXCEPT`` compares
   NULLs as equal), stamping ``loaded_at`` / ``batch_id`` on the rows it touches;
3. ``insert_only=True`` never updates (append-only facts such as ``fund.financial_fact``).

Re-loading identical data writes nothing, so loads can be re-run freely.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection

#: Columns maintained by the writer, never compared as data.
PROVENANCE = ("loaded_at", "batch_id")
CHUNK_ROWS = 50_000


@dataclass(frozen=True)
class UpsertResult:
    """Rows inserted and rows whose values changed."""

    inserted: int = 0
    updated: int = 0

    def __add__(self, other: UpsertResult) -> UpsertResult:
        return UpsertResult(self.inserted + other.inserted, self.updated + other.updated)


def _quote(name: str) -> str:
    return f"[{name}]"


def _stage_columns(table: sa.Table, cols: Sequence[str]) -> str:
    """Select list that copies the target's column types into a temp table, minus any IDENTITY property."""
    out = []
    for c in cols:
        is_identity = c in table.c and table.c[c].identity is not None
        out.append(f"{_quote(c)} + 0 AS {_quote(c)}" if is_identity else _quote(c))
    return ", ".join(out)


def _qualified(table: sa.Table) -> str:
    return f"{_quote(table.schema)}.{_quote(table.name)}" if table.schema else _quote(table.name)


def python_value(value: Any) -> Any:
    """A DataFrame cell as a DB-API parameter: NaN / NaT / NA -> None, numpy and pandas scalars -> Python."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()
    if isinstance(value, np.generic):
        item = value.item()
        return None if isinstance(item, float) and math.isnan(item) else item
    return value


def records(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple[Any, ...]]:
    """Rows of ``frame[columns]`` as tuples of DB-API values; DATE columns become ``datetime.date``."""
    out = []
    for row in frame[list(columns)].itertuples(index=False, name=None):
        out.append(tuple(python_value(v) for v in row))
    return out


def coerce_to_table(frame: pd.DataFrame, table: sa.Table) -> pd.DataFrame:
    """Cast columns to the target's types: DATE -> ``datetime.date``, integers -> nullable ``Int64``.

    A pandas integer column with a missing value turns float (``4.0``), which SQL Server refuses for an INT column.
    """
    frame = frame.copy()
    for col in frame.columns:
        if col not in table.c:
            continue
        sql_type = table.c[col].type
        if isinstance(sql_type, sa.Date):
            frame[col] = pd.to_datetime(frame[col]).dt.date.where(frame[col].notna(), None)
        elif isinstance(sql_type, sa.Integer) and not isinstance(sql_type, sa.Boolean):
            frame[col] = pd.to_numeric(frame[col]).astype("Int64")
    return frame


INTEGER_RANGES: dict[type[sa.types.TypeEngine[Any]], tuple[int, int]] = {
    sa.SmallInteger: (-(2**15), 2**15 - 1),
    sa.BigInteger: (-(2**63), 2**63 - 1),
    sa.Integer: (-(2**31), 2**31 - 1),
}


def _integer_range(sql_type: sa.types.TypeEngine[Any]) -> tuple[int, int] | None:
    for kind, bounds in INTEGER_RANGES.items():  # SmallInteger and BigInteger before their base Integer
        if isinstance(sql_type, kind):
            return bounds
    return None


def check_fits(frame: pd.DataFrame, table: sa.Table) -> None:
    """Raise ``ValueError`` naming the column when a value is wider than its column (string length, integer range).

    SQL Server would refuse the whole statement anyway, from inside the temp table and without saying which row.
    """
    for col in frame.columns:
        if col not in table.c:
            continue
        sql_type = table.c[col].type
        values = frame[col].dropna()
        if values.empty:
            continue
        if isinstance(sql_type, sa.String) and sql_type.length is not None:
            too_long = values[values.astype(str).str.len() > sql_type.length]
            if not too_long.empty:
                raise ValueError(
                    f"{table.fullname}.{col} holds {sql_type.length} characters: {python_value(too_long.iloc[0])!r} "
                    f"and {len(too_long) - 1} more are longer"
                )
        elif (bounds := _integer_range(sql_type)) is not None and not isinstance(sql_type, sa.Boolean):
            out = values[(values < bounds[0]) | (values > bounds[1])]
            if not out.empty:
                raise ValueError(
                    f"{table.fullname}.{col} holds {bounds[0]}..{bounds[1]}: {python_value(out.iloc[0])!r} "
                    f"and {len(out) - 1} more are outside"
                )


def _chunks(items: list[tuple[Any, ...]], size: int) -> Iterable[list[tuple[Any, ...]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def upsert(
    conn: Connection,
    table: sa.Table,
    frame: pd.DataFrame,
    *,
    key: Sequence[str],
    batch_id: int | None = None,
    insert_only: bool = False,
) -> UpsertResult:
    """Write ``frame`` into ``table`` (see the module docstring); returns inserted / updated counts.

    ``frame`` holds the key columns and any data columns of ``table``; columns not in ``table`` are an error.
    ``batch_id`` (when the table has the column) is stamped on inserted and changed rows.
    """
    if frame.empty:
        return UpsertResult()
    unknown = [c for c in frame.columns if c not in table.c]
    if unknown:
        raise ValueError(f"{table.fullname} has no columns {unknown}")
    missing_key = [k for k in key if k not in frame.columns]
    if missing_key:
        raise ValueError(f"frame lacks key columns {missing_key}")
    frame = frame.drop_duplicates(subset=list(key), keep="last")
    has_batch = "batch_id" in table.c
    if has_batch and batch_id is not None:
        frame = frame.assign(batch_id=batch_id)
    data_cols = [c for c in frame.columns if c not in key and c not in PROVENANCE]
    cols = [*key, *data_cols, *(["batch_id"] if has_batch and "batch_id" in frame.columns else [])]
    frame = coerce_to_table(frame, table)
    check_fits(frame, table)

    target = _qualified(table)
    col_list = ", ".join(_quote(c) for c in cols)
    conn.exec_driver_sql("IF OBJECT_ID('tempdb..#stage') IS NOT NULL DROP TABLE #stage")
    # copy the types (and collations) of the target; WHERE 1 = 0 copies no rows, identity is not involved
    conn.exec_driver_sql(f"SELECT {_stage_columns(table, cols)} INTO #stage FROM {target} WHERE 1 = 0")
    insert = f"INSERT INTO #stage ({col_list}) VALUES ({', '.join('?' for _ in cols)})"
    _executemany(conn, insert, records(frame, cols))

    on = " AND ".join(f"t.{_quote(k)} = s.{_quote(k)}" for k in key)
    insert_cols = ", ".join(_quote(c) for c in cols)
    insert_vals = ", ".join(f"s.{_quote(c)}" for c in cols)
    stamp = ", t.[loaded_at] = SYSUTCDATETIME()" if "loaded_at" in table.c else ""
    stamp += ", t.[batch_id] = s.[batch_id]" if "batch_id" in cols else ""
    matched = ""
    if data_cols and not insert_only:
        compare_s = ", ".join(f"s.{_quote(c)}" for c in data_cols)
        compare_t = ", ".join(f"t.{_quote(c)}" for c in data_cols)
        sets = ", ".join(f"t.{_quote(c)} = s.{_quote(c)}" for c in data_cols)
        matched = (
            f"WHEN MATCHED AND EXISTS (SELECT {compare_s} EXCEPT SELECT {compare_t}) THEN UPDATE SET {sets}{stamp}"
        )
    # a key on an IDENTITY column addresses existing rows only: the database generates new ids, so no insert branch
    keyed_on_identity = any(table.c[k].identity is not None for k in key)
    not_matched = (
        "" if keyed_on_identity else f"WHEN NOT MATCHED BY TARGET THEN INSERT ({insert_cols}) VALUES ({insert_vals}) "
    )
    if keyed_on_identity and not matched:
        return UpsertResult()
    # count the actions server-side: OUTPUT to the client would stream one row per changed record
    merge = (
        "SET NOCOUNT ON; DECLARE @actions TABLE (act NVARCHAR(10)); "
        f"MERGE {target} WITH (HOLDLOCK) AS t USING #stage AS s ON {on} "
        f"{matched} {not_matched}"
        "OUTPUT $action INTO @actions; "
        "SELECT COALESCE(SUM(CASE WHEN act = 'INSERT' THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN act = 'UPDATE' THEN 1 ELSE 0 END), 0) FROM @actions;"
    )
    counts = conn.exec_driver_sql(merge).fetchone()
    conn.exec_driver_sql("DROP TABLE #stage")
    inserted, updated = (int(counts[0]), int(counts[1])) if counts is not None else (0, 0)
    return UpsertResult(inserted=inserted, updated=updated)


def _executemany(conn: Connection, sql: str, rows: list[tuple[Any, ...]]) -> None:
    """Bulk parameterised execute on the connection's own DB-API cursor (pyodbc ``fast_executemany``)."""
    raw = conn.connection.dbapi_connection
    if raw is None:
        raise RuntimeError("connection is closed")
    cursor = raw.cursor()
    try:
        pyodbc_cursor: Any = cursor  # fast_executemany is pyodbc-specific (one round trip per chunk)
        pyodbc_cursor.fast_executemany = True
        for chunk in _chunks(rows, CHUNK_ROWS):
            cursor.executemany(sql, chunk)
    finally:
        cursor.close()


def insert_returning(
    conn: Connection, table: sa.Table, frame: pd.DataFrame, *, anchor: str, id_col: str
) -> dict[str, int]:
    """Insert every row of ``frame`` into ``table`` and return ``{anchor value: generated id}``.

    ``frame[anchor]`` is a caller's key that is not a column of ``table`` (e.g. the source system's id); it travels
    through the insert so each generated identity is matched back to its row in one round trip
    (``MERGE ... OUTPUT source.anchor, inserted.id`` - a plain ``INSERT ... OUTPUT`` cannot see source columns).
    """
    if frame.empty:
        return {}
    cols = [c for c in frame.columns if c != anchor]
    unknown = [c for c in cols if c not in table.c]
    if unknown:
        raise ValueError(f"{table.fullname} has no columns {unknown}")
    frame = coerce_to_table(frame, table).assign(**{anchor: frame[anchor].astype(str)})
    check_fits(frame, table)
    col_list = ", ".join(_quote(c) for c in cols)
    conn.exec_driver_sql("IF OBJECT_ID('tempdb..#stage_ids') IS NOT NULL DROP TABLE #stage_ids")
    conn.exec_driver_sql(
        f"SELECT {_stage_columns(table, cols)}, CAST(NULL AS NVARCHAR(256)) AS [__anchor] INTO #stage_ids "
        f"FROM {_qualified(table)} WHERE 1 = 0"
    )
    placeholders = ", ".join("?" for _ in [*cols, anchor])
    _executemany(
        conn,
        f"INSERT INTO #stage_ids ({col_list}, [__anchor]) VALUES ({placeholders})",
        records(frame, [*cols, anchor]),
    )
    values = ", ".join(f"s.{_quote(c)}" for c in cols)
    merge = (
        "SET NOCOUNT ON; DECLARE @ids TABLE (anchor NVARCHAR(256), id INT); "
        f"MERGE {_qualified(table)} AS t USING #stage_ids AS s ON 1 = 0 "
        f"WHEN NOT MATCHED THEN INSERT ({col_list}) VALUES ({values}) "
        f"OUTPUT s.[__anchor], inserted.{_quote(id_col)} INTO @ids; SELECT anchor, id FROM @ids;"
    )
    mapping = {str(a): int(i) for a, i in conn.exec_driver_sql(merge).fetchall()}
    conn.exec_driver_sql("DROP TABLE #stage_ids")
    return mapping


def today_utc() -> dt.datetime:
    """Now in UTC, without tzinfo (DATETIME2 columns are UTC by convention)."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)
