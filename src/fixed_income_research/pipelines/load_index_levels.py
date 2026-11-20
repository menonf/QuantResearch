"""Load the official total-return level of each bond index from FRED, month end by month end.

    report = load_index_levels(engine)

For every index we hold that has a FRED series (``ingest/reference.py``): the series is fetched, landed in the raw
zone as received, and each finished month's last level written to ``fixed_income.index_level``. FRED keeps three
years of these series, so a month loaded once stays when FRED drops it - run this every month.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fixed_income_research.data import schema
from fixed_income_research.ingest import fred
from fixed_income_research.ingest.reference import INDICES
from research_core.data import ops
from research_core.data.db.schema import vendor_id
from research_core.data.db.upsert import upsert
from research_core.data.raw_store import land_bytes, mark

VENDOR = "fred"
DATASET = "index_level"


def load_index_levels(
    engine: Engine, *, fetch: Callable[[str], bytes] = fred.fetch, today: dt.date | None = None
) -> dict[str, Any]:
    """Run the steps of the module docstring; returns, per index, the months written."""
    today = today or dt.datetime.now(dt.UTC).date()
    report: dict[str, Any] = {}
    with ops.job("load_index_levels", engine) as run:
        with engine.connect() as conn:
            held = dict(
                conn.execute(sa.select(schema.index_definition.c.code, schema.index_definition.c.index_id)).all()
            )
        for code, index_id in held.items():
            series = INDICES[code].fred_series if code in INDICES else None
            if series is None:
                continue
            payload = fetch(series)
            with engine.begin() as conn:
                batch = land_bytes(
                    conn,
                    vendor=VENDOR,
                    dataset=DATASET,
                    source=f"fred:{series}:{today:%Y-%m}",
                    payload=payload,
                    name=f"{series}.csv",
                )
                levels = fred.month_end_levels(payload, today).assign(index_id=index_id, vendor_id=vendor_id(VENDOR))
                written = upsert(
                    conn,
                    schema.index_level,
                    levels,
                    key=["index_id", "vendor_id", "as_of_date"],
                    batch_id=batch.batch_id,
                )
                mark(conn, batch.batch_id, "normalized")
            report[code] = {"months": len(levels), "inserted": written.inserted, "updated": written.updated}
        run.finish(rows=sum(r["inserted"] + r["updated"] for r in report.values()), detail=report)
    return report
