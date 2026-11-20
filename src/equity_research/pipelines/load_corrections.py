"""Write the repository's corrections (``ingest/corrections``) to ``equity.corporate_action``, vendor ``repository``.

    report = load_corrections(engine)     # {"written": n, "unknown_lines": [...], "unpriced": [...]}

Steps:

1. read and check ``corporate_actions.csv`` (:func:`~equity_research.ingest.corrections.read_corrections`);
2. land it in the raw zone like a vendor's answer, so every row it writes names its batch;
3. find each line by its RIC - a line this database does not hold (London's database lacks New York's) is skipped and
   reported;
4. a split is written with its ratio; a distribution with the ratio of the holder's value on its ex-date to the
   parent's price (:func:`~equity_research.ingest.corrections.distribution_ratio`), from both lines' closes that
   day - the first vendor that priced both, by ``vendor_id`` - and skipped and reported when either is unpriced.

Re-running is safe: an unchanged file lands no new batch and writes nothing new. Research reads a correction only where
the prices' vendor has no split or spin-off of its own near it (:func:`~equity_research.data.universe.price_actions`),
so a vendor that later records the event is not counted twice.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research.data import schema
from equity_research.ingest.corrections import PATH, distribution_ratio, read_corrections
from research_core.data import ops
from research_core.data.db import schema as core_schema
from research_core.data.db.schema import vendor_id
from research_core.data.db.upsert import upsert
from research_core.data.raw_store import land_frame, mark

log = logging.getLogger(__name__)

VENDOR = "repository"


def load_corrections(engine: Engine, path: Any = PATH) -> dict[str, Any]:
    """Run the steps above; returns what was written and what was skipped."""
    table = read_corrections(path)
    report: dict[str, Any] = {"corrections": len(table)}
    with ops.job("load_corrections", engine) as run, engine.begin() as conn:
        landed = land_frame(
            conn,
            vendor=VENDOR,
            dataset="corporate_actions",
            source=f"ingest/corrections/{path.name}",
            frame=table.assign(ex_date=table["ex_date"].dt.date.astype(str)),
        )
        rics = sorted(set(table["ric"]) | set(table["distributed_ric"].dropna()))
        i = core_schema.identifier
        found = conn.execute(
            sa.select(i.c.value, i.c.entity_id).where(
                i.c.entity_type == "listing", i.c.id_type == "RIC", i.c.value.in_(rics)
            )
        ).all()
        ids = {str(v): int(e) for v, e in found}
        report["unknown_lines"] = sorted(r for r in table["ric"] if r not in ids)
        known = table[table["ric"].isin(ids)]
        rows, unpriced = [], []
        for c in known.to_dict("records"):
            ratio = float(c["ratio"]) if c["action_type"] == "split" else _distribution(conn, ids, c)
            if ratio is None:
                unpriced.append(f"{c['ric']} {c['ex_date'].date()}")
                continue
            rows.append(
                {
                    "listing_id": ids[c["ric"]],
                    "vendor_id": vendor_id(VENDOR),
                    "action_type": c["action_type"],
                    "ex_date": c["ex_date"].date(),
                    "ratio": ratio,
                    "amount": None,
                }
            )
        report["unpriced"] = unpriced
        written = upsert(
            conn,
            schema.corporate_action,
            pd.DataFrame(rows, columns=["listing_id", "vendor_id", "action_type", "ex_date", "ratio", "amount"]),
            key=["listing_id", "vendor_id", "action_type", "ex_date"],
            batch_id=landed.batch_id,
        )
        mark(conn, landed.batch_id, "normalized")
        report["written"] = written.inserted + written.updated
        run.finish(rows=report["written"], detail=report)
    if report["unknown_lines"] or unpriced:
        log.info("corrections skipped: lines not held %s, unpriced %s", report["unknown_lines"], unpriced)
    return report


def _distribution(conn: sa.Connection, ids: dict[str, int], c: dict[Any, Any]) -> float | None:
    """The distribution's ratio from both lines' closes on its ex-date, ``None`` when either is not priced."""
    if c["distributed_ric"] not in ids:
        return None
    parent, child = ids[c["ric"]], ids[c["distributed_ric"]]
    b = schema.price_bar
    closes = pd.DataFrame(
        conn.execute(
            sa.select(b.c.listing_id, b.c.vendor_id, b.c.close).where(
                b.c.listing_id.in_([parent, child]), b.c.trade_date == c["ex_date"].date()
            )
        ).all(),
        columns=["listing_id", "vendor_id", "close"],
    )
    both = closes.pivot_table(index="vendor_id", columns="listing_id", values="close").dropna()
    if both.empty or parent not in both.columns or child not in both.columns:
        return None
    first = both.sort_index().iloc[0]
    return distribution_ratio(float(first[parent]), float(first[child]), float(c["per_share"]))
