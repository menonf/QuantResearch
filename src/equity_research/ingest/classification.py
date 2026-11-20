"""Sector classification over time: what each vendor said a company was, and from when.

    record(conn, frame, vendor="refinitiv", on=date.today(), batch_id=b)

``frame``: ``[issuer_id, sector, industry_group, industry, sub_industry]`` as the vendor classifies each company
today. A company seen for the first time opens a row valid from ``on``; one whose classification changed has its open
row closed on ``on`` and a new one opened; an unchanged one is left alone. History therefore starts on the day a
classification was first recorded - research asking about earlier days is told so (``known_from``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection

from equity_research.data import schema
from research_core.data.db.schema import vendor_id
from research_core.data.db.upsert import upsert

LEVELS = ("sector", "industry_group", "industry", "sub_industry")


@dataclass(frozen=True)
class Recorded:
    """What a recording changed."""

    opened: int
    closed: int


def record(
    conn: Connection,
    frame: pd.DataFrame,
    *,
    vendor: str,
    on: dt.date,
    scheme: str = "GICS",
    batch_id: int | None = None,
) -> Recorded:
    """Record today's classification (see the module docstring)."""
    t = schema.classification
    vid = vendor_id(vendor)
    rows = frame.reindex(columns=["issuer_id", *LEVELS]).dropna(subset=["issuer_id"])
    rows = rows.drop_duplicates("issuer_id").assign(issuer_id=lambda d: d["issuer_id"].astype(int))
    current = pd.DataFrame(
        conn.execute(
            sa.select(t.c.issuer_id, *[t.c[c] for c in LEVELS]).where(
                t.c.vendor_id == vid, t.c.scheme == scheme, t.c.valid_to.is_(None)
            )
        ).all(),
        columns=["issuer_id", *LEVELS],
    )
    both = rows.merge(current, on="issuer_id", how="left", suffixes=("", "_was"), indicator=True)
    same = pd.Series(True, index=both.index)
    for level in LEVELS:
        a, b = both[level], both[f"{level}_was"]
        same &= (a == b) | (a.isna() & b.isna())
    new = both["_merge"] == "left_only"
    changed = (both["_merge"] == "both") & ~same
    if changed.any():
        conn.execute(
            sa.update(t)
            .where(
                t.c.vendor_id == vid,
                t.c.scheme == scheme,
                t.c.valid_to.is_(None),
                t.c.issuer_id.in_([int(i) for i in both.loc[changed, "issuer_id"]]),
            )
            .values(valid_to=on)
        )
    opening = both.loc[new | changed, ["issuer_id", *LEVELS]]
    if not opening.empty:
        upsert(
            conn,
            t,
            opening.assign(vendor_id=vid, scheme=scheme, valid_from=on, valid_to=None),
            key=["issuer_id", "vendor_id", "scheme", "valid_from"],
            batch_id=batch_id,
        )
    return Recorded(opened=len(opening), closed=int(changed.sum()))
