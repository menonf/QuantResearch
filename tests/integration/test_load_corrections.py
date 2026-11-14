"""The repository's corrections reach ``market.corporate_action`` under vendor ``repository``, once."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import scratch_database

from equity_research.db import schema
from equity_research.db.upsert import upsert
from equity_research.ingest.identity import ensure_listings
from equity_research.pipelines.load_corrections import load_corrections
from equity_research.raw.store import land_frame

pytestmark = pytest.mark.integration


def _listings(engine: sa.Engine) -> dict[str, int]:
    """Three lines of the corrections file - Alphabet's two and C.H. Robinson - with LSEG's closes of 3 April 2014."""
    frame = pd.DataFrame(
        {
            "anchor": ["GOOGL.OQ", "GOOG.OQ", "CHRW.OQ"],
            "issuer_anchor": ["38259P508", "38259P508", "12541W209"],
            "issuer_name": ["Alphabet Inc", "Alphabet Inc", "C.H. Robinson"],
            "name": ["Alphabet A", "Alphabet C", "C.H. Robinson"],
            "instrument_type": "equity",
            "currency": "USD",
        }
    )
    with engine.begin() as conn:
        batch = land_frame(conn, vendor="refinitiv", dataset="instruments", source="test", frame=frame)
        ids = ensure_listings(
            conn, frame, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=batch.batch_id
        ).listing_ids
        bars = pd.DataFrame(
            {
                "listing_id": [ids["GOOGL.OQ"], ids["GOOG.OQ"]],
                "vendor_id": schema.vendor_id("refinitiv"),
                "trade_date": dt.date(2014, 4, 3),
                "close": [571.50, 569.74],
            }
        )
        upsert(conn, schema.price_bar, bars, key=["listing_id", "vendor_id", "trade_date"], batch_id=batch.batch_id)
    return {str(k): int(v) for k, v in ids.items()}


def test_corrections_are_written_once_and_what_cannot_be_is_reported(tmp_path: Path) -> None:
    with scratch_database("equity_research_test_corrections", tmp_path) as engine:
        ids = _listings(engine)
        report = load_corrections(engine)
        assert "TDS.N" in report["unknown_lines"] and "GOOGL.OQ" not in report["unknown_lines"]
        assert report["unpriced"] == [] and report["written"] == 2

        with engine.connect() as conn:
            rows = conn.execute(
                sa.text(
                    "SELECT a.listing_id, a.action_type, a.ratio, b.status FROM market.corporate_action a "
                    "JOIN ingest.batch b ON b.batch_id = a.batch_id WHERE a.vendor_id = 9 ORDER BY a.listing_id"
                )
            ).all()
        got = {int(r[0]): (r[1], float(r[2]), r[3]) for r in rows}
        assert got[ids["CHRW.OQ"]] == ("split", 2.0, "normalized")
        kind, ratio, status = got[ids["GOOGL.OQ"]]
        assert (kind, status) == ("spinoff", "normalized")
        assert ratio == pytest.approx(1 + 569.74 / 571.50)  # the class A holder's value over the class A price

        assert load_corrections(engine)["written"] == 0  # unchanged file: nothing new
