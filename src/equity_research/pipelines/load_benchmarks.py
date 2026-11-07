"""Load the benchmark series from Yahoo Finance: S&P index levels and the ETFs that track them.

    report = load_benchmarks(engine)

1. each symbol is matched to its listing: a Yahoo symbol already recorded, else the same instrument known under
   another code (``^GSPC`` is the listing whose RIC is ``.SPX``), else a new listing;
   An index known under another code that is not loaded yet waits (no second listing for one index);
2. its chart from 30 days before the latest stored bar (the whole history the first time) is landed, then
   normalised into ``market.price_bar`` and ``market.corporate_action`` with vendor ``yahoo``.

Run it after the LSEG load on a fresh database, so the index levels both vendors cover share one listing.
Re-running is safe: the overlap is rewritten only where Yahoo changed a value.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import UpsertResult, upsert
from equity_research.ingest.identity import OPEN_START, add_identifiers, anchor_map, ensure_listings
from equity_research.ingest.reference import write_reference
from equity_research.ingest.vendors.yahoo import VENDOR
from equity_research.ingest.vendors.yahoo.client import YahooClient
from equity_research.ingest.vendors.yahoo.normalize import parse_chart
from equity_research.raw.store import land_bytes, land_frame, mark

log = logging.getLogger(__name__)

OVERLAP_DAYS = 30


@dataclass(frozen=True)
class Benchmark:
    """A Yahoo symbol and what it is; ``same_as`` names the instrument under another code when we know it."""

    symbol: str
    name: str
    instrument_type: str
    same_as: tuple[tuple[str, str], ...] = ()  # codes another vendor's listing of the same instrument carries
    exchange_mic: str | None = None


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark("^GSPC", "S&P 500 Index", "index", same_as=(("RIC", ".SPX"), ("TICKER", "$SPX"))),
    Benchmark("^SP400", "S&P MidCap 400 Index", "index", same_as=(("RIC", ".MID"), ("TICKER", "$MID"))),
    Benchmark("^SP600", "S&P SmallCap 600 Index", "index", same_as=(("RIC", ".SP600"), ("TICKER", "$SML"))),
    Benchmark("^SPXEW", "S&P 500 Equal Weight Index", "index"),
    Benchmark("SPY", "SPDR S&P 500 ETF Trust", "etf", exchange_mic="ARCX"),
    Benchmark("RSP", "Invesco S&P 500 Equal Weight ETF", "etf", exchange_mic="ARCX"),
    Benchmark("SPTM", "SPDR Portfolio S&P 1500 Composite Stock Market ETF", "etf", exchange_mic="ARCX"),
    # tracks the S&P 500 GARP Index from 2019-06-24 only - before, as PXLG, a Russell pure-growth index
    Benchmark("SPGP", "Invesco S&P 500 GARP ETF", "etf", exchange_mic="ARCX"),
)


def resolve_listings(
    conn: Connection, benchmarks: tuple[Benchmark, ...], batch_id: int | None = None
) -> tuple[dict[str, int], list[str]]:
    """``({symbol: listing_id}, waiting)``, linking or creating listings as the module docstring describes.

    ``waiting``: symbols whose instrument is known under another code that is not loaded yet (``^GSPC`` before the
    LSEG load has created ``.SPX``). They are skipped rather than created, so one index never gets two listings.
    ``batch_id``: the landed benchmark list the links and listings are recorded from.
    """
    known = anchor_map(conn, "listing", "YAHOO")
    links, waiting = [], []
    for b in benchmarks:
        if b.symbol in known or not b.same_as:
            continue
        listing_id = next(
            (lid for t, v in b.same_as if (lid := anchor_map(conn, "listing", t).get(v)) is not None), None
        )
        if listing_id is None:
            waiting.append(b.symbol)
            continue
        links.append((b.symbol, listing_id))
        known[b.symbol] = listing_id
    if links:
        add_identifiers(
            conn,
            pd.DataFrame(
                {
                    "entity_type": "listing",
                    "entity_id": [lid for _, lid in links],
                    "id_type": "YAHOO",
                    "value": [s for s, _ in links],
                    "valid_from": OPEN_START,
                    "valid_to": None,
                    "vendor_id": vendor_id(VENDOR),
                }
            ),
            batch_id=batch_id,
        )
    new = [b for b in benchmarks if b.symbol not in known and b.symbol not in waiting]
    if new:
        frame = pd.DataFrame(
            {
                "anchor": [b.symbol for b in new],
                "issuer_anchor": None,  # an index level or a fund: no company behind it
                "issuer_name": None,
                "name": [b.name for b in new],
                "instrument_type": [b.instrument_type for b in new],
                "exchange_mic": [b.exchange_mic for b in new],
                "currency": "USD",
            }
        )
        ids = ensure_listings(
            conn, frame, anchor_type="YAHOO", issuer_anchor_type="CUSIP", vendor=VENDOR, batch_id=batch_id
        )
        known.update(ids.listing_ids)
    return {b.symbol: known[b.symbol] for b in benchmarks if b.symbol in known}, waiting


def _start(conn: Connection, listing_id: int) -> dt.date | None:
    """The first day to ask for: ``OVERLAP_DAYS`` before the latest stored bar, or ``None`` for everything."""
    p = schema.price_bar
    last = conn.execute(
        sa.select(sa.func.max(p.c.trade_date)).where(p.c.listing_id == listing_id, p.c.vendor_id == vendor_id(VENDOR))
    ).scalar()
    return None if last is None else last - dt.timedelta(days=OVERLAP_DAYS)


def load_benchmarks(
    engine: Engine, benchmarks: tuple[Benchmark, ...] = BENCHMARKS, client: YahooClient | None = None
) -> dict[str, Any]:
    """Run both steps for every benchmark; returns ``{symbol: {listing_id, bars, actions}}``."""
    client = client or YahooClient()
    report: dict[str, Any] = {}
    with ops.job("load_benchmarks", engine) as run:
        with engine.begin() as conn:
            # the list the links come from, landed like a vendor answer: every link and listing names its batch
            spec = pd.DataFrame([b.__dict__ for b in benchmarks]).astype({"same_as": str})
            listed = land_frame(
                conn, vendor=VENDOR, dataset="benchmarks", source="load_benchmarks.BENCHMARKS", frame=spec
            )
            listings, waiting = resolve_listings(conn, benchmarks, listed.batch_id)
            mark(conn, listed.batch_id, "normalized")
            # the universes name their benchmarks by code (SPTM for the S&P 1500): the first source wrote them before
            # these listings existed, so they are written again now that they do
            write_reference(conn)
        if waiting:
            report["waiting_for_other_vendor"] = waiting
            log.warning("not loaded yet (their index listing comes from LSEG): %s", waiting)
        total = UpsertResult()
        for b in benchmarks:
            if b.symbol not in listings:
                continue
            listing_id = listings[b.symbol]
            with engine.begin() as conn:
                start = _start(conn, listing_id)
            url, body = client.chart(b.symbol, start)
            bars, actions = parse_chart(body)
            with engine.begin() as conn:
                batch = land_bytes(
                    conn,
                    vendor=VENDOR,
                    dataset="chart",
                    source=url,
                    payload=body,
                    name=f"{b.symbol.lstrip('^')}.json",
                    rows=len(bars),
                )
                written = upsert(
                    conn,
                    schema.price_bar,
                    bars.assign(listing_id=listing_id, vendor_id=vendor_id(VENDOR)),
                    key=["listing_id", "vendor_id", "trade_date"],
                    batch_id=batch.batch_id,
                )
                acted = upsert(
                    conn,
                    schema.corporate_action,
                    actions.assign(listing_id=listing_id, vendor_id=vendor_id(VENDOR)),
                    key=["listing_id", "vendor_id", "action_type", "ex_date"],
                    batch_id=batch.batch_id,
                )
                mark(conn, batch.batch_id, "normalized")
            total += written
            report[b.symbol] = {
                "listing_id": listing_id,
                "from": start,
                "bars": written.__dict__,
                "actions": acted.__dict__,
            }
            log.info("%s (listing %d): bars %s, actions %s", b.symbol, listing_id, written, acted)
        run.finish(rows=total.inserted + total.updated, detail=report)
    return report
