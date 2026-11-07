"""Load from Norgate Data: index members with their history, listings, unadjusted prices, splits and dividends, GICS.

    report = load_norgate(engine)                          # everything since START
    report = load_norgate(engine, since=date(2026, 10, 1))  # a refresh

1. ``universe`` - every symbol that was ever in the S&P 500, 400 or 600 (Norgate's "current & past" watchlists),
   each security's description, and its daily membership of each index turned into intervals. A security is matched
   to one of our listings by its Norgate asset id, else by ticker on the day (its last trading day if delisted -
   ``LEH-200809`` is LEH in September 2008), the delisting month breaking a tie; only then is a listing created.
   The index levels ($SPX...) attach to the listings of our index definitions.
2. ``prices`` - per security, unadjusted daily bars and the splits, spin-offs and dividends read from them.

Norgate runs on this computer, so there is no daily allowance: a re-run simply asks again. Each answer is still
landed in the raw zone before it is used. Today's GICS and domicile are recorded too (Norgate keeps no history
of them, so ours starts today).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import UpsertResult, upsert
from equity_research.ingest.classification import record as record_classification
from equity_research.ingest.countries import country_code
from equity_research.ingest.identity import (
    OPEN_START,
    US_MARKETS,
    add_identifiers,
    anchor_map,
    ensure_listings,
    match_listings,
)
from equity_research.ingest.membership import close_at_delisting
from equity_research.ingest.reference import INDEXES as REFERENCE_INDEXES
from equity_research.ingest.reference import write_reference
from equity_research.ingest.vendors.norgate import VENDOR
from equity_research.ingest.vendors.norgate import normalize as norm
from equity_research.ingest.vendors.norgate.client import INDEXES, NorgateClient
from equity_research.raw.store import land_frame, mark

log = logging.getLogger(__name__)

START = "2005-01-01"
TIE_DAYS = 45  # a delisted symbol's month and our listing's last day may differ by this much and still match


def _land(engine: Engine, dataset: str, source: str, frame: pd.DataFrame) -> int:
    with engine.begin() as conn:
        landed = frame.reset_index() if not isinstance(frame.index, pd.RangeIndex) else frame
        return land_frame(conn, vendor=VENDOR, dataset=dataset, source=source, frame=landed).batch_id


class NorgateLoad:
    """The steps of the module docstring over one engine and client."""

    def __init__(self, engine: Engine, client: NorgateClient, *, start: str, end: str) -> None:
        self.engine = engine
        self.client = client
        self.start = start
        self.end = end
        self.assetids: dict[str, int] = {}  # symbol -> Norgate asset id, from the universe step

    # ------------------------------------------------------------ 1. universe

    def universe(self) -> dict[str, Any]:
        """Securities, their listings, memberships, today's GICS and share counts."""
        report: dict[str, Any] = {}
        members: dict[str, set[str]] = {}
        for code, (watchlist, _, _) in INDEXES.items():
            members[code] = set(self.client.symbols(watchlist))
        symbols = sorted(set().union(*members.values()))
        described = pd.DataFrame([self.client.describe(s) for s in symbols])
        self.assetids = dict(zip(described["symbol"], described["assetid"].astype(int), strict=True))
        batch_id = _land(self.engine, "securities", f"describe {len(symbols)} symbols", described)
        with self.engine.begin() as conn:
            line_of = self._identify(conn, described, batch_id, report)
            self._ensure_levels(conn, report)
            index_ids = write_reference(conn)
        spells = []
        for code, (_, index_name, _) in INDEXES.items():
            for symbol in sorted(members[code]):
                series = self.client.membership(symbol, index_name, self.start, self.end)
                for start, end in norm.memberships(series).itertuples(index=False):
                    spells.append((index_ids[code], line_of[symbol], start, end))
        rows = pd.DataFrame(spells, columns=["index_id", "listing_id", "start_date", "end_date"])
        gone = described[described["symbol"].map(lambda sym: norm.parse_symbol(sym)[1] is not None)]
        last_day = pd.Series(pd.to_datetime(gone["last_quoted"]).to_numpy(), index=gone["symbol"].map(line_of))
        rows, report["memberships_closed_at_delisting"] = close_at_delisting(rows, last_day)
        members_batch = _land(
            self.engine, "memberships", f"index_constituent_timeseries {self.start}..{self.end}", rows
        )
        with self.engine.begin() as conn:
            t = schema.index_membership
            conn.execute(sa.delete(t).where(t.c.vendor_id == vendor_id(VENDOR)))
            report["memberships"] = upsert(
                conn,
                t,
                rows.assign(vendor_id=vendor_id(VENDOR)).drop_duplicates(["index_id", "listing_id", "start_date"]),
                key=["index_id", "listing_id", "start_date"],
                batch_id=members_batch,
            ).__dict__
            self._describe_companies(
                conn, described.assign(listing_id=described["symbol"].map(line_of)), batch_id, report
            )
            for read in (batch_id, members_batch):
                mark(conn, read, "normalized")
        return report

    def _identify(
        self, conn: Connection, described: pd.DataFrame, batch_id: int, report: dict[str, Any]
    ) -> dict[str, int]:
        """``{symbol: listing_id}``: matched to our listings, or new ones (see the module docstring)."""
        known = anchor_map(conn, "listing", "NORGATE")
        parsed = described["symbol"].map(norm.parse_symbol)
        frame = described.assign(
            ticker=[t for t, _ in parsed],
            delisted=[m for _, m in parsed],
            key=described["assetid"].astype(int).astype(str),
        )
        line_of: dict[str, int] = {}
        to_match = frame[~frame["key"].isin(known)]
        for symbol, key in zip(frame["symbol"], frame["key"], strict=True):
            if key in known:
                line_of[symbol] = known[key]
        matched_by: dict[str, str] = {}
        for _, row in to_match.iterrows():
            day = pd.Timestamp(row["last_quoted"]) - pd.Timedelta(days=1) if row["delisted"] is not None else None
            hit = match_listings(
                conn,
                pd.DataFrame({"key": [row["key"]], "TICKER": [row["ticker"]]}),
                markets=US_MARKETS,
                on=day.date() if day is not None else None,
            )
            listing_id = hit["listing_id"].iloc[0]
            if pd.isna(listing_id) and bool(hit["ambiguous"].iloc[0]) and day is not None:
                listing_id = self._by_delisting(conn, str(row["ticker"]), pd.Timestamp(row["last_quoted"]))
            if pd.notna(listing_id):
                line_of[str(row["symbol"])] = int(listing_id)
                matched_by[str(row["symbol"])] = "ticker"
        new = frame[~frame["symbol"].isin(line_of)]
        created = ensure_listings(
            conn,
            pd.DataFrame(
                {
                    "anchor": new["key"],
                    "issuer_anchor": new["key"],  # no CUSIP: a company per security until the SEC's CIK joins them
                    "issuer_name": new["name"],
                    "name": new["name"].fillna(new["symbol"]).astype(str).str[:256],
                    "instrument_type": "equity",
                    "currency": new["currency"],
                    "delisted_on": pd.to_datetime(new["last_quoted"]).where(new["delisted"].notna()),
                }
            ),
            anchor_type="NORGATE",
            issuer_anchor_type="NORGATE",
            vendor=VENDOR,
            batch_id=batch_id,
        )
        for symbol, key in zip(new["symbol"], new["key"], strict=True):
            line_of[symbol] = created.listing_ids[key]
        self._date_tickers(conn, frame, line_of, batch_id)
        codes = frame[frame["symbol"].isin(matched_by)]
        add_identifiers(
            conn,
            pd.DataFrame(
                {
                    "entity_type": "listing",
                    "entity_id": codes["symbol"].map(line_of),
                    "id_type": "NORGATE",
                    "value": codes["key"],
                    "valid_from": OPEN_START,
                    "valid_to": None,
                    "vendor_id": vendor_id(VENDOR),
                }
            ),
            batch_id=batch_id,
        )
        report |= {
            "securities": len(frame),
            "known": len(frame) - len(to_match),
            "matched": len(matched_by),
            "created": created.listings_created,
        }
        return line_of

    def _date_tickers(self, conn: Connection, frame: pd.DataFrame, line_of: dict[str, int], batch_id: int) -> None:
        """Each security's ticker, valid from its first trading day to the day after its last (or still valid)."""
        first = pd.to_datetime(frame["first_quoted"], errors="coerce").dt.tz_localize(None).dt.normalize()
        last = pd.to_datetime(frame["last_quoted"], errors="coerce").dt.tz_localize(None).dt.normalize()
        add_identifiers(
            conn,
            pd.DataFrame(
                {
                    "entity_type": "listing",
                    "entity_id": frame["symbol"].map(line_of),
                    "id_type": "TICKER",
                    "value": frame["ticker"],
                    "valid_from": first.fillna(OPEN_START),
                    "valid_to": (last + pd.Timedelta(days=1)).where(frame["delisted"].notna()),
                    "vendor_id": vendor_id(VENDOR),
                }
            ).dropna(subset=["entity_id"]),
            batch_id=batch_id,
        )

    def _by_delisting(self, conn: Connection, ticker: str, last_quoted: pd.Timestamp) -> int | None:
        """Of the listings that held ``ticker``, the one that ended within :data:`TIE_DAYS` of ``last_quoted``."""
        rows = (
            conn.execute(
                sa.text(
                    "SELECT DISTINCT l.listing_id FROM ref.identifier i "
                    "JOIN ref.listing l ON l.listing_id = i.entity_id "
                    "WHERE i.entity_type = 'listing' AND i.id_type = 'TICKER' AND i.value = :t "
                    "AND l.delisted_on BETWEEN :lo AND :hi "
                    "AND (l.exchange_mic IN :markets OR l.exchange_mic IS NULL)"
                ).bindparams(sa.bindparam("markets", expanding=True)),
                {
                    "markets": sorted(US_MARKETS),
                    "t": ticker,
                    "lo": (last_quoted - pd.Timedelta(days=TIE_DAYS)).date(),
                    "hi": (last_quoted + pd.Timedelta(days=TIE_DAYS)).date(),
                },
            )
            .scalars()
            .all()
        )
        return int(rows[0]) if len(rows) == 1 else None

    def _ensure_levels(self, conn: Connection, report: dict[str, Any]) -> None:
        """Each index level ($SPX...) on a listing: the one another vendor's code names, else a new one."""
        attached = created = 0
        about_all = {symbol: self.client.describe(symbol) for _, _, symbol in INDEXES.values()}
        frame = pd.DataFrame(list(about_all.values()))
        landed = frame.reset_index() if not isinstance(frame.index, pd.RangeIndex) else frame
        batch_id = land_frame(
            conn, vendor=VENDOR, dataset="index_levels", source="describe levels", frame=landed
        ).batch_id
        for code, (_, _, symbol) in INDEXES.items():
            spec = next(i for i in REFERENCE_INDEXES if i.code == code)
            listing_id = next(
                (lid for t, v in spec.levels if (lid := anchor_map(conn, "listing", t).get(v)) is not None), None
            )
            about = about_all[symbol]
            key = str(int(about["assetid"]))
            if listing_id is None:
                made = ensure_listings(
                    conn,
                    pd.DataFrame(
                        {
                            "anchor": [key],
                            "issuer_anchor": [None],
                            "issuer_name": [None],
                            "name": [spec.name],
                            "instrument_type": ["index"],
                            "currency": ["USD"],
                        }
                    ),
                    anchor_type="NORGATE",
                    issuer_anchor_type="NORGATE",
                    vendor=VENDOR,
                    batch_id=batch_id,
                )
                listing_id, created = made.listing_ids[key], created + 1
            else:
                attached += 1
            add_identifiers(
                conn,
                pd.DataFrame(
                    {
                        "entity_type": "listing",
                        "entity_id": [listing_id, listing_id],
                        "id_type": ["TICKER", "NORGATE"],
                        "value": [symbol, key],
                        "valid_from": OPEN_START,
                        "valid_to": None,
                        "vendor_id": vendor_id(VENDOR),
                    }
                ),
                batch_id=batch_id,
            )
            self.assetids[symbol] = int(key)
        mark(conn, batch_id, "normalized")
        report["index_levels"] = {"attached": attached, "created": created}

    def _describe_companies(
        self, conn: Connection, described: pd.DataFrame, batch_id: int, report: dict[str, Any]
    ) -> None:
        issuer_of = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
        companies = described.assign(issuer_id=described["listing_id"].map(issuer_of)).dropna(subset=["issuer_id"])
        companies = companies.drop_duplicates("issuer_id")
        recorded = record_classification(
            conn,
            companies.dropna(subset=["sector"]),
            vendor=VENDOR,
            on=dt.date.fromisoformat(self.end),
            batch_id=batch_id,
        )
        report["classifications"] = {"opened": recorded.opened, "closed": recorded.closed}
        homes = companies.assign(domicile_country=companies["domicile"].map(country_code)).dropna(
            subset=["domicile_country"]
        )
        empty = {
            int(i)
            for i in conn.execute(
                sa.select(schema.issuer.c.issuer_id).where(schema.issuer.c.domicile_country.is_(None))
            ).scalars()
        }
        homes = homes[homes["issuer_id"].astype(int).isin(empty)]  # a source that has it is not overwritten
        report["domiciles"] = (
            upsert(
                conn,
                schema.issuer,
                homes[["issuer_id", "domicile_country"]].astype({"issuer_id": int}),
                key=["issuer_id"],
            ).updated
            if not homes.empty
            else 0
        )

    # ------------------------------------------------------------ 2. prices, splits, dividends

    def prices(self, since: dt.date | None = None) -> dict[str, Any]:
        """Per security: unadjusted bars and the actions read from them."""
        with self.engine.connect() as conn:
            line_of = anchor_map(conn, "listing", "NORGATE")
            levels = dict(
                conn.execute(
                    sa.select(schema.index_definition.c.code, schema.index_definition.c.level_listing_id)
                ).all()
            )
        targets = {s: line_of.get(str(a)) for s, a in (self.assetids or self._symbol_ids()).items()}
        level_symbols = {symbol for _, _, symbol in INDEXES.values()}
        targets |= {symbol: levels.get(code) for code, (_, _, symbol) in INDEXES.items()}  # the index levels
        first = since.isoformat() if since else self.start
        bars_total, actions_total = UpsertResult(), UpsertResult()
        for symbol, listing_id in sorted(targets.items()):
            if listing_id is None:
                continue
            assetid = self.assetids.get(symbol, symbol)
            raw = self.client.prices(symbol, first, self.end)
            batch_id = _land(self.engine, "prices", f"price_timeseries {assetid} {first}..{self.end}", raw)
            if symbol in level_symbols:
                raw = norm.as_level(raw)
            bars = norm.bars(raw).assign(listing_id=listing_id, vendor_id=vendor_id(VENDOR), currency="USD")
            acts = norm.actions(raw).assign(listing_id=listing_id, vendor_id=vendor_id(VENDOR))
            with self.engine.begin() as conn:
                bars_total += upsert(
                    conn, schema.price_bar, bars, key=["listing_id", "vendor_id", "trade_date"], batch_id=batch_id
                )
                actions_total += upsert(
                    conn,
                    schema.corporate_action,
                    acts,
                    key=["listing_id", "vendor_id", "action_type", "ex_date"],
                    batch_id=batch_id,
                )
                mark(conn, batch_id, "normalized")
        return {
            "securities": len(targets),
            "price_bars": bars_total.__dict__,
            "corporate_actions": actions_total.__dict__,
        }

    def _symbol_ids(self) -> dict[str, int]:
        """Asset ids when the universe step did not run in this load (a prices-only refresh)."""
        symbols = sorted(set().union(*(set(self.client.symbols(w)) for w, _, _ in INDEXES.values())))
        return {s: int(self.client.describe(s)["assetid"]) for s in symbols}


def load_norgate(
    engine: Engine,
    *,
    start: str = START,
    end: str | None = None,
    since: dt.date | None = None,
    client: NorgateClient | None = None,
) -> dict[str, Any]:
    """Run both steps (see the module docstring); ``since`` limits prices to that day on (a refresh)."""
    c = client or NorgateClient()
    c.check()
    load = NorgateLoad(engine, c, start=start, end=end or dt.date.today().isoformat())
    report: dict[str, Any] = {}
    with ops.job("load_norgate", engine) as run:
        report["universe"] = load.universe()
        report["prices"] = load.prices(since)
        run.finish(rows=report["prices"]["price_bars"]["inserted"], detail=report)
    return report


def probe(client: NorgateClient | None = None) -> pd.DataFrame:
    """Questions with known answers, read by the load's own code: ``[check, ok, detail]``. Nothing is stored.

    Run it once Norgate Data Updater is installed: it shows whether the answers mean what the adapter assumes -
    column names, O'Reilly's 15-for-1 split of 2025-06-10 read as a split, a delisted symbol described, the index
    levels. It also reports how far back the subscription's history reaches (a trial: about two years).
    """
    rows: list[dict[str, Any]] = []

    def check(name: str, run: Any) -> None:
        try:
            ok, detail = run()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:300]
        rows.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})

    try:
        c = client or NorgateClient()
        c.check()
    except Exception as exc:
        return pd.DataFrame([{"check": "Norgate Data Updater running", "ok": False, "detail": str(exc)[:300]}])
    rows.append({"check": "Norgate Data Updater running", "ok": True, "detail": ""})
    for code, (watchlist, _, _) in INDEXES.items():
        check(
            f"{code} current & past members",
            lambda w=watchlist: (len(c.symbols(w)) > 400, f"{len(c.symbols(w))} symbols"),
        )
    check("history reaches back to", lambda: (True, c.describe("AAPL")["first_quoted"]))

    def split() -> tuple[bool, str]:
        acts = norm.actions(c.prices("ORLY", "2025-06-02", "2025-06-13"))
        found = acts[acts["action_type"] == "split"]
        ok = len(found) == 1 and abs(float(found["ratio"].iloc[0]) - 15.0) < 0.01
        return ok, str(found.to_dict(orient="records"))

    check("ORLY 2025 split read as 15-for-1", split)
    check(
        "ORLY unadjusted close before the split",
        lambda: (
            (b := norm.bars(c.prices("ORLY", "2025-06-06", "2025-06-09")))["close"].iloc[-1] > 1000,
            b.to_dict(orient="records"),
        ),
    )
    check(
        "AAPL in the S&P 500",
        lambda: (
            not (m := norm.memberships(c.membership("AAPL", "S&P 500", "2025-01-01", "2025-12-31"))).empty,
            m.to_dict(orient="records"),
        ),
    )

    def delisted() -> tuple[bool, str]:
        gone = [s for s in c.symbols("S&P 500 Current & Past") if norm.parse_symbol(s)[1] is not None]
        about = c.describe(gone[0]) if gone else {}
        return bool(gone) and about.get("last_quoted") is not None, f"{len(gone)} delisted, e.g. {about}"

    check("delisted members described", delisted)
    for code, (_, _, symbol) in INDEXES.items():
        october = (symbol, "2025-10-01", "2025-10-31")
        check(f"{code} level {symbol}", lambda q=october: (len(norm.bars(norm.as_level(c.prices(*q)))) > 15, q[0]))
    check("domicile as a country code", lambda: ((d := country_code(c.describe("ACN")["domicile"])) == "IE", d))
    return pd.DataFrame(rows)
