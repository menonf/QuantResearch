"""Price from Tiingo the listings other vendors found: unadjusted daily bars, dividends and splits.

    report = load_tiingo(engine)                    # every index member still to price, then the daily top-up
    report = load_tiingo(engine, max_requests=45)   # stay inside a free account's hourly allowance

Tiingo knows only tickers, so it never creates a listing. It prices every equity listing that is or was in an index
we hold (whichever vendor recorded the membership), each under the ticker we hold for it on its last day:

1. Tiingo's list of supported tickers (a free file, no allowance used) is landed; a listing whose ticker Tiingo does
   not have, or had only outside the listing's life, is skipped without a request;
2. per listing one request, from :data:`OVERLAP_DAYS` before its newest Tiingo bar (from :data:`START` the first
   time) to its last day - landed as it came, then written to ``market.price_bar`` and ``market.corporate_action``;
3. where another vendor prices the same listing, the closes of the common days must agree
   (:data:`~equity_research.ingest.vendors.tiingo.normalize.AGREEMENT`): otherwise the ticker is another company's in
   Tiingo and nothing is written (reported as ``disagreed``).

A listing whose newest Tiingo bar is from the last business day is complete and not asked again, so a build stopped
by Tiingo's allowance (HTTP 429) continues on the next run where it stopped; run daily, the same call is the refresh.
"""

from __future__ import annotations

import datetime as dt
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import UpsertResult, upsert
from equity_research.ingest.identity import US_MARKETS
from equity_research.ingest.vendors.tiingo import VENDOR
from equity_research.ingest.vendors.tiingo import normalize as norm
from equity_research.ingest.vendors.tiingo.client import TiingoClient, TiingoThrottledError
from equity_research.raw.store import land_bytes, land_file, mark, recent_batch
from equity_research.settings import get_settings

log = logging.getLogger(__name__)

START = pd.Timestamp("2005-01-01")
OVERLAP_DAYS = 10  # vendors correct recent days: a top-up asks again for these
SAMPLE = 20  # tickers listed per kind of problem in the report


def targets(engine: Engine, today: pd.Timestamp) -> pd.DataFrame:
    """``[listing_id, ticker, first, last, newest]`` - the listings to price, their window, their newest Tiingo bar."""
    sql = sa.text(
        """
        SELECT l.listing_id, l.listed_on, l.delisted_on, t.value AS ticker, t.valid_from, t.valid_to
        FROM ref.listing l
        JOIN ref.identifier t ON t.entity_type = 'listing' AND t.entity_id = l.listing_id AND t.id_type = 'TICKER'
        WHERE l.instrument_type = 'equity'
          AND (l.exchange_mic IN :markets OR l.exchange_mic IS NULL)
          AND EXISTS (SELECT 1 FROM ref.index_membership m WHERE m.listing_id = l.listing_id)
        """
    ).bindparams(sa.bindparam("markets", sorted(US_MARKETS), expanding=True))
    newest_sql = sa.text(
        "SELECT listing_id, MAX(trade_date) AS newest FROM market.price_bar WHERE vendor_id = :v GROUP BY listing_id"
    )
    with engine.connect() as conn:
        rows = pd.DataFrame(
            conn.execute(sql).all(),
            columns=["listing_id", "listed_on", "delisted_on", "ticker", "valid_from", "valid_to"],
        )
        newest = pd.DataFrame(
            conn.execute(newest_sql, {"v": vendor_id(VENDOR)}).all(), columns=["listing_id", "newest"]
        )
    for c in ("listed_on", "delisted_on", "valid_from", "valid_to"):
        rows[c] = pd.to_datetime(rows[c])
    # the ticker it had last: still valid first, else the one that ended latest
    rows = rows.assign(open=rows["valid_to"].isna()).sort_values(["listing_id", "open", "valid_to", "valid_from"])
    rows = rows.drop_duplicates("listing_id", keep="last")
    last = pd.concat(
        [
            pd.Series(today, index=rows.index),
            rows["delisted_on"],
            rows["valid_to"] - pd.Timedelta(days=1),
        ],
        axis=1,
    ).min(axis=1)
    out = pd.DataFrame(
        {
            "listing_id": rows["listing_id"].astype(int),
            "ticker": rows["ticker"].map(norm.tiingo_ticker),
            "first": rows["listed_on"].fillna(START).where(rows["listed_on"] > START, START),
            "last": last,
        }
    ).merge(newest.assign(newest=pd.to_datetime(newest["newest"])), on="listing_id", how="left")
    return out.sort_values("listing_id").reset_index(drop=True)


def _supported(engine: Engine, client: TiingoClient) -> pd.DataFrame:
    """Tiingo's supported tickers, landed at most once a day."""
    with engine.begin() as conn:
        batch = recent_batch(conn, vendor=VENDOR, dataset="supported_tickers", max_age_days=1)
        if batch is None:
            staging = get_settings().raw_dir / "_staging" / "tiingo-supported_tickers.zip"
            url = client.download_supported_tickers(staging)
            batch = land_file(conn, vendor=VENDOR, dataset="supported_tickers", source=url, path=staging, move=True)
        mark(conn, batch.batch_id, "normalized")  # read in full to plan the requests
    return norm.supported(batch.path.read_bytes())


def plan(listings: pd.DataFrame, supported: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """``(requests to make [listing_id, ticker, start, end], skipped {reason: tickers})``."""
    df = listings.merge(supported[["ticker", "first_day", "last_day"]], on="ticker", how="left")
    df["start"] = df["first"].where(df["newest"].isna(), (df["newest"] - pd.Timedelta(days=OVERLAP_DAYS)))
    df["start"] = df[["start", "first"]].max(axis=1)
    complete = df["newest"].notna() & (df["newest"] >= df["last"] - pd.offsets.BDay(1))
    unknown = df["last_day"].isna() & df["first_day"].isna()
    outside = ~unknown & ((df["last_day"] < df["start"]) | (df["first_day"] > df["last"]))
    ended = df["start"] > df["last"]
    skipped = {
        "complete": df.loc[complete, "ticker"].tolist(),
        "not_at_tiingo": df.loc[~complete & unknown, "ticker"].tolist(),
        "outside_tiingo_dates": df.loc[~complete & ~unknown & (outside | ended), "ticker"].tolist(),
    }
    todo = df[~complete & ~unknown & ~outside & ~ended]
    return todo[["listing_id", "ticker", "start", "last"]].rename(columns={"last": "end"}), skipped


def _other_bars(engine: Engine, listing_id: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    p = schema.price_bar
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(p.c.trade_date, p.c.close)
            .where(
                p.c.listing_id == listing_id,
                p.c.vendor_id != vendor_id(VENDOR),
                p.c.trade_date.between(start.date(), end.date()),
            )
            .order_by(p.c.vendor_id)
        ).all()
    other = pd.DataFrame(rows, columns=["trade_date", "close"])
    return other.assign(trade_date=pd.to_datetime(other["trade_date"])).drop_duplicates("trade_date")


def _write(
    engine: Engine, listing_id: int, ticker: str, window: tuple[pd.Timestamp, pd.Timestamp], answer: tuple[str, bytes]
) -> tuple[str, UpsertResult, UpsertResult]:
    """Land one answer and write it unless another vendor's closes disagree: ``(outcome, bars, actions)``."""
    url, body = answer
    table = norm.frame(body)
    bars = norm.bars(table)
    with engine.begin() as conn:
        batch = land_bytes(
            conn, vendor=VENDOR, dataset="prices", source=url, payload=body, name=f"{ticker}.json", rows=len(bars)
        )
    if bars.empty:
        return "empty", UpsertResult(), UpsertResult()
    same = norm.same_stock(bars, _other_bars(engine, listing_id, *window))
    if same is False:
        with engine.begin() as conn:
            mark(conn, batch.batch_id, "failed", "closes disagree with another vendor's: another company?")
        return "disagreed", UpsertResult(), UpsertResult()
    keys = {"listing_id": listing_id, "vendor_id": vendor_id(VENDOR)}
    with engine.begin() as conn:
        written = upsert(
            conn,
            schema.price_bar,
            bars.assign(**keys, currency="USD"),
            key=["listing_id", "vendor_id", "trade_date"],
            batch_id=batch.batch_id,
        )
        acted = upsert(
            conn,
            schema.corporate_action,
            norm.actions(table).assign(**keys, currency="USD"),
            key=["listing_id", "vendor_id", "action_type", "ex_date"],
            batch_id=batch.batch_id,
        )
        mark(conn, batch.batch_id, "normalized")
    return ("verified" if same else "unverified"), written, acted


def load_tiingo(
    engine: Engine,
    *,
    client: TiingoClient | None = None,
    max_requests: int | None = None,
    today: dt.date | None = None,
) -> dict[str, Any]:
    """Run the steps of the module docstring; ``max_requests`` stops before Tiingo would (its allowance)."""
    client = client or TiingoClient()
    day = pd.Timestamp(today or dt.date.today())
    report: dict[str, Any] = {}
    with ops.job("load_tiingo", engine) as run:
        todo, skipped = plan(targets(engine, day), _supported(engine, client))
        report["skipped"] = {k: len(v) for k, v in skipped.items()}
        report["not_at_tiingo_sample"] = skipped["not_at_tiingo"][:SAMPLE]
        bars_total, actions_total = UpsertResult(), UpsertResult()
        outcome: dict[str, list[str]] = {
            "verified": [],
            "unverified": [],
            "disagreed": [],
            "empty": [],
            "not_found": [],
        }
        done = 0
        report["throttled"] = False
        for listing_id, ticker, start, end in todo.itertuples(index=False):
            if max_requests is not None and client.requests_made >= max_requests:
                break
            try:
                answer = client.prices(ticker, start.date().isoformat(), end.date().isoformat())
            except TiingoThrottledError as exc:
                report["throttled"] = True
                log.warning("%s - the next run continues", exc)
                break
            done += 1
            if answer is None:
                outcome["not_found"].append(ticker)
                continue
            kind, bars, acts = _write(engine, int(listing_id), ticker, (start, end), answer)
            outcome[kind].append(ticker)
            bars_total += bars
            actions_total += acts
        report |= {k: len(v) for k, v in outcome.items()}
        report["disagreed_sample"] = outcome["disagreed"][:SAMPLE]
        report["requests"] = client.requests_made
        report["remaining"] = len(todo) - done
        report["complete"] = report["remaining"] == 0
        report["price_bars"] = bars_total.__dict__
        report["corporate_actions"] = actions_total.__dict__
        log.info("Tiingo: %s", report)
        run.finish(rows=bars_total.inserted + bars_total.updated, detail=report)
    return report


def probe(client: TiingoClient | None = None) -> pd.DataFrame:
    """Questions with known answers, read by the load's own code: ``[check, ok, detail]``. Nothing is stored.

    The supported-tickers file, and - one request of the allowance - Apple's 4-for-1 split of 2020-08-31 and its
    $0.82 dividend of 2020-08-07 (in the shares of that day).
    """
    rows: list[dict[str, Any]] = []

    def check(name: str, run: Any) -> None:
        try:
            ok, detail = run()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:300]
        rows.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})

    try:
        c = client or TiingoClient()
    except Exception as exc:
        return pd.DataFrame([{"check": "API key configured", "ok": False, "detail": str(exc)[:300]}])
    rows.append({"check": "API key configured", "ok": True, "detail": ""})
    answer: dict[str, pd.DataFrame] = {"t": pd.DataFrame()}

    def supported() -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "supported.zip"
            c.download_supported_tickers(path)
            table = norm.supported(path.read_bytes())
        aapl = table[table["ticker"] == "AAPL"].to_dict(orient="records")
        return len(table) > 10_000 and bool(aapl), f"{len(table)} tickers; AAPL {aapl}"

    def fetch() -> tuple[bool, str]:
        got = c.prices("AAPL", "2020-08-03", "2020-09-04")
        answer["t"] = norm.frame(got[1]) if got else pd.DataFrame()
        return len(answer["t"]) > 20, f"{len(answer['t'])} days; columns {list(answer['t'].columns)}"

    def split() -> tuple[bool, str]:
        acts = norm.actions(answer["t"])
        found = acts[acts["action_type"] == "split"]
        ok = len(found) == 1 and found["ex_date"].iloc[0] == pd.Timestamp("2020-08-31") and found["ratio"].iloc[0] == 4
        return ok, str(found.to_dict(orient="records"))

    def dividend() -> tuple[bool, str]:
        acts = norm.actions(answer["t"])
        found = acts[acts["action_type"] == "cash_dividend"]
        ok = len(found) == 1 and abs(float(found["amount"].iloc[0]) - 0.82) < 0.01
        return ok, str(found.to_dict(orient="records"))

    def unadjusted() -> tuple[bool, str]:
        b = norm.bars(answer["t"]).set_index("trade_date")["close"]
        before, after = float(b[pd.Timestamp("2020-08-28")]), float(b[pd.Timestamp("2020-08-31")])
        return before > 400 and after < 150, f"close 2020-08-28 {before}, 2020-08-31 {after}"

    check("supported tickers file", supported)
    check("AAPL prices answered", fetch)
    check("AAPL 2020 split read as 4-for-1", split)
    check("AAPL dividend in the day's shares (0.82)", dividend)
    check("closes unadjusted around the split", unadjusted)
    rows.append({"check": "requests used", "ok": True, "detail": str(c.requests_made)})
    return pd.DataFrame(rows)
