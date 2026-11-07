"""Load SimFin's quarterly statements as financial facts of the companies we hold, and its sector labels.

    report = load_simfin(engine)                   # bulk files newer than a week are reused, not fetched again
    report = load_simfin(engine, max_age_days=0)   # always fetch

1. the companies, industries and quarterly statement files (general, banks, insurers) are landed as they came;
2. each SimFin company is matched to one of our issuers: by its SimFin id once known, else by CIK (recorded by the
   SEC load), else through one of its listings by ISIN, then by ticker on the day it last published (tickers are
   reused) or today - unmatched ones are counted, never created. The SimFin id becomes an issuer identifier;
3. US-dollar statements of matched companies become ``fund.financial_fact`` rows with vendor ``simfin``
   (insert-only: a later file adds a row only for a figure SimFin restated), its sector labels a
   ``ref.classification`` under scheme ``SF``.

Dates: SimFin re-dates a figure whenever a later report repeats it, so its restated date is usually a year after the
figure was public. A figure the SEC's first report confirms (same value, see
:func:`~equity_research.ingest.vendors.simfin.normalize.verified_dates`) is dated by its first publication; any other
keeps the restated date. Run it after ``load_sec``: run before, it dates conservatively, and the next run after the SEC
load adds the earlier-dated rows (research takes the first published).
"""

from __future__ import annotations

import datetime as dt
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import UpsertResult, upsert
from equity_research.ingest.classification import record as record_classification
from equity_research.ingest.identity import US_MARKETS, add_identifiers, anchor_map, match_listings
from equity_research.ingest.vendors.simfin import VENDOR
from equity_research.ingest.vendors.simfin import normalize as norm
from equity_research.ingest.vendors.simfin.client import DATASETS, SimFinClient
from equity_research.raw.store import Batch, land_file, mark, recent_batch
from equity_research.settings import get_settings

log = logging.getLogger(__name__)

SCHEME = "SF"
#: statement dataset -> the columns read from it
STATEMENTS = {
    name: norm.INCOME_ITEMS if name.startswith("income") else norm.BALANCE_ITEMS
    for name in DATASETS
    if name.endswith("_quarterly")
}
STILL_REPORTING_DAYS = 200  # a company that published this recently is matched on its ticker today
FACTS_PER_TRANSACTION = 200_000


def land_datasets(engine: Engine, client: SimFinClient, *, max_age_days: float) -> dict[str, Batch]:
    """Every dataset of :data:`DATASETS`, landed (or a landed one younger than ``max_age_days`` reused)."""
    out: dict[str, Batch] = {}
    for name in DATASETS:
        with engine.begin() as conn:
            batch = recent_batch(conn, vendor=VENDOR, dataset=name, max_age_days=max_age_days)
            if batch is None:
                staging = get_settings().raw_dir / "_staging" / f"simfin-{name}.zip"
                url = client.download(name, staging)
                batch = land_file(conn, vendor=VENDOR, dataset=name, source=url, path=staging, move=True)
        out[name] = batch
        log.info("SimFin %s: batch %d", name, batch.batch_id)
    return out


def _dotted(ticker: pd.Series) -> pd.Series:
    """The other spelling of a share class (``BRK-B`` <-> ``BRK.B``): vendors differ."""
    return ticker.str.replace("-", ".", regex=False).where(ticker.str.contains("-", regex=False))


def match_companies(conn: Connection, companies: pd.DataFrame, last_published: pd.Series) -> pd.DataFrame:
    """``companies`` with ``issuer_id`` and ``matched_by`` (missing when unmatched) - see the module docstring."""
    out = companies.assign(issuer_id=pd.Series(pd.NA, index=companies.index, dtype="Int64"), matched_by=None)

    def take(found: pd.Series, how: str) -> None:
        hit = out["issuer_id"].isna() & found.notna()
        out.loc[hit, "issuer_id"] = found[hit].astype(int)
        out.loc[hit, "matched_by"] = how

    known = anchor_map(conn, "issuer", "SIMFIN")
    take(out["simfin_id"].map(known), "SIMFIN")
    by_cik = anchor_map(conn, "issuer", "CIK")
    take(out["cik"].map(by_cik), "CIK")
    issuer_of = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
    last = out["simfin_id"].map(last_published)
    recent = last >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=STILL_REPORTING_DAYS)
    attempts = (
        (out["ticker"], last.where(~recent)),  # on its last report's day (today when still reporting)
        (out["ticker"], None),  # today
        (_dotted(out["ticker"].astype("string")), last.where(~recent)),
    )
    for tickers, day in attempts:
        todo = out["issuer_id"].isna()
        if not todo.any():
            break
        codes = pd.DataFrame(
            {"key": out.loc[todo, "simfin_id"], "ISIN": out.loc[todo, "isin"], "TICKER": tickers[todo]}
        )
        if day is not None:
            codes["on"] = day[todo]
        matched = match_listings(conn, codes.dropna(subset=["TICKER", "ISIN"], how="all"), markets=US_MARKETS)
        listing = pd.Series(matched["listing_id"].to_numpy(), index=matched.index)
        how = pd.Series(matched["matched_by"].to_numpy(), index=matched.index)
        found = listing.map(lambda x: issuer_of.get(int(x)) if pd.notna(x) else None).dropna()
        for method in ("ISIN", "TICKER"):
            ids = found[how.reindex(found.index) == method]
            take(ids.reindex(out.index), method)
    return out


def sec_first_reported(engine: Engine, items: set[str]) -> pd.DataFrame:
    """The SEC's first report of each quarter's flows and balances - what SimFin's figures are checked against.

    ``[issuer_id, item, period_type, period_end, value, published_at]``.
    """
    sql = sa.text(
        """
        SELECT issuer_id, item, period_type, period_end, value, published_at FROM (
            SELECT issuer_id, item, period_type, period_end, value, published_at,
                   ROW_NUMBER() OVER (PARTITION BY issuer_id, item, period_type, period_end ORDER BY published_at) AS n
            FROM fund.financial_fact
            WHERE vendor_id = :sec AND period_type IN ('Q', 'I') AND item IN :items
        ) f WHERE n = 1
        """
    ).bindparams(sa.bindparam("items", expanding=True))
    columns = ["issuer_id", "item", "period_type", "period_end", "value", "published_at"]
    with engine.connect() as conn:
        rows = conn.execute(sql, {"sec": vendor_id("sec"), "items": sorted(items)}).all()
    return pd.DataFrame(rows, columns=columns)


def load_simfin(
    engine: Engine, *, client: SimFinClient | None = None, max_age_days: float = 7, today: dt.date | None = None
) -> dict[str, Any]:
    """Run the steps of the module docstring; returns a report."""
    report: dict[str, Any] = {}
    on = today or dt.date.today()
    with ops.job("load_simfin", engine) as run:
        batches = land_datasets(engine, client or SimFinClient(), max_age_days=max_age_days)
        companies = norm.companies(norm.read_table(batches["companies"].path))
        statements = {name: norm.read_table(batches[name].path) for name in STATEMENTS}
        last_published = pd.concat([norm.last_published(t) for t in statements.values() if not t.empty])
        last_published = last_published.groupby(level=0).max()
        with engine.begin() as conn:
            matched = match_companies(conn, companies, last_published)
            found = matched.dropna(subset=["issuer_id"]).astype({"issuer_id": int})
            twins = found.duplicated("issuer_id", keep="first")  # two SimFin ids for one of our companies
            found = found[~twins]
            report["companies"] = len(companies)
            report["matched"] = len(found)
            report["by"] = found["matched_by"].value_counts().to_dict()
            report["one_issuer_two_simfin_ids"] = int(twins.sum())
            conflicts = add_identifiers(
                conn,
                pd.DataFrame(
                    {
                        "entity_type": "issuer",
                        "entity_id": found["issuer_id"],
                        "id_type": "SIMFIN",
                        "value": found["simfin_id"],
                        "valid_from": None,
                        "valid_to": None,
                        "vendor_id": vendor_id(VENDOR),
                    }
                ),
                batch_id=batches["companies"].batch_id,
            )
            report["identifier_conflicts"] = len(conflicts)
            sectors = found.merge(norm.industries(norm.read_table(batches["industries"].path)), on="industry_id")
            recorded = record_classification(
                conn,
                sectors.dropna(subset=["sector"]),
                vendor=VENDOR,
                on=on,
                scheme=SCHEME,
                batch_id=batches["industries"].batch_id,
            )
            report["classifications"] = {"opened": recorded.opened, "closed": recorded.closed}
        issuer_of = dict(zip(found["simfin_id"], found["issuer_id"], strict=True))
        report["facts"] = {}
        report["currencies"] = {}
        report["dates"] = {}
        sec_first = sec_first_reported(engine, {item for items in STATEMENTS.values() for item, _, _ in items.values()})
        for name, items in STATEMENTS.items():
            table = statements[name]
            report["currencies"][name] = norm.currencies(table)
            rows = norm.facts(table, items)
            rows = rows.assign(issuer_id=rows["simfin_id"].map(issuer_of)).dropna(subset=["issuer_id"])
            rows = norm.verified_dates(rows.astype({"issuer_id": int}), sec_first)
            outcome = rows["verified"].map({True: "confirmed_by_sec", False: "differs_from_sec"})
            report["dates"][name] = outcome.fillna("no_sec_figure").value_counts().to_dict()
            rows = rows.drop(columns=["simfin_id", "first_published", "verified"]).assign(vendor_id=vendor_id(VENDOR))
            written = UpsertResult()
            for start in range(0, len(rows), FACTS_PER_TRANSACTION):
                with engine.begin() as conn:
                    written += upsert(
                        conn,
                        schema.financial_fact,
                        rows.iloc[start : start + FACTS_PER_TRANSACTION],
                        key=["issuer_id", "vendor_id", "item", "period_type", "period_end", "published_at"],
                        batch_id=batches[name].batch_id,
                        insert_only=True,
                    )
            with engine.begin() as conn:
                mark(conn, batches[name].batch_id, "normalized")
            report["facts"][name] = written.__dict__
            log.info("SimFin %s: %s", name, written)
        inserted = sum(v["inserted"] for v in report["facts"].values())
        run.finish(rows=inserted, detail=report)
    return report


def probe(client: SimFinClient | None = None) -> pd.DataFrame:
    """Questions with known answers, read by the load's own code: ``[check, ok, detail]``. Nothing is stored.

    Downloads the companies file and the general quarterly income statements to a temporary folder: which codes the
    companies file carries (CIK / ISIN decide how companies are matched), Apple's 2024 quarterly revenue in a believable
    range with its cost positive, how often the restated date is far after the first publication, and whether the
    account may download the banks' file.
    """
    rows: list[dict[str, Any]] = []

    def check(name: str, run: Any) -> None:
        try:
            ok, detail = run()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:300]
        rows.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})

    try:
        c = client or SimFinClient()
    except Exception as exc:
        return pd.DataFrame([{"check": "API key configured", "ok": False, "detail": str(exc)[:300]}])
    rows.append({"check": "API key configured", "ok": True, "detail": ""})
    tables: dict[str, pd.DataFrame] = {}
    with tempfile.TemporaryDirectory() as tmp:

        def fetch(name: str) -> tuple[bool, str]:
            path = Path(tmp) / f"{name}.zip"
            c.download(name, path)
            tables[name] = norm.read_table(path)
            return not tables[name].empty, f"{len(tables[name])} rows; columns {list(tables[name].columns)}"

        def codes() -> tuple[bool, str]:
            return True, str({k: k in tables["companies"].columns for k in ("CIK", "ISIN")})

        def apple() -> tuple[bool, str]:
            sid = norm.companies(tables["companies"]).set_index("ticker").loc["AAPL", "simfin_id"]
            facts = norm.facts(tables["income_quarterly"], norm.INCOME_ITEMS)
            mine = facts[(facts["simfin_id"] == sid) & (facts["period_end"].dt.year == 2024)]
            revenue = mine.loc[mine["item"] == "revenue", "value"]
            cost = mine.loc[mine["item"] == "cost_of_revenue", "value"]
            ok = len(revenue) > 0 and bool(revenue.between(80e9, 130e9).all()) and bool((cost > 0).all())
            return ok, f"2024 quarters: revenue {revenue.tolist()}, cost {cost.tolist()}"

        def restated() -> tuple[bool, str]:
            t = tables["income_quarterly"]
            lag = pd.to_datetime(t["Restated Date"]) - pd.to_datetime(t["Publish Date"])
            return True, f"{(lag > pd.Timedelta(days=30)).mean():.0%} dated more than 30 days after first publication"

        check("companies file", lambda: fetch("companies"))
        check("companies carry CIK / ISIN", codes)
        check("quarterly income statements", lambda: fetch("income_quarterly"))
        check("AAPL quarterly revenue, cost positive", apple)
        check("restated dates", restated)
        check("banks' statements in this plan", lambda: fetch("income_banks_quarterly"))
    return pd.DataFrame(rows)
