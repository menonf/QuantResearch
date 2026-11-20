"""Load SEC XBRL financial statements for the issuers in the database.

Steps:

1. land the SEC's company-facts bulk file (re-downloaded when the landed one is older than ``max_age_days``), the
   SEC ticker file and the public CIK-CUSIP map as raw batches;
2. match issuers to CIKs (:mod:`equity_research.ingest.vendors.sec.cik`); issuers sharing a CIK are one company and
   are merged; the CIK becomes an issuer identifier;
3. filings and financial facts, as reported, for every matched issuer and each of its CIKs (a company that
   re-registered files its earlier accounts under the old one) - ``equity.filing``, ``equity.financial_fact``.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from typing import Any

import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from equity_research.data import schema
from equity_research.ingest.vendors.sec import VENDOR
from equity_research.ingest.vendors.sec import normalize as sec
from equity_research.ingest.vendors.sec.cik import match_issuers
from equity_research.ingest.vendors.sec.client import (
    CIK_CUSIP_MAP_MIRROR,
    CIK_CUSIP_MAP_URL,
    COMPANY_FACTS_BULK_URL,
    COMPANY_TICKERS_URL,
    SecClient,
)
from research_core.data import ops
from research_core.data.db.schema import vendor_id
from research_core.data.db.upsert import UpsertResult, upsert
from research_core.data.identity import US_MARKETS, add_identifiers, merge_issuers
from research_core.data.raw_store import Batch, land_file, mark, recent_batch
from research_core.settings import get_settings

log = logging.getLogger(__name__)

ISSUERS_PER_TRANSACTION = 100


def _land_download(conn: Connection, client: SecClient, urls: tuple[str, ...], dataset: str, filename: str) -> Batch:
    """Download from the first of ``urls`` that answers (a mirror after the original) and land the file."""
    staging = get_settings().raw_dir / "_staging" / filename
    failures = []
    for url in urls:
        try:
            client.download(url, staging)
        except (RuntimeError, requests.RequestException) as exc:
            log.warning("SEC %s: %s - trying the next source", url, str(exc)[:160])
            failures.append(f"{url}: {exc}")
            continue
        return land_file(conn, vendor=VENDOR, dataset=dataset, source=url, path=staging, move=True)
    raise RuntimeError(f"{dataset} could not be downloaded: " + " | ".join(failures)[:600])


def land_sources(engine: Engine, client: SecClient, *, max_age_days: float = 7) -> dict[str, Batch]:
    """The three raw inputs, landed (or the recent landed ones reused)."""
    out: dict[str, Batch] = {}
    sources = (
        ("companyfacts_bulk", (COMPANY_FACTS_BULK_URL,), "companyfacts.zip"),
        ("company_tickers", (COMPANY_TICKERS_URL,), "company_tickers_exchange.json"),
        ("cik_cusip_map", (CIK_CUSIP_MAP_URL, CIK_CUSIP_MAP_MIRROR), "cik-cusip-maps.csv"),
    )
    for dataset, urls, filename in sources:
        with engine.begin() as conn:
            batch = recent_batch(conn, vendor=VENDOR, dataset=dataset, max_age_days=max_age_days) or _land_download(
                conn, client, urls, dataset, filename
            )
        out[dataset] = batch
        log.info("%s: batch %d (%s)", dataset, batch.batch_id, batch.path.name)
    return out


def _issuer_tickers(conn: Connection) -> pd.DataFrame:
    """``[issuer_id, cusip6, ticker, name]``: each issuer's CUSIP issuer code, its listings' current tickers, its name.

    Tickers come from ``core.identifier`` whichever vendor recorded them - only those valid today (tickers are reused),
    and only of US listings: the SEC's ticker file is US tickers, and London's ``AAL`` is not American Airlines.
    """
    sql = sa.text(
        """
        SELECT l.issuer_id, cu.value AS cusip6, t.value AS ticker, i.name
        FROM core.listing l
        JOIN core.issuer i ON i.issuer_id = l.issuer_id
        LEFT JOIN core.identifier cu
            ON cu.entity_type = 'issuer' AND cu.entity_id = l.issuer_id AND cu.id_type = 'CUSIP'
        LEFT JOIN core.identifier t ON t.entity_type = 'listing' AND t.entity_id = l.listing_id AND t.id_type = 'TICKER'
             AND t.valid_to IS NULL AND (l.exchange_mic IN :markets OR l.exchange_mic IS NULL)
        WHERE l.issuer_id IS NOT NULL
        """
    ).bindparams(sa.bindparam("markets", sorted(US_MARKETS), expanding=True))
    return pd.DataFrame(conn.execute(sql).all(), columns=["issuer_id", "cusip6", "ticker", "name"])


_ENTITY_NAME = re.compile(rb'"entityName"\s*:\s*"((?:[^"\\]|\\.)*)"')


def registrant_names(z: zipfile.ZipFile, ciks: set[int]) -> dict[int, str]:
    """``{CIK: the name it files under}`` from the bulk file - read from the head of each company's file only."""
    out = {}
    for cik in sorted(ciks):
        member = f"CIK{cik:010d}.json"
        try:
            with z.open(member) as f:
                head = f.read(2048)
        except KeyError:
            continue
        if (m := _ENTITY_NAME.search(head)) is not None:
            out[cik] = json.loads(b'"' + m.group(1) + b'"')
    return out


#: a name-matched CIK's period must end this long before the company's main CIK's first (the same quarter's end
#: falls on different days: Disney's 52-week year)
CUTOVER_DAYS = 20
MAIN = ("ticker", "cusip")  # how a company's main CIK is found; ``cusip+name`` CIKs only fill in before it


def _companies(pairs: pd.DataFrame) -> dict[int, int]:
    """``{issuer_id: its company}``: issuers sharing a CIK, however chained, join the lowest id among them."""
    parent: dict[int, int] = {}

    def root(i: int) -> int:
        while parent.setdefault(i, i) != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _cik, grp in pairs.groupby("cik"):
        ids = sorted(int(i) for i in grp["issuer_id"])
        for other in ids[1:]:
            a, b = root(ids[0]), root(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {i: root(i) for i in list(parent)}


def match_and_record(engine: Engine, batches: dict[str, Batch]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Issuer -> CIKs ``[issuer_id, cik, main]``; main CIKs are recorded and issuers sharing one are merged.

    A ``cusip+name`` CIK is not recorded here: :func:`load_facts` records it only if it fills periods the main CIK
    does not cover, and it never merges two companies (one name can have passed to a subsidiary or a successor that
    is another company of ours). One already a main CIK of another company is dropped.
    """
    raw = json.loads(batches["company_tickers"].path.read_text(encoding="utf-8"))
    tickers = pd.DataFrame(raw["data"], columns=raw["fields"])
    cusip_map = pd.read_csv(batches["cik_cusip_map"].path, dtype={"cusip6": str, "cusip8": str}).dropna(subset=["cik"])
    cusip_map["cik"] = cusip_map["cik"].astype(float).astype("int64")
    cusip_map["cusip6"] = cusip_map["cusip6"].astype(str).str.strip().str.upper()
    with engine.connect() as conn:
        issuers = _issuer_tickers(conn)
    with zipfile.ZipFile(batches["companyfacts_bulk"].path) as z:
        filers = {int(n[3:13]) for n in z.namelist() if n.startswith("CIK") and n.endswith(".json")}
        candidates = set(cusip_map.loc[cusip_map["cusip6"].isin(set(issuers["cusip6"].dropna())), "cik"]) & filers
        names = registrant_names(z, candidates)
    matches = match_issuers(issuers, tickers, cusip_map, filers, names).dropna(subset=["cik"])
    matches = matches.assign(main=matches["method"].isin(MAIN))
    main = matches[matches["main"]]
    company_of = _companies(main)
    with engine.begin() as conn:
        merged = 0
        for survivor in sorted(set(company_of.values())):
            others = sorted(i for i, c in company_of.items() if c == survivor and i != survivor)
            if others:
                merge_issuers(conn, survivor, others)
                merged += len(others)
        matches = matches.assign(issuer_id=matches["issuer_id"].astype(int).map(lambda i: company_of.get(i, i)))
        taken = set(matches.loc[matches["main"], "cik"].astype(int))
        extra = matches[~matches["main"] & ~matches["cik"].astype(int).isin(taken)]
        links = pd.concat([matches[matches["main"]], extra], ignore_index=True)
        links = links.drop_duplicates(["issuer_id", "cik"]).sort_values(["issuer_id", "cik"])
        recorded = links[links["main"]]
        conflicts = _record_ciks(conn, recorded, batches["company_tickers"].batch_id)
    report = {
        "issuers_matched": links["issuer_id"].nunique(),
        "main_ciks": int(links["main"].sum()),
        "name_matched_ciks": int((~links["main"]).sum()),
        "by_method": links["method"].value_counts().to_dict(),
        "ticker_cusip_conflicts": int(links["conflict"].sum()),
        "issuers_merged": merged,
        "identifier_conflicts": len(conflicts),
    }
    return links[["issuer_id", "cik", "main"]].astype({"issuer_id": int, "cik": int}), report


def _record_ciks(conn: Connection, links: pd.DataFrame, batch_id: int) -> pd.DataFrame:
    return add_identifiers(
        conn,
        pd.DataFrame(
            {
                "entity_type": "issuer",
                "entity_id": links["issuer_id"].astype(int),
                "id_type": "CIK",
                "value": links["cik"].astype(int).astype(str),
                "valid_from": None,
                "valid_to": None,
                "vendor_id": vendor_id(VENDOR),
            }
        ),
        batch_id=batch_id,
    )


def company_facts(frames: dict[tuple[int, bool], pd.DataFrame]) -> tuple[pd.DataFrame, list[int]]:
    """One company's facts from its CIKs ``{(cik, main): canonical facts}``: ``(facts, name-matched CIKs used)``.

    Its main CIK's facts in full (with no main CIK, the one reporting latest stands in); every other CIK's only for
    periods ending :data:`CUTOVER_DAYS` before the main one's first - a predecessor's earlier years (Disney before
    2019), never a subsidiary's same years (Duke Energy Carolinas beside Duke Energy).
    """
    present = {k: f for k, f in frames.items() if not f.empty}
    if not present:
        return pd.DataFrame(), []
    mains = [k for k in present if k[1]] or [max(present, key=lambda k: pd.to_datetime(present[k]["period_end"]).max())]
    main = max(mains, key=lambda k: len(present[k]))
    first = pd.to_datetime(present[main]["period_end"]).min() - pd.Timedelta(days=CUTOVER_DAYS)
    parts, filled = [present[main]], ([] if main[1] else [main[0]])  # a stand-in main CIK is the company's too
    for key, f in present.items():
        if key == main:
            continue
        before = f[pd.to_datetime(f["period_end"]) < first]
        if not before.empty:
            parts.append(before)
            filled.append(key[0])
    return pd.concat(parts, ignore_index=True), filled


def load_facts(engine: Engine, links: pd.DataFrame, bulk: Batch) -> dict[str, Any]:
    """Filings and financial facts of each company from its CIKs, ``ISSUERS_PER_TRANSACTION`` companies at a time."""
    filings_total, facts_total = UpsertResult(), UpsertResult()
    missing, filled = 0, []
    by_issuer = {int(str(i)): g for i, g in links.groupby("issuer_id")}
    companies = sorted(by_issuer)
    with zipfile.ZipFile(bulk.path) as z:
        names = set(z.namelist())
        for start in range(0, len(companies), ISSUERS_PER_TRANSACTION):
            filing_rows, fact_rows = [], []
            for issuer_id in companies[start : start + ISSUERS_PER_TRANSACTION]:
                frames, filings = {}, []
                for cik, is_main in zip(by_issuer[issuer_id]["cik"], by_issuer[issuer_id]["main"], strict=True):
                    member = f"CIK{int(cik):010d}.json"
                    if member not in names:
                        missing += 1
                        continue
                    facts = sec.facts_frame(json.loads(z.read(member)))
                    frames[(int(cik), bool(is_main))] = sec.financial_facts(facts)
                    filings.append((int(cik), sec.filings(facts)))
                company, used = company_facts(frames)
                if company.empty:
                    continue
                kept = set(company["filing_ref"].dropna())
                filled += [(issuer_id, c) for c in used]
                for _cik, f in filings:  # the filings behind the facts kept
                    filing_rows.append(
                        f[f["filing_ref"].isin(kept)].assign(issuer_id=issuer_id, vendor_id=vendor_id(VENDOR))
                    )
                fact_rows.append(company.assign(issuer_id=issuer_id, vendor_id=vendor_id(VENDOR)))
            with engine.begin() as conn:
                if filing_rows:
                    filings_total += upsert(
                        conn,
                        schema.filing,
                        pd.concat(filing_rows, ignore_index=True),
                        key=["issuer_id", "vendor_id", "filing_ref"],
                        batch_id=bulk.batch_id,
                    )
                if fact_rows:
                    facts_total += upsert(
                        conn,
                        schema.financial_fact,
                        pd.concat(fact_rows, ignore_index=True),
                        key=["issuer_id", "vendor_id", "item", "period_type", "period_end", "published_at"],
                        batch_id=bulk.batch_id,
                        insert_only=True,
                    )
            log.info(
                "SEC facts: %d / %d companies, %d facts",
                min(start + ISSUERS_PER_TRANSACTION, len(companies)),
                len(companies),
                facts_total.inserted,
            )
    if filled:  # a name-matched CIK that filled in earlier years is the company's too
        with engine.begin() as conn:
            _record_ciks(conn, pd.DataFrame(filled, columns=["issuer_id", "cik"]), bulk.batch_id)
    return {
        "filings": filings_total.__dict__,
        "facts": facts_total.__dict__,
        "ciks_without_xbrl": missing,
        "companies_using_name_matched_ciks": len({i for i, _ in filled}),
    }


def load_sec(engine: Engine, *, client: SecClient | None = None, max_age_days: float = 7) -> dict[str, Any]:
    """Run every step (see the module docstring); returns a report."""
    with ops.job("load_sec", engine) as run:
        batches = land_sources(engine, client or SecClient(), max_age_days=max_age_days)
        links, identity = match_and_record(engine, batches)
        loaded = load_facts(engine, links, batches["companyfacts_bulk"])
        with engine.begin() as conn:  # all three inputs are read: the bulk file, the tickers, the CIK-CUSIP map
            for batch in batches.values():
                mark(conn, batch.batch_id, "normalized")
        report = {"identity": identity, **loaded}
        run.finish(rows=loaded["facts"]["inserted"], detail=report)
    return report
