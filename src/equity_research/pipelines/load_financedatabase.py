"""Enrich our listings from FinanceDatabase: codes, the country a company is run from, and its sector labels.

    report = load_financedatabase(engine)

1. its equities file (a bz2-compressed CSV from the project's GitHub, or the same file through the jsDelivr CDN) is
   landed in the raw zone as it came;
2. each US listing is matched to one of ours (:func:`~equity_research.ingest.identity.match_listings`: ISIN, then
   CUSIP, then ticker today) - unmatched ones are counted, never created (it has no history to give them);
3. for matched listings: its symbol (a Yahoo symbol) and FIGI become identifiers, its sector labels a
   ``ref.classification`` row under its own scheme (``FD`` - GICS-like names, not GICS), and its country fills a
   company's ``domicile_country`` where no other source has.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import upsert
from equity_research.ingest.classification import record as record_classification
from equity_research.ingest.identity import OPEN_START, US_MARKETS, add_identifiers, match_listings
from equity_research.ingest.vendors.financedatabase import VENDOR
from equity_research.ingest.vendors.financedatabase.normalize import us_equities
from equity_research.raw.store import land_file, land_frame, mark
from equity_research.settings import get_settings

log = logging.getLogger(__name__)

SCHEME = "FD"


#: The equities file as the project publishes it; the same file through a CDN when GitHub's file host is unreachable.
EQUITIES_URLS = (
    "https://raw.githubusercontent.com/JerBouma/FinanceDatabase/main/compression/equities.bz2",
    "https://cdn.jsdelivr.net/gh/JerBouma/FinanceDatabase@main/compression/equities.bz2",
)


def download_equities(target: Path) -> str:
    """Download the equities file (bz2-compressed CSV) to ``target``; returns the URL that answered."""
    errors = []
    for url in EQUITIES_URLS:
        try:
            response = requests.get(url, timeout=120, headers={"User-Agent": "equity_research"})
            response.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            return url
        except requests.RequestException as exc:
            log.warning("FinanceDatabase from %s failed: %s", url, str(exc)[:120])
            errors.append(f"{url}: {exc}")
    raise RuntimeError("FinanceDatabase could not be downloaded: " + " | ".join(errors)[:600])


def read_equities(path: Path) -> pd.DataFrame:
    """The landed file as a table indexed by symbol."""
    return pd.read_csv(path, compression="bz2", index_col=0, low_memory=False)


def load_financedatabase(engine: Engine, raw: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run the steps of the module docstring (``raw``: the equities table, fetched when not given)."""
    report: dict[str, Any] = {}
    with ops.job("load_financedatabase", engine) as run:
        with engine.begin() as conn:
            if raw is None:  # the delivery, kept as it came
                staging = get_settings().raw_dir / "_staging" / "financedatabase-equities.bz2"
                url = download_equities(staging)
                batch = land_file(conn, vendor=VENDOR, dataset="equities", source=url, path=staging, move=True)
                table = read_equities(batch.path)
            else:
                batch = land_frame(conn, vendor=VENDOR, dataset="equities", source="given", frame=raw)
                table = raw
        codes = us_equities(table)
        today = dt.date.today()
        with engine.begin() as conn:
            matched = match_listings(
                conn, codes.rename(columns={"symbol": "key", "ticker": "TICKER"}), markets=US_MARKETS, on=today
            )
            found = codes.assign(
                listing_id=matched["listing_id"].to_numpy(), matched_by=matched["matched_by"].to_numpy()
            )
            found = found.dropna(subset=["listing_id"]).astype({"listing_id": int})
            report["matched"] = len(found)
            report["by"] = found["matched_by"].value_counts().to_dict()
            report["ambiguous"] = int(matched["ambiguous"].sum())
            report["unmatched_us_listings"] = int(len(codes) - len(found) - report["ambiguous"])
            ids = pd.concat(
                [
                    found[["listing_id", "symbol"]].rename(columns={"symbol": "value"}).assign(id_type="YAHOO"),
                    found[["listing_id", "FIGI"]].rename(columns={"FIGI": "value"}).assign(id_type="FIGI"),
                ],
                ignore_index=True,
            ).dropna(subset=["value"])
            conflicts = add_identifiers(
                conn,
                ids.rename(columns={"listing_id": "entity_id"}).assign(
                    entity_type="listing", valid_from=OPEN_START, valid_to=None, vendor_id=vendor_id(VENDOR)
                ),
                batch_id=batch.batch_id,
            )
            report["identifier_conflicts"] = len(conflicts)
            issuer_of = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
            found["issuer_id"] = found["listing_id"].map(issuer_of)
            companies = found.dropna(subset=["issuer_id"]).drop_duplicates("issuer_id")
            recorded = record_classification(
                conn,
                companies.dropna(subset=["sector"]),
                vendor=VENDOR,
                on=today,
                scheme=SCHEME,
                batch_id=batch.batch_id,
            )
            report["classifications"] = {"opened": recorded.opened, "closed": recorded.closed}
            report["domiciles"] = _fill_domiciles(conn, companies)
            mark(conn, batch.batch_id, "normalized")
        log.info("FinanceDatabase: %s", report)
        run.finish(rows=report["matched"], detail=report)
    return report


def _fill_domiciles(conn: sa.Connection, companies: pd.DataFrame) -> int:
    """Set ``domicile_country`` where it is still empty (a source that has it is not overwritten)."""
    rows = companies.dropna(subset=["domicile"])[["issuer_id", "domicile"]]
    if rows.empty:
        return 0
    empty = {
        int(i)
        for i in conn.execute(
            sa.select(schema.issuer.c.issuer_id).where(schema.issuer.c.domicile_country.is_(None))
        ).scalars()
    }
    rows = rows[rows["issuer_id"].astype(int).isin(empty)].rename(columns={"domicile": "domicile_country"})
    if rows.empty:
        return 0
    return upsert(conn, schema.issuer, rows.astype({"issuer_id": int}), key=["issuer_id"]).updated
