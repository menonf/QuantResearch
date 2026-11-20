"""Load ICE's index constituent files from a folder: bonds, issuers, index weights and analytics, month by month.

    report = load_ice(engine)                      # the folder named by EQR_ICE_DIR
    report = load_ice(engine, Path("..."))         # or any folder

The files are downloads (one per index and month end, ``H0A0-01312019.xlsx``) that keep arriving, in any order:

1. **Land** - each file is copied unchanged into the raw zone and registered as a batch; a file landed and
   normalised before is skipped, so a second run reads only what is new.
2. **Who** - each of ICE's tickers is an issuer (``core.issuer``, code ``ICE``), each ISIN a bond (``core.listing``,
   instrument type ``bond``) with its CUSIP beside it. A bond's name and issuer follow the newest file that has it:
   an older file arriving later adds its month and leaves them alone.
3. **What** - the month's analytics per bond, the index's weights per bond, and the issuer's industry levels and
   country, each written change-only under the file's batch.

A file that cannot be read (a name without a month end, a missing column) is reported and skipped; the rest load.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from fixed_income_research.data import schema
from fixed_income_research.ingest import ice
from fixed_income_research.ingest.reference import INDICES, ensure_indices
from research_core.data import ops
from research_core.data.db.schema import vendor_id
from research_core.data.db.upsert import upsert
from research_core.data.identity import OPEN_START, add_identifiers, anchor_map, ensure_issuers, ensure_listings
from research_core.data.raw_store import land_file, mark, status
from research_core.settings import SettingMissingError

log = logging.getLogger(__name__)

VENDOR = "ice"
DATASET = "index_constituents"
FOLDER_SETTING = "EQR_ICE_DIR"
DELETE_CHUNK = 1000


def ice_folder() -> Path:
    """The folder the constituent files are downloaded to (``EQR_ICE_DIR``)."""
    value = os.environ.get(FOLDER_SETTING)
    if not value:
        raise SettingMissingError(f"the folder of ICE's constituent files is not configured: set {FOLDER_SETTING}")
    return Path(value)


def constituent_files(folder: Path) -> tuple[list[tuple[str, dt.date, Path]], dict[str, str]]:
    """``([(index code, month end, path)], {file name: why not})`` of a folder's workbooks, oldest month first."""
    found, refused = [], {}
    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):  # Excel's lock file of a workbook that is open
            continue
        try:
            code, as_of = ice.parse_name(path.name)
        except ice.LayoutError as bad:
            refused[path.name] = str(bad)
            continue
        if code not in INDICES:
            refused[path.name] = f"index {code} is not one we know (ingest/reference.py)"
            continue
        found.append((code, as_of, path))
    return sorted(found, key=lambda f: (f[1], f[0])), refused


def load_ice(engine: Engine, folder: Path | None = None) -> dict[str, Any]:
    """Run the steps of the module docstring; returns what was loaded, skipped and refused."""
    files, refused = constituent_files(folder or ice_folder())
    report: dict[str, Any] = {
        "files": len(files),
        "loaded": 0,
        "already_loaded": 0,
        "refused": refused,
        "bonds_created": 0,
        "issuers_created": 0,
        "identifier_conflicts": 0,
        "issuers_disagreeing": 0,
        "analytics": {"inserted": 0, "updated": 0},
        "constituents": {"inserted": 0, "updated": 0},
        "issuer_months": {"inserted": 0, "updated": 0},
    }
    with ops.job("load_ice", engine) as run:
        with engine.begin() as conn:
            indices = ensure_indices(conn)
        for code, as_of, path in files:
            with engine.begin() as conn:
                batch = land_file(conn, vendor=VENDOR, dataset=DATASET, source=path.name, path=path)
                if not batch.is_new and status(conn, batch.batch_id) == "normalized":
                    report["already_loaded"] += 1
                    continue
                try:
                    month = ice.normalize(ice.read_constituents(path))
                except ice.LayoutError as bad:
                    mark(conn, batch.batch_id, "failed", str(bad)[:400])
                    report["refused"][path.name] = str(bad)
                    continue
                written = write_month(conn, indices[code], as_of, month, batch.batch_id)
                mark(conn, batch.batch_id, "normalized")
            report["loaded"] += 1
            for key in ("bonds_created", "issuers_created", "identifier_conflicts"):
                report[key] += written[key]
            report["issuers_disagreeing"] += month.issuers_disagreeing
            for key in ("analytics", "constituents", "issuer_months"):
                report[key]["inserted"] += written[key].inserted
                report[key]["updated"] += written[key].updated
            log.info("ice %s %s: %d bonds", code, as_of, len(month.bonds))
        run.finish(rows=report["analytics"]["inserted"] + report["analytics"]["updated"], detail=report)
    return report


def write_month(
    conn: Connection, index_id: int, as_of: dt.date, month: ice.IndexMonth, batch_id: int
) -> dict[str, Any]:
    """Write one normalised file (steps 2 and 3 of the module docstring); returns what was created and written."""
    vendor = vendor_id(VENDOR)
    bonds = month.bonds.merge(month.issuers[["ticker", "name", "country"]], on="ticker", suffixes=("", "_issuer"))
    known_before = anchor_map(conn, "listing", "ISIN")
    newest = _newest_month(conn)
    # a bond's name and issuer are the newest file's: an older file arriving later does not take them back
    stale = bonds["isin"].map(lambda isin: newest.get(known_before.get(isin, -1), dt.date.min) > as_of)
    listings = pd.DataFrame(
        {
            "anchor": bonds["isin"],
            "issuer_anchor": bonds["ticker"],
            "issuer_name": bonds["name_issuer"],
            "issuer_domicile": bonds["country"],
            "name": bonds["name"].str[:256],
            "instrument_type": "bond",
            "currency": bonds["currency"],
        }
    )
    issuers, issuers_created = ensure_issuers(conn, listings, anchor_type="ICE", vendor=VENDOR, batch_id=batch_id)
    resolved = ensure_listings(
        conn, listings[~stale], anchor_type="ISIN", issuer_anchor_type="ICE", vendor=VENDOR, batch_id=batch_id
    )
    ids = {**known_before, **resolved.listing_ids}
    created = bonds[~bonds["isin"].isin(known_before) & bonds["cusip"].notna()]
    conflicts = add_identifiers(
        conn,
        pd.DataFrame(
            {
                "entity_type": "listing",
                "entity_id": created["isin"].map(ids),
                "id_type": "CUSIP",
                "value": created["cusip"],
                "valid_from": OPEN_START,
                "valid_to": None,
                "vendor_id": vendor,
            }
        ),
        batch_id=batch_id,
    )
    keys = pd.DataFrame({"listing_id": bonds["isin"].map(ids).astype(int), "vendor_id": vendor, "as_of_date": as_of})
    analytics = pd.concat(
        [keys, bonds["ticker"].map(issuers).rename("issuer_id"), bonds[list(ice.ANALYTICS.values())]], axis=1
    )
    weights = pd.concat([keys.assign(index_id=index_id), bonds[list(ice.CONSTITUENT.values())]], axis=1)
    issuer_months = month.issuers.drop(columns=["ticker", "name"]).assign(
        issuer_id=month.issuers["ticker"].map(issuers), vendor_id=vendor, as_of_date=as_of
    )
    _drop_leavers(conn, index_id, as_of, set(keys["listing_id"]))
    return {
        "bonds_created": resolved.listings_created,
        "issuers_created": issuers_created,
        "identifier_conflicts": len(conflicts),
        "analytics": upsert(
            conn, schema.bond_analytics, analytics, key=["listing_id", "vendor_id", "as_of_date"], batch_id=batch_id
        ),
        "constituents": upsert(
            conn,
            schema.index_constituent,
            weights,
            key=["index_id", "listing_id", "as_of_date"],
            batch_id=batch_id,
        ),
        "issuer_months": upsert(
            conn,
            schema.issuer_classification,
            issuer_months,
            key=["issuer_id", "vendor_id", "as_of_date"],
            batch_id=batch_id,
        ),
    }


def _newest_month(conn: Connection) -> dict[int, dt.date]:
    """``{listing_id: the newest month end it has analytics for}``."""
    t = schema.bond_analytics
    rows = conn.execute(sa.select(t.c.listing_id, sa.func.max(t.c.as_of_date)).group_by(t.c.listing_id))
    return {int(listing): day for listing, day in rows}


def _drop_leavers(conn: Connection, index_id: int, as_of: dt.date, members: set[int]) -> None:
    """Remove the index's members on ``as_of`` that a corrected file no longer lists."""
    t = schema.index_constituent
    held = {
        int(v)
        for (v,) in conn.execute(sa.select(t.c.listing_id).where(t.c.index_id == index_id, t.c.as_of_date == as_of))
    }
    gone = sorted(held - members)
    for start in range(0, len(gone), DELETE_CHUNK):
        conn.execute(
            sa.delete(t).where(
                t.c.index_id == index_id,
                t.c.as_of_date == as_of,
                t.c.listing_id.in_(gone[start : start + DELETE_CHUNK]),
            )
        )
