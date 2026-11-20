"""Land vendor responses unchanged and register them in ``ingest.batch``.

    batch = land_file(conn, vendor="sec", dataset="companyfacts", source=url, path=downloaded_zip)
    batch = land_frame(conn, vendor="refinitiv", dataset="prices", source="lseg:prices:2015:...", frame=df)

Files live under ``<raw zone>/<vendor>/<dataset>/<yyyy>/<mm>/<sha256[:16]>-<name>`` and are never modified. The
SHA-256 of the content is part of the batch key: landing the same content again returns the existing batch instead
of a duplicate - re-running a load never re-fetches or re-lands what is already there.

Files a vendor hands out again for free at any time (the SEC's monthly bulk file) need not be kept forever:
:func:`prune` removes all but the newest of them. Their ``ingest.batch`` rows stay, marked
``superseded``, so every canonical row still names the delivery it came from.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection

from research_core.data.db import schema
from research_core.data.db.schema import vendor_code, vendor_id
from research_core.settings import REPO_ROOT, get_settings

CHUNK = 8 << 20


@dataclass(frozen=True)
class Batch:
    """A landed raw file."""

    batch_id: int
    vendor: str
    dataset: str
    path: Path
    checksum: str
    rows: int | None
    is_new: bool


def code_version() -> str | None:
    """The git commit of the running code (``None`` outside a repository)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _target(raw_dir: Path, vendor: str, dataset: str, *, checksum: str, name: str, when: dt.datetime) -> Path:
    return raw_dir / vendor / dataset / f"{when:%Y}" / f"{when:%m}" / f"{checksum[:16]}-{name}"


def _existing(conn: Connection, dataset: str, checksum: str) -> tuple[int, str, str, int | None] | None:
    """``(batch_id, vendor, raw_path, rows)`` of the batch with this content, if landed before."""
    b = schema.batch
    row = conn.execute(
        sa.select(b.c.batch_id, b.c.vendor_id, b.c.raw_path, b.c.rows).where(
            b.c.checksum == checksum, b.c.dataset == dataset
        )
    ).first()
    if row is None:
        return None
    return int(row[0]), vendor_code(int(row[1])), str(row[2]), None if row[3] is None else int(row[3])


def land_file(
    conn: Connection,
    *,
    vendor: str,
    dataset: str,
    source: str,
    path: Path,
    fetched_at: dt.datetime | None = None,
    rows: int | None = None,
    move: bool = False,
    raw_dir: Path | None = None,
) -> Batch:
    """Copy (or move) ``path`` into the raw zone and register it; an identical file returns its existing batch."""
    raw_dir = raw_dir or get_settings().raw_dir
    checksum = sha256_file(path)
    found = _existing(conn, dataset, checksum)
    if found is not None:
        return Batch(found[0], found[1], dataset, raw_dir / found[2], checksum, found[3], is_new=False)
    when = fetched_at or dt.datetime.now(dt.UTC).replace(tzinfo=None)
    target = _target(raw_dir, vendor, dataset, checksum=checksum, name=path.name, when=when)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        (shutil.move if move else shutil.copy2)(path, target)
    relative = target.relative_to(raw_dir).as_posix()
    batch_id = conn.execute(
        sa.insert(schema.batch)
        .values(
            vendor_id=vendor_id(vendor),
            dataset=dataset,
            source=source,
            raw_path=relative,
            checksum=checksum,
            bytes=target.stat().st_size,
            rows=rows,
            fetched_at=when,
            code_version=code_version(),
        )
        .returning(schema.batch.c.batch_id)
    ).scalar_one()
    return Batch(int(batch_id), vendor, dataset, target, checksum, rows, is_new=True)


def land_bytes(
    conn: Connection,
    *,
    vendor: str,
    dataset: str,
    source: str,
    payload: bytes,
    name: str,
    rows: int | None = None,
    fetched_at: dt.datetime | None = None,
    raw_dir: Path | None = None,
) -> Batch:
    """Land a vendor response held in memory (a JSON body, a serialised frame) as the file ``name``."""
    raw_dir = raw_dir or get_settings().raw_dir
    # a staging folder per content keeps concurrent landings apart; land_file adds the hash to the final name
    staging = raw_dir / "_staging" / hashlib.sha256(payload).hexdigest()[:16]
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / name
    tmp.write_bytes(payload)
    try:
        return land_file(
            conn,
            vendor=vendor,
            dataset=dataset,
            source=source,
            path=tmp,
            fetched_at=fetched_at,
            rows=rows,
            move=True,
            raw_dir=raw_dir,
        )
    finally:
        tmp.unlink(missing_ok=True)
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()


def land_frame(
    conn: Connection,
    *,
    vendor: str,
    dataset: str,
    source: str,
    frame: pd.DataFrame,
    name: str | None = None,
    fetched_at: dt.datetime | None = None,
    raw_dir: Path | None = None,
) -> Batch:
    """Land a tabular vendor response as Parquet (deterministic bytes, so identical frames dedupe)."""
    buffer = io.BytesIO()
    frame.reset_index(drop=True).to_parquet(buffer, index=False, compression="zstd")
    return land_bytes(
        conn,
        vendor=vendor,
        dataset=dataset,
        source=source,
        payload=buffer.getvalue(),
        name=f"{name or dataset}.parquet",
        rows=len(frame),
        fetched_at=fetched_at,
        raw_dir=raw_dir,
    )


def read_frame(batch: Batch) -> pd.DataFrame:
    """The table landed by :func:`land_frame`."""
    return pd.read_parquet(batch.path)


def get_batch(conn: Connection, batch_id: int, raw_dir: Path | None = None) -> Batch:
    """A registered batch by id."""
    raw_dir = raw_dir or get_settings().raw_dir
    b = schema.batch
    row = conn.execute(
        sa.select(b.c.vendor_id, b.c.dataset, b.c.raw_path, b.c.checksum, b.c.rows).where(b.c.batch_id == batch_id)
    ).one()
    return Batch(batch_id, vendor_code(int(row[0])), row[1], raw_dir / row[2], row[3], row[4], is_new=False)


def mark(conn: Connection, batch_id: int, status: str, detail: str | None = None) -> None:
    """Record what happened to a batch (``normalized``, ``failed``, ``superseded``)."""
    conn.execute(
        sa.update(schema.batch).where(schema.batch.c.batch_id == batch_id).values(status=status, detail=detail)
    )


def status(conn: Connection, batch_id: int) -> str:
    """What has happened to a batch so far (``landed``, ``normalized``, ...)."""
    return str(conn.execute(sa.select(schema.batch.c.status).where(schema.batch.c.batch_id == batch_id)).scalar_one())


def find_batch(conn: Connection, *, vendor: str, dataset: str, source: str) -> Batch | None:
    """The latest batch landed for exactly this request (``source``), if any - to resume a load without asking twice."""
    b = schema.batch
    row = conn.execute(
        sa.select(b.c.batch_id)
        .where(b.c.vendor_id == vendor_id(vendor), b.c.dataset == dataset, b.c.source == source)
        .order_by(b.c.batch_id.desc())
    ).first()
    return None if row is None else get_batch(conn, int(row[0]))


def recent_batch(conn: Connection, *, vendor: str, dataset: str, max_age_days: float) -> Batch | None:
    """The newest batch of a dataset fetched within ``max_age_days``, if any - a bulk file recent enough to reuse."""
    b = schema.batch
    cutoff = dt.datetime.now(dt.UTC).replace(tzinfo=None) - dt.timedelta(days=max_age_days)
    row = conn.execute(
        sa.select(b.c.batch_id)
        .where(b.c.vendor_id == vendor_id(vendor), b.c.dataset == dataset, b.c.fetched_at >= cutoff)
        .order_by(b.c.fetched_at.desc())
    ).first()
    return None if row is None else get_batch(conn, int(row[0]))


@dataclass(frozen=True)
class Pruned:
    """A raw file removed (or, in a dry run, that would be)."""

    batch_id: int
    vendor: str
    dataset: str
    path: Path
    bytes: int


def prune(
    conn: Connection,
    *,
    retention: dict[tuple[str, str], int],
    raw_dir: Path | None = None,
    dry_run: bool = False,
) -> list[Pruned]:
    """Remove all but the newest files of each dataset in ``retention``; their batches become ``superseded``.

    ``retention`` is ``{(vendor, dataset): files kept}`` - only for files a vendor publishes again at any time; the
    package that knows the vendors says which those are.
    """
    raw_dir = raw_dir or get_settings().raw_dir
    b = schema.batch
    out = []
    for (vendor, dataset), keep in retention.items():
        rows = conn.execute(
            sa.select(b.c.batch_id, b.c.raw_path, b.c.bytes)
            .where(b.c.vendor_id == vendor_id(vendor), b.c.dataset == dataset, b.c.status != "superseded")
            .order_by(b.c.fetched_at.desc(), b.c.batch_id.desc())
        ).all()
        for batch_id, raw_path, size in rows[keep:]:
            out.append(Pruned(int(batch_id), vendor, dataset, raw_dir / raw_path, int(size)))
            if dry_run:
                continue
            (raw_dir / raw_path).unlink(missing_ok=True)
            mark(conn, int(batch_id), "superseded", "file pruned: the vendor publishes it again at any time")
    return out
