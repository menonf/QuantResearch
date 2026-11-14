"""Load the SEC's filing index for every company with a CIK, and measure the text of their annual reports.

Steps:

1. **Indexes** - land each CIK's EDGAR submissions index (and the older pages it points to) as raw batches; an index
   is fetched again when the landed one is older than ``max_age_days``, an older page never (it lists only filings
   long past).
2. **Documents** - ``fund.document``: every 10-K, 10-Q and 8-K (amendments too) filed since ``since``, with the
   instant EDGAR accepted it and an 8-K's item codes; each row names the index batch it came from.
3. **Annual reports** - land the primary document of every original annual report (10-K, 10-K405, 10-KT) since
   ``since``, gzip-compressed (the bytes are EDGAR's, unchanged once decompressed) - a few threads share the SEC's
   rate limit; a document already landed is never fetched again.

   Indexes and reports are downloaded into a staging folder first and landed afterwards, 500 to a transaction: the
   hours of downloading do no database work, so a serverless database can pause through them (its free compute is
   counted by the second), and a run that stops lands what it downloaded next time.
4. **Dictionary** - land the Loughran-McDonald master dictionary (vendor ``sraf``) when none is landed.
5. **Measures** - ``fund.text_measure``: for each annual report, the word counts of each Loughran-McDonald category
   in the whole report, its MD&A and its risk factors, and the similarity of each to the company's previous annual
   report (within :data:`PREVIOUS_DAYS`). Companies are measured in parallel processes; a company with a report not
   yet measured is measured again in full (its similarities need the report before).
6. **Tone** (``tone=True``, dependency group ``text``) - FinBERT's mean probabilities of positive, negative and
   neutral over sentences sampled from each MD&A (:func:`measure_tone`), kept in a local checkpoint while it runs
   and written at the end.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.engine import wake
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import UpsertResult, upsert
from equity_research.ingest.vendors import sraf
from equity_research.ingest.vendors.sec import VENDOR, text
from equity_research.ingest.vendors.sec import documents as docs
from equity_research.ingest.vendors.sec.client import SUBMISSIONS_PAGE_URL, SUBMISSIONS_URL, SecClient
from equity_research.raw.store import Batch, get_batch, land_bytes, land_file, mark
from equity_research.settings import get_settings

log = logging.getLogger(__name__)

INDEX_DATASET = "submissions"
ANNUAL_DATASET = "annual_report"
DICTIONARY_DATASET = "lm_dictionary"
ISSUERS_PER_TRANSACTION = 100
DOCUMENT_THREADS = 4
LAND_PER_TRANSACTION = 500
#: A report is compared with the company's previous annual report only when it came at most this long before.
PREVIOUS_DAYS = 550


def issuer_ciks(conn: Connection, issuers: Sequence[int] | None = None) -> pd.DataFrame:
    """``issuer_id``, ``cik`` of every company with a CIK (or of ``issuers`` only)."""
    i = schema.identifier
    query = sa.select(i.c.entity_id.label("issuer_id"), i.c.value.label("cik")).where(
        i.c.entity_type == "issuer", i.c.id_type == "CIK"
    )
    if issuers is not None:
        query = query.where(i.c.entity_id.in_([int(x) for x in issuers]))
    frame = pd.DataFrame(conn.execute(query).fetchall(), columns=["issuer_id", "cik"])
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce")
    return frame.dropna().astype(int).drop_duplicates().reset_index(drop=True)


def landed(conn: Connection, dataset: str, vendor: str = VENDOR) -> pd.DataFrame:
    """The latest batch per request (``source``) of a dataset: ``batch_id``, ``fetched_at``, ``raw_path`` by source."""
    b = schema.batch
    rows = conn.execute(
        sa.select(b.c.source, b.c.batch_id, b.c.fetched_at, b.c.raw_path).where(
            b.c.vendor_id == vendor_id(vendor), b.c.dataset == dataset
        )
    ).fetchall()
    frame = pd.DataFrame(rows, columns=["source", "batch_id", "fetched_at", "raw_path"])
    return frame.sort_values("batch_id").drop_duplicates("source", keep="last").set_index("source")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


@dataclass(frozen=True)
class StagedIndexes:
    """Indexes downloaded but not yet landed, and what each CIK's index is made of."""

    plan: dict[int, list[str]]  # CIK: the requests of its index, main first
    staged: dict[str, Path]  # request: file waiting to land
    known: dict[str, Batch]  # request: batch already landed
    unknown: list[int]  # CIKs EDGAR does not know


def land_indexes(engine: Engine, staged: StagedIndexes) -> dict[int, list[Batch]]:
    """Land the staged indexes; each CIK's index batches (main first, then older pages)."""
    new = _land_staged(engine, staged.staged, INDEX_DATASET)
    return {cik: [new.get(u) or staged.known[u] for u in urls] for cik, urls in staged.plan.items()}


def stage_indexes(engine: Engine, client: SecClient, ciks: Iterable[int], *, max_age_days: float = 30) -> StagedIndexes:
    """Download each CIK's index (and older pages) not landed, or landed more than ``max_age_days`` ago; no writes."""
    with engine.connect() as conn:
        have = landed(conn, INDEX_DATASET)
    raw_dir = get_settings().raw_dir
    staging = _staging(INDEX_DATASET)
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    cutoff = pd.Timestamp(now - dt.timedelta(days=max_age_days))
    batch_ids = {str(k): int(v) for k, v in have["batch_id"].items()}
    paths = {str(k): str(v) for k, v in have["raw_path"].items()}
    fetched = {str(k): pd.Timestamp(str(v)) for k, v in have["fetched_at"].items()}
    staged: dict[str, Path] = {}
    unknown: list[int] = []

    def local(url: str, name: str, *, page: bool) -> Path | None:
        """The index file to read: landed (fresh, or an older page), staged by an earlier run, or fetched now."""
        if url in batch_ids and (page or fetched[url] >= cutoff):
            return raw_dir / paths[url]
        target = staging / name
        if target.exists() and (page or pd.Timestamp(dt.datetime.fromtimestamp(target.stat().st_mtime)) >= cutoff):
            staged[url] = target
            return target
        try:
            target.write_bytes(client.get(url).content)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        staged[url] = target
        return target

    plan: dict[int, list[str]] = {}
    ciks = sorted(set(ciks))
    for n, cik in enumerate(ciks, start=1):
        url = SUBMISSIONS_URL.format(cik=cik)
        main = local(url, f"CIK{cik:010d}.json", page=False)
        if main is None:
            unknown.append(cik)
            continue
        urls = [url]
        for name in docs.older_pages(_read_json(main)):
            page_url = SUBMISSIONS_PAGE_URL.format(name=name)
            if local(page_url, name, page=True) is not None:
                urls.append(page_url)
        plan[cik] = urls
        if n % 250 == 0:
            log.info("SEC indexes: %d / %d CIKs", n, len(ciks))
    known = {
        u: Batch(batch_ids[u], VENDOR, INDEX_DATASET, raw_dir / paths[u], "", None, False)
        for urls in plan.values()
        for u in urls
        if u not in staged
    }
    return StagedIndexes(plan, staged, known, unknown)


def _staging(dataset: str) -> Path:
    """Where files wait between their download and their landing (a run that stops lands them next time)."""
    path = get_settings().raw_dir / "_staging" / "sec_text" / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def _land_staged(engine: Engine, staged: dict[str, Path], dataset: str) -> dict[str, Batch]:
    """Land downloaded files, :data:`LAND_PER_TRANSACTION` at a time: ``{request: batch}``.

    Downloads run with no database work between them, so a serverless database can pause through hours of them
    and the landings come in short bursts.
    """
    out: dict[str, Batch] = {}
    items = sorted(staged.items())
    for start in range(0, len(items), LAND_PER_TRANSACTION):
        chunk = items[start : start + LAND_PER_TRANSACTION]
        # files are copied, and the staged ones removed only once their batches are committed: a transaction that
        # fails (Azure drops an hour-long session now and then) leaves them staged, and the chunk is tried again
        for attempt in (1, 2):
            try:
                with engine.begin() as conn:
                    landed_now = {
                        url: land_file(conn, vendor=VENDOR, dataset=dataset, source=url, path=path)
                        for url, path in chunk
                    }
                break
            except sa.exc.OperationalError:
                if attempt == 2:
                    raise
                log.warning("SEC %s: connection lost while landing - trying the chunk again", dataset)
                wake(engine)
        out.update(landed_now)
        for _, path in chunk:
            path.unlink(missing_ok=True)
    if items:
        log.info("SEC %s: %d files landed", dataset, len(items))
    return out


def load_documents(
    engine: Engine, links: pd.DataFrame, indexes: dict[int, list[Batch]], *, since: str = "2003-01-01"
) -> UpsertResult:
    """``fund.document`` rows of every company from its CIKs' indexes, ``ISSUERS_PER_TRANSACTION`` at a time."""
    total = UpsertResult()
    by_issuer = {int(str(i)): [int(c) for c in g["cik"]] for i, g in links.groupby("issuer_id")}
    companies = sorted(by_issuer)
    for start in range(0, len(companies), ISSUERS_PER_TRANSACTION):
        parts, used = [], []
        for issuer_id in companies[start : start + ISSUERS_PER_TRANSACTION]:
            for cik in by_issuer[issuer_id]:
                for batch in indexes.get(cik, []):
                    rows = docs.filings(_read_json(batch.path), cik, since=since)
                    parts.append(rows.assign(issuer_id=issuer_id, vendor_id=vendor_id(VENDOR), batch_id=batch.batch_id))
                    used.append(batch.batch_id)
        with engine.begin() as conn:
            if parts:
                frame = pd.concat(parts, ignore_index=True)
                total += upsert(conn, schema.document, frame, key=["issuer_id", "vendor_id", "filing_ref"])
            for batch_id in sorted(set(used)):
                mark(conn, batch_id, "normalized")
    return total


def _annual_documents(conn: Connection, since: str, issuers: Sequence[int] | None = None) -> pd.DataFrame:
    d = schema.document
    query = sa.select(d.c.issuer_id, d.c.filing_ref, d.c.cik, d.c.published_at, d.c.primary_document).where(
        d.c.vendor_id == vendor_id(VENDOR),
        d.c.form.in_(docs.ANNUAL_FORMS),
        d.c.filing_date >= pd.Timestamp(since).date(),
        d.c.primary_document.is_not(None),
    )
    if issuers is not None:
        query = query.where(d.c.issuer_id.in_([int(x) for x in issuers]))
    frame = pd.DataFrame(
        conn.execute(query).fetchall(), columns=["issuer_id", "filing_ref", "cik", "published_at", "primary_document"]
    )
    frame["url"] = [
        docs.document_url(c, f, p)
        for c, f, p in zip(frame["cik"], frame["filing_ref"], frame["primary_document"], strict=True)
    ]
    return frame


def stage_annual_reports(
    engine: Engine, client: SecClient, *, since: str = "2003-01-01", issuers: Sequence[int] | None = None
) -> tuple[dict[str, Path], dict[str, int]]:
    """Download every annual report not yet landed, with no writes.

    Returns ``{request: staged file}`` and how many were downloaded, already landed and missing at EDGAR.
    """
    with engine.connect() as conn:
        wanted = _annual_documents(conn, since, issuers)
        have = landed(conn, ANNUAL_DATASET)
    # A combined report (a parent and its subsidiary) is one filing under two CIKs: it is fetched and landed once.
    done = set(wanted.loc[wanted["url"].isin(have.index), "filing_ref"])
    todo = wanted[~wanted["filing_ref"].isin(done)].drop_duplicates("filing_ref")
    log.info("SEC annual reports: %d wanted, %d to fetch", len(wanted), len(todo))
    staging = _staging(ANNUAL_DATASET)

    def fetch(url: str, filing_ref: str) -> Path | None:
        target = staging / f"{filing_ref}.htm.gz"
        if target.exists():  # downloaded by a run that stopped before landing it
            return target
        try:
            content = client.get(url).content
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        part = target.with_suffix(".part")
        part.write_bytes(gzip.compress(content, mtime=0))
        part.replace(target)
        return target

    staged: dict[str, Path] = {}
    missing = 0
    with ThreadPoolExecutor(DOCUMENT_THREADS) as pool:
        results = pool.map(fetch, todo["url"], todo["filing_ref"])
        for n, (url, path) in enumerate(zip(todo["url"], results, strict=True), start=1):
            if path is None:
                missing += 1
            else:
                staged[str(url)] = path
            if n % 500 == 0:
                log.info("SEC annual reports: %d / %d fetched", n, len(todo))
    return staged, {"landed": len(staged), "already": len(done), "missing": missing}


def land_dictionary(engine: Engine) -> Batch:
    """The Loughran-McDonald dictionary's batch, landing it first when none is."""
    with engine.begin() as conn:
        have = landed(conn, DICTIONARY_DATASET, vendor=sraf.VENDOR)
        if len(have):
            return get_batch(conn, int(have["batch_id"].max()))
        payload = sraf.fetch_dictionary()
        return land_bytes(
            conn,
            vendor=sraf.VENDOR,
            dataset=DICTIONARY_DATASET,
            source=sraf.LM_DICTIONARY_URL,
            payload=payload,
            name="Loughran-McDonald_MasterDictionary_1993-2025.csv",
        )


_DICTIONARY: dict[str, frozenset[str]] = {}


def _load_dictionary(path: str) -> None:
    _DICTIONARY.clear()
    _DICTIONARY.update(text.lm_dictionary(Path(path).read_bytes()))


def company_measures(reports: list[tuple[str, pd.Timestamp, str, int]]) -> list[tuple[str, str, str, str, float, int]]:
    """``(filing_ref, section, measure, method, value, batch_id)`` of one company's reports, oldest first.

    Runs in a worker process (the dictionary loaded by :func:`_load_dictionary`).
    """
    out: list[tuple[str, str, str, str, float, int]] = []
    previous: dict[str, Any] | None = None
    previous_at: pd.Timestamp | None = None
    for filing_ref, published_at, path, batch_id in sorted(reports, key=lambda r: r[1]):
        counts = text.report_counts(gzip.decompress(Path(path).read_bytes()))
        recent = previous_at is not None and (published_at - previous_at).days <= PREVIOUS_DAYS
        rows = text.measures(counts, previous if recent else None, _DICTIONARY)
        out += [(filing_ref, s, m, method, v, batch_id) for s, m, method, v in rows]
        previous, previous_at = counts, published_at
    return out


def _landed_by_filing(reports: pd.DataFrame, have: pd.DataFrame) -> pd.DataFrame:
    """Each filing's landed file, whichever of its CIKs' addresses it was fetched from: ``filing_ref``, batch, path."""
    found = reports[["filing_ref", "url"]].join(have[["batch_id", "raw_path"]], on="url", how="inner")
    return found.drop(columns="url").drop_duplicates("filing_ref")


def measure_reports(
    engine: Engine,
    dictionary: Batch,
    *,
    since: str = "2003-01-01",
    issuers: Sequence[int] | None = None,
    workers: int | None = None,
) -> UpsertResult:
    """``fund.text_measure`` for every landed annual report of a company with one not yet measured."""
    raw_dir = get_settings().raw_dir
    with engine.connect() as conn:
        reports = _annual_documents(conn, since, issuers)
        have = landed(conn, ANNUAL_DATASET)
        t = schema.text_measure
        done = pd.DataFrame(
            conn.execute(
                sa.select(t.c.issuer_id, t.c.filing_ref)
                .where(t.c.method == text.LM_METHOD, t.c.section == "all", t.c.measure == "words")
                .distinct()
            ).fetchall(),
            columns=["issuer_id", "filing_ref"],
        )
    reports = reports.merge(_landed_by_filing(reports, have), on="filing_ref", how="inner")
    measured = set(zip(done["issuer_id"], done["filing_ref"], strict=True))
    pending = {i for i, f in zip(reports["issuer_id"], reports["filing_ref"], strict=True) if (i, f) not in measured}
    jobs = {
        int(str(i)): [
            (str(f), pd.Timestamp(str(at)), str(raw_dir / str(path)), int(b))
            for f, at, path, b in zip(g["filing_ref"], g["published_at"], g["raw_path"], g["batch_id"], strict=True)
        ]
        for i, g in reports[reports["issuer_id"].isin(pending)].groupby("issuer_id")
    }
    log.info("SEC text measures: %d companies to measure", len(jobs))
    total = UpsertResult()
    companies = sorted(jobs)
    with ProcessPoolExecutor(
        workers or max(1, (os.cpu_count() or 2) - 1), initializer=_load_dictionary, initargs=(str(dictionary.path),)
    ) as pool:
        for start in range(0, len(companies), ISSUERS_PER_TRANSACTION):
            chunk = companies[start : start + ISSUERS_PER_TRANSACTION]
            parts = []
            for issuer_id, rows in zip(chunk, pool.map(company_measures, [jobs[i] for i in chunk]), strict=True):
                frame = pd.DataFrame(rows, columns=["filing_ref", "section", "measure", "method", "value", "batch_id"])
                parts.append(frame.assign(issuer_id=issuer_id, vendor_id=vendor_id(VENDOR)))
            with engine.begin() as conn:
                if parts:
                    total += upsert(
                        conn,
                        schema.text_measure,
                        pd.concat(parts, ignore_index=True),
                        key=["issuer_id", "vendor_id", "filing_ref", "section", "measure"],
                    )
                for batch_id in sorted({b for i in chunk for *_, b in jobs[i]}):
                    mark(conn, batch_id, "normalized")
            log.info("SEC text measures: %d / %d companies", min(start + len(chunk), len(companies)), len(companies))
    with engine.begin() as conn:
        mark(conn, dictionary.batch_id, "normalized")
    return total


def load_sec_text(
    engine: Engine,
    *,
    client: SecClient | None = None,
    since: str = "2003-01-01",
    max_age_days: float = 30,
    issuers: Sequence[int] | None = None,
    workers: int | None = None,
    tone: bool = False,
    tone_device: str = "cpu",
) -> dict[str, Any]:
    """Run every step (see the module docstring); ``issuers`` limits a trial to some companies. Returns a report.

    Downloads and scoring run outside the writer lock (whose keep-alive would hold a serverless database awake for
    their hours); each write phase takes the lock as a run of its own.
    """
    client = client or SecClient()
    with engine.connect() as conn:
        links = issuer_ciks(conn, issuers)
    staged = stage_indexes(engine, client, links["cik"], max_age_days=max_age_days)
    report: dict[str, Any] = {"ciks": len(links), "ciks_unknown_to_edgar": len(staged.unknown)}
    wake(engine)  # the downloads took long enough for a serverless database to pause
    with ops.job("load_sec_text", engine) as run:
        documents = load_documents(engine, links, land_indexes(engine, staged), since=since)
        report["documents"] = documents.__dict__
        run.finish(rows=documents.inserted, detail=dict(report))
    reports, report["annual_reports"] = stage_annual_reports(engine, client, since=since, issuers=issuers)
    wake(engine)
    with ops.job("load_sec_text_measures", engine) as run:
        _land_staged(engine, reports, ANNUAL_DATASET)
        dictionary = land_dictionary(engine)
        measures = measure_reports(engine, dictionary, since=since, issuers=issuers, workers=workers)
        report["measures"] = measures.__dict__
        run.finish(rows=measures.inserted, detail=dict(report))
    if tone:
        scores = score_tone(engine, since=since, issuers=issuers, workers=workers, device=tone_device)
        wake(engine)
        with ops.job("load_sec_text_tone", engine) as run:
            written = write_tone(engine, scores)
            report["tone"] = written.__dict__
            run.finish(rows=written.inserted, detail={"tone": written.__dict__})
    return report


#: FinBERT (Araci 2019, as ProsusAI publishes it on Hugging Face), pinned to one revision so a re-run scores alike.
FINBERT_MODEL = "ProsusAI/finbert"
TONE_METHOD = "finbert-prosus"
TONE_SENTENCES = 32


def _mdna_sentences(job: tuple[str, int]) -> list[str]:
    """The sentences FinBERT reads from one report: its MD&A's sampled prose (in a worker process)."""
    path, n = job
    plain = text.plain_text(gzip.decompress(Path(path).read_bytes()))
    return text.sample_sentences(text.section(plain, text.MDNA_START, text.MDNA_END), n)


def measure_tone(
    engine: Engine,
    *,
    since: str = "2003-01-01",
    issuers: Sequence[int] | None = None,
    sentences: int = TONE_SENTENCES,
    batch_size: int = 32,
    revision: str | None = None,
    device: str = "cpu",
) -> UpsertResult:
    """FinBERT's mean tone over ``sentences`` sampled from each annual report's MD&A not yet scored, written."""
    frame = score_tone(
        engine,
        since=since,
        issuers=issuers,
        sentences=sentences,
        batch_size=batch_size,
        revision=revision,
        device=device,
    )
    return write_tone(engine, frame)


def write_tone(engine: Engine, frame: pd.DataFrame) -> UpsertResult:
    """Write :func:`score_tone`'s rows and drop its checkpoint."""
    total = UpsertResult()
    for start in range(0, len(frame), 50_000):
        with engine.begin() as conn:
            total += upsert(
                conn,
                schema.text_measure,
                frame.iloc[start : start + 50_000],
                key=["issuer_id", "vendor_id", "filing_ref", "section", "measure"],
            )
    (_staging("tone") / f"{TONE_METHOD}.parquet").unlink(missing_ok=True)
    return total


def score_tone(
    engine: Engine,
    *,
    since: str = "2003-01-01",
    issuers: Sequence[int] | None = None,
    sentences: int = TONE_SENTENCES,
    batch_size: int = 32,
    revision: str | None = None,
    workers: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """FinBERT's tone of each annual report's MD&A not yet scored, as ``text_measure`` rows; reads, never writes.

    ``device`` is where the model runs (``"cuda"``: a GPU, with a CUDA build of torch); a GPU's scores differ from the
    CPU's only by rounding.
    """
    import torch  # noqa: PLC0415 - dependency groups deep / text
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    raw_dir = get_settings().raw_dir
    with engine.connect() as conn:
        reports = _annual_documents(conn, since, issuers)
        have = landed(conn, ANNUAL_DATASET)
        t = schema.text_measure
        done = pd.DataFrame(
            conn.execute(
                sa.select(t.c.issuer_id, t.c.filing_ref).where(t.c.method == TONE_METHOD).distinct()
            ).fetchall(),
            columns=["issuer_id", "filing_ref"],
        )
    reports = reports.merge(_landed_by_filing(reports, have), on="filing_ref", how="inner")
    measured = set(zip(done["issuer_id"], done["filing_ref"], strict=True))
    reports = reports[
        [(i, f) not in measured for i, f in zip(reports["issuer_id"], reports["filing_ref"], strict=True)]
    ]
    # scores wait in a local checkpoint, written to the database at the end: hours of scoring keep no database awake,
    # and a run that stops resumes from the checkpoint
    checkpoint = _staging("tone") / f"{TONE_METHOD}.parquet"
    rows: list[dict[str, Any]] = []
    if checkpoint.exists():
        rows = [{str(k): v for k, v in r.items()} for r in pd.read_parquet(checkpoint).to_dict("records")]
    scored = {(int(r["issuer_id"]), str(r["filing_ref"])) for r in rows}
    reports = reports[
        [(int(i), str(f)) not in scored for i, f in zip(reports["issuer_id"], reports["filing_ref"], strict=True)]
    ]
    log.info("FinBERT: %d reports to score", len(reports))
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL, revision=revision).eval().to(device)
    labels = [str(model.config.id2label[k]).lower() for k in range(model.config.num_labels)]
    torch.manual_seed(0)
    # reading a 10-K's HTML takes longer than scoring its sentences: worker processes read ahead while the model runs
    jobs = [(str(raw_dir / str(p)), sentences) for p in reports["raw_path"]]
    pool = ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) // 2))
    read = pool.map(_mdna_sentences, jobs, chunksize=4)
    try:
        rows = _score(reports, read, rows, checkpoint, tokenizer, model, labels, batch_size, device)
    finally:
        pool.shutdown(cancel_futures=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(checkpoint)
    return frame


def _score(  # noqa: PLR0917 - the scoring loop of score_tone, apart so its reader pool always shuts down
    reports: pd.DataFrame,
    read: Iterable[list[str]],
    rows: list[dict[str, Any]],
    checkpoint: Path,
    tokenizer: Any,
    model: Any,
    labels: list[str],
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    for n, (r, picked) in enumerate(zip(reports.itertuples(index=False), read, strict=True), start=1):
        if picked:
            probs = []
            with torch.inference_mode():
                for i in range(0, len(picked), batch_size):
                    enc = tokenizer(picked[i : i + batch_size], padding=True, truncation=True, max_length=128,
                                    return_tensors="pt").to(device)  # fmt: skip
                    probs.append(torch.softmax(model(**enc).logits.float(), dim=-1))
            mean = torch.cat(probs).mean(dim=0).cpu().tolist()
            base = {"issuer_id": int(str(r.issuer_id)), "vendor_id": vendor_id(VENDOR), "filing_ref": str(r.filing_ref),
                    "section": "mdna", "method": TONE_METHOD, "batch_id": int(str(r.batch_id))}  # fmt: skip
            rows += [
                base | {"measure": f"finbert_{lab}", "value": float(p)} for lab, p in zip(labels, mean, strict=True)
            ]
            rows.append(base | {"measure": "finbert_sentences", "value": float(len(picked))})
        if n % 200 == 0:
            pd.DataFrame(rows).to_parquet(checkpoint)
            log.info("FinBERT: %d / %d reports", n, len(reports))
    return rows
