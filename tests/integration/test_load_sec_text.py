"""The SEC text load end to end on a scratch database, EDGAR and the dictionary's host replaced by fakes.

One company files two annual reports a year apart (one listed in its index's older page) and an 8-K; a second CIK is
unknown to EDGAR; a third report's document is gone from EDGAR; a fourth company files its 2023 report combined with the
first's (one filing under two CIKs). A second run fetches nothing again and measures nothing again.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import requests
import sqlalchemy as sa
from conftest import scratch_database
from sqlalchemy.engine import Engine

from equity_research.ingest.identity import add_identifiers, ensure_listings
from equity_research.ingest.vendors import sraf
from equity_research.ingest.vendors.sec.client import SecClient
from equity_research.pipelines import load_sec_text as pipe

pytestmark = pytest.mark.integration

SCRATCH = "equity_research_test_sec_text"
LM_CSV = (
    b"Word,Seq_num,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining\n"
    b"LOSS,1,2009,0,0,0,0,0,0\nGAIN,2,0,2009,0,0,0,0,0\nMAY,3,0,0,2009,0,0,2009,0\n"
)
BASE = "https://www.sec.gov/Archives/edgar/data/1001"
SHARED = "https://www.sec.gov/Archives/edgar/data/1003/000100124000002/r2023.htm"  # the combined report under 1003


def _report(word: str) -> bytes:
    risk = " ".join([f"A {word} may occur"] * 100)
    mdna = " ".join(["Revenue grew while costs may rise and margins held"] * 40)
    return (
        f"<html><body><p>Item 1A. Risk Factors</p><p>{risk}</p><p>Item 1B. Unresolved Staff Comments</p>"
        f"<p>Item 7. Management's Discussion and Analysis</p><p>{mdna}</p><p>Item 8. Financial Statements</p>"
        "</body></html>"
    ).encode()


MAIN = {
    "cik": "1001",
    "filings": {
        "recent": {
            "accessionNumber": ["0001001-24-000002", "0001001-24-000003", "0001001-23-000009"],
            "filingDate": ["2024-02-20", "2024-04-30", "2023-11-01"],
            "reportDate": ["2023-12-31", "2024-04-30", "2023-09-30"],
            "acceptanceDateTime": ["2024-02-20T21:05:00.000Z", "2024-04-30T12:00:00.000Z", "2023-11-01T16:00:00.000Z"],
            "form": ["10-K", "8-K", "10-K"],
            "items": ["", "5.02,9.01", ""],
            "primaryDocument": ["r2023.htm", "e.htm", "gone.htm"],
        },
        "files": [{"name": "CIK0000001001-submissions-001.json"}],
    },
}
COMBINED = {
    "cik": "1003",
    "filings": {
        "recent": {
            "accessionNumber": ["0001001-24-000002"],
            "filingDate": ["2024-02-20"],
            "reportDate": ["2023-12-31"],
            "acceptanceDateTime": ["2024-02-20T21:05:00.000Z"],
            "form": ["10-K"],
            "items": [""],
            "primaryDocument": ["r2023.htm"],
        },
        "files": [],
    },
}
PAGE = {
    "accessionNumber": ["0001001-23-000001"],
    "filingDate": ["2023-02-21"],
    "reportDate": ["2022-12-31"],
    "acceptanceDateTime": ["2023-02-21T21:00:00.000Z"],
    "form": ["10-K"],
    "items": [""],
    "primaryDocument": ["r2022.htm"],
}


class _Answer:
    def __init__(self, content: bytes) -> None:
        self.content = content


class FakeSec:
    """Answers EDGAR's index, its older page and the annual reports; counts the requests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, **_: Any) -> _Answer:
        import json  # noqa: PLC0415

        self.calls.append(url)
        answers = {
            "https://data.sec.gov/submissions/CIK0000001001.json": json.dumps(MAIN).encode(),
            "https://data.sec.gov/submissions/CIK0000001001-submissions-001.json": json.dumps(PAGE).encode(),
            "https://data.sec.gov/submissions/CIK0000001003.json": json.dumps(COMBINED).encode(),
            f"{BASE}/000100124000002/r2023.htm": _report("gain"),
            SHARED: _report("gain"),
            f"{BASE}/000100123000001/r2022.htm": _report("loss"),
        }
        if url not in answers:
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError(f"404 {url}", response=response)
        return _Answer(answers[url])


@pytest.fixture(scope="module")
def loaded(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Engine, FakeSec, dict[str, Any], dict[str, Any]]]:
    mp = pytest.MonkeyPatch()
    mp.setattr(sraf, "fetch_dictionary", lambda *a, **k: LM_CSV)
    with scratch_database(SCRATCH, tmp_path_factory.mktemp("sec_text_data")) as engine:
        with engine.begin() as conn:
            ids = ensure_listings(
                conn,
                pd.DataFrame(
                    {
                        "anchor": ["AAA.N", "BBB.N", "CCC.N"],
                        "issuer_anchor": ["000001", "000002", "000003"],
                        "issuer_name": ["Alpha Inc", "Beta Inc", "Gamma Inc"],
                        "name": ["Alpha", "Beta", "Gamma"],
                        "instrument_type": "equity",
                        "currency": "USD",
                    }
                ),
                anchor_type="RIC",
                issuer_anchor_type="CUSIP",
                vendor="refinitiv",
                batch_id=None,
            )
            issuers = conn.execute(sa.text("SELECT issuer_id, name FROM ref.issuer")).all()
            by_name = {str(n): int(i) for i, n in issuers}
            add_identifiers(
                conn,
                pd.DataFrame(
                    {
                        "entity_type": "issuer",
                        "entity_id": [by_name["Alpha Inc"], by_name["Beta Inc"], by_name["Gamma Inc"]],
                        "id_type": "CIK",
                        "value": ["1001", "1002", "1003"],  # EDGAR does not know 1002
                        "valid_from": None,
                        "valid_to": None,
                        "vendor_id": 2,
                    }
                ),
                batch_id=None,
            )
            assert ids.listing_ids
        fake = FakeSec()
        client = cast(SecClient, fake)
        first = pipe.load_sec_text(engine, client=client, since="2003-01-01", workers=1)
        n_calls = len(fake.calls)
        second = pipe.load_sec_text(engine, client=client, since="2003-01-01", workers=1)
        # only what EDGAR lacked is asked again: the unknown CIK's index and the missing report
        assert fake.calls[n_calls:] == [
            "https://data.sec.gov/submissions/CIK0000001002.json",
            f"{BASE}/000100123000009/gone.htm",
        ]
        yield engine, fake, first, second
    mp.undo()


def _frame(engine: Engine, sql: str) -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql(sa.text(sql), c)


def test_documents_come_with_their_acceptance_instant_and_items(loaded: Any) -> None:
    engine, _, first, _ = loaded
    docs = _frame(
        engine, "SELECT filing_ref, form, published_at, items, batch_id FROM fund.document ORDER BY filing_ref"
    )
    assert list(docs["form"]) == ["10-K", "10-K", "10-K", "10-K", "8-K"]  # the combined report under both companies
    assert docs.set_index("filing_ref").loc["0001001-24-000003", "items"] == "5.02,9.01"
    shared = docs[docs["filing_ref"] == "0001001-24-000002"]
    assert list(shared["published_at"]) == [pd.Timestamp("2024-02-20 21:05")] * 2
    assert docs["batch_id"].notna().all() and docs["batch_id"].nunique() == 3  # two indexes and an older page
    assert first["ciks_unknown_to_edgar"] == 1
    assert first["annual_reports"] == {"landed": 2, "already": 0, "missing": 1}  # the combined report once


def test_each_report_is_measured_and_compared_with_the_year_before(loaded: Any) -> None:
    engine, *_ = loaded
    every = _frame(
        engine, "SELECT issuer_id, filing_ref, section, measure, method, value, batch_id FROM fund.text_measure"
    )
    alpha = _frame(engine, "SELECT issuer_id FROM ref.issuer WHERE name = 'Alpha Inc'")["issuer_id"].iloc[0]
    m = every[every["issuer_id"] == alpha]
    neg = m[(m["section"] == "risk") & (m["measure"] == "negative")].set_index("filing_ref")["value"]
    assert neg.to_dict() == {"0001001-23-000001": 100.0, "0001001-24-000002": 0.0}
    sim = m[m["measure"] == "sim_cosine"].set_index(["filing_ref", "section"])["value"]
    assert set(sim.index.get_level_values(0)) == {"0001001-24-000002"}  # the first report has none before it
    assert sim.loc[("0001001-24-000002", "mdna")] == pytest.approx(1.0)  # the same MD&A
    assert sim.loc[("0001001-24-000002", "risk")] < 1.0  # "loss" became "gain"
    batches = _frame(engine, "SELECT batch_id, dataset, status, raw_path FROM ingest.batch")
    reports = batches[batches["dataset"] == pipe.ANNUAL_DATASET]
    assert set(every["batch_id"]) == set(reports["batch_id"])  # each measure names its document's batch
    assert (batches["status"] == "normalized").all()
    assert all(str(p).endswith(".htm.gz") for p in reports["raw_path"])


def test_a_second_run_fetches_and_measures_nothing_again(loaded: Any) -> None:
    _, fake, first, second = loaded
    reports = [u for u in fake.calls if u.startswith(BASE)]
    assert reports.count(f"{BASE}/000100124000002/r2023.htm") + fake.calls.count(SHARED) == 1  # once, never again
    assert second["annual_reports"]["landed"] == 0 and second["annual_reports"]["already"] == 2
    assert second["measures"]["inserted"] == 0 and first["measures"]["inserted"] > 0
    assert second["documents"]["inserted"] == 0


def test_a_combined_report_is_measured_for_both_companies(loaded: Any) -> None:
    engine, *_ = loaded
    m = _frame(
        engine,
        "SELECT i.name, t.value FROM fund.text_measure t JOIN ref.issuer i ON i.issuer_id = t.issuer_id"
        " WHERE t.filing_ref = '0001001-24-000002' AND t.section = 'risk' AND t.measure = 'negative'",
    )
    assert dict(zip(m["name"], m["value"], strict=True)) == {"Alpha Inc": 0.0, "Gamma Inc": 0.0}


def test_raw_reports_are_edgars_bytes_compressed(loaded: Any, tmp_path: Path) -> None:
    engine, *_ = loaded
    from equity_research.settings import get_settings  # noqa: PLC0415

    path = _frame(engine, f"SELECT raw_path FROM ingest.batch WHERE dataset = '{pipe.ANNUAL_DATASET}'")["raw_path"]
    assert len(path) == 2
    contents = {gzip.decompress((get_settings().raw_dir / p).read_bytes()) for p in path}
    assert contents == {_report("gain"), _report("loss")}
