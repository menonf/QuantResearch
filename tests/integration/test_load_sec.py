"""The SEC load end to end on a scratch database, the SEC replaced by a fake answering with its three files.

Three companies show the matching rules: Disney re-registered in 2019 (a new holding company, a new CIK) and its old
CIK fills the years before; Duke Energy's subsidiary files its own accounts for the same years and adds nothing;
Stericycle, delisted, has no ticker of today and a CUSIP shared with a holder that filed about it - the name settles
it. A London company whose ticker is Disney's US ticker gets nothing: the SEC's tickers are US ones.
"""

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import scratch_database
from sqlalchemy.engine import Engine

from equity_research.ingest.vendors.sec.client import (
    CIK_CUSIP_MAP_URL,
    COMPANY_FACTS_BULK_URL,
    COMPANY_TICKERS_URL,
    SecClient,
)
from equity_research.pipelines.load_sec import load_sec
from research_core.data.identity import add_identifiers, ensure_listings

pytestmark = pytest.mark.integration

SCRATCH = "equity_research_test_sec"


def _company(cik: int, name: str, years: range, assets: float) -> dict[str, Any]:
    facts = [
        {
            "end": f"{y}-12-31",
            "val": assets + y,
            "accn": f"{cik}-{y}",
            "fy": y,
            "fp": "FY",
            "form": "10-K",
            "filed": f"{y + 1}-02-20",
        }
        for y in years
    ]
    return {"cik": cik, "entityName": name, "facts": {"us-gaap": {"Assets": {"units": {"USD": facts}}}}}


COMPANIES = [
    _company(1001039, "WALT DISNEY CO/", range(2009, 2019), 1_000.0),  # before the 2019 re-registration
    _company(1744489, "Walt Disney Co", range(2019, 2026), 2_000.0),
    _company(1326160, "Duke Energy CORP", range(2009, 2026), 3_000.0),
    _company(30371, "Duke Energy Carolinas, LLC", range(2009, 2026), 9.0),  # a subsidiary: not Duke's figures
    _company(861878, "STERICYCLE INC", range(2009, 2024), 4_000.0),
    _company(318306, "BARON CAPITAL GROUP INC", range(2012, 2020), 5.0),  # a holder that filed about Stericycle
]
TICKERS = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [[1744489, "Walt Disney Co", "DIS", "NYSE"], [1326160, "Duke Energy CORP", "DUK", "NYSE"]],
}
CUSIP_MAP = "cik,cusip6,cusip8\n" + "\n".join(
    f"{cik},{c6},{c6}10"
    for cik, c6 in (
        (1001039, "254687"),
        (1744489, "254687"),
        (1326160, "26441C"),
        (30371, "26441C"),
        (861878, "858912"),
        (318306, "858912"),
    )
)


class FakeSec:
    """Writes the SEC's three files where ``SecClient.download`` would."""

    def download(self, url: str, path: Path, *, chunk_bytes: int = 0) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if url == COMPANY_FACTS_BULK_URL:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as z:
                for c in COMPANIES:
                    z.writestr(f"CIK{c['cik']:010d}.json", json.dumps(c))
            path.write_bytes(buffer.getvalue())
        elif url == COMPANY_TICKERS_URL:
            path.write_text(json.dumps(TICKERS), encoding="utf-8")
        elif url == CIK_CUSIP_MAP_URL:
            path.write_text(CUSIP_MAP, encoding="utf-8")
        else:
            raise AssertionError(url)
        return path


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Engine, dict[str, Any]]]:
    with scratch_database(SCRATCH, tmp_path_factory.mktemp("sec_data")) as engine:
        with engine.begin() as conn:
            listings = pd.DataFrame(
                {
                    "anchor": ["DIS.N", "DUK.N", "SRCL.OQ^K24"],
                    "issuer_anchor": ["254687", "26441C", "858912"],
                    "issuer_name": ["The Walt Disney Company", "Duke Energy Corp", "Stericycle Inc"],
                    "name": ["Walt Disney", "Duke Energy", "Stericycle"],
                    "instrument_type": "equity",
                    "currency": "USD",
                }
            )
            ids = ensure_listings(
                conn, listings, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None
            )
            add_identifiers(
                conn,
                pd.DataFrame(
                    {
                        "entity_type": "listing",
                        "entity_id": [
                            ids.listing_ids["DIS.N"],
                            ids.listing_ids["DUK.N"],
                            ids.listing_ids["SRCL.OQ^K24"],
                        ],
                        "id_type": "TICKER",
                        "value": ["DIS", "DUK", "SRCL"],
                        "valid_from": pd.Timestamp("1990-01-01"),
                        "valid_to": [None, None, pd.Timestamp("2024-11-02")],  # Stericycle's ticker ended
                        "vendor_id": 1,
                    }
                ),
                batch_id=None,
            )
            london = ensure_listings(  # no CUSIP: named by its share's ISIN, as LSEG's London listings are
                conn,
                pd.DataFrame(
                    {
                        "anchor": ["TSTDIS.L"],
                        "issuer_anchor": ["GB00TEST0001"],
                        "issuer_name": ["Test London plc"],
                        "name": ["Test London"],
                        "instrument_type": "equity",
                        "exchange_mic": "XLON",
                        "currency": "GBX",
                    }
                ),
                anchor_type="RIC",
                issuer_anchor_type="ISIN",
                vendor="refinitiv",
                batch_id=None,
            )
            add_identifiers(
                conn,
                pd.DataFrame(
                    {
                        "entity_type": "listing",
                        "entity_id": [london.listing_ids["TSTDIS.L"]],
                        "id_type": "TICKER",
                        "value": ["DIS"],
                        "valid_from": pd.Timestamp("1990-01-01"),
                        "valid_to": [None],
                        "vendor_id": 1,
                    }
                ),
                batch_id=None,
            )
        yield engine, load_sec(engine, client=cast(SecClient, FakeSec()))


def _ciks(engine: Engine, company: str) -> list[str]:
    with engine.connect() as c:
        rows = c.execute(
            sa.text(
                "SELECT d.value FROM core.identifier d JOIN core.issuer i ON i.issuer_id = d.entity_id "
                "WHERE d.entity_type = 'issuer' AND d.id_type = 'CIK' AND i.name = :n ORDER BY d.value"
            ),
            {"n": company},
        ).scalars()
        return [str(v) for v in rows]


def _assets(engine: Engine, company: str) -> dict[int, float]:
    with engine.connect() as c:
        rows = c.execute(
            sa.text(
                "SELECT YEAR(f.period_end), f.value FROM equity.financial_fact f JOIN core.issuer i "
                "ON i.issuer_id = f.issuer_id WHERE i.name = :n AND f.item = 'total_assets'"
            ),
            {"n": company},
        ).all()
        return {int(y): float(v) for y, v in rows}


def test_a_reregistered_company_keeps_its_accounts_from_before(loaded: tuple[Engine, dict[str, Any]]) -> None:
    engine, report = loaded
    assert _ciks(engine, "The Walt Disney Company") == ["1001039", "1744489"]
    assets = _assets(engine, "The Walt Disney Company")
    assert sorted(assets) == list(range(2009, 2026))  # 2009-2018 from the old CIK, 2019 on from the new one
    assert report["companies_using_name_matched_ciks"] == 2  # Disney and Stericycle: Duke's subsidiary added nothing


def test_a_subsidiary_filing_its_own_accounts_is_not_the_company(loaded: tuple[Engine, dict[str, Any]]) -> None:
    engine, report = loaded
    assert _ciks(engine, "Duke Energy Corp") == ["1326160"]
    assert _assets(engine, "Duke Energy Corp")[2015] == 3_000.0 + 2015  # the parent's figure, not the subsidiary's
    assert report["identity"]["name_matched_ciks"] == 3  # Disney's old CIK, Duke's subsidiary, Stericycle


def test_a_delisted_company_is_found_by_its_cusip_and_name(loaded: tuple[Engine, dict[str, Any]]) -> None:
    engine, report = loaded
    assert _ciks(engine, "Stericycle Inc") == ["861878"]  # not the holder that shares its CUSIP rows
    assert len(_assets(engine, "Stericycle Inc")) == len(range(2009, 2024))
    assert report["identity"]["by_method"].get("cusip+name", 0) >= 1


def test_a_london_ticker_is_not_a_us_one(loaded: tuple[Engine, dict[str, Any]]) -> None:
    engine, _ = loaded
    # without the market check it takes Disney's CIK, and the two companies are merged into one
    with engine.connect() as c:
        companies = c.execute(
            sa.text("SELECT COUNT(DISTINCT issuer_id) FROM core.listing WHERE name IN ('Test London', 'Walt Disney')")
        ).scalar_one()
        assert companies == 2
    assert _ciks(engine, "Test London plc") == []
    assert _ciks(engine, "The Walt Disney Company") == ["1001039", "1744489"]
