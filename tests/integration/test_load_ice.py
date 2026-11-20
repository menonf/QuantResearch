"""ICE's constituent files into the shared security master and the bond tables, and back out through the cache."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import scratch_database

from bond_files import BOND, SECOND, write_file
from fixed_income_research.analytics import replication as rp
from fixed_income_research.data.cache import ResearchCache
from fixed_income_research.data.index_data import load_index_data
from fixed_income_research.pipelines.load_ice import load_ice
from fixed_income_research.pipelines.load_index_levels import load_index_levels
from fixed_income_research.quality.checks import BOND_CHECKS
from research_core.data.quality import run_checks

pytestmark = pytest.mark.integration

THIRD = BOND | {"Cusip": "1248EPBZ", "ISIN number": "US1248EPBZ53", "Ticker": "CHTR", "Description": "CCO Holdings LLC"}


def _one(engine: sa.Engine, sql: str) -> object:
    with engine.connect() as conn:
        return conn.execute(sa.text(sql)).scalar_one()


def test_files_load_once_in_any_order_and_come_back_as_panels(tmp_path: Path) -> None:
    folder = tmp_path / "files"
    folder.mkdir()
    # February first: the January file arrives later, as older history does
    february = [
        BOND
        | {"% Mkt Value": 50.0, "% MktVal-PrevMend": 60.0, "Total Return % MTD_LOC": 2.0, "Description": "Alcoa NV"},
        THIRD | {"% Mkt Value": 50.0, "% MktVal-PrevMend": None, "Total Return % MTD_LOC": None},
    ]
    write_file(folder / "H0A0-02282019.xlsx", february, "H0A0 for 02/28/2019  ( March Universe )")
    (folder / "notes.xlsx").write_bytes(b"")
    with scratch_database("equity_research_test_ice", tmp_path / "data") as engine:
        first = load_ice(engine, folder)
        assert (first["loaded"], first["bonds_created"], first["issuers_created"]) == (1, 2, 2)
        assert list(first["refused"]) == ["notes.xlsx"]

        write_file(folder / "H0A0-01312019.xlsx", [BOND, SECOND], "H0A0 for 01/31/2019  ( February Universe )")
        second = load_ice(engine, folder)
        assert (second["loaded"], second["already_loaded"], second["bonds_created"]) == (1, 1, 1)
        again = load_ice(engine, folder)  # nothing new: nothing read, nothing written
        assert (again["loaded"], again["already_loaded"]) == (0, 2)
        assert again["analytics"] == {"inserted": 0, "updated": 0}

        assert _one(engine, "SELECT COUNT(*) FROM core.listing WHERE instrument_type = 'bond'") == 3
        assert _one(engine, "SELECT COUNT(*) FROM fixed_income.bond_analytics") == 4
        assert _one(engine, "SELECT COUNT(*) FROM fixed_income.index_constituent") == 4
        assert _one(engine, "SELECT COUNT(*) FROM fixed_income.issuer_classification") == 3
        # the bond keeps the newest file's name, though the older file was loaded after it
        named = "SELECT l.name FROM core.listing l JOIN core.identifier i ON i.entity_id = l.listing_id "
        assert _one(engine, named + "AND i.entity_type = 'listing' WHERE i.value = 'US013822AC54'") == "Alcoa NV"
        cusips = "SELECT COUNT(*) FROM core.identifier WHERE id_type = 'CUSIP' AND value IN ('013822AC5', '013822AA9')"
        assert _one(engine, cusips) == 2
        assert (
            _one(engine, "SELECT COUNT(*) FROM core.identifier WHERE id_type = 'ICE' AND entity_type = 'issuer'") == 2
        )

        results = {r.name: r for r in run_checks(engine, BOND_CHECKS)}
        assert results["bond_index_weights_sum_to_100"].status == "pass"
        assert results["bond_constituents_without_analytics"].status == "pass"
        assert results["bond_rows_without_provenance"].status == "pass"
        assert results["bond_index_weight_leaving_without_return"].value == pytest.approx(40.0)  # the second bond left

        payload = b"observation_date,X\n2019-01-31,100.0\n2019-02-28,101.2\n2019-03-15,102.0\n"
        levels = load_index_levels(engine, fetch=lambda series: payload, today=dt.date(2019, 3, 20))
        assert levels["H0A0"]["inserted"] == 2

        cache = ResearchCache(tmp_path / "cache")
        cache.sync(engine)
        data = load_index_data(cache, "H0A0")
        assert list(data.dates) == list(pd.to_datetime(["2019-01-31", "2019-02-28"]))
        rebuilt = rp.index_returns(data.weights, data.total_returns)
        assert rebuilt.loc["2019-02-28", "return"] == pytest.approx(0.02)
        assert rebuilt.loc["2019-02-28", "covered"] == pytest.approx(0.6)
        assert rp.official_returns(data.official).iloc[0] == pytest.approx(0.012)
        assert set(data.rating.loc["2019-02-28"].dropna()) == {"BB1"}
