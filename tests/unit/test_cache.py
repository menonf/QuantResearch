from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa

from equity_research.research.cache import TABLES, ResearchCache


def test_views_keep_the_latest_version_of_each_key(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path)
    (tmp_path / "ref.vendor").mkdir()
    pd.DataFrame({"vendor_id": [1], "vendor_code": ["refinitiv"], "name": ["LSEG"]}).to_parquet(
        tmp_path / "ref.vendor" / "part-1.parquet"
    )
    folder = tmp_path / "market.price_bar"
    folder.mkdir()
    base = {
        "listing_id": [7, 7],
        "vendor_id": [1, 1],
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
    }
    pd.DataFrame({**base, "close": [10.0, 11.0], "loaded_at": pd.to_datetime(["2024-01-05"] * 2)}).to_parquet(
        folder / "part-1.parquet"
    )
    # a later sync brings a corrected close for 01-03 only
    pd.DataFrame(
        {
            "listing_id": [7],
            "vendor_id": [1],
            "trade_date": pd.to_datetime(["2024-01-03"]),
            "close": [11.5],
            "loaded_at": pd.to_datetime(["2024-02-01"]),
        }
    ).to_parquet(folder / "part-2.parquet")
    out = cache.query("SELECT trade_date, close, vendor FROM market.price_bar ORDER BY trade_date")
    assert out["close"].tolist() == [10.0, 11.5]
    assert out["vendor"].tolist() == ["refinitiv", "refinitiv"]  # the readable code next to the stored id


def test_a_table_nothing_was_loaded_into_yet_is_empty_not_missing(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path)
    (tmp_path / "ref.vendor").mkdir()
    pd.DataFrame({"vendor_id": [1], "vendor_code": ["refinitiv"], "name": ["LSEG"]}).to_parquet(
        tmp_path / "ref.vendor" / "part-1.parquet"
    )
    out = cache.query(
        "SELECT listing_id, shares, vendor FROM market.share_count WHERE kind = 'float' AND listing_id IN (1, 2)"
    )
    assert out.empty and list(out.columns) == ["listing_id", "shares", "vendor"]


def test_a_rebuilt_table_loses_the_rows_the_database_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ResearchCache(tmp_path)
    facts = next(t for t in TABLES if t.name == "fund.financial_fact")
    folder = tmp_path / facts.name
    folder.mkdir()
    old = pd.DataFrame(
        {
            "issuer_id": [1],
            "vendor_id": [1],
            "item": ["revenue"],
            "period_type": ["Q"],
            "period_end": pd.to_datetime(["2014-03-31"]),
            "published_at": pd.to_datetime(["2013-11-05"]),  # dated before its quarter ended: deleted by the repair
            "value": [9.0],
            "loaded_at": pd.to_datetime(["2026-10-25"]),
        }
    )
    old.to_parquet(folder / "part-old.parquet")
    now = old.assign(published_at=pd.Timestamp("2014-06-29"), loaded_at=pd.Timestamp("2026-10-26"))
    monkeypatch.setattr(pd, "read_sql_query", lambda *args, **kwargs: iter([now]))  # what the database holds now
    cache.sync(sa.create_engine("sqlite://"), tables=[facts], rebuild=["fund.financial_fact"])
    assert not (folder / "part-old.parquet").exists()
    assert cache.query("SELECT published_at FROM fund.financial_fact")["published_at"].tolist() == [
        pd.Timestamp("2014-06-29")
    ]
    with pytest.raises(KeyError, match="no cached table"):
        cache.sync(sa.create_engine("sqlite://"), tables=[facts], rebuild=["fund.nothing"])


def test_a_column_newer_than_the_cached_files_reads_as_empty(tmp_path: Path) -> None:
    # files synced before a migration added ref.issuer.incorporation_country
    (tmp_path / "ref.issuer").mkdir()
    pd.DataFrame({"issuer_id": [1], "name": ["Old Co"], "domicile_country": ["US"]}).to_parquet(
        tmp_path / "ref.issuer" / "part-1.parquet"
    )
    out = ResearchCache(tmp_path).query("SELECT issuer_id, incorporation_country FROM ref.issuer")
    assert out["issuer_id"].tolist() == [1] and out["incorporation_country"].isna().all()
