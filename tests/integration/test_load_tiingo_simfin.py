"""Tiingo and SimFin end to end on a scratch database whose listings came from (a fake) Norgate.

Neither vendor creates a listing or an issuer: both attach to what is there. Tiingo is stopped by its allowance and
continues, refuses prices that disagree with Norgate's, and asks nothing for listings it already has. SimFin's
companies are matched by ticker on the day they last published, then by their SimFin id.
"""

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import sqlalchemy as sa
import test_load_norgate as norgate_case
from conftest import scratch_database
from sqlalchemy.engine import Engine

from equity_research.ingest.vendors.norgate.client import NorgateClient
from equity_research.ingest.vendors.simfin.client import DATASETS, SimFinClient
from equity_research.ingest.vendors.tiingo.client import TiingoClient, TiingoThrottledError
from equity_research.pipelines.load_norgate import load_norgate
from equity_research.pipelines.load_simfin import load_simfin
from equity_research.pipelines.load_tiingo import load_tiingo

pytestmark = pytest.mark.integration

SCRATCH = "equity_research_test_tiingo_simfin"
TODAY = pd.Timestamp(norgate_case.END)


def _zip(path: Path, name: str, text: str) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(name, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


class FakeTiingo:
    """Answers like ``TiingoClient``: AAA and OLD as Norgate has them, CCC another company's prices, no DDD."""

    def __init__(self, allowance: int | None = None) -> None:
        self.allowance = allowance
        self.requests_made = 0
        self.asked: list[str] = []

    def download_supported_tickers(self, path: Path) -> str:
        rows = "".join(f"{t},NYSE,Stock,USD,2000-01-03,2024-03-29\n" for t in ("aaa", "ccc", "old"))
        _zip(path, "supported_tickers.csv", "ticker,exchange,assetType,priceCurrency,startDate,endDate\n" + rows)
        return "https://example.invalid/supported_tickers.zip"

    def prices(self, ticker: str, start: str, end: str) -> tuple[str, bytes] | None:
        if self.allowance is not None and self.requests_made >= self.allowance:
            raise TiingoThrottledError("allowance used up")
        self.requests_made += 1
        self.asked.append(ticker)
        symbol = {"OLD": "OLD-202402"}.get(ticker, ticker)
        real = norgate_case.FakeNorgate().prices(symbol, start, end)["Unadjusted Close"]
        if ticker == "CCC":
            real = real * 3  # the ticker belongs to someone else at Tiingo
        factor = real.shift(1) / real
        split = factor.where((factor - 1).abs() > 0.5, 1.0).fillna(1.0)
        frame = pd.DataFrame(
            {
                "date": [d.strftime("%Y-%m-%dT00:00:00.000Z") for d in real.index],
                "open": real.to_numpy(),
                "high": real.to_numpy(),
                "low": real.to_numpy(),
                "close": real.to_numpy(),
                "volume": 1000,
                "divCash": 0.0,
                "splitFactor": split.to_numpy(),
            }
        )
        return f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start}", frame.to_json(
            orient="records"
        ).encode()


STATEMENT_HEAD = "Ticker;SimFinId;Currency;Fiscal Year;Fiscal Period;Report Date;Publish Date;Restated Date"
SIMFIN_FILES = {
    "companies": "Ticker;SimFinId;Company Name;IndustryId\nAAA;11;AAA Inc;100001\nCCC;33;CCC plc;100001\n"
    "OLD;44;Old Co;100001\nZZZ;99;Unknown Corp;100001\n",
    "industries": "IndustryId;Sector;Industry\n100001;Industrials;Machinery\n",
    "income_quarterly": STATEMENT_HEAD + ";Revenue;Cost of Revenue;Net Income\n"
    "AAA;11;USD;2023;Q4;2023-12-31;2024-02-10;2025-02-07;1000;-600;100\n"  # re-dated by SimFin a year later
    "OLD;44;USD;2023;Q4;2023-12-31;2024-02-05;2024-02-05;500;-300;50\n"
    "CCC;33;USD;2023;Q4;2023-12-31;2024-03-01;2024-03-01;800;-400;80\n",
    "balance_quarterly": STATEMENT_HEAD + ";Total Assets;Total Equity\n"
    "AAA;11;USD;2023;Q4;2023-12-31;2024-02-10;2024-02-10;5000;2000\n",
}


class FakeSimFin:
    """Answers like ``SimFinClient``; the banks' and insurers' files have no rows."""

    def download(self, name: str, path: Path) -> str:
        text = SIMFIN_FILES.get(name, STATEMENT_HEAD + ";Revenue\n")
        _zip(path, f"us-{name}.csv", text)
        return f"https://prod.simfin.com/api/bulk-download/s3?dataset={DATASETS[name].dataset}"


@pytest.fixture(scope="module")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    with scratch_database(SCRATCH, tmp_path_factory.mktemp("tiingo_simfin_data")) as engine:
        load_norgate(
            engine,
            start=norgate_case.START,
            end=norgate_case.END,
            client=cast(NorgateClient, norgate_case.FakeNorgate()),
        )
        yield engine


def _one(engine: Engine, sql: str) -> Any:
    with engine.connect() as c:
        return c.execute(sa.text(sql)).scalar_one()


def _listing(ticker: str) -> str:
    return (
        "(SELECT entity_id FROM core.identifier WHERE entity_type = 'listing' AND id_type = 'TICKER' "
        f"AND value = '{ticker}')"
    )


def test_tiingo_prices_the_listings_it_can_and_continues_after_its_allowance(seeded: Engine) -> None:
    first = load_tiingo(seeded, client=cast(TiingoClient, FakeTiingo(allowance=1)), today=TODAY.date())
    assert first["throttled"] and not first["complete"]
    assert first["skipped"]["not_at_tiingo"] == 1  # DDD
    assert first["requests"] == 1 and first["remaining"] == 2

    second_client = FakeTiingo()
    second = load_tiingo(seeded, client=cast(TiingoClient, second_client), today=TODAY.date())
    assert second["complete"] and second["skipped"]["complete"] == 1  # AAA, done in the first run
    assert sorted(second_client.asked) == ["CCC", "OLD"]
    assert second["disagreed"] == 1 and second["disagreed_sample"] == ["CCC"]

    tiingo = "vendor_id = 6"
    assert _one(seeded, f"SELECT COUNT(*) FROM equity.price_bar WHERE {tiingo} AND listing_id = {_listing('CCC')}") == 0
    aaa_close = f"SELECT [close] FROM equity.price_bar WHERE {tiingo} AND listing_id = {_listing('AAA')}"
    assert _one(seeded, aaa_close + " AND trade_date = '2024-01-31'") == 200  # real, before the 2-for-1 split
    split = f"SELECT ratio FROM equity.corporate_action WHERE {tiingo} AND action_type = 'split'"
    assert float(_one(seeded, split)) == pytest.approx(2.0)
    old_last = f"SELECT MAX(trade_date) FROM equity.price_bar WHERE {tiingo} AND listing_id = {_listing('OLD')}"
    assert str(_one(seeded, old_last)) == "2024-02-15"  # its last day, nothing after it was delisted

    third_client = FakeTiingo()
    load_tiingo(seeded, client=cast(TiingoClient, third_client), today=TODAY.date())
    assert third_client.asked == ["CCC"]  # only the one still without Tiingo prices


def test_simfin_attaches_accounts_to_our_companies(seeded: Engine) -> None:
    with seeded.begin() as c:  # the SEC's first report of AAA's revenue: the same figure, two days earlier
        c.execute(
            sa.text(
                "INSERT INTO equity.financial_fact (issuer_id, vendor_id, item, period_type, period_end, published_at, "
                "period_start, value, unit) SELECT issuer_id, 2, 'revenue', 'Q', '2023-12-30', '2024-02-08', "
                f"'2023-10-01', 1000, 'USD' FROM core.listing WHERE listing_id = {_listing('AAA')}"
            )
        )
    first = load_simfin(seeded, client=cast(SimFinClient, FakeSimFin()), today=TODAY.date())
    assert first["companies"] == 4 and first["matched"] == 3  # ZZZ is not one of ours
    assert first["by"] == {"TICKER": 3}
    assert first["classifications"]["opened"] == 3
    facts = "SELECT COUNT(*) FROM equity.financial_fact WHERE vendor_id = 7"
    assert _one(seeded, facts) == 3 * 3 + 2
    cost = "SELECT value FROM equity.financial_fact WHERE vendor_id = 7 AND item = 'cost_of_revenue' AND value = 600"
    assert _one(seeded, cost) == 600  # positive, as the SEC's
    assert _one(seeded, "SELECT COUNT(*) FROM core.identifier WHERE id_type = 'SIMFIN'") == 3
    assert _one(seeded, "SELECT COUNT(*) FROM core.issuer") == 4  # nothing created
    dated = "SELECT published_at FROM equity.financial_fact WHERE vendor_id = 7 AND item = '{}' AND value = {}"
    assert str(_one(seeded, dated.format("revenue", 1000))) == "2024-02-10 00:00:00"  # confirmed: first publication
    assert str(_one(seeded, dated.format("net_income", 100))) == "2025-02-07 00:00:00"  # unconfirmed: restated
    assert first["dates"]["income_quarterly"] == {"no_sec_figure": 8, "confirmed_by_sec": 1}

    again = load_simfin(seeded, client=cast(SimFinClient, FakeSimFin()), max_age_days=0, today=TODAY.date())
    assert again["by"] == {"SIMFIN": 3}
    assert sum(v["inserted"] for v in again["facts"].values()) == 0
    assert _one(seeded, facts) == 3 * 3 + 2
