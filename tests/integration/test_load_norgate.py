"""The Norgate load end to end on a scratch database, Norgate replaced by a fake answering in Norgate's shapes.

Two runs: the first builds everything from nothing, the second finds every security by its Norgate id again.
"""

from collections.abc import Iterator
from typing import Any, cast

import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import scratch_database
from sqlalchemy.engine import Engine

from equity_research.ingest.vendors.norgate.client import NorgateClient
from equity_research.pipelines.load_norgate import load_norgate

pytestmark = pytest.mark.integration

SCRATCH = "equity_research_test_norgate"
START, END = "2024-01-01", "2024-03-29"
DAYS = pd.bdate_range(START, END)
LEFT = pd.Timestamp("2024-02-15")  # OLD's last day: delisted that month
SPLIT = pd.Timestamp("2024-02-01")  # AAA splits 2-for-1
WATCHLISTS = {
    "S&P 500 Current & Past": ["AAA", "OLD-202402"],
    "S&P MidCap 400 Current & Past": ["CCC"],
    "S&P SmallCap 600 Current & Past": ["DDD"],
}
ASSET_IDS = {"AAA": 101, "OLD-202402": 102, "CCC": 103, "DDD": 104, "$SPX": 901, "$MID": 902, "$SML": 903}
DOMICILES = {"AAA": "United States of America", "OLD-202402": "United States of America", "CCC": "Ireland"}


def _days(symbol: str) -> pd.DatetimeIndex:
    return DAYS[DAYS <= LEFT] if symbol == "OLD-202402" else DAYS


class FakeNorgate:
    """Answers like ``NorgateClient`` for a handful of symbols."""

    def check(self) -> None:
        pass

    def symbols(self, watchlist: str) -> list[str]:
        return WATCHLISTS[watchlist]

    def describe(self, symbol: str) -> dict[str, Any]:
        days = _days(symbol)
        index = symbol.startswith("$")
        return {
            "symbol": symbol,
            "assetid": ASSET_IDS[symbol],
            "name": f"{symbol} Inc",
            "exchange": "NYSE",
            "currency": "USD",
            "domicile": DOMICILES.get(symbol),
            "first_quoted": days[0].date(),
            "last_quoted": days[-1].date(),
            "sector": None if index else "Industrials",
            "industry_group": None if index else "Capital Goods",
            "industry": None if index else "Machinery",
            "sub_industry": None if index else "Industrial Machinery",
        }

    def prices(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        days = _days(symbol)
        days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
        real = pd.Series(100.0, index=days)
        factor = pd.Series(1.0, index=days)
        if symbol == "AAA":
            real[days < SPLIT] = 200.0
            factor[days < SPLIT] = 2.0
        adjusted = real / factor
        return pd.DataFrame(
            {
                "Open": adjusted,
                "High": adjusted,
                "Low": adjusted,
                "Close": adjusted,
                "Volume": 1000.0 * factor,
                "Turnover": 0.0,
                "Unadjusted Close": real,
                "Dividend": [0.5 if d == pd.Timestamp("2024-03-01") and symbol == "CCC" else 0.0 for d in days],
            },
            index=pd.DatetimeIndex(days, name="Date"),
        ).drop(columns=["Unadjusted Close", "Dividend"] if symbol.startswith("$") else [])  # as Norgate sends levels

    def membership(self, symbol: str, index_name: str, start: str, end: str) -> pd.DataFrame:
        days = _days(symbol)
        return pd.DataFrame({"Index Constituent": 1}, index=pd.DatetimeIndex(days, name="Date"))


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    with scratch_database(SCRATCH, tmp_path_factory.mktemp("norgate_data")) as engine:
        yield engine


def _one(engine: Engine, sql: str) -> Any:
    with engine.connect() as c:
        return c.execute(sa.text(sql)).scalar_one()


def test_a_norgate_build_from_nothing_and_a_repeat(scratch: Engine) -> None:
    client = cast(NorgateClient, FakeNorgate())
    first = load_norgate(scratch, start=START, end=END, client=client)
    assert first["universe"]["created"] == 4
    assert first["universe"]["index_levels"] == {"attached": 0, "created": 3}
    assert _one(scratch, "SELECT COUNT(*) FROM ref.listing") == 7  # 4 securities + 3 index levels
    assert _one(scratch, "SELECT COUNT(*) FROM ref.index_definition WHERE level_listing_id IS NOT NULL") == 3
    assert _one(scratch, "SELECT COUNT(*) FROM ref.index_membership WHERE vendor_id = 5") == 4

    # OLD left in February: delisted, its ticker valid only while it traded
    assert str(_one(scratch, "SELECT delisted_on FROM ref.listing WHERE name = 'OLD-202402 Inc'")) == "2024-02-15"
    old_spell = "SELECT end_date FROM ref.index_membership m JOIN ref.listing l ON l.listing_id = m.listing_id "
    assert str(_one(scratch, old_spell + "WHERE l.name = 'OLD-202402 Inc'")) == "2024-02-16"  # not a member for ever
    assert first["universe"]["memberships_closed_at_delisting"] == 1
    ticker_end = "SELECT valid_to FROM ref.identifier WHERE id_type = 'TICKER' AND value = 'OLD'"
    assert str(_one(scratch, ticker_end)) == "2024-02-16"

    # real prices: AAA at 200 before its split, the split recorded as 2-for-1
    aaa = (
        "(SELECT entity_id FROM ref.identifier WHERE entity_type = 'listing' AND id_type = 'NORGATE' AND value = '101')"
    )
    assert (
        _one(scratch, f"SELECT [close] FROM market.price_bar WHERE listing_id = {aaa} AND trade_date = '2024-01-31'")
        == 200
    )
    split = f"SELECT ratio FROM market.corporate_action WHERE listing_id = {aaa} AND action_type = 'split'"
    assert float(_one(scratch, split)) == pytest.approx(2.0)
    assert _one(scratch, "SELECT COUNT(*) FROM market.corporate_action WHERE action_type = 'cash_dividend'") == 1
    level = "SELECT COUNT(*) FROM market.price_bar p JOIN ref.identifier i ON i.entity_id = p.listing_id "
    assert _one(
        scratch, level + "WHERE i.entity_type = 'listing' AND i.id_type = 'TICKER' AND i.value = '$SPX'"
    ) == len(DAYS)

    # GICS and domicile from Norgate
    assert _one(scratch, "SELECT COUNT(*) FROM ref.classification WHERE vendor_id = 5 AND scheme = 'GICS'") == 4
    assert _one(scratch, "SELECT COUNT(*) FROM ref.issuer WHERE domicile_country = 'IE'") == 1

    again = load_norgate(scratch, start=START, end=END, client=client)
    assert again["universe"]["known"] == 4 and again["universe"]["created"] == 0
    assert again["universe"]["index_levels"] == {"attached": 3, "created": 0}
    assert again["prices"]["price_bars"]["inserted"] == 0
    assert _one(scratch, "SELECT COUNT(*) FROM ref.listing") == 7
    assert _one(scratch, "SELECT COUNT(*) FROM ref.index_membership WHERE vendor_id = 5") == 4
