from collections.abc import Sequence
from typing import cast

import pandas as pd

from equity_research.data.cache import ResearchCache
from equity_research.data.sectors import classification_on

ROWS = pd.DataFrame(
    {
        "issuer_id": [1, 1, 2, 3, 3],
        "vendor": ["refinitiv", "refinitiv", "refinitiv", "refinitiv", "other"],
        "valid_from": pd.to_datetime(["2015-01-01", "2018-09-28", "2026-10-24", "2026-10-24", "2010-01-01"]),
        "valid_to": pd.to_datetime(pd.Series(["2018-09-28", None, None, None, None])),
        "sector": ["Information Technology", "Communication Services", "Energy", "Financials", "Banks"],
        "industry_group": None,
        "industry": None,
        "sub_industry": None,
    }
)


class _Cache:
    def query(self, sql: str, params: Sequence[object] | None = None) -> pd.DataFrame:
        return ROWS


def test_the_sector_a_company_had_on_the_day() -> None:
    cache = cast(ResearchCache, _Cache())
    before = classification_on(cache, pd.Timestamp("2016-06-30"), vendors=("refinitiv",))
    assert before.loc[1, "sector"] == "Information Technology"  # before GICS moved it in 2018
    after = classification_on(cache, pd.Timestamp("2020-06-30"), vendors=("refinitiv",))
    assert after.loc[1, "sector"] == "Communication Services"
    # first recorded in 2026: a stand-in for 2016, and it says so
    assert before.loc[2, "sector"] == "Energy"
    assert before["known_from"][2] > pd.Timestamp("2016-06-30")


def test_a_classification_that_held_beats_a_preferred_vendors_stand_in() -> None:
    picked = classification_on(
        cast(ResearchCache, _Cache()), pd.Timestamp("2016-06-30"), vendors=("refinitiv", "other")
    )
    assert picked.loc[3, "sector"] == "Banks" and picked.loc[3, "vendor"] == "other"
