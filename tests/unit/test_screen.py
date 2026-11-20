from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import pytest

from equity_research.analytics.screen import screen, strategy_screen
from equity_research.data.cache import ResearchCache
from equity_research.data.universe import UniverseData
from equity_research.strategies.garp import GarpConfig
from equity_research.strategies.runs import Snapshot


class _Cache:
    """Answers the screen's two lookups (names, RICs) like the research cache would."""

    def query(self, sql: str, params: Sequence[object] | None = None) -> pd.DataFrame:
        ids = range(1, 31)
        if "core.identifier" in sql:
            return pd.DataFrame({"listing_id": list(ids), "value": [f"T{i}.N" for i in ids]})
        return pd.DataFrame({"listing_id": list(ids), "name": [f"Company {i}" for i in ids]})


def _data() -> UniverseData:
    days = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(1)
    drift = np.linspace(-0.002, 0.002, 30)  # listing 30 rises fastest, listing 1 falls
    returns = pd.DataFrame(rng.normal(0, 0.01, (300, 30)) + drift, index=days, columns=range(1, 31))
    adjusted = 100 * (1 + returns).cumprod()
    members = pd.DataFrame(True, index=days, columns=returns.columns)
    members[5] = False  # not in the index any more: never shown
    empty = pd.DataFrame(index=days, columns=returns.columns, dtype=float)
    return UniverseData(
        universe="TEST",
        close=adjusted,
        adjusted=adjusted,
        returns=returns,
        members=members,
        segment=pd.DataFrame("large", index=days, columns=returns.columns),
        cost_bps=empty,
        borrow_bps=empty,
        buy_tax_bps=empty,
        rebalances=pd.DatetimeIndex([days[-1]]),
        ratios=pd.DataFrame(),
        benchmark=pd.Series(dtype=float),
        market_cap=empty,
        share_change=empty,
        earnings_surprise=empty,
        eps_growth=empty,
        sales_growth=empty,
    )


def test_members_are_ranked_by_the_mix_with_their_numbers() -> None:
    table = screen(cast(ResearchCache, _Cache()), _data(), {"momentum": 1.0})
    assert 5 not in table.index and len(table) == 29
    assert table["rank"].tolist() == list(range(1, 30))
    assert table["score"].is_monotonic_decreasing
    assert table.iloc[0]["return_12_1"] == table["return_12_1"].max()  # momentum alone: the best run wins
    assert table.loc[30, "ticker"] == "T30" and table.loc[30, "name"] == "Company 30"


def test_a_strategy_screen_ranks_by_the_strategys_own_score_and_shows_what_it_would_hold() -> None:
    data = _data()
    day = data.rebalances.max()
    ids = data.members.columns
    score = pd.Series(range(len(ids)), index=ids, dtype=float).drop(7)  # a name the strategy cannot score
    today = Snapshot(
        config=GarpConfig(price_vendors=("refinitiv",), fundamentals_vendor="sec"),
        day=day,
        data=data,
        score=score,
        parts=pd.DataFrame({"growth": score * 2, "quality_value": score}),
        weights=pd.Series({30: 0.6, 29: 0.4}),
    )
    table = strategy_screen(cast(ResearchCache, _Cache()), today)
    assert table.index[0] == 30 and table["score"].is_monotonic_decreasing
    assert 7 not in table.index and 5 not in table.index  # unscored, and not a member
    assert list(table.columns[:7]) == ["rank", "ticker", "name", "segment", "score", "weight", "growth"]
    assert table["weight"].sum() == pytest.approx(1.0) and (table.loc[[28, 1], "weight"] == 0).all()
