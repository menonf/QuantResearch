"""The settings added to the factor strategy after the review of notebooks 3.1-3.6 (docs/STRATEGY_REVIEW.md)."""

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from equity_research.analytics.indices import cap_weighted
from equity_research.data.universe import UniverseData
from equity_research.strategies import construction as cn
from equity_research.strategies import factor_strategy as fs
from equity_research.strategies import factors as fx
from equity_research.strategies import learned_value as ml
from equity_research.strategies.runs import config_from

VENDORS = {"price_vendors": ("refinitiv",), "fundamentals_vendor": ("sec", "refinitiv")}
DAY = pd.Timestamp("2024-01-31")


def _config(**settings: object) -> fs.StrategyConfig:
    return fs.StrategyConfig(family="x", **(VENDORS | settings))  # type: ignore[arg-type]


def test_a_run_that_uses_no_new_setting_is_registered_as_before_they_existed() -> None:
    # runs registered before the settings existed hashed asdict() without them: the same dict keeps the same hash
    plain = fs.registered_settings(_config())
    assert set(fs.LATER).isdisjoint(plain)
    before = {k: v for k, v in asdict(_config()).items() if k not in fs.LATER}
    assert plain == before


def test_a_new_setting_is_registered_only_when_used_and_rebuilds_from_the_registry() -> None:
    config = _config(
        mix="standardised", construction="tilted", risk=0.04, neutral=("sector",), benchmark="cap_weighted"
    )
    stored = fs.registered_settings(config)
    assert {"mix", "construction", "risk", "neutral", "benchmark"} <= set(stored)
    assert "max_active" not in stored and "drift_band" not in stored  # still at their defaults
    rebuilt = config_from(json.loads(json.dumps(stored)))
    assert rebuilt == config


def test_an_unknown_mix_or_construction_is_refused_when_the_config_is_made() -> None:
    with pytest.raises(ValueError, match="unknown mix"):
        _config(mix="median")
    with pytest.raises(ValueError, match="unknown construction"):
        _config(construction="optimal")


def test_a_book_empty_on_a_rebalance_date_is_refused_naming_when_it_first_holds() -> None:
    days = pd.DatetimeIndex(["2004-01-30", "2004-02-27", "2004-03-31"])
    book = pd.DataFrame({1: [0.0, 0.0, 0.5], 2: [0.0, 0.0, 0.5]}, index=days)
    with pytest.raises(fs.EmptyBookError, match=r"2 rebalance date.*first holds names on 2004-03-31"):
        fs.check_book(book)
    fs.check_book(book.iloc[2:])  # a book that holds names every month passes


def test_a_learned_score_or_the_risk_model_starts_the_data_before_the_book() -> None:
    assert fs.data_start(_config(start="2004-01-01")) == "2004-01-01"
    learned = _config(start="2004-01-01", factors=(("ml_value", 1.0),))
    assert fs.data_start(learned) == "2004-01-01"  # TRAIN_FROM (2006) is later: the old rule, unchanged
    assert fs.data_start(replace(learned, train_from="2000-01-01")) == "2000-01-01"
    assert fs.data_start(_config(start="2004-01-01", construction="tilted")) == "2002-01-01"


def test_a_standardised_mix_weighs_a_four_part_composite_as_much_as_one_score() -> None:
    rng = np.random.default_rng(1)
    names = list(range(200))
    single = pd.DataFrame([rng.normal(size=200)], index=[DAY], columns=names)
    # the mean of four loosely related z-scores spreads less than one z-score
    composite = pd.DataFrame([rng.normal(size=(4, 200)).mean(axis=0)], index=[DAY], columns=names)
    members = pd.DataFrame(True, index=[DAY], columns=names)
    plain = fx.combine({"a": single, "b": composite})
    fair = fx.combine({"a": fx.cross_sectional_z(single, members), "b": fx.cross_sectional_z(composite, members)})
    share = {
        name: float(np.corrcoef(mix.loc[DAY], part.loc[DAY])[0, 1])
        for name, mix, part in (("plain_b", plain, composite), ("fair_b", fair, composite))
    }
    assert share["plain_b"] < 0.6 < share["fair_b"]  # the composite barely steers the plain mix


def test_a_missing_score_counts_as_neutral_when_asked() -> None:
    a = pd.DataFrame({1: [2.0], 2: [2.0]}, index=[DAY])
    b = pd.DataFrame({1: [2.0], 2: [np.nan]}, index=[DAY])
    assert fx.combine({"a": a, "b": b}).loc[DAY].to_numpy().tolist() == [2.0, 2.0]  # skipped: 2 has only its good score
    assert fx.combine({"a": a, "b": b}, missing="neutral").loc[DAY].to_numpy().tolist() == [2.0, 1.0]
    nothing = pd.DataFrame({1: [np.nan], 2: [np.nan]}, index=[DAY])
    assert fx.combine({"a": nothing, "b": nothing}, missing="neutral").isna().all().all()  # no score at all: none


def test_ranking_within_sectors_puts_the_cheapest_of_each_sector_on_top() -> None:
    # banks are all cheap, software all dear: across the universe the banks take the top, within sectors each leads
    score = pd.DataFrame([[10.0, 9.0, 8.0, 7.0, 6.0, 3.0, 2.0, 1.0, 0.5, 0.2]], index=[DAY], columns=range(10))
    sectors = pd.Series(["bank"] * 5 + ["software"] * 5, index=range(10))
    within = fx.within_groups(score, sectors)
    assert within.loc[DAY, 0] == pytest.approx(within.loc[DAY, 5])  # the best of each sector score alike
    assert within.loc[DAY, 4] == pytest.approx(within.loc[DAY, 9])
    across = fx.cross_sectional_z(score)
    assert across.to_numpy()[0, 5] < 0 < within.to_numpy()[0, 5]


def test_earnings_value_leaves_out_book_and_sales() -> None:
    frame = pd.DataFrame(
        {
            "EBIT/EV": [0.10, 0.02, 0.06],
            "P/E": [10.0, 50.0, 20.0],
            "FCF/EV": [0.08, 0.01, 0.05],
            "P/B": [10.0, 0.5, 1.0],  # the cheapest on book is the dearest on earnings: it does not count here
            "P/S": [5.0, 0.2, 1.0],
        },
        index=pd.MultiIndex.from_product([[DAY], [1, 2, 3]], names=["date", "listing_id"]),
    )
    score = fx.earnings_value(frame)
    assert pd.Series(score.loc[DAY]).sort_values(ascending=False).index.tolist() == [1, 3, 2]


def test_the_rank_target_orders_each_dates_members_from_zero_to_one() -> None:
    days = pd.bdate_range("2020-01-01", "2021-03-31")
    returns = pd.DataFrame({1: 0.001, 2: 0.0, 3: -0.001, 4: 0.002}, index=days)
    members = pd.DataFrame({1: True, 2: True, 3: True, 4: False}, index=days)
    dates = pd.DatetimeIndex([days[0]])
    target = ml.rank_target(returns, members, dates)
    assert target.loc[days[0], [3, 2, 1]].to_numpy().tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert pd.isna(target.loc[days[0], 4])  # not a member: no target


def _universe(days: pd.DatetimeIndex, returns: pd.DataFrame, caps: pd.DataFrame) -> UniverseData:
    """A universe whose market values are dated on its rebalance dates only, as ``load_universe_data`` gives them."""
    empty = pd.DataFrame(index=days, columns=returns.columns, dtype=float)
    return UniverseData(
        universe="TEST",
        close=100 * (1 + returns).cumprod(),
        adjusted=100 * (1 + returns).cumprod(),
        returns=returns,
        members=pd.DataFrame(True, index=days, columns=returns.columns),
        segment=pd.DataFrame("large", index=days, columns=returns.columns),
        cost_bps=empty,
        borrow_bps=empty,
        buy_tax_bps=empty,
        rebalances=pd.DatetimeIndex(caps.index),
        ratios=pd.DataFrame(),
        benchmark=pd.Series(dtype=float),
        market_cap=caps,
        share_change=empty,
        earnings_surprise=empty,
        eps_growth=empty,
        sales_growth=empty,
    )


def test_the_own_index_holds_month_end_market_weights_and_has_a_return_every_day() -> None:
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    returns = pd.DataFrame({1: 0.01, 2: -0.01}, index=days)
    ends = pd.DatetimeIndex([days[days.month == m][-1] for m in (1, 2)])
    caps = pd.DataFrame({1: [300.0, 300.0], 2: [100.0, 300.0]}, index=ends)
    index = cn.index_returns(_universe(days, returns, caps))
    assert index.notna().all() and index.index[0] == days[days > ends[0]][0]  # every day after the first month end
    assert index.iloc[0] == pytest.approx(0.75 * 0.01 + 0.25 * -0.01)  # 300 : 100 at the end of January
    after = index[index.index > ends[1]]
    assert after.iloc[0] == pytest.approx(0.0)  # 300 : 300 from the end of February
    with pytest.raises(ValueError, match="cap_weighted needs daily"):
        cap_weighted(returns, caps, pd.DataFrame(True, index=days, columns=[1, 2]))
