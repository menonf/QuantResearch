import numpy as np
import pandas as pd
import pytest

from equity_research.data.universe import per_share_growth
from equity_research.strategies import garp

DAY = pd.DatetimeIndex([pd.Timestamp("2015-05-29")])
NAMES = list(range(1, 41))


def _panel(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([values], index=DAY, columns=NAMES[: len(values)])


def _members(n: int) -> pd.DataFrame:
    return pd.DataFrame(True, index=DAY, columns=NAMES[:n])


def test_z_scores_are_winsorized_and_standardised_among_members() -> None:
    raw = _panel([*range(1, 40), 10_000])  # one wild value
    z = garp.standard_z(raw, _members(40)).iloc[0]
    assert z.mean() == pytest.approx(0.0, abs=1e-9) and z.std() == pytest.approx(1.0)
    assert z[40] < 2.0  # clipped to the 97.5th percentile before standardising, not a z of 6
    reverse = garp.standard_z(raw, _members(40), reverse=True).iloc[0]
    assert reverse[1] > reverse[40]


def test_a_loss_over_negative_equity_and_negative_book_leverage_score_worst() -> None:
    n = 20
    index = pd.MultiIndex.from_product([DAY, NAMES[:n]], names=["date", "listing_id"])
    ratios = pd.DataFrame(
        {
            "RoE": np.linspace(0.02, 0.30, n),
            "Book Equity/TL": 0.5,
            "Debt/Equity": np.linspace(0.1, 2.0, n),
            "P/E": 15.0 + np.arange(n),
        },
        index=index,
    )
    ratios.loc[(DAY[0], n), ["RoE", "Book Equity/TL", "Debt/Equity"]] = [0.9, -0.3, -4.0]  # loss over negative equity
    clean = ratios.drop(index=[(DAY[0], n)])
    flagged = garp.quality_value_z(ratios, _members(n), DAY).iloc[0]
    assert flagged[n] == flagged.min()  # its "high" RoE and "low" leverage count against it, not for it
    assert garp.quality_value_z(clean, _members(n), DAY).iloc[0][1] == pytest.approx(flagged[1], abs=0.2)


def test_quality_and_value_needs_earnings_to_price_and_one_quality_score() -> None:
    index = pd.MultiIndex.from_product([DAY, [1, 2, 3, 4]], names=["date", "listing_id"])
    ratios = pd.DataFrame(
        {
            "RoE": [0.1, 0.2, np.nan, 0.15],
            "Book Equity/TL": [0.5, 0.5, 0.5, 0.5],
            "Debt/Equity": [1.0, 0.5, np.nan, 0.8],
            "P/E": [10.0, np.nan, 12.0, 20.0],
        },
        index=index,
    )
    qv = garp.quality_value_z(ratios, _members(4), DAY).iloc[0]
    assert np.isnan(qv[2])  # no earnings to price
    assert np.isnan(qv[3])  # no quality score at all
    assert qv[[1, 4]].notna().all()


def test_composite_weights_change_the_score_and_only_a_variant_is_registered() -> None:
    eps, sales = _panel([float(v) for v in range(1, 41)]), _panel([float(v) for v in range(40, 0, -1)])
    members = _members(40)
    equal = garp.growth_z(eps, sales, members).iloc[0]
    assert equal.abs().max() == pytest.approx(0.0, abs=1e-9)  # opposite growths cancel at equal weights...
    tilted = garp.growth_z(eps, sales, members, {"eps_growth": 3.0, "sales_growth": 1.0}).iloc[0]
    assert tilted[40] > tilted[1]  # ...and EPS growth decides once it weighs more
    eps_only = garp.growth_z(eps, sales, members, {"eps_growth": 1.0}).iloc[0]  # a z-score not named weighs zero
    assert eps_only.tolist() == pytest.approx(garp.standard_z(eps, members).iloc[0].clip(-4, 4).tolist())

    published = garp.GarpConfig()
    assert "growth_weights" not in published.registered()  # runs from before the setting stay the same trial
    variant = garp.GarpConfig(quality_value_weights={"earnings_yield": 2.0, "roe": 1.0, "leverage": 1.0})
    assert variant.registered()["quality_value_weights"] == (("leverage", 1.0), ("roe", 1.0), ("earnings_yield", 2.0))
    for wrong in ({"eps": 1.0}, {"eps_growth": -1.0}, {"eps_growth": 0.0}):
        with pytest.raises(ValueError):
            garp.GarpConfig(growth_weights=wrong)


def test_the_growth_score_is_positive_and_rewards_growth() -> None:
    score = garp.growth_score(pd.Series([-1.0, 0.0, 2.0]))
    assert score.tolist() == [0.5, 1.0, 3.0]


def test_the_buffer_keeps_a_constituent_within_the_wider_band() -> None:
    config = garp.GarpConfig(growth_count=8, count=3, automatic=2, keep_within=4)
    growth = pd.Series({i: float(10 - i) for i in range(1, 11)})  # 1..8 pass the growth filter
    quality = pd.Series({i: float(i) for i in range(1, 11)})  # among them 8 is best, then 7, 6, 5
    assert garp.select(growth, quality, set(), config) == [8, 7, 6]
    assert garp.select(growth, quality, {5}, config) == [8, 7, 5]  # 5 is held and within the top 4: it stays
    assert garp.select(growth, quality, {4}, config) == [8, 7, 6]  # 4 is fifth: outside the band, replaced
    assert 9 not in garp.select(growth, quality, {9}, config)  # failed the growth filter: out, held or not


def test_weights_respect_the_stock_and_sector_caps() -> None:
    score = pd.Series({i: (10.0 if i <= 3 else 1.0) for i in range(1, 41)})
    sectors = pd.Series({i: ("tech" if i <= 20 else f"s{i % 4}") for i in range(1, 41)})
    w = garp.capped_weights(score, sectors, stock_cap=0.05, sector_cap=0.40)
    assert w.sum() == pytest.approx(1.0)
    assert w.max() <= 0.05 + 1e-9
    assert w.groupby(sectors).sum()["tech"] <= 0.40 + 1e-9


def test_trades_follow_the_third_friday_after_each_reference_date() -> None:
    days = pd.bdate_range("2015-05-01", "2015-12-31")
    rebalances = pd.DatetimeIndex([pd.Timestamp("2015-05-29"), pd.Timestamp("2015-06-30"), pd.Timestamp("2015-11-30")])
    trades = garp.schedule(rebalances, days, (5, 11))
    assert trades.to_dict() == {
        pd.Timestamp("2015-05-29"): pd.Timestamp("2015-06-19"),
        pd.Timestamp("2015-11-30"): pd.Timestamp("2015-12-18"),
    }


def test_per_share_growth_compounds_and_needs_positive_amounts() -> None:
    now = pd.DataFrame({1: [133.1], 2: [100.0], 3: [-5.0]}, index=DAY)
    shares_now = pd.DataFrame({1: [100.0], 2: [200.0], 3: [10.0]}, index=DAY)
    then = pd.DataFrame({1: [100.0], 2: [100.0], 3: [5.0]}, index=DAY)
    shares_then = pd.DataFrame({1: [100.0], 2: [100.0], 3: [10.0]}, index=DAY)  # already in today's shares
    growth = per_share_growth(now, shares_now, then, shares_then, 3).iloc[0]
    assert growth[1] == pytest.approx(0.10)  # 1.331x in three years: 10 % a year
    assert growth[2] == pytest.approx(0.5 ** (1 / 3) - 1)  # the same profit over twice the shares: per share fell
    assert np.isnan(growth[3])  # a loss now: no compound rate
