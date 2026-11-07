import numpy as np
import pandas as pd
import pytest

from equity_research.research import model_choice as mc
from equity_research.research import model_choice_v2 as v2
from equity_research.research import model_choice_v3 as v3

DATES = pd.date_range("2020-01-31", periods=3, freq="ME")
NAMES = list(range(1, 101))


def _members() -> pd.DataFrame:
    return pd.DataFrame(True, index=DATES, columns=NAMES)


def test_the_target_keeps_the_average_but_clips_the_extremes() -> None:
    rng = np.random.default_rng(0)
    names = list(range(1, 1001))  # an index's worth of members: the 99th percentile is the tenth best
    returns = pd.DataFrame(rng.normal(0.08, 0.3, (3, 1000)), index=DATES, columns=names)
    returns.iloc[0, 0] = 83.6  # +8,360 %
    members = pd.DataFrame(True, index=DATES, columns=names)
    members.iloc[:, -1] = False
    target = v3.trimmed_excess(returns, members)
    assert target.mean(axis=1).abs().max() < 1e-12  # nothing common to the date
    assert target[1000].isna().all()  # non-members are not targets
    assert target.to_numpy()[0, 0] < 1.5  # the outlier is clipped to the date's 99th percentile, not ranked away
    # within the clipping bounds the distances are kept: a return 0.1 higher is a target 0.1 higher
    middle = returns.iloc[1, :999].sort_values().index[[400, 600]]
    gap = returns.loc[DATES[1], middle[1]] - returns.loc[DATES[1], middle[0]]
    assert target.loc[DATES[1], middle[1]] - target.loc[DATES[1], middle[0]] == pytest.approx(gap)


def test_flat_finds_a_forecast_that_ranks_nobody() -> None:
    p = pd.DataFrame(np.tile(np.arange(100.0), (3, 1)), index=DATES, columns=NAMES)
    p.iloc[1] = 0.25
    members = _members()
    assert v3.flat(p, members).tolist() == [False, True, False]
    p.iloc[2, 5] = 7.0  # differs only on a non-member: still flat among members
    p.iloc[2] = 0.5
    p.iloc[2, 5] = 7.0
    members.iloc[2, 5] = False
    assert v3.flat(p, members).iloc[2]


def test_the_top_spread_is_what_the_top_thirty_percent_earned_over_the_average() -> None:
    outcome = pd.DataFrame(np.tile(np.arange(100.0), (3, 1)), index=DATES, columns=NAMES)
    perfect = outcome.copy()
    p = perfect.copy()
    p.iloc[1] = 1.0  # flat on the second date
    spread = v3.top_spread(p, outcome, _members())
    assert spread.iloc[0] == pytest.approx(np.arange(70, 100).mean() - np.arange(100).mean())
    assert spread.iloc[1] == 0.0  # a flat forecast earns nothing over the average, rather than going missing
    members = _members()
    members.iloc[2, 20:] = False  # 20 members: too few to judge
    assert np.isnan(v3.top_spread(perfect, outcome, members).iloc[2])
    table = v3.top_spreads({"right": perfect, "wrong": -perfect}, outcome, _members())
    assert bool((table["right"].to_numpy() > 0).all()) and bool((table["wrong"].to_numpy() < 0).all())


def test_a_flat_forecast_has_an_ic_of_zero_not_a_missing_one() -> None:
    returns = pd.DataFrame(np.tile(np.arange(100.0), (3, 1)), index=DATES, columns=NAMES)
    p = returns.copy()
    p.iloc[1] = 3.0
    ic = v3.information_coefficients({"m": p}, returns, _members())
    assert ic["m"].tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert bool(mc.information_coefficients({"m": p}, returns, _members()).isna().iloc[1, 0])  # version 1's rule


def test_a_flat_score_holds_the_largest_companies_instead_of_cash() -> None:
    score = pd.DataFrame(np.tile(np.arange(100.0), (3, 1)), index=DATES, columns=NAMES)
    score.iloc[1] = 0.0
    caps = pd.DataFrame(np.tile(np.arange(100.0)[::-1] + 1, (3, 1)), index=DATES, columns=NAMES)
    out = v3.flat_to_index(score, _members(), caps)
    pd.testing.assert_series_equal(out.iloc[1], caps.iloc[1])
    pd.testing.assert_frame_equal(out.iloc[[0, 2]], score.iloc[[0, 2]])


def test_quintile_outcomes_separate_the_typical_year_from_the_average_one() -> None:
    rng = np.random.default_rng(1)
    feature = pd.DataFrame(np.tile(np.arange(1, 101) / 100, (3, 1)), index=DATES, columns=NAMES)
    # the top fifth (most volatile) usually lose a little, and one in ten soars
    outcome = pd.DataFrame(rng.normal(0, 0.01, (3, 100)), index=DATES, columns=NAMES)
    outcome.iloc[:, 80:] = -0.05
    outcome.iloc[:, 80::10] = 1.0
    table = v3.quintile_outcomes(feature, outcome, _members())
    low, high = table.loc[1].to_numpy(dtype=float), table.loc[5].to_numpy(dtype=float)  # [mean, median]
    assert high[1] < low[1] and high[0] > low[0]


def test_leanings_measure_what_each_model_follows() -> None:
    f = pd.DataFrame(np.random.default_rng(2).normal(size=(3, 100)), index=DATES, columns=NAMES)
    table = v3.leanings({"follows": f * 3, "opposes": -f}, {"volatility": f})
    assert table.loc["follows", "volatility"] == pytest.approx(1.0)
    assert table.loc["opposes", "volatility"] == pytest.approx(-1.0)


def test_version_3_keeps_version_2s_learners_in_a_family_of_its_own() -> None:
    assert v3.CANDIDATES == v2.CANDIDATES
    assert len({mc.FAMILY, v2.FAMILY, v3.FAMILY}) == 3
    assert set(v3.PROTOCOL["candidates"]) == set(v3.CANDIDATES)
    long = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_product([DATES[:2], [1, 2]], names=["date", "listing_id"]),
    )
    assert v3.feature_panel(long, "x").to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]
