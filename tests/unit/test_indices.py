import numpy as np
import pandas as pd
import pytest

from equity_research.analytics.indices import cap_weighted, equal_weighted, segments
from equity_research.data.panels import splits_between
from research_core.analytics.performance import tracking as ix
from research_core.analytics.performance.tracking import tracking

DAYS = pd.to_datetime(["2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02"])


def test_each_day_uses_yesterdays_market_values() -> None:
    returns = pd.DataFrame({1: [np.nan, 0.10, 0.0, 0.0], 2: [np.nan, 0.0, 0.10, 0.0]}, index=DAYS)
    caps = pd.DataFrame({1: [300.0, 330.0, 330.0, 330.0], 2: [100.0, 100.0, 110.0, 110.0]}, index=DAYS)
    members = pd.DataFrame(True, index=DAYS, columns=[1, 2])
    out = cap_weighted(returns, caps, members)
    assert out.loc[DAYS[1], "return"] == pytest.approx(0.75 * 0.10)  # 300 of 400 yesterday
    assert out.loc[DAYS[2], "return"] == pytest.approx(100 / 430 * 0.10)
    assert out["covered"].tolist() == [1.0, 1.0, 1.0]


def test_a_member_without_a_share_count_is_left_out_and_counted() -> None:
    returns = pd.DataFrame({1: [np.nan, 0.10], 2: [np.nan, 0.50]}, index=DAYS[:2])
    caps = pd.DataFrame({1: [300.0, 330.0], 2: [np.nan, np.nan]}, index=DAYS[:2])
    out = cap_weighted(returns, caps, pd.DataFrame(True, index=DAYS[:2], columns=[1, 2]))
    assert out["return"].tolist() == [pytest.approx(0.10)] and out["covered"].tolist() == [0.5]


def test_equal_weights_drift_hand_over_and_reset_each_quarter() -> None:
    returns = pd.DataFrame(
        {1: [np.nan, 0.10, 0.0, 0.0], 2: [np.nan, 0.0, 0.0, 0.10], 3: [np.nan, np.nan, 0.20, 0.0]}, index=DAYS
    )
    # 2 leaves after the 28th (the quarter's last day) and 3 joins: the reset gives 1 and 3 half each
    members = pd.DataFrame({1: [True] * 4, 2: [True, True, False, False], 3: [False, False, True, True]}, index=DAYS)
    out = equal_weighted(returns, members)
    assert out.loc[DAYS[1], "return"] == pytest.approx(0.05)  # half in each, 1 up 10 %
    assert out.loc[DAYS[2], "return"] == pytest.approx(0.10)  # reset: half in 3, up 20 %
    assert out["names"].tolist() == [2, 2, 2]


def test_a_leaver_between_resets_hands_its_weight_to_the_joiner() -> None:
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    returns = pd.DataFrame({1: [np.nan, 0.0, 0.0], 2: [np.nan, 1.0, 0.0], 3: [np.nan, np.nan, 0.5]}, index=days)
    members = pd.DataFrame({1: [True] * 3, 2: [True, True, False], 3: [False, False, True]}, index=days)
    out = equal_weighted(returns, members)
    # after the 3rd, 2 holds 2/3 of the book (it doubled); 3 takes that weight over and gains 50 %
    assert out.loc[days[2], "return"] == pytest.approx(2 / 3 * 0.5)


def test_share_counts_follow_splits_after_their_date_only() -> None:
    actions = pd.DataFrame(
        {
            "listing_id": [1, 1],
            "action_type": ["split", "split"],
            "ex_date": ["2024-02-01", "2024-06-03"],
            "ratio": [2.0, 3.0],
        }
    )
    factor = splits_between(
        actions,
        pd.Series([1, 1, 1, 2]),
        pd.Series(pd.to_datetime(["2024-01-15", "2024-02-01", "2024-01-15", "2024-01-15"])),
        pd.Series(pd.to_datetime(["2024-03-01", "2024-03-01", "2024-07-01", "2024-07-01"])),
    )
    assert factor.tolist() == [2.0, 1.0, 6.0, 1.0]  # a count dated on the ex-day is already in the new shares


def test_tracking_of_an_identical_series_is_perfect() -> None:
    r = pd.Series([0.01, -0.02, 0.015] * 100, index=pd.bdate_range("2024-01-01", periods=300))
    t = tracking(r, r)
    assert t["tracking_error"] == 0.0 and t["correlation"] == pytest.approx(1.0) and t["beta"] == pytest.approx(1.0)


def test_a_listing_in_two_indices_belongs_to_the_first_by_priority() -> None:
    days = pd.to_datetime(["2008-01-02", "2008-01-03", "2008-01-04"])
    spells = pd.DataFrame(
        {
            "index_id": [4, 6, 6],
            "listing_id": [1, 1, 2],
            "start_date": ["2005-01-01", "2005-01-01", "2008-01-03"],
            "end_date": ["2008-01-03", None, None],
        }
    )
    # 1 is left in the SmallCap (6) while in the FTSE 100 (4), as LSEG's events leave 3i Group in 2005-2009
    out = segments(spells, pd.DataFrame({"index_id": [4, 6], "code": ["FTSE100", "FTSESC"]}), days, pd.Index([1, 2, 3]))
    assert out[1].tolist() == ["FTSE100", "FTSESC", "FTSESC"]
    assert out[2].isna().tolist() == [True, False, False] and out[3].isna().all()


def test_the_tracking_summary_has_a_row_per_comparison_and_whole_days() -> None:
    days = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    theirs = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    pairs = {"same": (theirs, theirs), "noisy": (theirs + rng.normal(0, 0.001, len(days)), theirs)}
    summary = ix.tracking_summary(pairs)
    assert list(summary.index) == ["same", "noisy"] and summary["days"].dtype.kind == "i"
    assert summary.loc["same", "tracking_error"] == 0.0
    assert set(ix.TRACKING_FORMAT) == set(summary.columns)


def test_yearly_tracking_error_finds_the_year_a_copy_drifted() -> None:
    days = pd.bdate_range("2020-01-01", "2022-12-31")
    rng = np.random.default_rng(1)
    theirs = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    drift = np.where(days.year == 2021, rng.normal(0, 0.003, len(days)), 0.0)  # only 2021 goes wrong
    by_year = ix.yearly_tracking_errors({"copy": (theirs + drift, theirs)})["copy"]
    assert list(by_year.index) == [2020, 2021, 2022]
    assert by_year.loc[2020] == 0.0 and by_year.loc[2022] == 0.0
    assert by_year.loc[2021] == pytest.approx(0.003 * np.sqrt(252), rel=0.15)
