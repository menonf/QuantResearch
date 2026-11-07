from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.research import ml
from equity_research.research import model_choice as mc
from equity_research.research.data import UniverseData
from equity_research.research.registry import HoldoutClosedError
from equity_research.research.strategy import StrategyConfig

MONTH_ENDS = pd.date_range("2006-01-31", "2011-12-31", freq="ME")
NAMES = list(range(1, 41))


def _features(values: np.ndarray, dates: pd.DatetimeIndex = MONTH_ENDS) -> pd.DataFrame:
    """``(date, listing_id)`` x one feature ``f``, from a ``date x listing`` array."""
    wide = pd.DataFrame(values, index=dates, columns=NAMES).rename_axis(index="date", columns="listing_id")
    long = wide.reset_index().melt(id_vars="date", var_name="listing_id", value_name="f")
    return long.set_index(["date", "listing_id"]).sort_index()


def test_every_candidate_fits_and_predicts_with_its_registered_settings() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    x[::7, 1] = np.nan  # a missing ratio is imputed, not an error
    y = x[:, 0] + rng.normal(0, 0.1, 300)
    for name, candidate in mc.CANDIDATES.items():
        predicted = candidate.build().fit(x, y).predict(x[:5])
        assert predicted.shape == (5,) and np.isfinite(predicted).all(), name
    assert set(mc.PROTOCOL["candidates"]) == set(mc.CANDIDATES)  # the protocol registers every candidate


def test_a_sampled_model_is_fitted_on_at_most_its_rows() -> None:
    class Counter:
        rows = 0

        def fit(self, x: np.ndarray, y: np.ndarray) -> "Counter":
            Counter.rows = len(x)
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    mc.Sampled(Counter(), 50).fit(np.zeros((200, 2)), np.zeros(200))
    assert Counter.rows == 50
    mc.Sampled(Counter(), 500).fit(np.zeros((200, 2)), np.zeros(200))
    assert Counter.rows == 200


def test_rank_ic_is_spearman_across_members_and_needs_enough_names() -> None:
    dates = MONTH_ENDS[:3]
    truth = pd.DataFrame(np.tile(np.arange(40.0), (3, 1)), index=dates, columns=NAMES)
    members = pd.DataFrame(True, index=dates, columns=NAMES)
    assert mc.rank_ic(truth**3, truth, members).tolist() == pytest.approx([1.0] * 3)  # ranks, not values
    assert mc.rank_ic(-truth, truth, members).tolist() == pytest.approx([-1.0] * 3)
    members.iloc[2, 15:] = False  # 15 members left on the last date: too few to judge
    assert np.isnan(mc.rank_ic(truth, truth, members).iloc[2])
    noisy = truth.copy()
    noisy[40] = -1000.0  # a non-member's wild prediction does not count
    members.iloc[:, 39] = False
    assert mc.rank_ic(noisy, truth, members).iloc[0] == pytest.approx(1.0)


def test_newey_west_t_is_the_plain_t_without_lags_and_smaller_with_overlap() -> None:
    rng = np.random.default_rng(3)
    iid = pd.Series(0.05 + rng.normal(0, 0.1, 240))
    plain = iid.mean() / (iid.std(ddof=0) / np.sqrt(len(iid)))
    assert mc.newey_west_t(iid, lags=0) == pytest.approx(plain)
    overlapping = pd.Series(np.convolve(rng.normal(0, 0.1, 251), np.ones(12) / 12, mode="valid") + 0.05)
    assert mc.newey_west_t(overlapping) < mc.newey_west_t(overlapping, lags=0) / 2


def test_the_choice_uses_only_ics_known_by_january() -> None:
    dates = pd.date_range("2010-01-31", "2016-12-31", freq="ME")
    later = dates.year >= 2013
    ic = pd.DataFrame({"a": 0.1, "b": np.where(later, 0.5, 0.0)}, index=dates)
    table = mc.selection_table(ic, dates)
    # a date's IC is known a year after it: each January sees the dates up to two Decembers before
    assert table["dates"].tolist() == [0, 0, 12, 24, 36, 48, 60]
    assert table["chosen"].tolist() == [mc.FALLBACK, mc.FALLBACK, "a", "a", "a", "b", "b"]
    # in 2014 b's better 2013 was in the table but not yet known - so a was kept
    assert table.loc[2015, "b"] == pytest.approx(0.5 * 12 / 48)


def test_the_chosen_score_takes_each_year_from_that_years_choice_or_the_ensemble() -> None:
    dates = pd.date_range("2010-01-31", "2011-12-31", freq="ME")
    members = pd.DataFrame(True, index=dates, columns=NAMES)
    up = pd.DataFrame(np.tile(np.arange(40.0), (len(dates), 1)), index=dates, columns=NAMES)
    predictions = {"up": up, "down": -up}
    score = mc.chosen_score(predictions, pd.Series({2010: mc.FALLBACK, 2011: "down"}), members)
    later = dates.year == 2011
    pd.testing.assert_frame_equal(score[later], -up[later])
    assert np.allclose(score.loc["2010"].to_numpy(), 41 / 80)  # up and down cancel: every rank averages to the middle


def test_the_ensemble_averages_ranks_of_members_only() -> None:
    dates = MONTH_ENDS[:2]
    members = pd.DataFrame(True, index=dates, columns=NAMES)
    members[1] = False
    a = pd.DataFrame(np.tile(np.arange(40.0), (2, 1)), index=dates, columns=NAMES)
    b = a * 1000  # far wider predictions, the same ranks: no extra say
    reversed_ = -a
    pooled = mc.ensemble({"a": a, "b": b, "reversed": reversed_}, members)
    assert pooled[1].isna().all()
    # a and b agree and outvote the reversed model two to one, whatever the spread of b's predictions
    assert pooled.iloc[0, 1:].is_monotonic_increasing
    assert pooled.iloc[0, 1:].to_numpy() == pytest.approx((2 * np.arange(1, 40) + (40 - np.arange(1, 40))) / 3 / 39)


def test_predictions_are_stored_and_read_back_for_the_same_inputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(len(MONTH_ENDS), len(NAMES)))
    target = pd.DataFrame(0.1 * x + rng.normal(0, 0.02, x.shape), index=MONTH_ENDS, columns=NAMES)
    members = pd.DataFrame(True, index=MONTH_ENDS, columns=NAMES)
    seen: list[tuple[str, bool]] = []
    first = mc.candidate_predictions(
        _features(x),
        target,
        members,
        MONTH_ENDS,
        ["linear"],
        root=tmp_path,
        progress=lambda n, _, s: seen.append((n, s)),
    )
    again = mc.candidate_predictions(
        _features(x),
        target,
        members,
        MONTH_ENDS,
        ["linear"],
        root=tmp_path,
        progress=lambda n, _, s: seen.append((n, s)),
    )
    assert seen == [("linear", False), ("linear", True)]
    pd.testing.assert_frame_equal(first["linear"], again["linear"], check_names=False, check_freq=False)
    assert first["linear"].loc["2011"].notna().all().all() and first["linear"].loc[:"2008"].isna().all().all()
    ic = mc.information_coefficients(first, target, members)
    assert ic["linear"].dropna().min() > 0.8  # the relation holds through time, so it is learned
    changed = mc.candidate_predictions(_features(x * 2), target, members, MONTH_ENDS, ["linear"], root=tmp_path)
    assert not changed["linear"].equals(first["linear"])  # other inputs, other predictions - not the stored ones
    # the same inputs in another row and column order, and differing in the last bits (as a database load may return
    # them), are the same inputs
    shuffled = _features(x * (1 + 1e-15)).sample(frac=1.0, random_state=1)
    seen.clear()
    mc.candidate_predictions(
        shuffled,
        target.iloc[::-1, ::-1],
        members.iloc[:, ::-1],
        MONTH_ENDS,
        ["linear"],
        root=tmp_path,
        progress=lambda n, _, s: seen.append((n, s)),
    )
    assert seen == [("linear", True)]
    assert len(list((tmp_path / "ml_candidates").glob("linear-*.parquet"))) == 2


def test_distressed_companies_rank_last() -> None:
    dates = MONTH_ENDS[-1:]
    index = pd.MultiIndex.from_product([dates, [1, 2]], names=["date", "listing_id"])
    ratios = pd.DataFrame(
        {"(CA-CL)/TA": [-0.5, 0.2], "RE/TA": [-1.0, 0.3], "EBIT/TA": [-0.2, 0.1], "Book Equity/TL": [0.1, 1.0]},
        index=index,
    )
    score = pd.DataFrame({1: [5.0], 2: [1.0]}, index=dates)
    out = ml.distress_last(score, ratios)
    assert out.loc[dates[0], 1] == -np.inf and out.loc[dates[0], 2] == 1.0


def _universe() -> UniverseData:
    days = pd.bdate_range("2019-12-02", "2021-12-31")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (len(days), len(NAMES))), index=days, columns=NAMES)
    adjusted = 100 * (1 + returns).cumprod()
    members = pd.DataFrame(True, index=days, columns=NAMES)
    zero = pd.DataFrame(0.0, index=days, columns=NAMES)
    rebalances = pd.DatetimeIndex(returns.groupby(days.to_period("M")).apply(lambda m: m.index.max()))
    panel = pd.DataFrame(1.0, index=rebalances, columns=NAMES)
    return UniverseData(
        universe="TEST",
        close=adjusted,
        adjusted=adjusted,
        returns=returns,
        members=members,
        segment=pd.DataFrame("large", index=days, columns=NAMES),
        cost_bps=zero,
        borrow_bps=zero,
        buy_tax_bps=zero,
        rebalances=rebalances,
        ratios=pd.DataFrame(),
        benchmark=returns.mean(axis=1),
        market_cap=panel,
        share_change=panel,
        earnings_surprise=panel,
        eps_growth=panel,
        sales_growth=panel,
    )


def test_a_score_is_run_as_a_fenced_long_only_book_from_the_start_date() -> None:
    data = _universe()
    config = StrategyConfig(
        family=mc.FAMILY, start="2020-06-01", end="2021-12-31", factors=(("ml:test", 1.0),), enter=0.3, exit_=0.4,
        long_only=True, beta_neutral=False, price_vendors=("x",), fundamentals_vendor="x",
    )  # fmt: skip
    score = pd.DataFrame(np.tile(np.arange(40.0), (len(data.rebalances), 1)), index=data.rebalances, columns=NAMES)
    result = mc.run_score(data, config, score, protocol={"test": True})
    held = result.book.loc[result.book.index.min()]
    assert result.book.index.min() >= pd.Timestamp("2020-06-01")
    assert set(held[held > 0].index) == set(NAMES[-12:])  # the top 30 %: the highest scores
    assert result.metrics["cagr"] == result.metrics["cagr"] and result.run_id is None  # measured, not registered
    with pytest.raises(HoldoutClosedError):
        mc.run_score(data, StrategyConfig(family=mc.FAMILY, end="2030-01-01"), score, protocol={})


def test_toy_fits_draw_every_candidate_through_the_same_points() -> None:
    x, y = mc.toy_data(200)
    grid = np.linspace(-3, 3, 25)
    fits = mc.toy_fits(x, y, grid)
    assert list(fits.columns) == list(mc.CANDIDATES) and fits.shape == (25, len(mc.CANDIDATES))
    assert np.isfinite(fits.to_numpy()).all()
