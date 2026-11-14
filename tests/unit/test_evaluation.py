import numpy as np
import pandas as pd
import pytest

from equity_research.research import evaluation as ev

DATES = pd.bdate_range("2020-01-01", periods=40)
NAMES = list(range(1, 101))


def _panel(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES[: len(values)], columns=NAMES)


def test_spearman_and_pearson_ics_and_their_limits() -> None:
    base = np.tile(np.arange(100.0), (5, 1))
    score, outcome = _panel(base), _panel(base**3)  # the same order, curved
    assert ev.ic_series(score, outcome).tolist() == pytest.approx([1.0] * 5)
    pearson = ev.ic_series(score, outcome, method="pearson")
    assert (pearson < 1.0).all() and (pearson > 0.8).all()  # values, not ranks: the curve costs a little
    members = pd.DataFrame(True, index=score.index, columns=NAMES)
    members.iloc[0, 20:] = False  # 20 members on the first date: too few
    assert np.isnan(ev.ic_series(score, outcome, members).iloc[0])
    with pytest.raises(ValueError, match="spearman"):
        ev.ic_series(score, outcome, method="kendall")


def test_newey_west_and_its_p_value() -> None:
    rng = np.random.default_rng(0)
    iid = pd.Series(0.02 + rng.normal(0, 0.1, 400))
    plain = iid.mean() / (iid.std(ddof=0) / np.sqrt(len(iid)))
    assert ev.newey_west_t(iid, 0) == pytest.approx(plain)
    overlapping = pd.Series(np.convolve(rng.normal(0, 0.1, 420), np.ones(21) / 21, mode="valid") + 0.01)
    assert abs(ev.newey_west_t(overlapping, 20)) < abs(ev.newey_west_t(overlapping, 0)) / 2
    assert ev.p_value(1.959964) == pytest.approx(0.05, abs=1e-5)
    assert np.isnan(ev.p_value(float("nan")))


def test_overlap_lags_follow_the_label_and_the_step() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=300)
    assert ev.overlap_lags(calendar, calendar, 21) == 20  # daily dates, 21-day label
    months = calendar[::21]
    assert ev.overlap_lags(months, calendar, 21) == 0  # monthly dates, monthly label: no overlap
    assert ev.overlap_lags(months, calendar, 252) == 11


def test_ic_summary_and_decay() -> None:
    rng = np.random.default_rng(1)
    returns = _panel(rng.normal(0, 0.01, (40, 100)))
    tomorrow = returns.shift(-1)
    score = tomorrow + rng.normal(0, 0.01, tomorrow.shape)  # knows tomorrow's return, nothing after
    decay = ev.ic_decay(score, returns, horizons=(1, 5, 10))
    one, five, ten = decay["mean_ic"].to_numpy(dtype=float)
    assert one > 0.5
    assert one > five > ten > 0  # the information fades
    summary = ev.ic_summary(ev.ic_series(score, tomorrow), lags=0)
    assert summary.loc["ic", "positive"] == 1.0 and summary.loc["ic", "dates"] == 39


def test_quantiles_turnover_and_rank_autocorrelation() -> None:
    score = _panel(np.tile(np.arange(100.0), (6, 1)))
    outcome = _panel(np.tile(np.arange(100.0) / 100, (6, 1)))
    table = ev.quantile_returns(score, outcome)
    assert np.array_equal(table.columns.to_numpy(), np.arange(1, 6))
    assert table.iloc[0].is_monotonic_increasing
    assert ev.quantile_spread(table).iloc[0] == pytest.approx(0.895 - 0.095)
    assert ev.top_turnover(score).iloc[1:].tolist() == [0.0] * 5  # the same top every day
    flipped = score.copy()
    flipped.iloc[1::2] = -flipped.iloc[1::2]  # the order reverses every other day
    assert ev.top_turnover(flipped).iloc[1:].tolist() == [1.0] * 5
    assert ev.rank_autocorrelation(score).iloc[1:].tolist() == pytest.approx([1.0] * 5)
    assert ev.rank_autocorrelation(flipped).iloc[1:].tolist() == pytest.approx([-1.0] * 5)


def test_benjamini_hochberg_controls_the_share_of_false_discoveries() -> None:
    p = pd.Series({"a": 0.001, "b": 0.008, "c": 0.039, "d": 0.041, "e": 0.30})
    found = ev.benjamini_hochberg(p, q=0.05)
    # step-up: the largest k with p_(k) <= q k / m is 2 (0.008 <= 0.02; 0.039 > 0.03, 0.041 > 0.04)
    assert found.to_dict() == {"a": True, "b": True, "c": False, "d": False, "e": False}
    assert ev.benjamini_hochberg(pd.Series(dtype=float)).empty


def test_triage_lists_signals_with_their_verdicts() -> None:
    rng = np.random.default_rng(2)
    outcome = _panel(rng.normal(0, 0.02, (40, 100)))
    good = outcome + rng.normal(0, 0.02, outcome.shape)
    noise = _panel(rng.normal(size=(40, 100)))
    ledger = ev.triage({"noise": noise, "good": good}, outcome, lags=0, q=0.10)
    assert ledger.index[0] == "good" and bool(ledger.loc["good", "discovery"])
    assert not bool(ledger.loc["noise", "discovery"])
    assert list(ledger.columns) == [
        "mean_ic",
        "t_nw",
        "p_value",
        "discovery",
        "spread",
        "turnover",
        "rank_autocorrelation",
    ]


def test_moves_against_costs_and_the_ic_the_law_requires() -> None:
    returns = _panel(np.tile(np.r_[0.01, -0.01], (40, 50)))
    cost = _panel(np.full((40, 100), 10.0))  # 10 bp each way
    table = ev.move_against_cost(returns, cost, DATES[:30], horizons=(1, 2))
    assert table.loc[1, "median move"] == pytest.approx(0.01)
    assert table.loc[1, "round trip"] == pytest.approx(0.002)
    assert table.loc[1, "move / cost"] == pytest.approx(5.0)
    # 500 names a month: breadth 6,000 a year, so an information ratio of 0.5 needs an IC of 0.5 / sqrt(6000)
    assert ev.required_ic(0.5, 500, 12) == pytest.approx(0.5 / np.sqrt(6000))


def test_rank_correlations_between_signals() -> None:
    rng = np.random.default_rng(4)
    a = _panel(rng.normal(size=(40, 100)))
    table = ev.rank_correlations(
        {"a": a, "b": a * 2 + 1, "c": -a}, {"a": a, "noise": _panel(rng.normal(size=(40, 100)))}
    )
    assert table.loc["b", "a"] == pytest.approx(1.0) and table.loc["c", "a"] == pytest.approx(-1.0)
    assert abs(float(table.to_numpy()[0, 1])) < 0.05  # a against noise
    assert ev.rank_correlations({"a": a}).loc["a", "a"] == pytest.approx(1.0)


def test_a_flat_score_has_an_ic_of_zero_not_a_missing_one() -> None:
    outcome = _panel(np.tile(np.arange(100.0), (3, 1)))
    score = outcome.copy()
    score.iloc[1] = 7.0  # ranks nobody on the second date
    assert np.isnan(ev.ic_series(score, outcome).iloc[1])
    assert ev.ic_flat_zero(score, outcome).tolist() == pytest.approx([1.0, 0.0, 1.0])
    few = pd.DataFrame(True, index=score.index, columns=NAMES)
    few.iloc[1, 10:] = False  # too few members to judge: missing, flat or not
    assert np.isnan(ev.ic_flat_zero(score, outcome, few).iloc[1])
