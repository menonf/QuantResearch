import numpy as np
import pandas as pd
import pytest

from fixed_income_research.analytics import replication as rp
from fixed_income_research.data.index_data import MEASURES, IndexData, broad_rating
from fixed_income_research.strategies import factor_strategy as fs
from fixed_income_research.strategies import factors as fx
from research_core.engine.backtest import run_backtest

NOTCHES = ("BB1", "BB2", "BB3", "B1", "B2", "B3", "CCC1")
INDUSTRIES = ("Energy", "Media", "Retail", "Healthcare")


def _index(months: int = 14, bonds: int = 80, seed: int = 0, missing: str | None = None) -> IndexData:
    """A small index: bonds of three each per issuer, most of them in most months, random but plausible numbers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    if missing is not None:
        dates = dates.drop([pd.Timestamp(missing)])
    ids = list(range(1, bonds + 1))
    n = len(dates)
    member = rng.random((n, bonds)) > 0.06

    def panel(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=dates, columns=ids).where(member)

    raw = rng.uniform(0.5, 2.0, (n, bonds)) * member
    weights = panel(raw / raw.sum(axis=1, keepdims=True))
    excess = panel(rng.normal(0.003, 0.02, (n, bonds)))
    rating = panel(np.tile(rng.choice(NOTCHES, bonds), (n, 1)))
    due = pd.Timestamp("2019-01-31") + pd.to_timedelta(rng.uniform(365, 3650, bonds), unit="D")
    maturity = pd.DataFrame(np.tile(due.to_numpy(), (n, 1)), index=dates, columns=ids).where(member)
    issuer = panel(np.tile([100.0 + (i - 1) // 3 for i in ids], (n, 1)))
    by_issuer = {100.0 + k: rng.choice(INDUSTRIES) for k in range(bonds)}
    industry = issuer.apply(lambda column: column.map(by_issuer))
    oas = panel(rng.uniform(200.0, 800.0, (n, bonds)))
    return IndexData(
        code="TEST",
        weights=weights,
        total_returns=excess + 0.002,
        excess_returns=excess,
        analytics={m: (oas if m == "oas_bp" else panel(rng.uniform(1.0, 5.0, (n, bonds)))) for m in MEASURES},
        rating=rating,
        maturity=maturity,
        issuer=issuer,
        industry=industry,
        names=pd.Series({i: f"bond {i}" for i in ids}),
        official=pd.Series(dtype=float, index=pd.DatetimeIndex([])),
    )


def test_months_are_counted_on_the_calendar_not_in_files() -> None:
    months = pd.to_datetime(["2019-01-31", "2019-02-28", "2019-04-30"])  # no file for March
    panel = pd.DataFrame({1: [1.0, 2.0, 4.0]}, index=months)
    back = fx.calendar_shift(panel, 1)[1]
    assert back.iloc[1] == 1.0 and np.isnan(back.iloc[2])  # April's month before is March, which has no file
    assert fx.calendar_shift(panel, 2).iloc[2, 0] == 2.0  # April's two months back is February
    assert fx.calendar_shift(panel, -1).iloc[0, 0] == 2.0 and pd.isna(fx.calendar_shift(panel, -1).iloc[1, 0])


def test_momentum_compounds_the_months_before_the_one_just_ended() -> None:
    months = pd.date_range("2019-01-31", periods=6, freq="ME")
    returns = pd.DataFrame({1: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}, index=months)
    mom = fx.momentum(returns, months=3, skip=1)
    assert mom.iloc[:3, 0].isna().all()  # three months, the latest skipped: April is the first with a window
    assert mom.iloc[3, 0] == pytest.approx(1.01 * 1.02 * 1.03 - 1)  # April: January to March
    assert mom.iloc[4, 0] == pytest.approx(1.02 * 1.03 * 1.04 - 1)  # May: February to April, May itself left out
    assert fx.momentum(returns, months=3, skip=0).iloc[2, 0] == pytest.approx(1.01 * 1.02 * 1.03 - 1)
    gap = returns.drop(months[2])  # March has no file: no window that needs it is complete
    assert fx.momentum(gap, months=3, skip=1).iloc[:, 0].isna().all()


def test_size_is_the_issuers_weight_all_its_bonds_together() -> None:
    month = pd.to_datetime(["2019-01-31"])
    weights = pd.DataFrame({1: [0.2], 2: [0.3], 3: [0.4], 4: [0.1]}, index=month)
    issuer = pd.DataFrame({1: [7.0], 2: [7.0], 3: [8.0], 4: [np.nan]}, index=month)
    assert fx.issuer_weight(weights, issuer).iloc[0].tolist() == pytest.approx([0.5, 0.5, 0.4, 0.1])


def test_low_risk_prefers_short_well_rated_bonds_and_fills_what_is_missing_with_the_median() -> None:
    month = pd.to_datetime(["2019-01-31"])
    years = pd.DataFrame({1: [1.0], 2: [9.0], 3: [5.0], 4: [np.nan], 5: [np.nan]}, index=month)
    rating = pd.DataFrame({1: ["BB1"], 2: ["CCC2"], 3: ["B1"], 4: ["BB1"], 5: [np.nan]}, index=month)
    risk = fx.riskiness(years, rating).iloc[0]
    assert risk[1] < risk[3] < risk[2]  # short and BB the safest, long and CCC the riskiest
    assert risk[4] == pytest.approx(0.5 * 2 / 3 + 0.5 * 2 / 12)  # median maturity rank, its own BB1 (2nd of 12)
    assert np.isnan(risk[5])  # not in the index that month: nothing to score
    position = fx.rating_position(pd.DataFrame({1: ["BBB3"], 2: ["C"], 3: ["D"]}))
    assert position.iloc[0, :2].tolist() == pytest.approx([1 / 12, 1.0]) and pd.isna(position.iloc[0, 2])


def test_value_finds_the_bond_whose_spread_is_above_its_peers() -> None:
    rng = np.random.default_rng(1)
    month = pd.DataFrame(
        {
            "years": rng.uniform(1, 10, 60),
            "change": rng.normal(0, 30, 60),
            "rating": rng.choice(["BB1", "B1", "CCC1"], 60),
            "industry": rng.choice(["Energy", "Media"], 60),
        }
    )
    step = month["rating"].map({"BB1": 0.0, "B1": 150.0, "CCC1": 500.0})
    month["oas"] = 200 + 15 * month["years"] + 0.5 * month["change"] + step + (month["industry"] == "Media") * 40
    exact = fx.cheapness(month)
    assert exact.abs().max() < 1e-6  # every spread on the fitted line: nothing is cheap
    month["oas"] = month["oas"].where(month.index != 7, month["oas"] * 1.5)
    gap = fx.cheapness(month)
    assert gap.idxmax() == 7 and gap[7] > 30  # well above its peers' line
    assert fx.cheapness(month.head(5)).isna().all()  # fewer bonds than regressors: no fit


def test_the_composite_counts_a_missing_score_as_average() -> None:
    month = pd.to_datetime(["2019-01-31"])
    members = pd.DataFrame({1: [True], 2: [True], 3: [True], 4: [False]}, index=month)
    a = pd.DataFrame({1: [3.0], 2: [2.0], 3: [1.0], 4: [9.0]}, index=month)
    b = pd.DataFrame({1: [np.nan], 2: [1.0], 3: [2.0], 4: [9.0]}, index=month)
    blend = fx.composite({"a": a, "b": b}, {"a": 3.0, "b": 1.0}, members).iloc[0]
    assert blend[1] == pytest.approx((3 * 1.0 + 0.5) / 4)  # top of a, average on b
    assert blend[3] == pytest.approx((3 * (1 / 3) + 1.0) / 4)
    assert np.isnan(blend[4])  # not a member
    with pytest.raises(ValueError, match="positive"):
        fx.composite({"a": a}, {"a": 0.0}, members)


def test_ratings_fall_into_their_bands() -> None:
    rating = pd.DataFrame({1: ["AAA", "BBB2"], 2: ["BB3", "CC"], 3: ["B1", "D"], 4: [np.nan, "A1"]})
    assert broad_rating(rating).fillna("-").to_numpy().tolist() == [
        ["AAA", "BB", "B", "-"],
        ["BBB", "CCC and below", "CCC and below", "A"],
    ]


def test_a_bond_that_left_earns_the_index_and_the_index_book_rebuilds_the_index() -> None:
    data = _index(missing="2019-09-30")
    earned = rp.with_leavers(data.weights, data.excess_returns)
    rebuilt = rp.index_returns(data.weights, data.excess_returns)["return"]
    held = data.weights.shift(1)
    left = held.gt(0) & data.excess_returns.isna()
    d = left.index[int(left.any(axis=1).to_numpy().argmax())]
    gone = left.loc[[d]].to_numpy()[0]
    assert gone.sum() > 0
    assert earned.loc[[d]].to_numpy()[0][gone] == pytest.approx(np.full(int(gone.sum()), rebuilt[d]))
    assert earned.loc["2019-10-31"].isna().all()  # the month after a missing file is not measured
    book = data.weights.fillna(0.0)
    index = run_backtest(book, earned, cost_bps=0.0, periods_per_year=12).net
    both = pd.concat({"book": index, "rebuilt": rebuilt}, axis=1).dropna()
    assert len(both) == len(rebuilt) and both["book"].to_numpy() == pytest.approx(both["rebuilt"].to_numpy())


def test_no_score_reads_a_later_file() -> None:
    data = _index(seed=2)
    config = fs.FactorConfig(family="test", momentum_months=3)
    check = fs.point_in_time_check(data, config, data.dates[[4, 8, 12]])
    assert (check.drop(columns="bonds") == 0.0).all().all()
    assert (check["bonds"] > 50).all()


def test_a_run_backtests_every_fifth_and_measures_the_best_against_the_index() -> None:
    data = _index(seed=3)
    config = fs.FactorConfig(family="test", start=str(data.dates[7].date()), end=str(data.dates[-1].date()))
    run = fs.run_on(data, config)
    assert set(run.backtests) == {"value", "momentum", "size", "low_risk", "composite"}
    for fifths in run.books.values():
        assert sorted(fifths) == [1, 2, 3, 4, 5]
        assert all(b.sum(axis=1).to_numpy() == pytest.approx(1.0) for b in fifths.values())
    assert list(run.index.net.index) == list(data.dates[8:])  # the first book earns the month after it is formed
    rebuilt = rp.index_returns(data.weights, data.excess_returns)["return"]
    assert run.benchmark.to_numpy() == pytest.approx(rebuilt.reindex(run.benchmark.index).to_numpy())
    small = run.books["size"][5].iloc[0]
    sizes = fx.issuer_weight(data.weights, data.issuer).loc[run.dates[0]]
    assert sizes[small > 0].max() <= sizes[run.books["size"][1].iloc[0] > 0].min()  # best fifth: smallest issuers
    table = fs.quantile_table(run)
    assert len(table) == 25 and {"information_ratio", "up_capture", "turnover"} <= set(table.columns)
    assert table.loc[("composite", 5), "cagr"] == pytest.approx(run.metrics["composite"]["cagr"])
    sorted_by = fs.fan(table)
    assert list(sorted_by.index) == list(run.backtests) and list(sorted_by.columns) == [1, 2, 3, 4, 5, "5 less 1"]
    assert sorted_by.loc["size", "5 less 1"] == pytest.approx(
        np.subtract(*table["active_return"].loc[[("size", 5), ("size", 1)]].to_numpy(dtype=float))
    )
    assert run.run_ids == dict.fromkeys(run.backtests)  # no engine: measured, not registered
    spread = fs.spreads(run)
    assert spread["value"].to_numpy() == pytest.approx(
        (run.backtests["value"][5].gross - run.backtests["value"][1].gross).to_numpy()
    )
    assert set(fs.spread_table(spread).columns) == {"cagr", "vol", "sharpe", "positive", "t", "months"}
    tilt = fs.active_weights(run, "composite", 5, data.industry)
    assert tilt.sum(axis=1).to_numpy() == pytest.approx(0.0)  # a fifth's overweights are the index's underweights
    formed, after = data.dates[[3]], data.dates[[4]]
    later = data.excess_returns.loc[after].to_numpy()[0]
    stay = data.weights.loc[formed].notna().to_numpy()[0] & ~np.isnan(later)
    assert stay.sum() > 50  # what a bond held at a month end earned the month after, for the ICs
    assert run.outcome.loc[formed].to_numpy()[0][stay] == pytest.approx(later[stay])
    by_value = fs.run_on(
        data, fs.FactorConfig(family="test", start=config.start, end=config.end, weighting="market_value")
    )
    best = by_value.books["value"][5].iloc[0]
    held = best[best > 0]
    w = data.weights.loc[run.dates[[0]]].iloc[0][held.index]
    assert (held / held.sum()).to_numpy() == pytest.approx((w / w.sum()).to_numpy())


def test_a_screen_ranks_the_members_best_first() -> None:
    data = _index(seed=4)
    scores = fs.factor_scores(data, fs.FactorConfig(family="test", momentum_months=3))
    last = data.dates[-1]
    table = fs.screen(data, scores, last, pd.Series({100: "Issuer A"}))
    assert len(table) == int(data.weights.loc[[last]].notna().to_numpy().sum())
    assert table["composite"].is_monotonic_decreasing
    assert set(table.loc[table.index.isin([1, 2, 3]), "issuer"]) <= {"Issuer A"}


def test_a_config_refuses_what_it_does_not_know() -> None:
    with pytest.raises(ValueError, match="weighting"):
        fs.FactorConfig(family="x", weighting="by_magic")
    with pytest.raises(ValueError, match="carry"):
        fs.FactorConfig(family="x", factors=(("carry", 1.0),))
    assert "factors" not in fs.registered_settings(fs.FactorConfig(family="x"), "value")
    assert fs.registered_settings(fs.FactorConfig(family="x"), "composite")["signal"] == "composite"
