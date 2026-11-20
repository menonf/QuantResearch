import numpy as np
import pandas as pd
import pytest

from equity_research.analytics import dashboard as db
from equity_research.strategies.factors import dividend_yield
from research_core.analytics.performance import holdings, periods

DAYS = pd.bdate_range("2019-01-01", "2024-12-31")
ASOF = pd.Timestamp("2024-12-31")


def test_period_returns_compound_and_turn_into_yearly_rates_beyond_a_year() -> None:
    daily = pd.Series(0.0004, index=DAYS)
    late = daily[daily.index >= pd.Timestamp("2023-06-01")]  # does not cover three years
    table = periods.period_returns({"steady": daily, "late": late}, ASOF)
    days_in_year = ((ASOF - pd.DateOffset(years=1) < DAYS) & (DAYS <= ASOF)).sum()
    assert table.loc["steady", "12M"] == pytest.approx(1.0004**days_in_year - 1)
    three = ((ASOF - pd.DateOffset(years=3) < DAYS) & (DAYS <= ASOF)).sum()
    assert table.loc["steady", "3Y"] == pytest.approx((1.0004**three) ** (1 / 3.0) - 1, rel=1e-3)
    assert table[["3Y"]].loc[["late"]].isna().all().all() and table[["10Y"]].loc[["steady"]].isna().all().all()
    assert table.loc["steady", "YTD"] == pytest.approx(table.loc["steady", "12M"], rel=1e-2)


def test_a_book_measured_against_itself_has_no_excess_and_no_tracking_error() -> None:
    rng = np.random.default_rng(1)
    daily = pd.Series(rng.normal(0.0004, 0.01, len(DAYS)), index=DAYS)
    excess, te = periods.relative_to({"same": daily}, daily, ASOF)
    assert excess.loc["same"].dropna().abs().max() == pytest.approx(0.0, abs=1e-12)
    assert te.loc["same"].dropna().max() == pytest.approx(0.0, abs=1e-12)
    trailing = periods.trailing_excess(daily + 0.001, daily, days=63)
    assert (trailing > 0).all()


def test_overlap_active_share_and_concentration() -> None:
    a = pd.Series({1: 0.5, 2: 0.5})
    b = pd.Series({2: 0.5, 3: 0.5})
    overlap = holdings.portfolio_overlap({"a": a, "b": b})
    assert overlap.loc["a", "a"] == pytest.approx(1.0) and overlap.loc["a", "b"] == pytest.approx(0.5)
    assert holdings.active_share(a, a) == 0.0 and holdings.active_share(a, b) == pytest.approx(0.5)
    assert holdings.concentration(pd.Series(0.01, index=range(100))) == pytest.approx(100.0)
    assert holdings.concentration(pd.Series({1: 1.0})) == pytest.approx(10_000.0)
    sectors = pd.Series({1: "IT", 2: "IT", 3: "Energy"})
    assert holdings.sector_weights(b, sectors).to_dict() == {"Energy": 0.5, "IT": 0.5}


def test_the_benchmark_has_no_exposure_of_its_own() -> None:
    weights = pd.Series({1: 0.6, 2: 0.4})
    scores = pd.DataFrame({"Value": [1.0, -1.0], "Quality": [0.5, 0.0]}, index=[1, 2])
    assert holdings.exposures(weights, weights, scores).abs().max() == 0.0
    tilted = holdings.exposures(pd.Series({1: 1.0}), weights, scores)
    assert tilted["Value"] == pytest.approx(1.0 - 0.2)  # all in the cheap stock, against 0.6 * 1 + 0.4 * -1


def test_the_dividend_yield_is_what_total_returns_add_to_price_returns() -> None:
    days = pd.bdate_range("2020-01-01", periods=300)
    adjusted = pd.DataFrame({1: np.full(300, 100.0)}, index=days)
    returns = pd.DataFrame({1: np.zeros(300)}, index=days)
    returns.iloc[[50, 110, 170, 230], 0] = 0.01  # four quarterly dividends of 1 % of the price
    assert dividend_yield(returns, adjusted).iloc[-1, 0] == pytest.approx(0.04)


def test_the_dashboards_charts_draw() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    from research_core import charts  # noqa: PLC0415

    charts.style()
    fig = charts.ranked_bars(pd.Series({"a": 0.1, "b": -0.02}), "Bars", percent=True, reference=("Index", 0.05))
    assert any("Index" in t.get_text() for t in fig.axes[0].texts)
    points = pd.DataFrame({"x": [0.1, 0.15], "y": [0.12, 0.09]}, index=["Book", "Benchmark"])
    fig = charts.labeled_scatter(points, "Risk and return", xlabel="vol", ylabel="return", highlight="Benchmark")
    assert {t.get_text() for t in fig.axes[0].texts} == {"Book", "Benchmark"}
    fig = charts.radar({"Book": pd.Series({"A": 0.5, "B": -0.3, "C": 2.0})}, "Exposures")
    assert fig.axes[0].name == "polar"


def test_factor_scores_rank_the_members_on_each_trait() -> None:
    n = 40
    rng = np.random.default_rng(2)
    traits = pd.DataFrame(
        {
            "volatility": np.linspace(0.1, 0.6, n),
            "return_12_1": rng.normal(0.1, 0.2, n),
            "beta": np.linspace(0.5, 1.5, n),
            "earnings_to_price": rng.normal(0.05, 0.02, n),
            "book_to_price": rng.normal(0.3, 0.1, n),
            "sales_to_price": rng.normal(0.5, 0.2, n),
            "dividend_yield": np.linspace(0.0, 0.05, n),
            "roe": rng.normal(0.15, 0.05, n),
            "accruals": rng.normal(0.0, 0.03, n),
            "leverage": rng.uniform(0.0, 2.0, n),
            "market_cap": np.linspace(1e9, 1e12, n),
        },
        index=range(1, n + 1),
    )
    members = pd.Series(True, index=traits.index)
    members[n] = False  # not in the index that day
    scores = db.factor_z(traits, members)
    assert scores.loc[: n - 1].notna().all().all()
    assert scores.loc[n].isna().all()
    calm, beta, small = (scores[f].to_numpy(dtype=float) for f in ("Low Volatility", "High Beta", "Small Size"))
    assert calm[0] > calm[n - 2]  # the calmest member scores highest
    assert beta[n - 2] > beta[0]
    assert small[0] > small[n - 2]


def test_books_are_compared_over_the_longest_period_they_all_cover() -> None:
    table = pd.DataFrame(
        {"1M": [0.01, 0.02], "YTD": [0.1, 0.2], "12M": [0.1, 0.2], "5Y": [0.05, 0.06], "15Y": [0.04, np.nan]},
        index=["since 2009", "since 2015"],
    )
    assert periods.longest_period(table) == "5Y"  # the younger book has no fifteen years
    with pytest.raises(ValueError, match="no period"):
        periods.longest_period(pd.DataFrame({"1M": [np.nan]}))


def test_a_book_and_its_copy_have_perfectly_correlated_excess_returns() -> None:
    days = pd.bdate_range("2020-01-01", "2024-12-31")
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    book = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    corr = periods.excess_correlations({"a": book, "b": book, "c": bench}, bench, "3Y", days[-1])
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    assert pd.isna(corr.loc["a", "c"])  # the benchmark itself has no excess return to correlate
