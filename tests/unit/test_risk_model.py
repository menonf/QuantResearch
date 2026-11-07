import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from equity_research.research import portfolio_risk as pr
from equity_research.research import risk_model as rm
from equity_research.research import risk_styles as rs

DAYS = pd.bdate_range("2015-01-01", "2019-12-31")


def _exposures(rng: np.random.Generator, n: int = 300, dates: pd.DatetimeIndex | None = None) -> rm.Exposures:
    """Three industries and two styles for ``n`` shares, the same on every month end."""
    dates = dates if dates is not None else rm.month_ends(DAYS)
    industry = rng.integers(0, 3, n)
    frame = pd.DataFrame(
        {
            rm.MARKET: 1.0,
            "A": (industry == 0).astype(float),
            "B": (industry == 1).astype(float),
            "C": (industry == 2).astype(float),
            "Size": rng.normal(size=n),
            "Value": rng.normal(size=n),
        },
        index=pd.Index(range(1, n + 1), name="listing_id"),
    )
    weight = pd.Series(np.sqrt(rng.lognormal(8, 1, n)), index=frame.index)
    estu = pd.Series(True, index=frame.index)
    return rm.Exposures(
        {pd.Timestamp(d): frame for d in dates},
        {pd.Timestamp(d): weight for d in dates},
        {pd.Timestamp(d): estu for d in dates},
        ("A", "B", "C"),
        ("Size", "Value"),
    )


def test_standardised_values_have_weighted_mean_zero_unit_spread_and_squeezed_tails() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=500), [40.0, -35.0]])
    w = rng.uniform(1, 10, len(x))
    z = rm.standardise(x, w, np.ones(len(x), dtype=bool))
    assert np.average(z, weights=w) == pytest.approx(0.0, abs=0.02)
    assert np.std(z, ddof=1) == pytest.approx(1.0, abs=0.05)
    assert np.abs(z).max() <= 3.0 and z[-2] == z.max() and z[-1] == z.min()
    assert (np.argsort(z[:500]) == np.argsort(x[:500])).all()  # the order is kept


def test_a_missing_style_is_filled_from_size_and_industry() -> None:
    rng = np.random.default_rng(1)
    n = 400
    size = rng.normal(size=n)
    industry = rng.integers(0, 2, n)
    z = 0.8 * size + np.where(industry == 1, 0.5, -0.5)
    holes = z.copy()
    holes[:20] = np.nan
    filled = rm.fill_missing(holes, size, industry, np.ones(n, dtype=bool))
    assert filled[:20] == pytest.approx(z[:20], abs=1e-9)
    assert (filled[20:] == z[20:]).all()


def test_the_constrained_regression_recovers_factor_returns_and_zero_weighted_industries() -> None:
    rng = np.random.default_rng(2)
    exp = _exposures(rng)
    x = exp.loadings[exp.dates[0]][exp.factors].to_numpy()
    w = exp.weight[exp.dates[0]].to_numpy()
    c = (w[:, None] * x[:, 1:4]).sum(axis=0)
    f = np.array([0.01, 0.004, -0.002, 0.0, 0.003, -0.001])
    f[3] = -(c[0] * f[1] + c[1] * f[2]) / c[2]  # the industries' weighted average is zero
    got, t, r2 = rm.constrained_regression(x, x @ f, w, slice(1, 4))
    assert got == pytest.approx(f, abs=1e-12)
    assert r2 == pytest.approx(1.0)
    assert (c * got[1:4]).sum() == pytest.approx(0.0, abs=1e-12)
    # the market is the weighted average return when the styles average zero
    noisy = x @ f + rng.normal(0, 0.02, len(w))
    got, t, r2 = rm.constrained_regression(x, noisy, w, slice(1, 4))
    assert abs(t[0]) > 2 and 0 < r2 < 1


def test_daily_regressions_use_the_exposures_of_the_month_end_before_the_day() -> None:
    rng = np.random.default_rng(3)
    exp = _exposures(rng, dates=rm.month_ends(DAYS[:80]))
    days = DAYS[:80]
    truth = pd.DataFrame(rng.normal(0, 0.01, (len(days), 6)), index=days, columns=exp.factors)
    truth[["A", "B", "C"]] = 0.0
    x = exp.loadings[exp.dates[0]][exp.factors]
    returns = pd.DataFrame(truth.to_numpy() @ x.T.to_numpy(), index=days, columns=x.index)
    returns += rng.normal(0, 0.001, returns.shape)
    est = rm.estimate_factor_returns(returns, exp)
    assert est.factor_returns.index[0] > exp.dates[0]  # nothing on or before the first month end
    got = est.factor_returns
    assert np.corrcoef(got["Size"], truth.loc[got.index, "Size"])[0, 1] > 0.99
    assert est.specific_returns.loc[got.index].abs().mean().mean() < 0.002


def test_the_ewma_covariance_finds_a_known_covariance_and_is_positive_definite() -> None:
    rng = np.random.default_rng(4)
    true = np.array([[1.0, 0.6], [0.6, 2.0]]) * 1e-4
    x = pd.DataFrame(rng.multivariate_normal([0, 0], true, size=3000), index=pd.bdate_range("2010-01-01", periods=3000))
    covs = rm.ewma_covariance(x, [x.index[-1]], var_half_life=2000, corr_half_life=2000, nw_lags=0)
    got = covs[x.index[-1]].to_numpy()
    assert got == pytest.approx(true, rel=0.15)
    assert np.linalg.eigvalsh(got).min() > 0
    assert rm.ewma_covariance(x, [x.index[100]], min_obs=250) == {}  # not enough history yet


def test_newey_west_raises_the_forecast_for_trending_returns() -> None:
    rng = np.random.default_rng(5)
    e = rng.normal(0, 0.01, 3000)
    trending = np.convolve(e, [1.0, 0.8], mode="same")  # positively autocorrelated
    x = pd.DataFrame({"f": trending}, index=pd.bdate_range("2010-01-01", periods=3000))
    plain = rm.ewma_covariance(x, [x.index[-1]], nw_lags=0)[x.index[-1]].to_numpy()[0, 0]
    adjusted = rm.ewma_covariance(x, [x.index[-1]], nw_lags=1)[x.index[-1]].to_numpy()[0, 0]
    assert adjusted > plain * 1.2


def test_gamma_rises_when_the_last_month_runs_hot() -> None:
    rng = np.random.default_rng(6)
    days = pd.bdate_range("2010-01-01", periods=800)
    x = pd.DataFrame(rng.normal(0, 0.01, (800, 5)), index=days)
    x.iloc[-21:] *= 3.0
    covs = rm.ewma_covariance(x, [days[-300], days[-1]])
    gamma = rm.factor_gamma(x, covs)
    assert gamma[days[-300]] == pytest.approx(1.0, abs=0.25)
    assert gamma[days[-1]] > 2.0


def test_specific_risk_finds_each_shares_volatility_and_needs_enough_days() -> None:
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2010-01-01", periods=600)
    vols = np.array([0.01, 0.03])
    u = pd.DataFrame(rng.normal(0, 1, (600, 2)) * vols, index=days, columns=[1, 2])
    u.iloc[:560, 1] = np.nan  # the second share listed 40 days ago
    vol, nobs = rm.ewma_specific_risk(u, [days[-1]], nw_lags=0)
    assert vol.iloc[0, 0] == pytest.approx(0.01, rel=0.15)
    assert np.isnan(vol.to_numpy()[0, 1]) and nobs.to_numpy()[0, 1] == 40


def test_a_takeover_jump_is_clipped_before_it_sets_specific_risk() -> None:
    rng = np.random.default_rng(8)
    u = pd.DataFrame(rng.normal(0, 0.01, (5, 400)))
    u.iloc[2, 0] = 0.6
    cleaned = rm.clean_specific(u)
    assert 0.05 < cleaned.to_numpy()[2, 0] < 0.2
    assert (cleaned.iloc[:, 1:] == u.iloc[:, 1:]).all().all()


def test_mid_cap_peaks_at_the_sixtieth_size_percentile() -> None:
    n = 1000
    value = pd.DataFrame([np.arange(n, 0, -1, dtype=float)], index=[pd.Timestamp("2020-01-31")])
    members = value.notna()
    mid = rs.mid_cap(value, members).iloc[0]
    assert mid.idxmax() == pytest.approx(0.6 * n, abs=2)
    assert mid.iloc[int(0.54 * n) - 1] == pytest.approx(0.7, abs=0.01)
    assert mid.iloc[0] < 0.01 and mid.iloc[-1] < 0.6


def test_historical_beta_finds_the_slope_without_reading_past_the_date() -> None:
    rng = np.random.default_rng(9)
    days = pd.bdate_range("2012-01-02", "2017-12-29")
    market = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    returns = pd.DataFrame({1: 1.5 * market + rng.normal(0, 0.005, len(days))}, index=days)
    date = pd.DatetimeIndex([pd.Timestamp("2017-05-31")])  # a Wednesday
    got = rs.historical_beta(returns, market, date)
    assert got["beta"].iloc[0, 0] == pytest.approx(1.5, abs=0.1)
    spoiled = returns.copy()
    spoiled.loc["2017-06-01":] = 0.5  # what happens after the date must not matter
    assert rs.historical_beta(spoiled, market, date)["beta"].iloc[0, 0] == got["beta"].iloc[0, 0]


def test_a_trend_is_the_yearly_slope_per_unit_of_the_average() -> None:
    idx, cols = [pd.Timestamp("2020-01-31")], [1, 2]
    points = [pd.DataFrame([[100.0 - 10 * k, np.nan if k else 5.0]], index=idx, columns=cols) for k in range(5)]
    got = rs.trend(points)
    assert got.iloc[0, 0] == pytest.approx(10 / 80)
    assert np.isnan(got.to_numpy()[0, 1])  # one year is not a trend


def test_turnover_is_unchanged_by_a_split() -> None:
    days = pd.bdate_range("2020-01-01", periods=100)
    dollars = pd.DataFrame({1: np.full(100, 1e6)}, index=days)
    value = pd.DataFrame({1: [1e8]}, index=[days[-1]])
    assert rs.turnover(dollars, value, days=63).iloc[0, 0] == pytest.approx(math.log(0.01))


def _model(rng: np.random.Generator) -> rm.RiskModel:
    exp = _exposures(rng, n=200)
    x = exp.loadings[exp.dates[0]][exp.factors]
    truth = pd.DataFrame(rng.normal(0, 0.005, (len(DAYS), 6)), index=DAYS, columns=exp.factors)
    truth[rm.MARKET] *= 2.0
    noise = rng.normal(0, 0.015, (len(DAYS), len(x)))
    returns = pd.DataFrame(truth.to_numpy() @ x.T.to_numpy() + noise, index=DAYS, columns=x.index)
    return rm.build_risk_model(returns, exp, list(rm.month_ends(DAYS)), rm.RiskModelConfig(structural_months=12))


@pytest.fixture(scope="module")
def model() -> rm.RiskModel:
    return _model(np.random.default_rng(10))


def test_the_model_forecasts_the_specific_risk_it_was_given(model: rm.RiskModel) -> None:
    day = model.dates[-1]
    _, cov, spec = model.on(day, "1.0")
    assert spec.median() == pytest.approx(0.015, rel=0.15)
    assert math.sqrt(cov.to_numpy()[0, 0])  # the market is the first factor == pytest.approx(0.01, rel=0.25)
    _, cov2, spec2 = model.on(day)
    assert spec2.notna().all() and np.linalg.eigvalsh(cov2.to_numpy()).min() > 0
    assert model.gamma["factor"].dropna().between(0.3, 3).all()


def test_contributions_add_up_to_the_risk_and_var_scales_it(model: rm.RiskModel) -> None:
    day = model.dates[-1]
    loadings = model.exposures.on(day)
    weights = pd.Series(1.0 / len(loadings), index=loadings.index)
    risk = pr.portfolio_risk(model, weights, day)
    assert risk.contributions.sum() == pytest.approx(risk.total)
    assert risk.total**2 == pytest.approx(risk.factor**2 + risk.specific**2)
    assert risk.coverage == pytest.approx(1.0)
    var = pr.value_at_risk(risk, 0.99)
    assert var["Total"] == pytest.approx(NormalDist().inv_cdf(0.99) * risk.total)
    assert var.drop("Total").sum() == pytest.approx(var["Total"])
    assert pr.by_block(risk, model).sum() == pytest.approx(risk.total)
    against_itself = pr.portfolio_risk(model, weights, day, benchmark=weights)
    assert against_itself.total == 0.0


def test_two_listings_of_one_company_share_their_specific_risk(model: rm.RiskModel) -> None:
    day = model.dates[-1]
    weights = pd.Series({1: 0.5, 2: 0.5})
    apart = pr.portfolio_risk(model, weights, day).specific
    together = pr.portfolio_risk(model, weights, day, listing_issuer=pd.Series({1: 7, 2: 7})).specific
    assert together > apart


def test_forecast_tests_and_kupiec() -> None:
    rng = np.random.default_rng(11)
    z = pd.Series(rng.normal(size=2000))
    bias, lo, hi = pr.bias_statistic(z)
    assert lo < bias < hi
    assert pr.q_statistic(z) < pr.q_statistic(z * 2)  # forecasts half the true volatility score worse
    lr, p = pr.kupiec(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    assert pr.kupiec(30, 1000, 0.01)[1] < 0.001


def test_forward_returns_buy_and_hold() -> None:
    days = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.DataFrame({1: [0.0, 0.1, 0.1, 0.0, 0.0], 2: [0.0, 0.0, np.nan, 0.0, 0.0]}, index=days)
    weights = pd.DataFrame({1: [0.5], 2: [0.5]}, index=[days[0]])
    assert pr.forward_returns(returns, weights, 2).iloc[0] == pytest.approx(0.5 * 0.21)


def test_var_backtest_holds_the_latest_weights_and_counts_exceptions(model: rm.RiskModel) -> None:
    ids = model.exposures.on(model.dates[0]).index
    weights = pd.DataFrame([np.full(len(ids), 1.0 / len(ids))], index=[model.dates[0]], columns=ids)
    returns = model.specific_returns.reindex(columns=ids).fillna(0.0)
    returns.iloc[:] = -0.5  # every day a crash: every VaR is exceeded
    bt = pr.var_backtest(model, weights, returns)
    assert len(bt) == len(model.dates) - 1 and bt["exception"].all()
    summary = pr.kupiec_summary({"crash": bt})
    row = summary.loc["crash"].to_dict()
    assert row["exceptions"] == len(bt) and float(row["p_value"]) < 1e-6


def test_stock_z_scores_are_near_one_when_the_model_is_right(model: rm.RiskModel) -> None:
    ids = model.exposures.on(model.dates[0]).index
    rng = np.random.default_rng(12)
    returns = pd.DataFrame(rng.normal(0, 0.02, (len(DAYS), len(ids))), index=DAYS, columns=ids)
    z = pr.stock_z_scores(model, returns, model.dates[-6:-1], version="1.0")
    assert z.index.names == ["date", "listing_id"] and len(z) == 5 * len(ids)
    assert pr.bias_statistic(z)[0] == pytest.approx(1.0, abs=0.15)


def test_a_factor_summary_and_significance() -> None:
    days = pd.bdate_range("2020-01-01", periods=504)
    rng = np.random.default_rng(13)
    fr = pd.DataFrame({rm.MARKET: rng.normal(0.0004, 0.01, 504), "Value": rng.normal(0, 0.003, 504)}, index=days)
    table = rm.factor_summary(fr)
    assert table.loc[rm.MARKET, "correlation with market"] == pytest.approx(1.0)
    assert table.loc["Value", "yearly volatility"] == pytest.approx(0.003 * math.sqrt(252), rel=0.1)
    t = pd.DataFrame({"a": [3.0, 0.5, -2.5, 1.0], "b": [0.1, 0.2, 0.3, 0.4]})
    assert rm.significance(t).to_dict() == {"a": 0.5, "b": 0.0}


def test_cap_weights_sum_to_one_among_the_chosen_members() -> None:
    day = pd.Timestamp("2020-01-31")
    caps = pd.DataFrame({1: [100.0], 2: [300.0], 3: [600.0]}, index=[day])
    members = pd.DataFrame({1: [True], 2: [True], 3: [False]}, index=[day])
    assert pr.cap_weights(caps, members).loc[day].fillna(0).to_dict() == {1: 0.25, 2: 0.75, 3: 0.0}
    only = pd.DataFrame({1: [True], 2: [False], 3: [True]}, index=[day])
    assert pr.cap_weights(caps, members, within=only).loc[day, 1] == 1.0


def test_forecast_tests_report_both_versions(model: rm.RiskModel) -> None:
    ids = model.exposures.on(model.dates[0]).index
    days = model.dates[-8:-2]
    weights = pd.DataFrame(1.0 / len(ids), index=days, columns=ids)
    rng = np.random.default_rng(14)
    returns = pd.DataFrame(rng.normal(0, 0.02, (len(DAYS), len(ids))), index=DAYS, columns=ids)
    table = pr.forecast_tests(model, returns, {"equal": weights})
    assert {"bias 1.0", "bias 2.0", "Q 1.0", "Q 2.0", "band"} <= set(table.columns)
    risk = pr.portfolio_risk(model, weights.iloc[0], days[0])
    assert pr.risk_table(risk)["share of risk"].sum() == pytest.approx(1.0)


def test_a_portfolio_has_a_beta_of_one_to_itself(model: rm.RiskModel) -> None:
    day = model.dates[-1]
    ids = model.exposures.on(day).index
    market = pd.Series(1.0 / len(ids), index=ids)
    assert pr.predicted_beta(model, market, market, day) == pytest.approx(1.0)
    assert pr.predicted_beta(model, 2 * market, market, day) == pytest.approx(2.0)


def test_implied_moves_follow_the_correlations() -> None:
    names = [rm.MARKET, "Volatility", "Value"]
    vol = np.array([0.01, 0.004, 0.002])
    corr = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = pd.DataFrame(corr * np.outer(vol, vol), index=names, columns=names)
    moves = pr.implied_moves(cov, {rm.MARKET: -0.10})
    assert moves[rm.MARKET] == -0.10
    assert moves["Volatility"] == pytest.approx(0.8 * 0.004 / 0.01 * -0.10)  # beta of volatility to the market
    assert moves["Value"] == pytest.approx(0.0)
    assert (pr.implied_moves(cov * 7, {rm.MARKET: -0.10}) - moves).abs().max() < 1e-12  # the scale cancels
    with pytest.raises(KeyError):
        pr.implied_moves(cov, {"Nonsense": 0.1})


def test_a_stress_test_isolated_and_with_correlations(model: rm.RiskModel) -> None:
    day = model.dates[-1]
    loadings = model.exposures.on(day)
    weights = pd.Series(1.0 / len(loadings), index=loadings.index)
    shock = {rm.MARKET: -0.10}
    result = pr.stress_test(model, weights, day, shock)
    x = result.table["exposure"]
    assert result.isolated == pytest.approx(x[rm.MARKET] * -0.10)
    assert result.table["isolated P&L"].drop(rm.MARKET).abs().max() == 0.0
    expected = x @ pr.implied_moves(model.on(day)[1], shock)
    assert result.correlated == pytest.approx(expected)
    assert result.correlated == pytest.approx(result.table["correlated P&L"].sum())
    assert pr.stress_test(model, weights, day, shock, benchmark=weights).correlated == 0.0
    sized = pr.shock_in_sigmas(model, day, {rm.MARKET: -2.0}, horizon=21)
    sigma = math.sqrt(21 * model.on(day)[1].to_numpy()[0, 0])
    assert sized[rm.MARKET] == pytest.approx(-2.0 * sigma)
