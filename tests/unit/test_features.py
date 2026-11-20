import numpy as np
import pandas as pd
import pytest

from equity_research.data.universe import UniverseData
from equity_research.ml import features as ft
from equity_research.ml import model_features as mf
from equity_research.strategies import learned_value as ml

DAYS = pd.bdate_range("2018-01-01", "2021-12-31")
NAMES = list(range(1, 31))


def _universe(seed: int = 0) -> UniverseData:
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(rng.normal(0.0004, 0.015, (len(DAYS), len(NAMES))), index=DAYS, columns=NAMES)
    returns[30] = 0.0  # a share whose price never moves
    adjusted = 100 * (1 + returns).cumprod()
    members = pd.DataFrame(True, index=DAYS, columns=NAMES)
    zero = pd.DataFrame(0.0, index=DAYS, columns=NAMES)
    rebalances = pd.DatetimeIndex(returns.groupby(DAYS.to_period("M")).apply(lambda m: m.index.max()))
    panel = pd.DataFrame(rng.uniform(1e9, 1e10, (len(rebalances), len(NAMES))), index=rebalances, columns=NAMES)
    index = pd.MultiIndex.from_product([rebalances, NAMES], names=["date", "listing_id"])
    ratios = pd.DataFrame(rng.normal(size=(len(index), len(ml.BOOK_CLIPS))), index=index, columns=list(ml.BOOK_CLIPS))
    return UniverseData(
        universe="TEST", close=adjusted, adjusted=adjusted, returns=returns, members=members,
        segment=pd.DataFrame("large", index=DAYS, columns=NAMES), cost_bps=zero, borrow_bps=zero, buy_tax_bps=zero,
        rebalances=rebalances, ratios=ratios, benchmark=returns.mean(axis=1), market_cap=panel,
        share_change=panel * 0, earnings_surprise=panel * 0, eps_growth=panel * 0, sales_growth=panel * 0,
    )  # fmt: skip


def test_roll_spread_reads_the_bid_ask_bounce() -> None:
    bounce = pd.DataFrame({1: np.tile([0.004, -0.004], 50)}, index=DAYS[:100])
    assert ft.roll_spread(bounce).iloc[-1, 0] == pytest.approx(2 * 0.004, rel=0.05)  # cov = -s^2: spread 2s
    trending = pd.DataFrame({1: np.full(100, 0.01)}, index=DAYS[:100])
    assert ft.roll_spread(trending).iloc[-1, 0] == 0.0  # no negative covariance: no measurable spread


def test_amihud_and_idiosyncratic_volatility() -> None:
    r = pd.DataFrame({1: np.tile([0.01, -0.01], 50)}, index=DAYS[:100])
    dollars = pd.DataFrame({1: np.full(100, 2e6)}, index=DAYS[:100])
    assert ft.amihud(r, dollars).iloc[-1, 0] == pytest.approx(0.005)  # 1 % per two million: 0.5 % a million
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, 300), index=DAYS[:300])
    own = pd.DataFrame({1: 1.5 * market, 2: market + rng.normal(0, 0.02, 300)}, index=DAYS[:300])
    idio = ft.idiosyncratic_volatility(own, market).iloc[-1]
    assert idio[1] == pytest.approx(0.0, abs=1e-6)  # all of it is the market
    assert idio[2] == pytest.approx(0.02 * np.sqrt(252), rel=0.25)


def test_price_features_are_the_documented_set_on_the_dates() -> None:
    data = _universe()
    dates = data.rebalances[12:]
    dollars = data.adjusted * 1e5
    out = ft.price_features(data, dollars, dates)
    assert list(out) == list(ft.PRICE_FEATURES)
    assert all(panel.index.equals(dates) and panel.columns.equals(pd.Index(NAMES)) for panel in out.values())
    d = dates[-1]
    p = data.adjusted[1].to_numpy()
    i = int(data.adjusted.index.searchsorted(d))
    assert out["mom_12_1"].loc[d, 1] == pytest.approx(p[i - 21] / p[i - 252] - 1)
    assert out["ret_1w"].loc[d, 1] == pytest.approx(p[i] / p[i - 5] - 1)
    assert out["zero_days_3m"].loc[d, 30] == 1.0 and out["zero_days_3m"].loc[d, 1] == 0.0
    assert out["vol_1y"].loc[d, 1] == pytest.approx(0.015 * np.sqrt(252), rel=0.15)
    assert out["turnover_1m"].loc[d].notna().all()


def test_fundamental_and_calendar_features_and_stacking() -> None:
    data = _universe()
    dates = data.rebalances[-3:]
    accounts = ft.fundamental_features(data, dates)
    assert list(accounts) == [*ml.BOOK_CLIPS, *ft.ACCOUNT_EXTRAS]
    calendar = ft.calendar_features(pd.DatetimeIndex(["2021-01-29", "2021-07-30"]))
    assert calendar.loc["2021-01-29", "month_sin"] == pytest.approx(0.0)
    assert calendar.loc["2021-01-29", "month_cos"] == pytest.approx(1.0)
    assert calendar.loc["2021-07-30", "month_cos"] == pytest.approx(-1.0)
    assert calendar["january"].tolist() == [1.0, 0.0]
    members = data.members.copy()
    members[5] = False
    long = ft.stack(accounts, members)
    assert 5 not in long.index.get_level_values("listing_id") and list(long.columns) == list(accounts)
    assert len(long) == len(dates) * (len(NAMES) - 1)
    back = ft.unstack(long)
    assert list(back) == list(accounts) and back["size"].shape == (len(dates), len(NAMES) - 1)


def test_fractional_differencing() -> None:
    assert mf.ffd_weights(1.0).tolist() == [1.0, -1.0]  # d = 1 is the ordinary difference
    assert mf.ffd_weights(0.0).tolist() == [1.0]
    w = mf.ffd_weights(0.4, 1e-3)
    assert w[1] == pytest.approx(-0.4) and abs(w[-1]) >= 1e-3 and len(w) > 10
    x = pd.DataFrame({1: np.arange(10.0) ** 2}, index=DAYS[:10])
    diff = mf.frac_diff(x, d=1.0)
    assert diff[1].iloc[1:].tolist() == pytest.approx(np.diff(np.arange(10.0) ** 2).tolist())
    assert bool(diff.isna().iloc[0, 0])  # the window is not complete


def test_kalman_trend_follows_a_straight_line() -> None:
    line = pd.DataFrame(
        {1: 0.001 * np.arange(800), 2: -0.002 * np.arange(800)}, index=pd.bdate_range("2015-01-01", periods=800)
    )
    slope, gap = mf.kalman_trend(line)
    assert slope.iloc[-1, 0] == pytest.approx(0.001 * 252, rel=1e-3)
    assert slope.iloc[-1, 1] == pytest.approx(-0.002 * 252, rel=1e-3)
    assert gap.iloc[-1].abs().max() < 1e-6


def test_break_statistic_flags_a_sudden_run() -> None:
    rng = np.random.default_rng(2)
    r = pd.DataFrame(
        {1: rng.normal(0, 0.01, 400), 2: rng.normal(0, 0.01, 400)}, index=pd.bdate_range("2015-01-01", periods=400)
    )
    r.iloc[-5:, 0] = 0.05  # five days of +5 %
    z = mf.break_statistic(r).iloc[-1]
    assert z[1] > 8 and abs(z[2]) < 4


def test_har_volatility_forecasts_persistent_volatility() -> None:
    rng = np.random.default_rng(3)
    vols = np.where(np.arange(60) < 30, 0.01, 0.03)  # half calm shares, half volatile ones
    r = pd.DataFrame(rng.normal(0, 1, (len(DAYS), 60)) * vols, index=DAYS, columns=range(60))
    members = pd.DataFrame(True, index=DAYS, columns=r.columns)
    dates = pd.DatetimeIndex(r.groupby(DAYS.to_period("M")).apply(lambda m: m.index.max()))
    har = mf.har_volatility(r, members, dates)
    assert har.loc["2018"].isna().all().all()  # no year of history before 2018
    late = har.loc["2021"].to_numpy().mean(axis=0)
    calm, volatile = late[:30].mean(), late[30:].mean()
    # daily squared returns measure volatility with much noise, so the pooled fit shrinks forecasts towards the
    # average (regression dilution): the order is right and the gap wide, the levels pulled together
    assert volatile > 1.8 * calm
    assert 0.01 * np.sqrt(252) < calm < 0.03 * np.sqrt(252) and volatile < 0.03 * np.sqrt(252) * 1.1


def test_garch_volatility_tracks_simulated_garch_variance() -> None:
    rng = np.random.default_rng(4)
    omega, alpha, beta = 0.05, 0.08, 0.9  # percent-squared a day
    n, k = len(DAYS), 3
    s2 = np.full(k, omega / (1 - alpha - beta))
    true = np.zeros((n, k))
    x = np.zeros((n, k))
    for t in range(n):
        true[t] = s2
        x[t] = np.sqrt(s2) * rng.standard_normal(k)
        s2 = omega + alpha * x[t] ** 2 + beta * s2
    r = pd.DataFrame(x / 100, index=DAYS, columns=[1, 2, 3])
    members = pd.DataFrame(True, index=DAYS, columns=r.columns)
    dates = DAYS[DAYS.year == 2021]
    forecast = mf.garch_volatility(r, members, dates, n_jobs=1)
    expected = pd.DataFrame(np.sqrt(true * 252) / 100, index=DAYS, columns=r.columns).shift(-1).reindex(dates)
    for name in r.columns:
        both = pd.concat([forecast[name], expected[name]], axis=1).dropna()
        assert np.corrcoef(both.to_numpy().T)[0, 1] > 0.9  # the forecast follows tomorrow's true volatility
    assert forecast.loc["2021"].notna().all().all()


def test_market_regime_separates_calm_from_turbulent_years() -> None:
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2015-01-01", "2019-12-31")
    turbulent = (days.year == 2018) | ((days.year == 2016) & (days.month <= 6))
    market = pd.Series(
        np.where(turbulent, rng.normal(-0.001, 0.025, len(days)), rng.normal(0.0005, 0.007, len(days))), index=days
    )
    dates = pd.DatetimeIndex(market.groupby(days.to_period("M")).apply(lambda m: m.index.max()))
    p = mf.market_regime(market, dates)
    assert p.loc[:"2016-12-31"].isna().all()  # under two years of history before 2017
    assert p.loc["2018"].mean() > 0.8 and p.loc["2019"].mean() < 0.2


def test_style_panels_turn_exposures_into_one_panel_per_style() -> None:
    from equity_research.analytics.risk import risk_model as rm  # noqa: PLC0415

    dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]
    loadings = {d: pd.DataFrame({"Value": [0.5, -0.5], "Size": [1.0, 2.0]}, index=[1, 2]) for d in dates}
    weights = {d: pd.Series(1.0, index=[1, 2]) for d in dates}
    exposures = rm.Exposures(loadings, weights, weights, industries=(), styles=("Value", "Size"))
    panels = rm.style_panels(exposures)
    assert list(panels) == ["Value", "Size"]
    assert panels["Size"].to_numpy()[1].tolist() == [1.0, 2.0]
    assert panels["Value"].index.equals(pd.DatetimeIndex(dates))
