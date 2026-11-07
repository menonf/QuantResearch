import numpy as np
import pandas as pd
import pytest

from equity_research.research.backtest import active_performance, decile_returns, performance, run_backtest
from equity_research.research.factors import combine, cross_sectional_z, magic_formula, momentum, net_issuance, quality
from equity_research.research.portfolio import buffered_selection, quantile_book, rebalance_dates, weighted_by

DAYS = pd.bdate_range("2024-01-01", periods=6)


def _ratios(rows: dict[int, dict[str, float]]) -> pd.DataFrame:
    """``(date, listing_id)`` x ratios on one day, as :func:`~equity_research.research.ratios.ratio_panel` gives."""
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_product([DAYS[:1], frame.index], names=["date", "listing_id"])
    return frame


def test_quality_ignores_returns_on_negative_equity_and_negative_capital() -> None:
    sound = {"Book Equity/TL": 0.5, "EBIT/TA": 0.1, "Gross Profit/TA": 0.3, "Debt/Equity": 0.5}
    ratios = _ratios(
        {
            1: sound | {"RoE": 0.10, "ROC": 0.20},
            2: sound | {"RoE": 0.05, "ROC": 0.10},
            # a loss over negative equity and EBIT over negative capital: both "returns" look high
            3: sound | {"RoE": 0.90, "Book Equity/TL": -0.2, "ROC": 0.90, "EBIT/TA": -0.1},
        }
    )
    q = quality(ratios).iloc[0]
    assert q[3] == pytest.approx(0.0, abs=1e-9)  # only its (average) gross profitability and leverage count
    assert q[1] > q[3]
    magic = magic_formula(ratios.assign(**{"EBIT/EV": 0.1})).iloc[0]
    assert magic[3] == pytest.approx(0.0, abs=1e-9) and magic[1] > magic[3]  # its yield counts, its ROC does not


def test_quality_prefers_gross_profit_per_dollar_of_assets() -> None:
    base = {"RoE": 0.1, "ROC": 0.1, "Book Equity/TL": 0.5, "EBIT/TA": 0.1, "Debt/Equity": 0.5}
    ratios = _ratios({1: base | {"Gross Profit/TA": 0.6}, 2: base | {"Gross Profit/TA": 0.2}})
    q = quality(ratios).iloc[0]
    assert q[1] > q[2]


def test_a_company_buying_back_shares_scores_above_one_issuing_them() -> None:
    change = pd.DataFrame({1: [np.log(0.95)], 2: [0.0], 3: [np.log(1.30)]}, index=DAYS[:1])
    z = cross_sectional_z(net_issuance(change)).iloc[0]
    assert z[1] > z[2] > z[3]


def test_weighting_by_market_value_keeps_each_legs_names_and_total() -> None:
    book = pd.DataFrame({1: [0.5], 2: [0.5], 3: [0.0], 4: [-0.5], 5: [-0.5]}, index=DAYS[:1])
    size = pd.DataFrame({1: [300.0], 2: [100.0], 3: [1e6], 4: [np.nan], 5: [50.0]}, index=DAYS[:1])
    w = weighted_by(book, size).iloc[0]
    assert w[1] == pytest.approx(0.75) and w[2] == pytest.approx(0.25)  # 3 : 1, the leg still sums to 1
    assert w[3] == 0.0  # not chosen, however large
    assert w[4] == pytest.approx(-0.5) and w[5] == pytest.approx(-0.5)  # no size: the leg's median, not dropped
    nothing = weighted_by(book, size * np.nan).iloc[0]
    assert nothing[[1, 2]].tolist() == [0.5, 0.5]  # no sizes at all: stays equal-weighted


def test_z_scores_rank_members_only() -> None:
    raw = pd.DataFrame({1: [1.0], 2: [2.0], 3: [3.0], 4: [100.0]}, index=DAYS[:1])
    members = pd.DataFrame({1: [True], 2: [True], 3: [True], 4: [False]}, index=DAYS[:1])
    z = cross_sectional_z(raw, members).iloc[0]
    assert np.isnan(z[4])  # not a member that day
    assert z[1] < z[2] < z[3] and z[2] == pytest.approx(0.0, abs=1e-9)


def test_momentum_skips_the_latest_month() -> None:
    prices = pd.DataFrame({1: [100.0, 110.0, 121.0, 200.0]}, index=pd.bdate_range("2024-01-01", periods=4))
    m = momentum(prices, lookback=3, skip=1)[1]
    assert m.iloc[-1] == pytest.approx(121.0 / 100.0 - 1)  # ignores the last day's jump to 200


def test_combine_ignores_a_missing_component() -> None:
    a = pd.DataFrame({1: [1.0]}, index=DAYS[:1])
    b = pd.DataFrame({1: [np.nan]}, index=DAYS[:1])
    assert combine({"a": a, "b": b}).iloc[0, 0] == 1.0


def test_buffer_keeps_a_holding_until_it_leaves_the_exit_band() -> None:
    score = pd.Series({i: float(i) for i in range(1, 11)})  # 10 is best
    assert buffered_selection(score, set(), enter=0.1, exit_=0.3, top=True) == {10}
    assert buffered_selection(score, {8}, enter=0.1, exit_=0.3, top=True) == {8, 10}  # 8 is still in the top 30 %
    assert buffered_selection(score, {6}, enter=0.1, exit_=0.3, top=True) == {10}  # 6 is not


def test_books_are_fully_invested_on_each_side() -> None:
    score = pd.DataFrame({i: [float(i)] for i in range(1, 11)}, index=DAYS[:1])
    book = quantile_book(score, DAYS[:1], enter=0.2, exit_=0.2)
    assert book.iloc[0].clip(lower=0).sum() == pytest.approx(1.0)
    assert book.iloc[0].clip(upper=0).sum() == pytest.approx(-1.0)
    assert rebalance_dates(pd.bdate_range("2024-01-01", "2024-03-31")).strftime("%m-%d").tolist() == [
        "01-31",
        "02-29",
        "03-29",
    ]


def test_a_book_earns_from_the_next_day_and_pays_costs_on_the_trade() -> None:
    returns = pd.DataFrame({1: [0.0, 0.10, 0.10, 0.0, 0.0, 0.0]}, index=DAYS)
    book = pd.DataFrame({1: [1.0]}, index=DAYS[1:2])  # set at the close of day 1
    result = run_backtest(book, returns, cost_bps=10.0)
    assert DAYS[1] not in result.gross.index  # the rebalance day itself earns nothing
    assert result.gross.loc[DAYS[2]] == pytest.approx(0.10)
    assert result.net.loc[DAYS[2]] == pytest.approx(0.10 - 0.001)  # 1 unit traded at 10 bps


def test_short_positions_pay_borrow_and_weights_drift() -> None:
    returns = pd.DataFrame({1: [0.0, 0.0, 0.02, -0.01], 2: [0.0, 0.0, 0.0, 0.0]}, index=DAYS[:4])
    book = pd.DataFrame({1: [0.5], 2: [-0.5]}, index=DAYS[:1])
    result = run_backtest(book, returns, cost_bps=0.0, borrow_bps=252.0)  # 1 bp a day on the short leg
    assert result.net.loc[DAYS[1]] == pytest.approx(-0.5 * 1e-4)
    w = result.weights.loc[DAYS[3]]
    assert w[1] > 0.5  # the long grew with its return


def test_performance_of_a_flat_series() -> None:
    perf = performance(pd.Series(0.0, index=DAYS))
    assert perf["cagr"] == 0.0 and np.isnan(perf["sharpe"]) and perf["max_drawdown"] == 0.0


def test_active_performance_measures_the_bet_against_the_benchmark() -> None:
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2015-01-01", periods=1500)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    same = active_performance(bench, bench)
    assert same["beta"] == pytest.approx(1.0) and same["tracking_error"] == 0.0 and np.isnan(same["information_ratio"])
    calm = active_performance(0.5 * bench + 0.0001, bench)  # half the market plus 1 bp a day
    assert calm["beta"] == pytest.approx(0.5, abs=0.03)
    assert calm["alpha"] == pytest.approx(0.0001 * 252, rel=0.2)
    assert calm["active_return"] < 0  # in a rising market, half the exposure grows more slowly
    assert np.isnan(active_performance(bench, bench.iloc[:5])["beta"])  # one common month measures nothing


def test_a_benchmark_with_stale_daily_closes_still_measures_beta_one() -> None:
    rng = np.random.default_rng(4)
    days = pd.bdate_range("2015-01-01", periods=1500)
    market = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    stale = market.shift(1).fillna(0.0)  # a thin fund's close lags the market by a day
    measured = active_performance(market, stale)
    daily_beta = market.cov(stale) / stale.var()
    assert abs(daily_beta) < 0.1  # day by day the lag hides the whole market move
    assert measured["beta"] > 0.8  # month by month it is only a day in twenty-one


def test_residual_momentum_ranks_a_companys_own_run_not_its_beta() -> None:
    from equity_research.research.factors import residual_momentum  # noqa: PLC0415

    rng = np.random.default_rng(5)
    days = pd.bdate_range("2017-01-02", "2020-12-31")
    market = pd.Series(rng.normal(0.0012, 0.01, len(days)), index=days)  # a strongly rising market
    lately = (days >= pd.Timestamp("2020-01-01")).astype(float)
    returns = pd.DataFrame(
        {
            1: 2.0 * market + rng.normal(0, 0.004, len(days)),  # high beta, nothing of its own
            2: 0.5 * market + rng.normal(0, 0.004, len(days)) + 0.001 * lately,  # calm, but rising on its own
        },
        index=days,
    )
    when = pd.DatetimeIndex([days[-1]])
    plain = momentum((1 + returns).cumprod()).loc[when[0]]
    residual = residual_momentum(returns, market, when).iloc[0]
    assert plain[1] > plain[2]  # plain momentum prefers the share that rode the market
    assert residual[2] > residual[1]  # the residual prefers the one that rose by itself
    assert residual_momentum(returns.iloc[:100], market.iloc[:100], pd.DatetimeIndex([days[99]])).isna().all().all()


def test_deciles_show_where_a_score_earns() -> None:
    days = pd.bdate_range("2020-01-01", periods=120)
    dates = pd.DatetimeIndex([days[0], days[40], days[80], days[119]])
    names = range(1, 21)
    # name i earns i bp a day: the score that ranks by i is right
    returns = pd.DataFrame({i: [i * 1e-4] * len(days) for i in names}, index=days)
    score = pd.DataFrame({i: [float(i)] * len(dates) for i in names}, index=dates)
    members = pd.DataFrame(True, index=days, columns=list(names))
    deciles = decile_returns(score, returns, members)
    assert deciles["excess"].is_monotonic_increasing
    excess = deciles["excess"].to_numpy(dtype=float)
    assert excess[-1] > 0 > excess[0]


def test_profit_backed_by_cash_scores_above_profit_made_of_accruals() -> None:
    from equity_research.research.factors import accruals, cash_flow_yield  # noqa: PLC0415

    ratios = _ratios(
        {
            1: {"Accruals/TA": -0.05, "FCF/EV": 0.08},  # more cash than profit
            2: {"Accruals/TA": 0.10, "FCF/EV": -0.02},  # profit not yet cash; burning cash
        }
    )
    assert accruals(ratios).iloc[0][1] > accruals(ratios).iloc[0][2]
    assert cash_flow_yield(ratios).iloc[0][1] > cash_flow_yield(ratios).iloc[0][2]


def test_stamp_duty_is_paid_on_purchases_only() -> None:
    days = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame(0.0, index=days, columns=[1, 2])
    # buy 1, short 2; then sell 1 and buy back 2
    book = pd.DataFrame({1: [1.0, 0.0], 2: [-1.0, 0.0]}, index=[days[0], days[3]])
    result = run_backtest(book, returns, cost_bps=0.0, buy_tax_bps=50.0)
    assert result.costs.sum() == pytest.approx(0.005)  # the purchase of 1 only: no tax on a sale or on a short
