import numpy as np
import pandas as pd
import pytest

from equity_research.strategies.factors import (
    combine,
    cross_sectional_z,
    magic_formula,
    momentum,
    net_issuance,
    quality,
)
from research_core.analytics.performance import metrics as bt
from research_core.analytics.performance.metrics import active_performance, decile_returns, performance
from research_core.data.calendars import rebalance_dates
from research_core.engine.backtest import BacktestResult, banded, run_backtest
from research_core.engine.finish import deflated_active, finish_run
from research_core.engine.portfolio import buffered_selection, capped, quantile_book, quantile_books, weighted_by

DAYS = pd.bdate_range("2024-01-01", periods=6)


def _ratios(rows: dict[int, dict[str, float]]) -> pd.DataFrame:
    """``(date, listing_id)`` x ratios on one day, as :func:`~equity_research.data.ratios.ratio_panel` gives."""
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
    from equity_research.strategies.factors import residual_momentum  # noqa: PLC0415

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
    from equity_research.strategies.factors import accruals, cash_flow_yield  # noqa: PLC0415

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


def test_periods_against_measures_the_book_before_and_from_the_hold_out() -> None:
    days = pd.bdate_range("2024-01-01", "2025-12-31")
    bench = pd.Series(0.0005, index=days)
    book = pd.Series(np.where(days < pd.Timestamp("2025-01-01"), 0.0005, 0.001), index=days)  # ahead only from 2025
    table = bt.periods_against(book, bench, "2025-01-01")
    rows = {str(k): {str(c): float(v) for c, v in r.items()} for k, r in table.to_dict("index").items()}
    before, after = rows["before 2025-01-01"], rows["from 2025-01-01"]
    assert list(rows) == ["before 2025-01-01", "from 2025-01-01"]
    assert before["months"] == 12 and after["months"] == 12
    assert abs(before["active_return"]) < 1e-12  # the same returns: nothing added
    assert after["active_return"] > 0.1
    assert after["benchmark_cagr"] == pytest.approx(before["cagr"])


def test_deflating_by_more_trials_lowers_the_probability() -> None:
    days = pd.bdate_range("2015-01-01", "2020-12-31")
    rng = np.random.default_rng(4)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    net = benchmark + rng.normal(0.0001, 0.002, len(days))
    ten, twenty = deflated_active(net, benchmark, 0.4, 10), deflated_active(net, benchmark, 0.4, 20)
    assert 0 < twenty < ten < 1


def test_a_cap_holds_each_name_under_it_and_shares_out_the_excess() -> None:
    book = pd.DataFrame({1: [0.6], 2: [0.25], 3: [0.15], 4: [-0.7], 5: [-0.3]}, index=DAYS[:1])
    out = capped(book, 0.4)
    assert out.loc[DAYS[0], 1] == pytest.approx(0.4)
    # the 0.2 over the cap goes to the others in proportion: 0.25 and 0.15 become 0.375 and 0.225
    assert out.loc[DAYS[0], 2] == pytest.approx(0.375) and out.loc[DAYS[0], 3] == pytest.approx(0.225)
    assert out.iloc[0].clip(lower=0).sum() == pytest.approx(1.0)
    # a short leg of two names cannot fit under 0.4 each: it ends equally weighted, its total kept
    assert out.loc[DAYS[0], [4, 5]].to_numpy().tolist() == pytest.approx([-0.5, -0.5])


def test_a_drift_band_leaves_holdings_alone_and_trades_only_what_changed() -> None:
    returns = pd.DataFrame({1: [0.0, 0.05, 0.0, 0.0, 0.0, 0.0], 2: [0.0, -0.05, 0.0, 0.0, 0.0, 0.0]}, index=DAYS)
    book = pd.DataFrame({1: [0.5, 0.5], 2: [0.5, 0.5]}, index=[DAYS[0], DAYS[2]])  # the same names, equal weights
    every = run_backtest(book, returns, cost_bps=10.0)
    loose = run_backtest(book, returns, cost_bps=10.0, drift_band=0.2)
    assert every.turnover.loc[DAYS[3]] == pytest.approx(0.05)  # back to 50/50 from 52.5/47.5
    assert loose.turnover.loc[DAYS[3]] == 0.0  # 5 % off target is inside the band: no trade
    # an entry or an exit trades; the kept names keep their drifted weights, the entry absorbs the rest
    out = banded(np.array([0.5, 0.0, 0.5]), np.array([0.55, 0.45, 0.0]), 0.2)
    assert out.tolist() == pytest.approx([0.55, 0.0, 0.45])
    assert banded(np.array([0.5, 0.5]), np.array([0.7, 0.3]), 0.2).tolist() == pytest.approx([0.5, 0.5])


def test_quantile_books_hold_each_fifth_equally() -> None:
    days = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    score = pd.DataFrame([np.arange(1.0, 11.0), np.arange(10.0, 0.0, -1.0), [1.0, 2.0] + [np.nan] * 8], index=days)
    books = quantile_books(score, pd.DatetimeIndex(days), quantiles=5)
    assert sorted(books) == [1, 2, 3, 4, 5]
    assert list(books[5].index) == list(days[:2])  # two names on the last date: too few for five fifths
    assert books[5].iloc[0].to_dict() == {i: (0.5 if i in (8, 9) else 0.0) for i in range(10)}  # the highest scores
    assert books[1].iloc[1].to_dict() == {i: (0.5 if i in (8, 9) else 0.0) for i in range(10)}  # reversed: lowest
    for book in books.values():
        assert book.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    held = np.sum([(b > 0).to_numpy() for b in books.values()], axis=0)
    assert (held == 1).all()  # every name in one fifth on every date


def test_capture_measures_the_rising_and_falling_months_apart() -> None:
    months = pd.date_range("2020-01-31", periods=24, freq="ME")
    bench = pd.Series(np.where(np.arange(24) % 3 == 0, -0.02, 0.015), index=months)
    same = bt.capture(bench, bench)
    assert (same["up_capture"], same["down_capture"]) == pytest.approx((1.0, 1.0))
    cash = bt.capture(bench * 0.0, bench)
    assert (cash["up_capture"], cash["down_capture"]) == pytest.approx((0.0, 0.0))
    defensive = bt.capture(bench.clip(lower=0.0), bench)  # all of the rises, none of the falls
    assert defensive["up_capture"] == pytest.approx(1.0) and defensive["down_capture"] == pytest.approx(0.0)
    assert np.isnan(bt.capture(bench.iloc[:2], bench.iloc[:2])["down_capture"])  # one falling month measures nothing


def test_a_monthly_backtest_is_measured_at_twelve_periods_a_year() -> None:
    months = pd.date_range("2020-01-31", periods=36, freq="ME")
    r = pd.Series(np.random.default_rng(5).normal(0.005, 0.02, 36), index=months)
    result = BacktestResult(r, r, pd.Series(0.5, index=months), r * 0.0, pd.DataFrame(index=months))
    metrics, run_id = finish_run(
        result, r * 0.5, {"x": 1}, asset_class="fixed_income", family="f", engine=None, holdout=False,
        periods_per_year=12,
    )  # fmt: skip
    assert run_id is None
    assert metrics["cagr"] == pytest.approx(performance(r, periods_per_year=12)["cagr"])
    assert metrics["turnover"] == pytest.approx(6.0)  # 0.5 a month
    assert metrics["beta"] == pytest.approx(2.0)
