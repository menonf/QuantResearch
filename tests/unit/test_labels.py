import numpy as np
import pandas as pd
import pytest

from equity_research.research import labels

DAYS = pd.bdate_range("2020-01-01", periods=60)


def _returns(columns: dict[int, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=DAYS[: len(next(iter(columns.values())))])


def test_forward_returns_compound_the_next_days_and_stop_at_the_data() -> None:
    r = _returns({1: np.full(20, 0.01), 2: np.r_[np.nan, np.nan, np.full(18, -0.02)]})
    fwd = labels.forward_returns(r, 5)
    assert fwd.loc[DAYS[0], 1] == pytest.approx(1.01**5 - 1)
    assert fwd[1].iloc[-5:].isna().all() and fwd[1].iloc[:-5].notna().all()  # the window must exist
    assert fwd[2].iloc[:2].isna().all()  # not yet listed: no label
    assert fwd.loc[DAYS[2], 2] == pytest.approx(0.98**5 - 1)
    one = labels.forward_returns(r, 1)
    assert one.loc[DAYS[3], 1] == pytest.approx(0.01)  # tomorrow's return, not today's
    with pytest.raises(ValueError, match="at least 1"):
        labels.forward_returns(r, 0)


def test_a_listing_that_stops_trading_keeps_what_it_made() -> None:
    r = _returns({1: np.r_[np.full(5, 0.10), np.full(15, np.nan)]})
    fwd = labels.forward_returns(r, 5)
    assert fwd.loc[DAYS[2], 1] == pytest.approx(1.1**2 - 1)  # two days of +10 %, then delisted: flat
    assert bool(fwd.isna().loc[DAYS[6], 1])  # no return of its own on the day: no label


def test_period_returns_run_from_one_decision_date_to_the_next() -> None:
    r = _returns({1: np.full(60, 0.01)})
    dates = pd.DatetimeIndex([DAYS[9], DAYS[29], DAYS[59]])
    out = labels.period_returns(r, dates)
    assert out.loc[DAYS[9], 1] == pytest.approx(1.01**20 - 1)
    assert out.loc[DAYS[29], 1] == pytest.approx(1.01**30 - 1)
    assert bool(out.isna().loc[DAYS[59], 1])  # the last date has no next one


def test_triple_barrier_closes_at_the_first_barrier_or_the_deadline() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, 60)  # a year's volatility history of about 1 % a day
    rising = np.r_[noise[:40], np.full(20, 0.03)]
    falling = np.r_[noise[:40], np.full(20, -0.03)]
    still = np.r_[noise[:40], np.full(20, 0.0001)]
    tb = labels.triple_barrier(_returns({1: rising, 2: falling, 3: still}), horizon=10, width=1.0, vol_days=20)
    day = DAYS[39]  # a trade opened just before the run
    sigma = pd.Series(noise[:40]).iloc[-20:].std()
    assert tb.barrier.loc[day, 1] == pytest.approx(sigma * np.sqrt(10))
    expected_days = int(np.ceil(np.log1p(sigma * np.sqrt(10)) / np.log1p(0.03)))
    assert tb.label.loc[day, 1] == 1.0 and tb.days.loc[day, 1] == expected_days
    assert tb.label.loc[day, 2] == -1.0
    assert tb.days.loc[day, 3] == 10 and tb.label.loc[day, 3] == 1.0  # the deadline: the sign of a tiny gain
    assert tb.touch.loc[day, 1] == 1.0 and tb.touch.loc[day, 2] == -1.0 and tb.touch.loc[day, 3] == 0.0
    assert tb.ret.loc[day, 3] == pytest.approx(1.0001**10 - 1)
    assert tb.label.iloc[-10:].isna().all().all()  # deadlines past the data
    assert tb.label.iloc[:9].isna().all().all()  # too little history for a volatility


def test_trend_scan_finds_the_direction_and_clarity_of_the_coming_trend() -> None:
    rng = np.random.default_rng(1)
    up = 0.01 + rng.normal(0, 0.002, 60)
    down = -0.01 + rng.normal(0, 0.002, 60)
    walk = rng.normal(0, 0.02, 60)
    ts = labels.trend_scan(_returns({1: up, 2: down, 3: walk}), spans=range(5, 21))
    early = ts.t_value.iloc[:30]
    assert (early[1] > 10).all() and (early[2] < -10).all()  # clear trends
    assert early[3].abs().median() < early[1].abs().median() / 3  # a random walk's trends are murky
    spans = ts.span.iloc[:30].to_numpy()
    assert ((spans >= 5) & (spans <= 20)).all()  # every span is one of those scanned
    assert ts.t_value.iloc[-5:].isna().all().all()  # no window of even the shortest span left


def test_trend_scan_matches_a_least_squares_fit() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 30)
    ts = labels.trend_scan(_returns({1: r}), spans=[10])
    y = np.cumsum(np.log1p(r))[3:14]  # the window of day 3: days 3 .. 13
    x = np.arange(11.0)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = np.sqrt(resid @ resid / 9 / ((x - x.mean()) @ (x - x.mean())))
    assert ts.t_value.loc[DAYS[3], 1] == pytest.approx(slope / se)


def test_label_and_barrier_summaries() -> None:
    r = _returns({1: np.full(30, 0.01), 2: np.full(30, -0.01)})
    members = pd.DataFrame(True, index=r.index, columns=r.columns)
    table = labels.label_summary({"fwd_5d": labels.forward_returns(r, 5)}, members)
    assert table.loc["fwd_5d", "coverage"] == pytest.approx(25 / 30)
    assert table.loc["fwd_5d", "mean"] == pytest.approx(((1.01**5 - 1) + (0.99**5 - 1)) / 2)
    rng = np.random.default_rng(3)
    noisy = _returns({1: np.r_[rng.normal(0, 0.01, 30), np.full(30, 0.03)][:60]})
    tb = labels.triple_barrier(noisy, horizon=5, width=1.0, vol_days=20)
    shares = labels.barrier_summary(tb, pd.DataFrame(True, index=noisy.index, columns=noisy.columns))
    assert shares[["profit barrier", "loss barrier", "deadline"]].sum() == pytest.approx(1.0)
    assert 1 <= shares["days open"] <= 5
