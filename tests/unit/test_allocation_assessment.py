import numpy as np
import pandas as pd
import pytest

from equity_research.ml import assessment as asm
from research_core.analytics.risk import statistics as risk_stats
from research_core.engine import allocation as al
from research_core.engine.backtest import BacktestResult

DAYS = pd.bdate_range("2018-01-01", "2020-12-31")
NAMES = list(range(1, 41))


def _returns(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = np.linspace(0.005, 0.03, len(NAMES))  # name 1 calmest, name 40 wildest
    return pd.DataFrame(rng.normal(0, 1, (len(DAYS), len(NAMES))) * vol, index=DAYS, columns=NAMES)


def test_select_and_simple_weights() -> None:
    score = pd.Series(np.arange(40.0), index=NAMES)
    assert al.select(score, fraction=0.1) == [37, 38, 39, 40]
    assert al.select(score, fraction=0.1, top=False) == [1, 2, 3, 4]
    r = _returns()
    day = DAYS[-1]
    eq = al.leg_weights([1, 40], "equal", day=day, returns=r)
    assert eq.tolist() == [0.5, 0.5]
    iv = al.leg_weights([1, 40], "inverse_vol", day=day, returns=r)
    assert iv[1] > 4 * iv[40] and iv.sum() == pytest.approx(1.0)  # the calm name gets more capital
    with pytest.raises(ValueError, match="unknown method"):
        al.leg_weights([1], "magic", day=day, returns=r)
    with pytest.raises(ValueError, match="score and the risk model"):
        al.leg_weights([1, 2], "mean_variance", day=day, returns=r)


def test_hrp_splits_risk_between_uncorrelated_clusters() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 0.01, len(DAYS)), rng.normal(0, 0.03, len(DAYS))
    # names 1-3 move with a (calm), names 4-5 with b (three times as volatile)
    r = pd.DataFrame({n: (a if n <= 3 else b) + rng.normal(0, 0.001, len(DAYS)) for n in range(1, 6)}, index=DAYS)
    w = al.leg_weights([1, 2, 3, 4, 5], "hrp", day=DAYS[-1], returns=r)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert w[[1, 2, 3]].sum() > 0.8  # the calm cluster carries most of the capital


def test_mean_variance_follows_score_over_risk() -> None:
    names = [1, 2, 3]
    loadings = pd.DataFrame({"market": [1.0, 1.0, 1.0]}, index=names)
    factor_cov = pd.DataFrame([[1e-6]], index=["market"], columns=["market"])
    specific = pd.Series([0.01, 0.01, 0.02], index=names)

    def risk(day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        return loadings, factor_cov, specific

    score = pd.Series([1.0, 2.0, 3.0], index=names)
    w = al.leg_weights(names, "mean_variance", day=DAYS[-1], returns=_returns(), score=score, risk=risk)
    assert w.sum() == pytest.approx(1.0) and w[1] == 0.0  # the lowest score is clipped out
    assert w[2] > 0 and w[3] > 0


def test_decile_and_top_k_books() -> None:
    r = _returns()
    dates = pd.DatetimeIndex([DAYS[300], DAYS[400]])
    score = pd.DataFrame(np.tile(np.arange(40.0), (2, 1)), index=dates, columns=NAMES)
    members = pd.DataFrame(True, index=DAYS, columns=NAMES)
    long_only = al.decile_book(score, members, dates, r)
    assert long_only.sum(axis=1).tolist() == pytest.approx([1.0, 1.0]) and (long_only.iloc[0, -4:] == 0.25).all()
    both = al.decile_book(score, members, dates, r, method="inverse_vol", long_only=False)
    assert both.sum(axis=1).tolist() == pytest.approx([0.0, 0.0]) and both.abs().sum(axis=1).tolist() == pytest.approx(
        [2.0, 2.0]
    )
    k = al.top_k_book(score, members, 5)
    assert (k.iloc[0, -5:] == 0.2).all() and (k.iloc[0, :5] == -0.2).all()


def test_exposure_overlays() -> None:
    rng = np.random.default_rng(2)
    net = pd.Series(rng.normal(0.0004, 0.02, len(DAYS)), index=DAYS)
    e = al.vol_target(net, target=0.10)
    assert e.iloc[-1] == pytest.approx(0.10 / (0.02 * np.sqrt(252)), rel=0.25)
    crash = pd.Series(np.r_[np.full(100, 0.001), np.full(20, -0.02), np.full(50, 0.0)], index=DAYS[:170])
    dd = al.drawdown_control(crash, threshold=0.15, cut=0.5)
    assert dd.iloc[:100].eq(1.0).all() and dd.iloc[-1] == 0.5
    kelly = al.kelly(net, fraction=0.5)
    assert kelly.iloc[:252].eq(1.0).all() and (kelly.iloc[253:] >= 0).all()
    scaled = al.apply_exposure(pd.Series(0.01, index=DAYS[:3]), pd.Series([1.0, 0.5, 0.5], index=DAYS[:3]), cost_bps=10)
    assert scaled.tolist() == pytest.approx([0.01, 0.005 - 0.5 * 10 / 1e4, 0.005])


def test_break_even_and_tail_risk() -> None:
    benchmark = pd.Series(0.0003, index=DAYS)
    gross = benchmark + 0.0002
    net = gross - 0.0001  # costs take half of the edge
    result = BacktestResult(gross=gross, net=net, turnover=pd.Series(0.01, index=DAYS), costs=gross - net,
                            weights=pd.DataFrame())  # fmt: skip
    even = asm.break_even(result, benchmark)
    assert even["break-even multiple"] == pytest.approx(2.0, rel=0.05)
    assert even["cost charged (bp)"] == pytest.approx(100.0)
    r = pd.Series(np.tile(np.r_[np.full(98, 0.001), -0.05, -0.05], 10), index=pd.RangeIndex(1000))
    tails = risk_stats.tail_risk(r, confidence=0.99)  # 2 % of days lose 5 %: the worst 1 % all do
    assert tails["VaR"] == pytest.approx(0.05) and tails["CVaR"] == pytest.approx(0.05)


def test_drift_transfer_coefficient_and_the_law() -> None:
    months = pd.date_range("2010-01-31", periods=120, freq="ME")
    fading = pd.Series(np.linspace(0.06, -0.02, 120) + np.random.default_rng(3).normal(0, 0.01, 120), index=months)
    drift = asm.ic_drift(fading)
    assert drift["trend a year"] == pytest.approx(-0.08 / (119 / 12), rel=0.2) and drift["t_nw"] < -5
    dates = months[:3]
    score = pd.DataFrame(np.random.default_rng(4).normal(size=(3, 40)), index=dates, columns=NAMES)
    vol = pd.DataFrame(0.02, index=dates, columns=NAMES)
    members = pd.DataFrame(True, index=dates, columns=NAMES)
    bench = pd.DataFrame(1 / 40, index=dates, columns=NAMES)
    book = bench + score / 100  # active weights proportional to score / vol (vol is the same for all)
    tc = asm.transfer_coefficient(book, score, vol, members, bench)
    assert tc.tolist() == pytest.approx([1.0] * 3)
    assert asm.fundamental_law(0.02, 1500 * 12, 0.5) == pytest.approx(0.5 * 0.02 * np.sqrt(18000))


def test_reality_check_and_rademacher_bounds() -> None:
    rng = np.random.default_rng(5)
    noise = pd.DataFrame(rng.normal(0, 0.02, (240, 10)), columns=[f"s{i}" for i in range(10)])
    assert asm.reality_check(noise, n_boot=500)["p_value"] > 0.05  # the best of ten noises is luck
    skilled = noise.assign(s0=noise["s0"] + 0.01)
    rc = asm.reality_check(skilled, n_boot=500)
    assert rc["best"] == "s0" and rc["p_value"] < 0.01
    # one loud, unskilled strategy dominates White's raw maximum and hides a quiet skilled one; studentised, it does not
    quiet = noise.assign(s0=noise["s0"] * 0.25 + 0.004, s1=noise["s1"] * 10)
    assert asm.reality_check(quiet, n_boot=500)["p_value"] > 0.05
    rc = asm.reality_check(quiet, n_boot=500, studentise=True)
    assert rc["best"] == "s0" and rc["p_value"] < 0.01 and abs(rc["mean"] - quiet["s0"].mean()) < 1e-12
    ics = pd.DataFrame(rng.uniform(-0.2, 0.2, (5000, 5)), columns=list("abcde"))
    ics["a"] += 0.15
    bounds = asm.rademacher_bounds(ics, draws=500)
    lower = bounds["lower bound"].to_numpy()
    assert lower[0] > 0 and (lower[1:] < 0).all()  # only a clears the bound
    idx = asm.stationary_bootstrap(100, 5.0, np.random.default_rng(6))
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100


def test_ml_strategy_measures_long_only_against_the_index_and_long_short_against_cash() -> None:
    from equity_research.ml import ml_strategy as ms  # noqa: PLC0415

    rng = np.random.default_rng(7)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(DAYS)), index=DAYS)
    net = bench + rng.normal(0.0001, 0.002, len(DAYS))
    lo = ms.measure(net, bench, long_only=True)
    ls = ms.measure(net - bench, bench, long_only=False)
    assert 0 < lo["tracking_error"] < 0.05 and ls["information_ratio"] == ls["sharpe"]
    assert len(ms.monthly_active(net, bench, long_only=True)) == 36
    assert len(ms.monthly_active(net - bench, bench, long_only=False)) == 36
    assert 0 < ms.deflated_sharpe_of(net, 0.8, 30) < ms.deflated_sharpe_of(net, 0.8, 1) <= 1
