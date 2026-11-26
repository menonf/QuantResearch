import numpy as np
import pandas as pd
import pytest

from research_core.engine.tilt import predicted_risk, tilt

N = 300
SECTORS = ["Energy", "Tech", "Banks"]


def _market(seed: int = 7) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """A benchmark, a score and a risk model of ``N`` names in three sectors (daily units)."""
    rng = np.random.default_rng(seed)
    names = pd.Index(range(1, N + 1))
    sector = rng.integers(0, len(SECTORS), N)
    loadings = pd.DataFrame(0.0, index=names, columns=["Market", *SECTORS, "Size", "Value"])
    loadings["Market"] = 1.0
    for i, s in enumerate(SECTORS):
        loadings[s] = (sector == i).astype(float)
    loadings["Size"] = rng.normal(size=N)
    loadings["Value"] = rng.normal(size=N)
    vols = np.array([0.012, 0.008, 0.010, 0.009, 0.004, 0.003])
    factor_cov = pd.DataFrame(np.diag(vols**2), index=loadings.columns, columns=loadings.columns)
    specific = pd.Series(rng.uniform(0.01, 0.03, N), index=names)
    cap = np.exp(1.5 * loadings["Size"].to_numpy())
    benchmark = pd.Series(cap / cap.sum(), index=names)
    score = pd.Series(rng.normal(size=N), index=names)
    return benchmark, score, loadings, factor_cov, specific


def test_a_long_only_tilt_holds_the_benchmark_plus_neutral_tilts_at_the_risk_asked() -> None:
    benchmark, score, loadings, factor_cov, specific = _market()
    w = tilt(benchmark, score, loadings, factor_cov, specific, neutral=[*SECTORS, "Size"], risk=0.02, max_active=0.01)
    active = (w - benchmark).to_numpy()
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    assert np.abs(active).max() <= 0.01 + 1e-12
    x = loadings.to_numpy()
    for name in [*SECTORS, "Size"]:  # no sector or size bet
        assert abs(float(loadings[name].to_numpy() @ active)) < 1e-6
    # no active beta: the book moves with its benchmark one for one
    f, s, b = factor_cov.to_numpy(), specific.to_numpy(), benchmark.to_numpy()
    sigma_b = x @ (f @ (x.T @ b)) + s**2 * b
    assert abs(float(active @ sigma_b) / float(b @ sigma_b)) < 1e-6
    yearly = predicted_risk(active, x, f, s) * np.sqrt(252)
    assert yearly == pytest.approx(0.02, rel=1e-3)
    assert np.corrcoef(active, score.to_numpy())[0, 1] > 0.3  # tilted towards the score


def test_a_budget_the_bounds_cannot_reach_takes_as_much_risk_as_they_allow() -> None:
    benchmark, score, loadings, factor_cov, specific = _market()
    x, f, s = loadings.to_numpy(), factor_cov.to_numpy(), specific.to_numpy()
    risks = []
    for budget in (0.05, 0.50):
        w = tilt(benchmark, score, loadings, factor_cov, specific, neutral=SECTORS, risk=budget, max_active=0.01)
        active = (w - benchmark).to_numpy()
        assert abs(active.sum()) < 1e-9 and (w >= 0).all()
        risks.append(predicted_risk(active, x, f, s) * np.sqrt(252))
    assert risks[0] < 0.05 and risks[0] == pytest.approx(risks[1], rel=1e-4)  # the most a 1 % cap allows here


def test_a_name_is_underweighted_at_most_by_what_the_benchmark_holds() -> None:
    benchmark, score, loadings, factor_cov, specific = _market(seed=3)
    w = tilt(benchmark, score, loadings, factor_cov, specific, neutral=SECTORS, risk=0.08, max_active=0.05)
    worst = score.nsmallest(20).index
    assert (w[worst] >= 0).all()
    assert (w[worst] <= benchmark[worst] + 1e-12).all()  # the worst names are cut, never shorted


def test_a_calmer_name_gets_a_bigger_tilt_for_the_same_score() -> None:
    benchmark, _, loadings, factor_cov, specific = _market(seed=5)
    score = pd.Series(0.0, index=benchmark.index)
    order = specific.sort_values().index
    calm, wild = order[0], order[-1]
    score[[calm, wild]] = 3.0
    w = tilt(benchmark, score, loadings, factor_cov, specific, risk=0.02, max_active=0.05)
    assert (w - benchmark)[calm] > (w - benchmark)[wild] > 0


def test_a_long_short_tilt_is_dollar_and_beta_neutral_and_capped_in_gross() -> None:
    benchmark, score, loadings, factor_cov, specific = _market(seed=11)
    w = tilt(
        benchmark,
        score,
        loadings,
        factor_cov,
        specific,
        neutral=[*SECTORS, "Size"],
        risk=0.30,
        max_active=0.02,
        long_only=False,
        gross=2.0,
    )
    assert w.sum() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(w).sum() <= 2.0 + 1e-9
    assert (w < 0).any() and (w > 0).any()
    for name in SECTORS:
        assert abs(float(loadings[name].to_numpy() @ w.to_numpy())) < 1e-6


def test_names_the_model_does_not_cover_stay_at_their_benchmark_weight() -> None:
    benchmark, score, loadings, factor_cov, specific = _market(seed=2)
    specific = specific.copy()
    specific.iloc[:5] = np.nan
    w = tilt(benchmark, score, loadings, factor_cov, specific, neutral=SECTORS, risk=0.03)
    assert w.iloc[:5].to_numpy() == pytest.approx(benchmark.iloc[:5].to_numpy())
    assert w.sum() == pytest.approx(1.0)
