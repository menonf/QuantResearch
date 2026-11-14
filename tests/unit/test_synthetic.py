"""Synthetic markets: generators recover what they were fitted to, fidelity measures tell real from fake."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_research.research import synthetic as syn


def _panel(t: int = 1500, n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """A two-factor market with GARCH-like market volatility; returns the panel and the true market betas."""
    rng = np.random.default_rng(seed)
    garch = syn.GarchT(mu=0.0004, omega=2e-6, alpha=0.08, beta=0.9, nu=5.0)
    market = garch.simulate(t, rng)
    style = rng.normal(0, 0.004, t)
    betas = rng.uniform(0.5, 1.5, n)
    tilts = rng.normal(0, 1, n)
    x = market[:, None] * betas + style[:, None] * tilts + rng.normal(0, 0.01, (t, n))
    index = pd.bdate_range("2015-01-01", periods=t)
    return pd.DataFrame(x, index=index, columns=range(100, 100 + n)), betas


def test_garch_t_fit_recovers_persistence_and_tails() -> None:
    truth = syn.GarchT(mu=0.0, omega=2e-6, alpha=0.08, beta=0.9, nu=5.0)
    fit = syn.fit_garch_t(truth.simulate(5000, np.random.default_rng(1)))
    assert fit.alpha + fit.beta == pytest.approx(0.98, abs=0.02)
    assert 3.5 < fit.nu < 8.0


def test_bootstrap_moves_whole_days_and_keeps_names_and_dates() -> None:
    panel, _ = _panel(t=300, n=10)
    path = syn.fit_bootstrap(panel, mean_block=10).sample(np.random.default_rng(2))
    assert path.index.equals(panel.index) and path.columns.equals(panel.columns)
    rows = {tuple(np.round(r, 12)) for r in panel.to_numpy()}
    assert all(tuple(np.round(r, 12)) in rows for r in path.to_numpy())  # each day is a real day, every name together


def test_principal_factors_recover_the_market_betas() -> None:
    panel, betas = _panel()
    factors, loadings, resid = syn.principal_factors(panel, 2)
    assert np.corrcoef(loadings.loc["f1"].to_numpy(), betas)[0, 1] > 0.9
    assert factors.shape == (len(panel), 2) and resid.between(0.005, 0.02).all()


def test_baum_welch_finds_the_turbulent_days() -> None:
    rng = np.random.default_rng(3)
    states = np.repeat([0, 1, 0, 1, 0], [400, 100, 300, 150, 300])
    x = rng.normal(0, np.where(states == 1, 0.03, 0.008))
    transition, turbulent = syn.baum_welch(x)
    assert ((turbulent > 0.5) == (states == 1)).mean() > 0.9
    assert transition[0, 0] > 0.95 and transition[1, 1] > 0.9


def test_factor_models_reproduce_clustering_and_the_regime_model_raises_correlation_in_turbulence() -> None:
    panel, _ = _panel()
    real = syn.stylized_facts(panel, sample=40)
    garch = syn.stylized_facts(syn.fit_factor_garch(panel, factors=2).sample(np.random.default_rng(4)), sample=40)
    assert garch["market_acf_abs_1"] > 0.05 and real["market_acf_abs_1"] > 0.05  # volatility clusters in both
    assert garch["mean_corr"] == pytest.approx(real["mean_corr"], abs=0.1)
    regime = syn.fit_regime(panel, factors=2)
    calm, turbulent = regime.regimes.covs[0], regime.regimes.covs[1]
    assert turbulent[0, 0] > 2 * calm[0, 0]
    stressed = syn.RegimeFactors(regime.index, regime.loadings, regime.resid_sd, regime.regimes, 4.0, (0, len(panel)))
    path = stressed.sample(np.random.default_rng(5))
    assert path.std().mean() > panel.std().mean()  # held turbulent throughout: a rougher market


def test_iid_noise_shows_no_clustering_and_thin_tails() -> None:
    rng = np.random.default_rng(6)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 30)), index=pd.bdate_range("2010-01-01", periods=2000))
    facts = syn.stylized_facts(noise, sample=30)
    assert abs(facts["kurtosis"]) < 0.3 and abs(facts["acf_abs_1"]) < 0.05 and abs(facts["mean_corr"]) < 0.05


def test_discriminator_cannot_tell_a_market_from_itself_but_can_from_a_wilder_one() -> None:
    panel, _ = _panel(t=2500, n=20)
    same = syn.fit_bootstrap(panel).sample(np.random.default_rng(7))
    assert syn.discriminator_score(panel, [same]) < 0.65
    assert syn.discriminator_score(panel, [same * 3]) > 0.9


def test_book_returns_hold_month_end_weights_through_the_month() -> None:
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    daily = pd.DataFrame(0.0, index=days, columns=[1, 2])
    daily.loc["2020-02-03", 1] = 0.10  # in February
    daily.loc["2020-02-04", 2] = -0.05
    daily.loc["2020-03-02", 1] = 0.20  # in March
    month_ends = pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"])
    book = pd.DataFrame({1: [0.5, 1.0], 2: [0.5, 0.0]}, index=month_ends[:2])
    got = syn.book_returns(book, daily, month_ends)
    assert got.loc["2020-01-31"] == pytest.approx(0.5 * 0.10 + 0.5 * -0.05)
    assert got.loc["2020-02-28"] == pytest.approx(0.20)
    costs = pd.Series(0.001, index=days)
    summed = syn.month_sum(costs, month_ends)
    assert summed.loc["2020-01-31"] == pytest.approx(
        0.001 * len(days[(days > pd.Timestamp("2020-01-31")) & (days <= pd.Timestamp("2020-02-28"))])
    )


def test_outcomes_summarise_paths() -> None:
    months = pd.RangeIndex(120)
    paths = pd.DataFrame({"good": 0.01, "bad": -0.005}, index=months)
    bench = pd.DataFrame({"good": 0.005, "bad": 0.0}, index=months)
    out = syn.outcomes(paths, bench)
    assert out["p_loss"] == 0.5 and out["p_trail"] == 0.5
    assert out["max_dd_p95"] > 0.4  # the bad path lost about 45 % over ten years


def test_factor_drift_is_the_sample_mean_and_a_rougher_market_keeps_it() -> None:
    panel, _ = _panel(t=2500, n=40)
    gen = syn.fit_factor_garch(panel, factors=2)
    rng = np.random.default_rng(8)
    calm = np.mean([gen.sample(rng).mean(axis=1).mean() for _ in range(20)])
    rough_gen = syn.FactorGarch(gen.index, gen.loadings, gen.resid_sd, gen.factor_models, gen.resid_nu, vol_scale=2.0)
    rough = [rough_gen.sample(rng) for _ in range(20)]
    real = panel.mean(axis=1).mean()
    assert calm == pytest.approx(real, abs=2e-4)
    assert np.mean([p.mean(axis=1).mean() for p in rough]) == pytest.approx(real, abs=4e-4)  # drift unscaled
    assert np.mean([p.std().mean() for p in rough]) > 1.8 * panel.std().mean()


def test_synthetic_listings_exist_only_where_the_real_ones_traded() -> None:
    panel, _ = _panel(t=400, n=10)
    panel.iloc[:100, 0] = np.nan  # listed late
    panel.iloc[300:, 1] = np.nan  # delisted
    for gen in (syn.fit_bootstrap(panel), syn.fit_factor_garch(panel, factors=2), syn.fit_regime(panel, factors=2)):
        path = gen.sample(np.random.default_rng(9))
        assert path.isna().equals(panel.isna())
