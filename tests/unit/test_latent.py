import numpy as np
import pandas as pd
import pytest

from equity_research.ml import latent as lt

MONTHS = pd.date_range("2008-01-31", "2016-12-31", freq="ME")


def _months(seed: int = 0, n: int = 300, chars: int = 4) -> list[lt.Month]:
    """Returns from two factors whose loadings are linear in the characteristics: IPCA's own model, plus noise."""
    rng = np.random.default_rng(seed)
    gamma = np.zeros((chars + 1, 2))
    gamma[0, 0], gamma[1, 0], gamma[2, 1] = 1.0, 0.5, 1.0
    out = []
    for date in MONTHS:
        z = np.column_stack([np.ones(n), rng.uniform(-0.5, 0.5, (n, chars))])
        f = np.array([0.01, 0.005]) + rng.normal(0, [0.04, 0.03])
        r = z @ gamma @ f + rng.normal(0, 0.01, n)
        out.append(lt.Month(date, z, r))
    return out


def test_monthly_panel_and_managed_portfolios() -> None:
    dates = MONTHS[:2]
    index = pd.MultiIndex.from_product([dates, range(150)], names=["date", "listing_id"])
    x = pd.DataFrame({"a": np.tile(np.linspace(-0.5, 0.5, 150), 2)}, index=index)
    returns = pd.DataFrame(0.01, index=dates, columns=range(150))
    returns.iloc[1, :60] = np.nan  # 90 left: too few for the second month
    months = lt.monthly_panel(x, returns)
    assert [m.date for m in months] == [dates[0]] and months[0].z.shape == (150, 2)
    managed = lt.managed_portfolios(months, ["a"])
    assert list(managed.columns) == ["market", "a"] and managed.iloc[0, 0] == pytest.approx(0.01)
    clipped = lt.clipped_returns(
        pd.DataFrame([[0.0, 1.0, 50.0]] * 2, index=dates), pd.DataFrame(True, index=dates, columns=range(3))
    )
    assert clipped.to_numpy()[0, 2] < 50.0


def test_pca_and_rp_pca_loadings() -> None:
    rng = np.random.default_rng(1)
    loud = rng.normal(0, 0.10, (500, 1))
    earner = 0.02 + rng.normal(0, 0.03, (500, 1))
    x = np.hstack([loud, earner, rng.normal(0, 0.001, (500, 2))])
    pca = lt.pca_loadings(x, 1)
    rp = lt.pca_loadings(x, 1, gamma=50.0)
    assert abs(pca[0, 0]) > 0.9  # PCA follows the variance
    assert abs(rp[1, 0]) > abs(pca[1, 0])  # RP-PCA leans towards what earns
    w = lt.tangency_weights(np.column_stack([earner[:, 0], loud[:, 0]]))
    assert np.abs(w).sum() == pytest.approx(1.0) and w[0] > w[1]


def test_ipca_recovers_a_two_factor_model_out_of_sample() -> None:
    months = _months()
    names = [f"c{i}" for i in range(4)]
    one = lt.walk_forward_ipca(months, 1, first_test=2015, last_test=2016)
    two = lt.walk_forward_ipca(months, 2, first_test=2015, last_test=2016)
    assert two["total R2"] > 0.9 and two["total R2"] > one["total R2"]
    managed = lt.managed_portfolios(months, names)
    pca = lt.walk_forward_pca(managed, 2, first_test=2015, last_test=2016)
    assert pca["total R2"] > 0.9
    observed = lt.walk_forward_observed(managed, managed[["market"]], first_test=2015, last_test=2016)
    assert 0 < observed["total R2"] < two["total R2"]  # one observed factor misses the second


def test_conditional_autoencoder_learns_the_factors() -> None:
    pytest.importorskip("torch")
    from equity_research.ml import autoencoder as ae  # noqa: PLC0415

    months = _months(2, n=200)
    out = ae.walk_forward(months, 2, first_test=2016, last_test=2016, epochs=40, patience=8)
    assert out["total R2"] > 0.5
    beta, f = ae.predict(ae.fit(months[:60], 2, epochs=5), months[60])
    assert beta.shape == (200, 2) and f.shape == (2,)
