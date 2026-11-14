"""Synthetic markets for stress-testing strategies: three generators, their fidelity, and books run on their paths.

    gen = fit_factor_garch(returns, factors=5)               # or fit_bootstrap / fit_regime
    path = gen.sample(rng)                                    # a synthetic date x listing panel, real dates and names
    facts = stylized_facts(path)                              # fat tails, volatility clustering, correlation...
    monthly = book_returns(book, path, month_ends)            # a strategy's books held on the synthetic market

A backtest is one path through history; a strategy judged on it is judged on one draw of luck. A generator fitted to
the history makes other paths with the same statistical character, and a strategy's outcomes across them - its
median, its bad tail, how often it trails - say how much of the backtest was the strategy and how much the path.

Steps:

1. **Generators** - each keeps the real panel's dates and listings, so a strategy's real books apply unchanged:

   * :func:`fit_bootstrap` - stationary block bootstrap of whole days (Politis and Romano): every listing's return
     of a day moves together, so cross-sectional correlation, fat tails and short volatility clusters survive; long
     regimes and anything slower than a block do not.
   * :func:`fit_factor_garch` - a statistical factor model: principal components of the returns, each factor a
     GARCH(1,1) with Student-t shocks (volatility clustering and fat tails), each listing its loadings and a
     Student-t residual of its own size.
   * :func:`fit_regime` - the same factors driven by a two-state Markov chain (calm, turbulent) fitted by
     Baum-Welch to the first factor; each state has its own factor means and covariance, so correlations rise in
     the turbulent state, as they do in crises.

2. **Fidelity** - :func:`stylized_facts` measures what a synthetic market must reproduce (Cont, 2001): heavy tails
   (kurtosis), no autocorrelation of returns, slowly decaying autocorrelation of absolute returns (volatility
   clustering), the leverage effect (falls raise later volatility), the average correlation between stocks and the
   share of variance the first component carries. :func:`discriminator_score` asks a classifier to tell real
   months from synthetic ones by their statistics: 50 % accuracy means it cannot.

3. **Books on paths** - :func:`book_returns` holds a strategy's real month-end weights through each synthetic
   month (each listing's month compounded from its synthetic days, missing days counting as cash);
   :func:`outcomes` summarises many paths - growth, drawdown, how often a book trailed its benchmark.

Generators are fitted to the period they simulate: they describe a market, they do not forecast one.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

SEED = 7


class Generator(Protocol):
    """Anything that draws a synthetic ``date x listing`` return panel."""

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        """One synthetic path."""
        ...


def _present(returns: pd.DataFrame) -> np.ndarray:
    """Where the real panel has a return: a synthetic listing exists only where the real one traded."""
    return returns.notna().to_numpy()


def _frame(values: np.ndarray, like: pd.DatetimeIndex, columns: pd.Index, present: np.ndarray | None) -> pd.DataFrame:
    """A synthetic panel, missing where the real listing did not trade."""
    if present is not None:
        values = np.where(present, values, np.nan)
    return pd.DataFrame(values, index=like, columns=columns)


def _clean(returns: pd.DataFrame) -> pd.DataFrame:
    """Returns with missing values as 0 (no position) and extremes cut at -95 % / +200 % (data errors)."""
    return returns.fillna(0.0).clip(-0.95, 2.0)


# ---------------------------------------------------------------- 1a. block bootstrap


@dataclass(frozen=True)
class Bootstrap:
    """Stationary block bootstrap of whole days."""

    returns: pd.DataFrame
    mean_block: float
    present: np.ndarray | None = None

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        """Days resampled in blocks of geometric length (mean ``mean_block``), every listing together."""
        t = len(self.returns)
        idx = np.empty(t, dtype=int)
        i = int(rng.integers(t))
        restart = rng.random(t) < 1.0 / self.mean_block
        starts = rng.integers(t, size=t)
        for k in range(t):
            idx[k] = i
            i = int(starts[k]) if restart[k] else (i + 1) % t
        values = self.returns.to_numpy()[idx]
        return _frame(values, pd.DatetimeIndex(self.returns.index), self.returns.columns, self.present)


def fit_bootstrap(returns: pd.DataFrame, *, mean_block: float = 21.0) -> Bootstrap:
    """A block bootstrap of ``returns`` (blocks of about a month by default)."""
    return Bootstrap(_clean(returns), mean_block, _present(returns))


# ---------------------------------------------------------------- 1b. factor model with GARCH-t factors


@dataclass(frozen=True)
class GarchT:
    """GARCH(1,1) with Student-t shocks: ``x_t = mu + s_t z_t``, ``s_t^2 = omega + alpha (x - mu)^2 + beta s^2``."""

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float

    def simulate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` draws, started at the unconditional variance."""
        z = rng.standard_t(self.nu, size=n) * math.sqrt((self.nu - 2) / self.nu)  # unit variance
        var = self.omega / max(1e-12, 1 - self.alpha - self.beta)
        out = np.empty(n)
        for k in range(n):
            e = math.sqrt(var) * z[k]
            out[k] = self.mu + e
            var = self.omega + self.alpha * e * e + self.beta * var
        return out


def _garch_nll(params: np.ndarray, x: np.ndarray) -> float:
    mu, omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999 or nu <= 2.05:
        return 1e10
    e = x - mu
    var = np.empty_like(x)
    var[0] = e.var()
    for k in range(1, len(x)):
        var[k] = omega + alpha * e[k - 1] ** 2 + beta * var[k - 1]
    scale2 = var * (nu - 2) / nu
    ll = (
        special.gammaln((nu + 1) / 2)
        - special.gammaln(nu / 2)
        - 0.5 * np.log(np.pi * nu * scale2)
        - (nu + 1) / 2 * np.log1p(e * e / (nu * scale2))
    )
    return float(-ll.sum())


def fit_garch_t(x: np.ndarray) -> GarchT:
    """Maximum-likelihood GARCH(1,1)-t of a return series."""
    x = np.asarray(x, dtype=float)
    v = float(x.var())
    start = np.array([float(x.mean()), 0.05 * v, 0.08, 0.90, 6.0])
    res = optimize.minimize(_garch_nll, start, args=(x,), method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-6})  # fmt: skip
    mu, omega, alpha, beta, nu = res.x
    return GarchT(float(mu), float(omega), float(alpha), float(beta), float(nu))


def principal_factors(returns: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """``k`` principal-component factors of ``returns``: factor returns, loadings and residual sizes.

    Returns the factor returns (date x k), the loadings (k x listing) and each listing's residual standard deviation
    over its own days. The components come from the listings present on at least 90 % of days (missing days as 0);
    each listing's loadings are then its regression on the factors over the days it has.
    """
    raw = returns.clip(-0.95, 2.0)
    present = raw.notna().mean() >= 0.9
    core = raw.loc[:, present].fillna(0.0)
    centred = core - core.mean()
    _, _, vt = np.linalg.svd(centred.to_numpy(), full_matrices=False)
    weights = vt[:k].T / np.abs(vt[:k].T).sum(axis=0)  # each factor a portfolio whose weights' sizes sum to 1
    weights *= np.sign(weights.sum(axis=0))  # the first is long the market
    factors = pd.DataFrame(core.to_numpy() @ weights, index=returns.index, columns=[f"f{i + 1}" for i in range(k)])
    f = factors.to_numpy()
    loadings = np.full((k, raw.shape[1]), np.nan)
    resid = np.full(raw.shape[1], np.nan)
    for j, col in enumerate(raw.columns):
        y = raw[col].to_numpy()
        ok = ~np.isnan(y)
        if ok.sum() < 60:
            continue
        x = np.column_stack([np.ones(ok.sum()), f[ok]])
        coef, *_ = np.linalg.lstsq(x, y[ok], rcond=None)
        loadings[:, j] = coef[1:]
        resid[j] = float(np.std(y[ok] - x @ coef, ddof=k + 1))
    return (
        factors,
        pd.DataFrame(loadings, index=factors.columns, columns=raw.columns).fillna(0.0),
        pd.Series(resid, index=raw.columns).fillna(float(np.nanmedian(resid))),
    )


@dataclass(frozen=True)
class FactorGarch:
    """Factor model: GARCH-t factors, fixed loadings, Student-t residuals."""

    index: pd.DatetimeIndex
    loadings: pd.DataFrame
    resid_sd: pd.Series
    factor_models: tuple[GarchT, ...]
    resid_nu: float
    vol_scale: float = 1.0  # a scenario: every factor's and residual's swings (not the drift) scaled
    present: np.ndarray | None = None

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        """One path: simulated factors times loadings, plus each listing's residual."""
        t = len(self.index)
        mu = np.array([m.mu for m in self.factor_models])
        f = mu + (np.column_stack([m.simulate(t, rng) for m in self.factor_models]) - mu) * self.vol_scale
        z = rng.standard_t(self.resid_nu, size=(t, len(self.resid_sd))) * math.sqrt((self.resid_nu - 2) / self.resid_nu)
        values = f @ self.loadings.to_numpy() + z * self.resid_sd.to_numpy() * self.vol_scale
        return _frame(np.clip(values, -0.95, 2.0), self.index, self.loadings.columns, self.present)


def fit_factor_garch(returns: pd.DataFrame, *, factors: int = 5, resid_nu: float = 4.0) -> FactorGarch:
    """A factor model of ``returns``: ``factors`` principal components, each fitted a GARCH(1,1)-t."""
    f, loadings, resid = principal_factors(returns, factors)
    # each factor's drift is its sample mean: a Student-t's fitted centre sits nearer the median, above the mean
    # of negatively skewed returns, and over a decade that difference compounds into a market that rises too fast
    models = tuple(dataclasses.replace(fit_garch_t(f[c].to_numpy()), mu=float(f[c].mean())) for c in f.columns)
    return FactorGarch(pd.DatetimeIndex(returns.index), loadings, resid, models, resid_nu, present=_present(returns))


# ---------------------------------------------------------------- 1c. regime-switching factors


@dataclass(frozen=True)
class Regimes:
    """A two-state Markov chain: ``transition[i, j]`` = P(state j tomorrow | state i today); state 1 is turbulent."""

    transition: np.ndarray
    start: np.ndarray
    means: np.ndarray  # state x factor
    covs: np.ndarray  # state x factor x factor


def baum_welch(x: np.ndarray, *, iterations: int = 200, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Gaussian hidden Markov model of ``x`` by expectation-maximisation.

    Returns the transition matrix and each day's smoothed probability of the high-variance state.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = np.array([x.mean(), x.mean()])
    sd = np.array([x.std() * 0.7, x.std() * 1.8])
    a = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = np.array([0.8, 0.2])
    previous = -np.inf
    gamma = np.zeros((n, 2))
    for _ in range(iterations):
        dens = np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)) + 1e-300
        alpha = np.empty((n, 2))
        scale = np.empty(n)
        alpha[0] = pi * dens[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for k in range(1, n):
            alpha[k] = (alpha[k - 1] @ a) * dens[k]
            scale[k] = alpha[k].sum()
            alpha[k] /= scale[k]
        beta = np.ones((n, 2))
        for k in range(n - 2, -1, -1):
            beta[k] = (a @ (dens[k + 1] * beta[k + 1])) / scale[k + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * a[None] * (dens[1:] * beta[1:])[:, None, :]) / scale[1:, None, None]
        a = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        pi = gamma[0]
        w = gamma.sum(axis=0)
        mu = (gamma * x[:, None]).sum(axis=0) / w
        sd = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w)
        ll = float(np.log(scale).sum())
        if ll - previous < tol:
            break
        previous = ll
    if sd[0] > sd[1]:  # state 1 is the turbulent one
        a = a[::-1, ::-1]
        gamma = gamma[:, ::-1]
    return a, gamma[:, 1]


@dataclass(frozen=True)
class RegimeFactors:
    """Factors drawn from the state of a Markov chain; listings as in :class:`FactorGarch`."""

    index: pd.DatetimeIndex
    loadings: pd.DataFrame
    resid_sd: pd.Series
    regimes: Regimes
    resid_nu: float
    force_turbulent: tuple[int, int] | None = None  # (first day, days): a scenario holds the chain turbulent
    present: np.ndarray | None = None

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        """One path: a state path, factors from each state's distribution, plus residuals."""
        t = len(self.index)
        r = self.regimes
        states = np.empty(t, dtype=int)
        s = int(rng.random() < r.start[1])
        u = rng.random(t)
        for k in range(t):
            states[k] = s
            s = int(u[k] < r.transition[s, 1])
        if self.force_turbulent is not None:
            first, days = self.force_turbulent
            states[first : first + days] = 1
        k_f = r.means.shape[1]
        f = np.empty((t, k_f))
        for state in (0, 1):
            on = states == state
            f[on] = rng.multivariate_normal(r.means[state], r.covs[state], size=int(on.sum()))
        z = rng.standard_t(self.resid_nu, size=(t, len(self.resid_sd))) * math.sqrt((self.resid_nu - 2) / self.resid_nu)
        scale = np.where(states == 1, 1.5, 1.0)[:, None]  # residuals are larger in turbulence too
        values = f @ self.loadings.to_numpy() + z * self.resid_sd.to_numpy() * scale
        return _frame(np.clip(values, -0.95, 2.0), self.index, self.loadings.columns, self.present)


def fit_regime(returns: pd.DataFrame, *, factors: int = 5, resid_nu: float = 4.0) -> RegimeFactors:
    """A regime-switching factor model of ``returns``: states from the first factor, moments per state."""
    f, loadings, resid = principal_factors(returns, factors)
    transition, turbulent = baum_welch(f.iloc[:, 0].to_numpy())
    x = f.to_numpy()
    means, covs = [], []
    for weight in (1 - turbulent, turbulent):
        w = weight / weight.sum()
        m = w @ x
        means.append(m)
        covs.append(((x - m) * w[:, None]).T @ (x - m))
    # resid_sd was measured across both states; scale it so its average over the states is unchanged
    share = float(turbulent.mean())
    resid = resid / math.sqrt((1 - share) + share * 1.5**2)
    regimes = Regimes(transition, np.array([1 - share, share]), np.array(means), np.array(covs))
    return RegimeFactors(pd.DatetimeIndex(returns.index), loadings, resid, regimes, resid_nu, present=_present(returns))


# ---------------------------------------------------------------- 2. fidelity


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (0 for a normal distribution), corrected for sample size."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(stats.kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else float("nan")


def _autocorr(x: np.ndarray, lag: int) -> float:
    a, b = x[:-lag], x[lag:]
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def stylized_facts(returns: pd.DataFrame, *, sample: int = 200, seed: int = SEED) -> pd.Series:
    """The statistics a realistic market shows, averaged over ``sample`` listings (and the equal-weighted market).

    ``kurtosis`` (excess, of daily returns), ``acf_ret_1`` (lag-1 autocorrelation of returns), ``acf_abs_1`` /
    ``acf_abs_20`` (of absolute returns: volatility clustering and its persistence), ``leverage_5`` (correlation of
    today's return with absolute returns over the next five days), ``mean_corr`` (average correlation between pairs
    of listings), ``first_pc`` (the first component's share of the variance), ``market_*`` the same for the market.
    """
    rng = np.random.default_rng(seed)
    data = returns.dropna(axis=1, thresh=int(0.9 * len(returns)))
    cols = rng.choice(data.columns.to_numpy(), size=min(sample, data.shape[1]), replace=False)
    x = data[cols].to_numpy(dtype=float)
    facts: dict[str, list[float]] = {"kurtosis": [], "acf_ret_1": [], "acf_abs_1": [], "acf_abs_20": [],
                                     "leverage_5": []}  # fmt: skip
    for j in range(x.shape[1]):
        r = x[:, j]
        r = r[~np.isnan(r)]
        facts["kurtosis"].append(_kurtosis(r))
        facts["acf_ret_1"].append(_autocorr(r, 1))
        facts["acf_abs_1"].append(_autocorr(np.abs(r), 1))
        facts["acf_abs_20"].append(_autocorr(np.abs(r), 20))
        future = pd.Series(np.abs(r)).rolling(5).mean().shift(-5).to_numpy()
        facts["leverage_5"].append(float(pd.Series(r).corr(pd.Series(future))))
    out = {k: float(np.nanmean(v)) for k, v in facts.items()}
    filled = np.nan_to_num(x)
    corr = np.corrcoef(filled.T)
    out["mean_corr"] = float(corr[np.triu_indices_from(corr, 1)].mean())
    eig = np.linalg.eigvalsh(np.cov(filled.T))
    out["first_pc"] = float(eig[-1] / eig.sum())
    market = np.nanmean(returns.to_numpy(dtype=float), axis=1)
    out["market_kurtosis"] = _kurtosis(market)
    out["market_acf_abs_1"] = _autocorr(np.abs(market), 1)
    out["market_vol"] = float(np.nanstd(market) * math.sqrt(252))
    return pd.Series(out)


def window_statistics(returns: pd.DataFrame, *, days: int = 21) -> pd.DataFrame:
    """Summary statistics of each ``days``-day window of the equal-weighted market and the cross-section."""
    x = returns.to_numpy(dtype=float)
    rows = []
    for start in range(0, len(x) - days + 1, days):
        w = x[start : start + days]
        m = np.nanmean(w, axis=1)
        rows.append(
            {
                "mean": float(m.mean()),
                "vol": float(m.std()),
                "abs_acf": _autocorr(np.abs(m), 1),
                "max_abs": float(np.abs(m).max()),
                "dispersion": float(np.nanmean(np.nanstd(w, axis=1))),
                "kurt": _kurtosis(m),
            }
        )
    return pd.DataFrame(rows).fillna(0.0)


def discriminator_score(real: pd.DataFrame, synthetic: list[pd.DataFrame], *, seed: int = SEED) -> float:
    """Cross-validated accuracy of a classifier telling real from synthetic months (0.5: indistinguishable)."""
    from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415
    from sklearn.model_selection import cross_val_score  # noqa: PLC0415

    r = window_statistics(real)
    s = pd.concat([window_statistics(p) for p in synthetic], ignore_index=True)
    s = s.sample(n=min(len(s), len(r)), random_state=seed)  # as many of each
    x = pd.concat([r, s], ignore_index=True)
    y = np.r_[np.ones(len(r)), np.zeros(len(s))]
    model = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=seed)
    return float(cross_val_score(model, x, y, cv=5, scoring="accuracy").mean())


# ---------------------------------------------------------------- 3. books on paths


def month_compound(daily: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Each listing's compounded return from one month end to the next (``date x listing``, at the earlier end)."""
    clean = daily.fillna(0.0).clip(-0.95, 2.0)
    cumulative = pd.DataFrame(np.log1p(clean.to_numpy()), index=clean.index, columns=clean.columns).cumsum()
    at = cumulative.reindex(month_ends, method="ffill").fillna(0.0)
    change = at.shift(-1) - at
    return pd.DataFrame(np.expm1(change.to_numpy()), index=at.index, columns=at.columns).iloc[:-1]


def month_sum(daily: pd.Series, month_ends: pd.DatetimeIndex) -> pd.Series:
    """A daily series (costs, say) summed over each month after a month end (at the earlier end)."""
    cumulative = daily.fillna(0.0).cumsum().reindex(month_ends, method="ffill").fillna(0.0)
    return (cumulative.shift(-1) - cumulative).iloc[:-1]


def book_returns(book: pd.DataFrame, daily: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.Series:
    """A book of month-end weights (``date x listing``) held for each following month on ``daily`` returns."""
    monthly = month_compound(daily, month_ends)
    weights = book.reindex(index=monthly.index, columns=monthly.columns).fillna(0.0)
    return (weights * monthly).sum(axis=1)


def outcomes(paths: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.Series:
    """Across paths (``month x path`` net monthly returns): growth, drawdown and, given a benchmark, trailing it.

    ``cagr_median`` / ``cagr_p05`` / ``cagr_p95``, ``max_dd_median`` / ``max_dd_p95`` (the drawdown of the worst
    twentieth), ``p_loss`` (the share of paths that lost money over the whole period) and, with ``benchmark`` (its
    returns on the same paths), ``active_median`` (yearly) and ``p_trail`` (the share of paths that trailed it).
    """
    years = len(paths) / 12
    wealth = (1 + paths).cumprod()
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    drawdown = (1 - wealth / wealth.cummax()).max()
    out = {
        "cagr_median": float(cagr.median()),
        "cagr_p05": float(cagr.quantile(0.05)),
        "cagr_p95": float(cagr.quantile(0.95)),
        "max_dd_median": float(drawdown.median()),
        "max_dd_p95": float(drawdown.quantile(0.95)),
        "p_loss": float((wealth.iloc[-1] < 1).mean()),
    }
    if benchmark is not None:
        bench = (1 + benchmark).cumprod().iloc[-1] ** (1 / years) - 1
        active = cagr - bench
        out["active_median"] = float(active.median())
        out["p_trail"] = float((active < 0).mean())
    return pd.Series(out)
