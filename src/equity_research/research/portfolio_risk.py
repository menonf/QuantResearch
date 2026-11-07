"""A portfolio's risk from the factor model: its volatility, where it comes from, its value at risk, and tests.

    risk = portfolio_risk(model, weights, day, benchmark=index_weights)    # active risk when a benchmark is given
    risk.total, risk.contributions                                         # volatility over the horizon, by factor
    value_at_risk(risk, 0.99)                                              # analytical VaR, by factor and specific

For holdings ``h`` (or active holdings, ``h`` less the benchmark's) with factor exposures ``x = X'h``, the variance
over ``H`` days is ``H (x'Fx + sum_i h_i^2 s_i^2)`` - the factors' part and the specific part, which share no
covariance. Each factor's contribution to volatility is ``H x_k (Fx)_k / sigma`` and the specific part's
``H sum h_i^2 s_i^2 / sigma``; together they add up to ``sigma`` (Euler), so a contribution is the factor's share of
the risk, not just its size.

**Analytical value at risk** treats the horizon's return as normal with mean zero: the loss exceeded with
probability ``1 - c`` is ``z_c sigma``, and the same Euler split gives each factor's and the specific part's share of
it. Daily equity returns have fatter tails than a normal, so a 99% VaR is exceeded more often than 1% of days;
:func:`kupiec` tests the count of exceptions, as BFRE's surveillance does (99% one-day VaR over 252 days).

**Stress tests** shock one or more factors by a return and read the portfolio's profit or loss, ``x's``, in two
flavours. *Isolated*: only the shocked factors move - what the bet on that factor alone would lose. *Correlated*:
the other factors move too, by what the covariance expects given the shock (``F_us F_ss^-1 s``, the conditional mean
of jointly normal returns) - a market fall also drags the factors that usually fall with it. Specific returns are
expected to be zero either way.

**Forecast tests** (BFRE 5): a return divided by its forecast volatility should have a standard deviation of one -
the bias statistic, inside ``1 +- sqrt(2 / T)`` for ``T`` observations - and Patton's Q-statistic,
``mean(z^2 - ln z^2)``, is lowest when the forecasts match the true volatility and punishes under-forecasts more.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from equity_research.research.risk_model import MARKET, RiskModel

SPECIFIC = "Specific"


@dataclass(frozen=True)
class PortfolioRisk:
    """One portfolio's risk on one day over ``horizon`` days; volatilities as fractions of the portfolio's value."""

    day: pd.Timestamp
    horizon: int
    total: float
    factor: float
    specific: float
    #: factor -> the portfolio's (active) exposure
    exposures: pd.Series
    #: factor (and "Specific") -> contribution to ``total``; they add up to it
    contributions: pd.Series
    #: the share of the portfolio's gross weight the model covers (the rest is left out of every figure)
    coverage: float


def portfolio_risk(
    model: RiskModel,
    weights: pd.Series,
    day: pd.Timestamp,
    *,
    benchmark: pd.Series | None = None,
    horizon: int = 21,
    version: str = "2.0",
    listing_issuer: pd.Series | None = None,
) -> PortfolioRisk:
    """The risk of ``weights`` (``listing_id -> weight``) on ``day``; against ``benchmark``, the active risk.

    With ``listing_issuer``, two listings of one company (GOOG and GOOGL) have perfectly correlated specific returns,
    as in BFRE's structural treatment; without it, every listing's specific return is its own.
    """
    h = weights.astype(float)
    if benchmark is not None:
        h = h.sub(benchmark.astype(float), fill_value=0.0)
    h = h[h != 0]
    loadings, cov, spec = model.on(day, version)
    covered = h.index[h.index.isin(loadings.index[spec.notna().to_numpy()])]
    gross = float(h.abs().sum())
    hc = h.loc[covered]
    x = loadings.loc[covered].T @ hc
    fx = cov.to_numpy() @ x.to_numpy()
    by_factor = pd.Series(x.to_numpy() * fx, index=cov.index)
    hs = hc * spec.loc[covered]
    if listing_issuer is not None:
        groups = listing_issuer.reindex(covered).to_numpy(dtype=float)
        groups = np.where(np.isnan(groups), -covered.to_numpy(dtype=float), groups)  # no company: a group of its own
        specific_var = float((hs.groupby(groups).sum() ** 2).sum())
    else:
        specific_var = float((hs**2).sum())
    factor_var = float(by_factor.sum())
    total = math.sqrt(horizon * max(factor_var + specific_var, 0.0))
    parts = pd.concat([by_factor, pd.Series({SPECIFIC: specific_var})])
    return PortfolioRisk(
        day=pd.Timestamp(day),
        horizon=horizon,
        total=total,
        factor=math.sqrt(horizon * max(factor_var, 0.0)),
        specific=math.sqrt(horizon * specific_var),
        exposures=x,
        contributions=parts * horizon / total if total > 0 else parts * 0.0,
        coverage=float(hc.abs().sum()) / gross if gross else float("nan"),
    )


def predicted_beta(
    model: RiskModel, weights: pd.Series, benchmark: pd.Series, day: pd.Timestamp, *, version: str = "2.0"
) -> float:
    """The portfolio's beta to ``benchmark`` the model predicts: their covariance over the benchmark's variance.

    BFRE has no beta factor; beta follows from the two portfolios' exposures, the factor covariance and the specific
    risk of the shares they share.
    """
    loadings, cov, spec = model.on(day, version)
    ids = loadings.index[spec.notna().to_numpy()]
    h = weights.reindex(ids).fillna(0.0)
    b = benchmark.reindex(ids).fillna(0.0)
    xh, xb = loadings.loc[ids].T @ h, loadings.loc[ids].T @ b
    f = cov.to_numpy()
    s2 = spec.loc[ids].to_numpy() ** 2
    together = float(xh.to_numpy() @ f @ xb.to_numpy() + (h.to_numpy() * b.to_numpy() * s2).sum())
    alone = float(xb.to_numpy() @ f @ xb.to_numpy() + (b.to_numpy() ** 2 * s2).sum())
    return together / alone


def risk_table(risk: PortfolioRisk) -> pd.DataFrame:
    """Per factor (and the specific part): the exposure, the contribution to volatility and its share of the total."""
    out = pd.DataFrame({"exposure": risk.exposures, "contribution": risk.contributions})
    out["share of risk"] = out["contribution"] / risk.total if risk.total else np.nan
    return out


def cap_weights(market_cap: pd.DataFrame, members: pd.DataFrame, within: pd.DataFrame | None = None) -> pd.DataFrame:
    """``date x listing_id`` weights by market value on ``market_cap``'s dates, among the members (and ``within``)."""
    keep = members.reindex(index=market_cap.index, columns=market_cap.columns).fillna(False).astype(bool)
    if within is not None:
        keep &= within.reindex(index=market_cap.index, columns=market_cap.columns).fillna(False).astype(bool)
    value = market_cap.where(keep & (market_cap > 0))
    return value.div(value.sum(axis=1), axis=0)


def by_block(risk: PortfolioRisk, model: RiskModel) -> pd.Series:
    """The contributions summed by block: Market, Industry, Style and Specific."""
    block = {MARKET: MARKET, SPECIFIC: SPECIFIC} | dict.fromkeys(model.exposures.industries, "Industry")
    block |= dict.fromkeys(model.exposures.styles, "Style")
    return (
        risk.contributions.groupby(risk.contributions.index.map(block))
        .sum()
        .reindex([MARKET, "Industry", "Style", SPECIFIC])
    )


def value_at_risk(risk: PortfolioRisk, confidence: float = 0.99) -> pd.Series:
    """Analytical VaR over the risk's horizon, as a fraction of value: ``Total`` and each contribution to it."""
    z = NormalDist().inv_cdf(confidence)
    out: pd.Series = pd.concat([pd.Series({"Total": z * risk.total}), z * risk.contributions])
    return out


# ---------------------------------------------------------------- stress tests


@dataclass(frozen=True)
class StressResult:
    """A scenario's profit or loss as a fraction of value, isolated and with the factors' correlations."""

    #: factor -> ``[exposure, isolated move, isolated P&L, correlated move, correlated P&L]``
    table: pd.DataFrame
    isolated: float
    correlated: float


def shock_in_sigmas(
    model: RiskModel, day: pd.Timestamp, sigmas: Mapping[str, float], *, horizon: int = 21, version: str = "2.0"
) -> dict[str, float]:
    """Each factor's shock as a return: ``sigmas[k]`` standard deviations of factor ``k`` over ``horizon`` days."""
    _, cov, _ = model.on(day, version)
    variance = pd.Series(np.diag(cov.to_numpy()), index=cov.index)
    return {k: v * math.sqrt(horizon * float(variance[k])) for k, v in sigmas.items()}


def implied_moves(cov: pd.DataFrame, shocks: Mapping[str, float]) -> pd.Series:
    """Every factor's expected return given ``shocks`` to some (equal to the shocks on those).

    ``F_us F_ss^-1 s``: the regression of the other factors on the shocked ones. The covariance's scale cancels, so
    the moves depend only on the correlations and relative volatilities - not on the horizon or BFRE 2.0's gamma.
    """
    unknown = set(shocks) - set(cov.index)
    if unknown:
        raise KeyError(f"no such factors: {sorted(unknown)}")
    shocked = list(shocks)
    s = np.array([shocks[k] for k in shocked], dtype=float)
    f = cov.to_numpy()
    idx = cov.index.get_indexer(pd.Index(shocked))
    moves = f[:, idx] @ (np.linalg.pinv(f[np.ix_(idx, idx)]) @ s)
    exact = dict(zip(shocked, s.tolist(), strict=True))  # exactly the shock, whatever rounding the inverse left
    return pd.Series([exact.get(k, m) for k, m in zip(cov.index, moves.tolist(), strict=True)], index=cov.index)


def stress_test(
    model: RiskModel,
    weights: pd.Series,
    day: pd.Timestamp,
    shocks: Mapping[str, float],
    *,
    benchmark: pd.Series | None = None,
    version: str = "2.0",
) -> StressResult:
    """The P&L of ``weights`` (against ``benchmark``: the active P&L) when ``shocks`` hit the factors on ``day``.

    ``shocks`` maps factors to returns (-0.10: the factor loses 10%); :func:`shock_in_sigmas` turns standard
    deviations into returns. Only the shares the model covers count, as in :func:`portfolio_risk`.
    """
    h = weights.astype(float)
    if benchmark is not None:
        h = h.sub(benchmark.astype(float), fill_value=0.0)
    loadings, cov, _ = model.on(day, version)
    covered = h.index[h.index.isin(loadings.index)]
    x = loadings.loc[covered, list(cov.index)].T @ h.loc[covered]
    alone = pd.Series([float(shocks.get(k, 0.0)) for k in cov.index], index=cov.index)
    together = implied_moves(cov, shocks)
    table = pd.DataFrame(
        {
            "exposure": x,
            "isolated move": alone,
            "isolated P&L": x * alone,
            "correlated move": together,
            "correlated P&L": x * together,
        }
    )
    return StressResult(table, float(table["isolated P&L"].sum()), float(table["correlated P&L"].sum()))


# ---------------------------------------------------------------- forecast tests


def forward_returns(returns: pd.DataFrame, weights: pd.DataFrame, horizon: int) -> pd.Series:
    """For each row date of ``weights``: the return of buying those weights at its close and holding ``horizon`` days.

    A share with no return on a day (delisted, suspended) counts as flat. Missing where fewer than ``horizon`` days
    follow.
    """
    index = pd.DatetimeIndex(returns.index)
    filled = returns.fillna(0.0)
    grown = pd.DataFrame(np.log1p(filled.to_numpy()), index=filled.index, columns=filled.columns).cumsum()
    out = {}
    for d, row in weights.iterrows():
        p = int(index.searchsorted(pd.Timestamp(str(d)), side="right")) - 1
        if p < 0 or p + horizon >= len(index):
            continue
        held = row[row != 0].dropna()
        held = held[held.index.isin(returns.columns)]
        change = (grown.iloc[p + horizon][held.index] - grown.iloc[p][held.index]).apply(math.expm1)
        out[pd.Timestamp(str(d))] = float((held * change).sum())
    return pd.Series(out, dtype=float)


def forecast_volatility(
    model: RiskModel,
    weights: pd.DataFrame,
    *,
    horizon: int = 21,
    version: str = "2.0",
    benchmark: pd.DataFrame | None = None,
) -> pd.Series:
    """The model's volatility forecast for each row of ``weights`` on its date; days without a forecast are left out."""
    out = {}
    have = set(model.covariance)
    for d, row in weights.iterrows():
        day = pd.Timestamp(str(d))
        if day not in have:
            continue
        bench = pd.Series(benchmark.loc[day]).dropna() if benchmark is not None else None
        out[day] = portfolio_risk(model, row.dropna(), day, benchmark=bench, horizon=horizon, version=version).total
    return pd.Series(out, dtype=float)


def bias_statistic(z: pd.Series) -> tuple[float, float, float]:
    """The standard deviation of standardised returns and the band ``1 +- sqrt(2 / T)`` it should fall in."""
    z = z.dropna()
    band = math.sqrt(2.0 / len(z)) if len(z) else float("nan")
    return float(z.std(ddof=1)), 1.0 - band, 1.0 + band


def q_statistic(z: pd.Series) -> float:
    """Patton's Q: the mean of ``z^2 - ln z^2`` (lower is better; 1 for perfect forecasts)."""
    z2 = z.dropna() ** 2
    z2 = z2[z2 > 0]
    return float((z2 - np.log(z2)).mean())


def kupiec(exceptions: int, observations: int, probability: float) -> tuple[float, float]:
    """Kupiec's proportion-of-failures test: the likelihood ratio and its p-value (chi-square, one degree).

    ``probability`` is the share of exceptions the VaR allows (0.01 for 99%); a small p-value says the count is not
    what the VaR promised.
    """
    n, x, p = observations, exceptions, probability
    if n == 0:
        return float("nan"), float("nan")
    rate = x / n

    def loglik(q: float) -> float:
        return (n - x) * math.log(1 - q) + (x * math.log(q) if x else 0.0) if 0 < q < 1 else 0.0

    lr = -2.0 * (loglik(p) - (loglik(rate) if 0 < rate < 1 else 0.0))
    return lr, math.erfc(math.sqrt(max(lr, 0.0) / 2.0))


def var_backtest(
    model: RiskModel, weights: pd.DataFrame, returns: pd.DataFrame, *, confidence: float = 0.99, version: str = "2.0"
) -> pd.DataFrame:
    """One-day VaR on every forecast day against the next day's return.

    ``weights`` is ``rebalance date x listing_id``: each day holds the latest row on or before it (drift inside the
    month is ignored). ``[var, realised, exception]`` by forecast day; an exception is a loss beyond the VaR.
    """
    index = pd.DatetimeIndex(returns.index)
    rebalances = pd.DatetimeIndex(weights.index)
    rows = {}
    for day in model.dates:
        k, p = int(rebalances.searchsorted(day, side="right")), int(index.searchsorted(day, side="right"))
        if k == 0 or p >= len(index):
            continue
        held = weights.iloc[k - 1].dropna()
        held = held[held != 0]
        risk = portfolio_risk(model, held, day, horizon=1, version=version)
        nxt = returns.iloc[p].reindex(held.index).fillna(0.0)
        var = float(value_at_risk(risk, confidence)["Total"])
        realised = float((held * nxt).sum())
        rows[day] = {"var": var, "realised": realised, "exception": realised < -var}
    return pd.DataFrame.from_dict(rows, orient="index")


def stock_z_scores(
    model: RiskModel, returns: pd.DataFrame, dates: pd.DatetimeIndex, *, horizon: int = 21, version: str = "2.0"
) -> pd.Series:
    """Every covered share's next ``horizon``-day return over its total-risk forecast, pooled over ``dates``.

    ``(date, listing_id)``-indexed; BFRE's asset-level test. A share leaving in the window counts as flat after.
    """
    index = pd.DatetimeIndex(returns.index)
    filled = returns.fillna(0.0)
    grown = pd.DataFrame(np.log1p(filled.to_numpy()), index=filled.index, columns=filled.columns).cumsum()
    parts = {}
    for d in dates:
        p = int(index.searchsorted(d, side="right")) - 1
        if d not in model.covariance or p + horizon >= len(index):
            continue
        loadings, cov, spec = model.on(d, version)
        x = loadings[cov.index].to_numpy()
        var = horizon * (np.einsum("ij,jk,ik->i", x, cov.to_numpy(), x) + spec.to_numpy() ** 2)
        ids = loadings.index[loadings.index.isin(returns.columns)]
        keep = loadings.index.isin(ids)
        change = (grown.iloc[p + horizon][ids] - grown.iloc[p][ids]).apply(math.expm1)
        parts[pd.Timestamp(d)] = pd.Series(change.to_numpy() / np.sqrt(var[keep]), index=ids).dropna()
    if not parts:
        return pd.Series(dtype=float)
    out: pd.Series = pd.concat(parts, names=["date", "listing_id"])
    return out


def forecast_tests(
    model: RiskModel, returns: pd.DataFrame, portfolios: Mapping[str, pd.DataFrame], *, horizon: int = 21
) -> pd.DataFrame:
    """Per portfolio: realised and forecast yearly volatility, bias statistic and Q, for BFRE 1.0 and 2.0.

    Each portfolio is ``date x listing_id`` weights on the test dates (month ends); its return over the next
    ``horizon`` days is divided by each version's forecast for it. A bias inside ``band`` says the forecasts were
    right on average; the lower Q says which version tracked the changes in risk better.
    """
    rows = {}
    per_year = math.sqrt(252 / horizon)
    for name, weights in portfolios.items():
        realised = forward_returns(returns, weights, horizon)
        row: dict[str, object] = {"realised volatility": realised.std() * per_year}
        for version in ("1.0", "2.0"):
            forecast = forecast_volatility(model, weights, horizon=horizon, version=version)
            z = (realised / forecast).dropna()
            bias, lo, hi = bias_statistic(z)
            row[f"forecast {version}"] = forecast.mean() * per_year
            row[f"bias {version}"] = bias
            row[f"Q {version}"] = q_statistic(z)
        row["band"] = f"{lo:.2f}-{hi:.2f}"
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def kupiec_summary(backtests: Mapping[str, pd.DataFrame], confidence: float = 0.99) -> pd.DataFrame:
    """Per backtest: days, exceptions, the share expected and found, and Kupiec's p-value."""
    out = {}
    for name, bt in backtests.items():
        n, x = len(bt), int(bt["exception"].sum())
        lr, pvalue = kupiec(x, n, 1 - confidence)
        out[name] = {
            "days": n,
            "exceptions": x,
            "expected": n * (1 - confidence),
            "rate": x / n if n else np.nan,
            "kupiec_lr": lr,
            "p_value": pvalue,
        }
    return pd.DataFrame.from_dict(out, orient="index")
