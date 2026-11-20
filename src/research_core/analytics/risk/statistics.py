"""Risk statistics that need no risk model: tail risk, tests of risk forecasts, conditional stress moves.

    tail_risk(returns)                        # historical one-day value at risk and expected shortfall
    bias_statistic(z), q_statistic(z)         # were standardised returns as wide as the forecasts said?
    kupiec(exceptions, days, 0.01)            # was a value at risk exceeded as often as it should be?
    implied_moves(cov, {"Market": -0.10})     # every factor's expected move given shocks to some

Pure functions. A model that forecasts risk - an equity factor model, a curve and spread model - supplies the
forecasts and covariances these judge.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


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


def tail_risk(returns: pd.Series, confidence: float = 0.99) -> dict[str, float]:
    """Historical one-day value at risk and conditional value at risk (expected shortfall), as positive losses."""
    r = returns.dropna().to_numpy(dtype=float)
    cut = np.quantile(r, 1 - confidence)
    return {"VaR": float(-cut), "CVaR": float(-r[r <= cut].mean())}
