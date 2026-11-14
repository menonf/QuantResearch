"""From predictions to portfolios (docs/ML4T_PLAN.md, P4): entry, weighting within each side, exposure overlays.

    book = decile_book(score, members, dates, returns, method="hrp", long_only=False)   # monthly deciles, HRP weights
    book = decile_book(score, members, dates, returns, method="mean_variance", risk=model.on)
    exposure = vol_target(net, target=0.15)                       # scale a strategy to a volatility, from its past
    scaled = apply_exposure(net, exposure, cost_bps=10)           # the strategy at that exposure, its trading charged

A score ranks companies; a portfolio needs weights. *Machine Learning for Trading* (chapter 17) separates the two:

1. **Entry** - which names are held: each decision date the top fraction of members by score (long), and in a
   long-short book the bottom fraction too (:func:`select`); or the top and bottom ``k`` (:func:`top_k_book`).
2. **Weighting** (:func:`leg_weights`) - how much of each side's capital each name gets, each side summing to 1:

   * **equal** - one share each;
   * **inverse volatility** - in proportion to one over each name's volatility over the past 63 days, so each adds
     about the same risk;
   * **hierarchical risk parity** (Lopez de Prado, 2016) - names clustered by the correlation of their past year's
     returns, then capital split down the cluster tree in inverse proportion to each branch's variance: diversified
     without inverting a covariance matrix;
   * **mean-variance** - weights proportional to ``inverse(Sigma) alpha``, with ``Sigma`` the risk model's covariance
     of the side's names (factor part plus specific variance) and ``alpha`` the names' score z-scores times their
     volatility (Grinold's forecast rule: alpha = IC x volatility x score); negative weights clipped to 0 (a side holds
     only its own direction) and the rest scaled to 1.
3. **Exposure overlays** - scaling the whole strategy by a number known the day before:

   * **volatility target** - the target over the strategy's volatility over the past 63 days;
   * **drawdown control** - full exposure until the strategy is ``threshold`` below its peak, then ``cut``;
   * **fractional Kelly** - a fraction of the Kelly exposure ``mean / variance`` of the past year's returns.

   Each is capped, and :func:`apply_exposure` charges the cost of changing the exposure.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

#: The weighting methods :func:`leg_weights` knows.
METHODS = ("equal", "inverse_vol", "hrp", "mean_variance")
#: Days of returns for inverse-volatility weights and for HRP's correlations.
VOL_DAYS, CORR_DAYS = 63, 252

RiskOn = Callable[[pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame, pd.Series]]


def select(score: pd.Series, *, fraction: float, top: bool = True) -> list[int]:
    """The members (``score`` non-missing) in the top (or bottom) ``fraction`` by score."""
    s = score.dropna()
    if s.empty:
        return []
    pct = s.rank(pct=True, ascending=top)
    return [int(n) for n in pct[pct > 1 - fraction].index]


def _hrp(cov: np.ndarray) -> np.ndarray:
    """Hierarchical risk parity weights of a covariance matrix (Lopez de Prado, 2016)."""
    from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: PLC0415
    from scipy.spatial.distance import squareform  # noqa: PLC0415

    n = len(cov)
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    distance = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    weights = np.ones(n)
    clusters = [order]
    while clusters:
        split = []
        for c in clusters:
            if len(c) < 2:
                continue
            left, right = c[: len(c) // 2], c[len(c) // 2 :]

            def variance(members: np.ndarray) -> float:
                sub = cov[np.ix_(members, members)]
                w = 1.0 / np.diag(sub)
                w /= w.sum()
                return float(w @ sub @ w)

            vl, vr = variance(left), variance(right)
            share = 1.0 - vl / (vl + vr)
            weights[left] *= share
            weights[right] *= 1.0 - share
            split += [left, right]
        clusters = split
    return weights / weights.sum()


def leg_weights(
    names: list[int],
    method: str,
    *,
    day: pd.Timestamp,
    returns: pd.DataFrame,
    score: pd.Series | None = None,
    risk: RiskOn | None = None,
) -> pd.Series:
    """One side's weights (summing to 1) for ``names`` on ``day``, by ``method`` (one of :data:`METHODS`).

    Volatilities and correlations use the returns up to and including ``day``; a name without enough history gets the
    side's median volatility. ``mean_variance`` needs ``score`` (higher is better on this side) and ``risk`` (the risk
    model's ``on``: exposures, factor covariance, specific volatility on ``day``).
    """
    if not names:
        return pd.Series(dtype=float)
    if method == "equal":
        return pd.Series(1.0 / len(names), index=names)
    past = returns.loc[:day, names]
    if method == "inverse_vol":
        vol = past.iloc[-VOL_DAYS:].std()
        vol = vol.fillna(vol.median()).where(lambda v: v > 0, vol.median())
        w = 1.0 / vol
        return w / w.sum()
    if method == "hrp":
        window = past.iloc[-CORR_DAYS:]
        cov = window.cov(min_periods=CORR_DAYS // 2)
        keep = cov.index[np.isfinite(np.diag(cov.to_numpy()))]
        cov = cov.loc[keep, keep]
        cov = cov.fillna(0.0) + np.diag(np.full(len(keep), 1e-8))
        out = pd.Series(0.0, index=names)
        if len(keep):
            out.loc[keep] = _hrp(cov.to_numpy())
        else:
            out[:] = 1.0 / len(names)
        return out
    if method == "mean_variance":
        if score is None or risk is None:
            raise ValueError("mean_variance needs a score and the risk model")
        loadings, factor_cov, specific = risk(day)
        covered = [n for n in names if n in loadings.index and np.isfinite(specific.get(n, np.nan))]
        if not covered:
            return pd.Series(1.0 / len(names), index=names)
        x = loadings.loc[covered].to_numpy(dtype=float)
        sigma = x @ factor_cov.to_numpy() @ x.T + np.diag(specific.loc[covered].to_numpy() ** 2)
        z = score.reindex(covered)
        z = ((z - z.mean()) / z.std()).fillna(0.0) if z.std() > 0 else z * 0
        alpha = z.to_numpy() * np.sqrt(np.diag(sigma))
        raw = np.linalg.solve(sigma + np.eye(len(covered)) * 1e-10, alpha + 1e-12)
        raw = np.clip(raw, 0.0, None)
        out = pd.Series(0.0, index=names)
        out.loc[covered] = raw / raw.sum() if raw.sum() > 0 else 1.0 / len(covered)
        return out
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def decile_book(
    score: pd.DataFrame,
    members: pd.DataFrame,
    dates: pd.DatetimeIndex,
    returns: pd.DataFrame,
    *,
    method: str = "equal",
    fraction: float = 0.1,
    long_only: bool = True,
    risk: RiskOn | None = None,
) -> pd.DataFrame:
    """``date x listing_id`` target weights on ``dates``: the top and bottom deciles (by default), weighted.

    The top ``fraction`` of members is long (summing to 1) and, unless ``long_only``, the bottom ``fraction`` short
    (summing to -1); each side is weighted by ``method``.
    """
    on = members.reindex(index=dates, columns=score.columns, fill_value=False).astype(bool)
    rows = {}
    for d in dates:
        if d not in score.index:
            continue
        s = pd.Series(score.loc[d], index=score.columns).where(on.loc[d].to_numpy())
        w = pd.Series(0.0, index=score.columns)
        longs = select(s, fraction=fraction, top=True)
        w = w.add(leg_weights(longs, method, day=d, returns=returns, score=s, risk=risk), fill_value=0.0)
        if not long_only:
            shorts = [n for n in select(s, fraction=fraction, top=False) if n not in set(longs)]
            w = w.sub(leg_weights(shorts, method, day=d, returns=returns, score=-s, risk=risk), fill_value=0.0)
        rows[d] = w
    return pd.DataFrame(rows).T.reindex(columns=score.columns).fillna(0.0)


def top_k_book(score: pd.DataFrame, members: pd.DataFrame, k: int, *, long_only: bool = False) -> pd.DataFrame:
    """The book's daily scheme: equal weights in the top ``k`` members on every date of ``score``.

    Unless ``long_only``, the bottom ``k`` are shorted the same way.
    """
    on = members.reindex(index=score.index, columns=score.columns, fill_value=False).astype(bool)
    s = score.where(on)
    top = s.rank(axis=1, ascending=False) <= k
    out = top.astype(float).div(top.sum(axis=1).replace(0, np.nan), axis=0)
    if not long_only:
        bottom = s.rank(axis=1, ascending=True) <= k
        out = out - bottom.astype(float).div(bottom.sum(axis=1).replace(0, np.nan), axis=0)
    return out.fillna(0.0)


def vol_target(net: pd.Series, *, target: float = 0.15, window: int = 63, cap: float = 1.5) -> pd.Series:
    """Each day's exposure: ``target`` over the annualised volatility of the past ``window`` days (to yesterday)."""
    vol = net.rolling(window, min_periods=window // 2).std().shift(1) * np.sqrt(252)
    exposure: pd.Series = (target / vol).clip(upper=cap).fillna(1.0)
    return exposure.rename("exposure")


def drawdown_control(net: pd.Series, *, threshold: float = 0.15, cut: float = 0.5) -> pd.Series:
    """Each day's exposure: 1, or ``cut`` while the strategy is more than ``threshold`` below its peak.

    The drawdown is the strategy's at full exposure, as of yesterday's close.
    """
    wealth = (1 + net.fillna(0.0)).cumprod()
    drawdown = (wealth / wealth.cummax() - 1).shift(1).fillna(0.0)
    return pd.Series(np.where(drawdown < -threshold, cut, 1.0), index=net.index, name="exposure")


def kelly(net: pd.Series, *, fraction: float = 0.5, window: int = 252, cap: float = 1.5) -> pd.Series:
    """Each day's exposure: ``fraction`` of the Kelly exposure, between 0 and ``cap``.

    The Kelly exposure is the mean over the variance of the past ``window`` days' returns; 1 until they exist.
    """
    mean = net.rolling(window, min_periods=window).mean().shift(1)
    var = net.rolling(window, min_periods=window).var().shift(1)
    exposure: pd.Series = (fraction * mean / var).clip(lower=0.0, upper=cap).fillna(1.0)
    return exposure.rename("exposure")


def apply_exposure(net: pd.Series, exposure: pd.Series, *, cost_bps: float = 10.0) -> pd.Series:
    """The strategy's daily return at ``exposure``, less the cost of each change of exposure (one-way ``cost_bps``)."""
    e = exposure.reindex(net.index).ffill().fillna(1.0)
    change = e.diff().abs().fillna(0.0)
    return (net * e - change * cost_bps / 1e4).rename("net")
