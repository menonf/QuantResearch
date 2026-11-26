"""Index weights plus active tilts: a book built on a factor risk model, for any asset class. Pure functions.

    weights = tilt(benchmark, score, loadings, factor_cov, specific, neutral=["Energy", ..., "Size"], risk=0.03)

A long-only book holds its benchmark and moves away from it only where the score says so, by as much as a budget of
active risk allows (Grinold and Kahn, *Active Portfolio Management*, ch. 14):

* **Forecast.** Each name's expected active return is its score's z-score times its specific volatility (Grinold's
  rule, the information coefficient being one number for all names), so the tilt it earns is proportional to the
  score and inversely proportional to its specific risk: ``a_i ~ (z_i - c_i'mu) / s_i``.
* **Constraints** ``C'a = 0``: the active weights sum to zero (fully invested; dollar-neutral for a long-short book),
  have no active beta (the book's predicted beta to the benchmark is the benchmark's, 1), and no exposure to each of
  the ``neutral`` factors - sectors and size, so the book is not a sector or a size bet in disguise.
* **Bounds** per name: no active weight beyond ``max_active`` either way, and a long-only book never short
  (``b_i + a_i >= 0``: a name can be underweighted by at most what the benchmark holds of it).
* **Budget.** The tilts are scaled until the predicted active risk - from the whole covariance, factors and specific
  - is ``risk`` a year (or as near as the bounds allow).

For a given scale the problem is a quadratic programme with a diagonal Hessian (the specific variances), linear
equalities and box bounds; it is solved through its dual by damped Newton steps (:func:`_solve`), and the scale by
bisection on the predicted risk. A long-short book is the same with no benchmark to hold: ``long_only=False`` returns
the active weights alone (dollar-, beta- and factor-neutral), with an optional cap on gross exposure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_core.data.calendars import TRADING_DAYS

#: Equality constraints are met to this (in weight).
TOLERANCE = 1e-9
#: Bisection steps on the scale (each halves a bracket four-fold wide: 30 fix it to a few parts in a billion).
SCALE_STEPS = 30


def _solve(
    kappa: float,
    alpha: np.ndarray,
    d: np.ndarray,
    c: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int = 200,
) -> np.ndarray:
    """``argmax alpha'a - a'Da / (2 kappa)`` subject to ``C'a = 0`` and ``lower <= a <= upper``.

    Through the dual: for multipliers ``mu`` the box-constrained optimum is ``a(mu) = clip(kappa (alpha - C mu) / d)``,
    and the dual function ``g(mu) = max_a (alpha - C mu)'a - a'Da / (2 kappa)`` is convex and smooth with gradient
    ``-C'a(mu)``. Newton steps (the free names' Hessian ``kappa C_f' D_f^-1 C_f``; a collinear ``C`` - industries
    summing to the column of ones - in the least-squares sense) with an Armijo line search on ``g`` converge to the
    ``mu`` whose ``a(mu)`` meets the equalities.
    """
    k = c.shape[1]
    mu = np.zeros(k)

    def dual(m: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        u = alpha - c @ m
        raw = kappa * u / d
        a = np.clip(raw, lower, upper)
        return float(u @ a - np.sum(a * a * d) / (2 * kappa)), a, (raw > lower) & (raw < upper)

    g, a, free = dual(mu)
    for _ in range(max_iter):
        grad = -(c.T @ a)
        if float(np.abs(grad).max()) < TOLERANCE:
            break
        hess = kappa * (c[free].T / d[free]) @ c[free] if free.any() else np.zeros((k, k))
        step = -np.linalg.lstsq(hess + np.eye(k) * 1e-12 * max(float(np.trace(hess)), 1.0), grad, rcond=None)[0]
        slope = float(grad @ step)
        if not slope < 0:  # not a descent direction (every name at a bound): steepest descent
            step = -grad / max(float(np.abs(grad).max()), 1e-300) * max(float(np.abs(mu).max()), 1e-6)
            slope = float(grad @ step)
        t = 1.0
        while t > 1e-14:
            g_new, a_new, free_new = dual(mu + t * step)
            if g_new <= g + 1e-4 * t * slope:
                break
            t /= 2
        else:
            break  # no step lowers the dual: as close as the bounds allow
        mu, g, a, free = mu + t * step, g_new, a_new, free_new
    return a


def predicted_risk(active: np.ndarray, loadings: np.ndarray, factor_cov: np.ndarray, specific: np.ndarray) -> float:
    """Per-period volatility of ``active`` weights: factor part ``(X'a)'F(X'a)`` plus specific ``sum (s_i a_i)^2``."""
    exposure = loadings.T @ active
    variance = float(exposure @ factor_cov @ exposure + np.sum((specific * active) ** 2))
    return float(np.sqrt(max(variance, 0.0)))


@dataclass(frozen=True)
class _Problem:
    """A day's programme over the covered names: exposures, covariance, specific vol, benchmark, alpha, constraints."""

    x: np.ndarray
    f: np.ndarray
    s: np.ndarray
    b: np.ndarray
    alpha: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def solved(self, kappa: float, target: float) -> tuple[np.ndarray, float, bool]:
        """The tilts at scale ``kappa``, their predicted risk, and whether they fit: constraints met, within budget."""
        a = _solve(kappa, self.alpha, self.s**2, self.c, lower=self.lower, upper=self.upper)
        r = predicted_risk(a, self.x, self.f, self.s)
        return a, r, bool(np.abs(self.c.T @ a).max() < 1e-7) and r <= target

    def scaled(self, target: float) -> np.ndarray:
        """The tilts at the largest scale that fits: widened while they fit and the risk still grows, then bisected."""
        a, lo, hi = np.zeros(len(self.b)), 0.0, 1e-6
        previous = 0.0
        for _ in range(80):
            trial, r, fits = self.solved(hi, target)
            if not fits:
                break
            a, lo = trial, hi
            if r <= previous * (1 + 1e-9):  # the bounds stop the risk growing: as much as can be had
                return a
            previous, hi = r, hi * 4
        for _ in range(SCALE_STEPS):
            mid = (lo + hi) / 2
            trial, _, fits = self.solved(mid, target)
            if fits:
                a, lo = trial, mid
            else:
                hi = mid
        return a


def _problem(
    covered: list[object],
    benchmark: pd.Series,
    score: pd.Series,
    *,
    loadings: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific: pd.Series,
    neutral: Sequence[str],
    max_active: float,
    long_only: bool,
) -> _Problem:
    x = np.nan_to_num(loadings.loc[covered].to_numpy(dtype=float))
    f = factor_cov.reindex(index=loadings.columns, columns=loadings.columns).fillna(0.0).to_numpy(dtype=float)
    s = specific.reindex(covered).to_numpy(dtype=float)
    b = benchmark.reindex(covered).to_numpy(dtype=float)
    z = score.reindex(covered).astype(float)
    spread = float(z.std())
    z = ((z - z.mean()) / spread).fillna(0.0) if spread > 0 else z * 0.0
    # beta of each name to the benchmark (its covered part), from the whole covariance: (Sigma b)_i / b'Sigma b
    reference = b / b.sum() if b.sum() > 0 else np.full(len(b), 1.0 / len(b))
    sigma_b = x @ (f @ (x.T @ reference)) + s**2 * reference
    beta = sigma_b / float(reference @ sigma_b)
    columns = [np.ones(len(covered)), beta]
    columns += [np.nan_to_num(loadings.loc[covered, name].to_numpy(dtype=float)) for name in neutral]
    return _Problem(
        x=x,
        f=f,
        s=s,
        b=b,
        alpha=z.to_numpy() * s,
        c=np.column_stack(columns),
        lower=np.maximum(-b, -max_active) if long_only else np.full(len(b), -max_active),
        upper=np.full(len(b), max_active),
    )


def tilt(
    benchmark: pd.Series,
    score: pd.Series,
    loadings: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific: pd.Series,
    *,
    neutral: Sequence[str] = (),
    risk: float = 0.03,
    max_active: float = 0.02,
    long_only: bool = True,
    gross: float | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """The book's weights: ``benchmark`` plus active tilts by ``score`` (long-only), or the tilts alone (long-short).

    Args:
        benchmark: name -> weight of the benchmark (the universe by market value); it sets the names, what a
            long-only book holds before tilting, and what beta is measured against.
        score: name -> score, higher better; standardised among the names the model covers, a missing one neutral.
        loadings: the risk model's ``name x factor`` exposures on the day. A name it does not cover (or without a
            specific volatility) is held at its benchmark weight (long-only) or not at all (long-short).
        factor_cov: the factor covariance, per period.
        specific: each name's specific volatility, per period.
        neutral: columns of ``loadings`` the active weights have no exposure to (sectors, size).
        risk: the predicted active risk a year to scale the tilts to (a long-short book's predicted volatility).
        max_active: the largest active weight in a name, either way.
        long_only: hold the benchmark plus the tilts (never short), or the tilts alone.
        gross: a long-short book's cap on the sum of its absolute weights.
        periods_per_year: periods of the risk model's units in a year (trading days for a daily model).
    """
    names = benchmark.index
    b_all = benchmark.astype(float).fillna(0.0)
    covered = [
        n for n in names if n in loadings.index and np.isfinite(specific.get(n, np.nan)) and specific.get(n, 0) > 0
    ]
    out = b_all.copy() if long_only else pd.Series(0.0, index=names)
    if len(covered) < 2:
        return out
    problem = _problem(
        covered,
        b_all,
        score,
        loadings=loadings,
        factor_cov=factor_cov,
        specific=specific,
        neutral=neutral,
        max_active=max_active,
        long_only=long_only,
    )
    a = problem.scaled(risk / np.sqrt(periods_per_year))
    if not long_only and gross is not None and np.abs(a).sum() > gross:
        a = a * gross / np.abs(a).sum()
    out.loc[covered] = np.maximum(problem.b + a, 0.0) if long_only else a
    return out
