"""Latent factors (docs/ML4T_PLAN.md, P5): PCA, RP-PCA and IPCA, out of sample, set against the BFRE risk model.

    months = monthly_panel(x, returns_next)                    # per month end: characteristics Z_t, next returns r_t+1
    X = managed_portfolios(months)                              # month x characteristic: Z_t' r_t+1 / N_t
    v = pca_loadings(X_train, k=3, gamma=10)                    # RP-PCA (gamma = 0: PCA)
    gamma = ipca_fit(train_months, k=3)                         # IPCA: loadings = characteristics x Gamma
    out = evaluate(fitted_X, X_test, factors_test, factors_train)   # total R2, pricing errors, the SDF's Sharpe

How many hidden factors drive the cross-section of returns, and do they price it better than the risk model's
named ones? *Machine Learning for Trading* (chapter 14) estimates them three ways; every way is fitted on years
already over and judged on the next:

1. **Managed portfolios** - each month, one portfolio per characteristic: every member weighted by its centred
   characteristic (and one equal-weighted portfolio for the constant): ``X_t+1 = Z_t' r_t+1 / N_t``. Forty-six
   portfolios that span what the characteristics can earn, the common ground every method is judged on.
2. **PCA** - the eigenvectors of the managed portfolios' second-moment matrix ``X'X / T``: the directions of greatest
   variation. **RP-PCA** (Lettau and Pelger, 2020) adds ``gamma`` times the outer product of the mean, so a direction
   that *earns* counts as well as one that varies: ``X'X / T + gamma xbar xbar'``.
3. **IPCA** (Kelly, Pruitt and Su, 2019) - each stock's loadings on ``K`` factors are a linear function of its
   characteristics, ``beta_i,t = z_i,t' Gamma``. ``Gamma`` (``L x K``) and the factors are found by alternating least
   squares: given ``Gamma``, each month's factors are a cross-sectional regression of returns on ``Z Gamma``; given
   the factors, ``Gamma`` solves the pooled regression of returns on ``z (x) f``.
4. **Out-of-sample measures** (Kelly, Pruitt and Su's and Lettau and Pelger's): the **total R-squared** - the share
   of the test months' managed-portfolio returns the model's fitted values explain; the **pricing error** - the root
   mean square, across portfolios, of the test months' mean residual; and the **SDF Sharpe ratio** - the annualised
   Sharpe ratio, in the test months, of the factors' tangency portfolio, its weights ``Sigma_f^-1 mu_f`` estimated on
   the training months only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Month:
    """One month end: its members' characteristics (with a constant) and their returns over the next month."""

    date: pd.Timestamp
    z: np.ndarray  # N x L
    r: np.ndarray  # N


def clipped_returns(returns: pd.DataFrame, members: pd.DataFrame, trim: float = 0.01) -> pd.DataFrame:
    """``date x listing_id``: members' returns clipped to the date's ``trim`` quantiles (not demeaned)."""
    on = members.reindex(index=returns.index, columns=returns.columns, fill_value=False).astype(bool)
    known = returns.where(on)
    return known.clip(lower=known.quantile(trim, axis=1), upper=known.quantile(1 - trim, axis=1), axis=0)


def monthly_panel(x: pd.DataFrame, returns: pd.DataFrame, *, min_names: int = 100) -> list[Month]:
    """Per month end of ``x`` (``(date, listing_id) x characteristic``, centred): the members with a next return.

    A constant column is appended, so the first managed portfolio is the equal-weighted market.
    """
    out = []
    for key, part in x.groupby(level="date"):
        date = pd.Timestamp(str(key))
        if date not in returns.index:
            continue
        r = returns.loc[date].reindex(part.index.get_level_values("listing_id"))
        keep = r.notna().to_numpy()
        if keep.sum() < min_names:
            continue
        z = np.column_stack([np.ones(int(keep.sum())), part.to_numpy(dtype=float)[keep]])
        out.append(Month(date, z, r.to_numpy(dtype=float)[keep]))
    return out


def managed_portfolios(months: Sequence[Month], names: Sequence[str]) -> pd.DataFrame:
    """``month x portfolio``: ``Z_t' r_t+1 / N_t`` for each month (the constant first, named ``market``)."""
    rows = {m.date: m.z.T @ m.r / len(m.r) for m in months}
    return pd.DataFrame(rows, index=["market", *names]).T


def pca_loadings(x: np.ndarray, k: int, *, gamma: float = 0.0) -> np.ndarray:
    """``L x k``: the top ``k`` eigenvectors of ``X'X / T + gamma xbar xbar'`` (PCA at ``gamma = 0``; RP-PCA above)."""
    t = len(x)
    mean = x.mean(axis=0)
    matrix = x.T @ x / t + gamma * np.outer(mean, mean)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1][:k]
    v = vectors[:, order]
    signs = np.sign(v.T @ mean)  # orient each factor to earn a positive mean in training
    oriented: np.ndarray = v * np.where(signs == 0, 1.0, signs)
    return oriented


def ipca_factors(month: Month, gamma: np.ndarray) -> np.ndarray:
    """``K``: the month's factor returns given ``Gamma`` - the regression of returns on the loadings ``Z Gamma``."""
    beta = month.z @ gamma
    f: np.ndarray = np.linalg.lstsq(beta, month.r, rcond=None)[0]
    return f


def ipca_fit(months: Sequence[Month], k: int, *, iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """``L x K`` ``Gamma`` by alternating least squares (Kelly, Pruitt and Su, 2019), started from PCA.

    Normalised as theirs is: ``Gamma' Gamma = I``, and the factors' second moment diagonal with descending entries.
    """
    x = np.array([m.z.T @ m.r / len(m.r) for m in months])
    gamma = pca_loadings(x, k)
    zz = [m.z.T @ m.z / len(m.r) for m in months]
    zr = [m.z.T @ m.r / len(m.r) for m in months]
    lsize = gamma.shape[0]
    for _ in range(iterations):
        f = np.array([np.linalg.lstsq(zz_t @ gamma, zr_t, rcond=None)[0] for zz_t, zr_t in zip(zz, zr, strict=True)])
        a = np.zeros((lsize * k, lsize * k))
        b = np.zeros(lsize * k)
        for zz_t, zr_t, f_t in zip(zz, zr, f, strict=True):
            a += np.kron(zz_t, np.outer(f_t, f_t))
            b += np.kron(zr_t, f_t)
        new = np.linalg.lstsq(a, b, rcond=None)[0].reshape(lsize, k)
        # normalise: orthonormal Gamma, factors' second moment diagonal and ordered
        q, r = np.linalg.qr(new)
        f2 = (f @ r.T).T @ (f @ r.T) / len(f)
        values, vectors = np.linalg.eigh(f2)
        order = np.argsort(values)[::-1]
        new = q @ vectors[:, order]
        change = float(np.max(np.abs(new - gamma)))
        gamma = new
        if change < tol:
            break
    return gamma


def tangency_weights(factors: np.ndarray) -> np.ndarray:
    """The factors' tangency (maximum Sharpe) weights ``Sigma^-1 mu``, scaled to unit gross weight."""
    f = np.atleast_2d(factors.T).T
    mu = f.mean(axis=0)
    sigma = np.atleast_2d(np.cov(f, rowvar=False))
    w = np.linalg.solve(sigma + 1e-12 * np.eye(len(mu)), mu)
    scaled: np.ndarray = w / np.abs(w).sum()
    return scaled


def evaluate(fitted: np.ndarray, actual: np.ndarray, sdf: np.ndarray) -> dict[str, float]:
    """Out-of-sample measures over the test months.

    The total R2 of ``actual`` by ``fitted`` (both ``T x L``), the RMS pricing error (each portfolio's mean
    residual) and the annualised Sharpe ratio of ``sdf``: the tangency portfolio's returns in the test months, each
    year's weights estimated on that year's training months.
    """
    residual = actual - fitted
    total_r2 = 1.0 - float((residual**2).sum() / (actual**2).sum())
    pricing = float(np.sqrt(np.mean(residual.mean(axis=0) ** 2)))
    sharpe = float(sdf.mean() / sdf.std() * math.sqrt(12)) if sdf.std() > 0 else float("nan")
    return {"total R2": total_r2, "pricing error": pricing, "SDF Sharpe": sharpe}


def walk_forward_pca(
    managed: pd.DataFrame, k: int, *, gamma: float = 0.0, first_test: int, last_test: int
) -> dict[str, float]:
    """PCA (or RP-PCA) refitted each January on all earlier months; its out-of-sample measures over the test years.

    A test month's fitted values are its managed returns projected on the training loadings, its factors their
    coordinates; the tangency weights come from the training months' factors.
    """
    fitted, actual, sdf = [], [], []
    for year in range(first_test, last_test + 1):
        years = pd.DatetimeIndex(managed.index).year
        train = managed[years < year].to_numpy()
        test = managed[years == year].to_numpy()
        if len(test) == 0:
            continue
        v = pca_loadings(train, k, gamma=gamma)
        f = test @ v
        fitted.append(f @ v.T)
        actual.append(test)
        sdf.append(f @ tangency_weights(train @ v))
    return evaluate(np.vstack(fitted), np.vstack(actual), np.concatenate(sdf))


def walk_forward_ipca(months: Sequence[Month], k: int, *, first_test: int, last_test: int) -> dict[str, float]:
    """IPCA refitted each January on all earlier months; its out-of-sample measures over the test years.

    A test month's factors are its own cross-sectional regression on the training ``Gamma`` (the total R-squared of
    Kelly, Pruitt and Su); its fitted managed returns are ``(Z'Z / N) Gamma f``.
    """
    fitted, actual, sdf = [], [], []
    for year in range(first_test, last_test + 1):
        train = [m for m in months if m.date.year < year]
        test = [m for m in months if m.date.year == year]
        if not test:
            continue
        gamma = ipca_fit(train, k)
        weights = tangency_weights(np.array([ipca_factors(m, gamma) for m in train]))
        for m in test:
            f = ipca_factors(m, gamma)
            fitted.append(m.z.T @ m.z @ gamma @ f / len(m.r))
            actual.append(m.z.T @ m.r / len(m.r))
            sdf.append(float(f @ weights))
    return evaluate(np.array(fitted), np.array(actual), np.array(sdf))


def walk_forward_observed(
    managed: pd.DataFrame, factors: pd.DataFrame, *, first_test: int, last_test: int
) -> dict[str, float]:
    """Observed factors (the risk model's), regressed on each January over all earlier months, measured out of sample.

    Each portfolio is regressed on the factors; the test months' fitted values use the realised factors (a total
    R-squared, as for the latent models), and the intercept is left out as the pricing error.
    """
    named = factors.rename(columns=lambda c: f"factor: {c}")  # a factor may share a portfolio's name
    both = managed.join(named, how="inner")
    x_cols, f_cols = list(managed.columns), list(named.columns)
    fitted, actual, sdf = [], [], []
    for year in range(first_test, last_test + 1):
        years = pd.DatetimeIndex(both.index).year
        train = both[years < year]
        test = both[years == year]
        if test.empty:
            continue
        design = np.column_stack([np.ones(len(train)), train[f_cols].to_numpy()])
        loadings = np.linalg.lstsq(design, train[x_cols].to_numpy(), rcond=None)[0][1:]
        fitted.append(test[f_cols].to_numpy() @ loadings)
        actual.append(test[x_cols].to_numpy())
        sdf.append(test[f_cols].to_numpy() @ tangency_weights(train[f_cols].to_numpy()))
    return evaluate(np.vstack(fitted), np.vstack(actual), np.concatenate(sdf))
