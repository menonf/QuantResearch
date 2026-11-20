"""A fundamental equity risk model after BlackRock's BFRE 2.0 Short Horizon (one month). Pure functions.

    raw = risk_styles.substyles(inputs)
    exposures = style_exposures(raw, inputs)                   # month ends: listing_id x factor
    model = build_risk_model(inputs.returns, exposures, forecast_dates)
    loadings, covariance, specific = model.on(day)              # daily units; x252 a year, x21 a month

The steps follow the paper's sections:

1. **Exposures** (8.1). Each substyle is standardised among the members: its mean weighted by the square root of
   market value is zero, its plain standard deviation one, and values beyond 2.5 are squeezed smoothly towards 3
   (BFRE's "Huberisation" - its exact curve is not published). A style is the average of the substyles a share
   has, standardised again; a share with none takes what its size and industry predict ("fill-miss"). Every share
   has 1 in the Market factor and 1 in its own industry.
2. **Factor returns** (3.2, equations 9-10). Each day, that day's returns are regressed on the exposures of the
   latest month end before it, weighted by the square root of market value, with the industry returns constrained
   to a weighted average of zero - so the Market factor is the weighted average return and an industry's return is
   its excess over it. A one-country model needs no country factors. Returns beyond +-25% are clipped in the
   regression only (one company's news would tilt its industry); every share's specific return is its own return
   less the factors'.
3. **Factor covariance** (1.0): exponentially weighted, variances halving every 250 days and correlations every 500,
   with a Newey-West term for serial correlation. **2.0**: blended equally with the same matrix scaled by gamma
   squared, gamma being how far the past month's factor volatilities ran above (or below) the forecasts
   (equations 12, 58-61).
4. **Specific risk** (1.0): each share's specific returns, outliers clipped, exponentially weighted (half-life 125
   days over 375), with a cross-sectional estimate from size and industry for shares with a short history. **2.0**:
   the average of that and a structural forecast (equation 13) - a panel regression of the next month's realised
   specific volatility on size, volatility, liquidity, momentum and the gamma-scaled forecast (Appendix 8.7).

Choices the paper leaves open: gamma scales volatility, so the covariance by gamma squared; the factor covariance
uses one Newey-West lag like the specific risk of the 2.0 models; no prior for thin industries (the GICS sectors of
1,500 shares are not thin).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from equity_research.analytics.risk.risk_styles import STYLES, RiskInputs, decay_weights, mid_cap
from research_core.data.calendars import rebalance_dates

MARKET = "Market"
MID_CAP = "Mid-Cap"

# ---------------------------------------------------------------- exposures


def _squash(z: np.ndarray, knee: float = 2.5, limit: float = 3.0) -> np.ndarray:
    """Values beyond ``knee`` squeezed smoothly towards ``limit``: the order in the tails is kept, unlike a clip."""
    a = np.abs(z)
    room = limit - knee
    with np.errstate(invalid="ignore"):
        return np.where(a > knee, np.sign(z) * (knee + room * np.tanh((a - knee) / room)), z)


def standardise(
    values: np.ndarray, weights: np.ndarray, universe: np.ndarray, *, mad_limit: float = 5.0, rounds: int = 2
) -> np.ndarray:
    """Values on a common scale: weighted mean 0 and standard deviation 1 among ``universe``, tails squeezed.

    The mean is weighted by ``weights`` (the square root of market value), the standard deviation is the plain one,
    as in BFRE: +1 is one standard deviation above the average share. Raw values beyond ``mad_limit`` robust
    standard deviations of the median are pulled in first, so one wild ratio cannot set everyone's scale.
    """
    x = values.astype(float).copy()
    inside = universe & ~np.isnan(x)
    if inside.sum() < 3:
        return np.full_like(x, np.nan)
    median = np.median(x[inside])
    mad = 1.4826 * np.median(np.abs(x[inside] - median))
    if mad > 0:
        x = np.clip(x, median - mad_limit * mad, median + mad_limit * mad)
    w = np.where(weights[inside] > 0, weights[inside], 0.0)
    for _ in range(rounds):
        mean = float(np.average(x[inside], weights=w)) if w.sum() > 0 else float(np.mean(x[inside]))
        sd = float(np.std(x[inside], ddof=1))
        if not sd > 0:
            return np.where(np.isnan(x), np.nan, 0.0)
        x = _squash((x - mean) / sd)
    return x


def fill_missing(z: np.ndarray, log_value: np.ndarray, industry: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """``z`` with the covered shares' gaps filled by what size and industry predict (BFRE's fill-miss).

    A least-squares fit of the known values on the log of market value and industry dummies; a share whose value or
    industry is unknown too takes the average, zero.
    """
    missing = covered & np.isnan(z)
    if not missing.any():
        return z
    out = z.copy()
    groups = np.unique(industry[industry >= 0])
    dummies = (industry[:, None] == groups[None, :]).astype(float)
    design = np.column_stack([np.ones(len(z)), np.nan_to_num(log_value), dummies])
    known = covered & ~np.isnan(z) & ~np.isnan(log_value)
    if known.sum() > design.shape[1] + 10:
        coef = np.linalg.lstsq(design[known], z[known], rcond=None)[0]
        guess = np.where(np.isnan(log_value) | (industry < 0), 0.0, design @ coef)
    else:
        guess = np.zeros(len(z))
    out[missing] = guess[missing]
    return out


@dataclass(frozen=True)
class Exposures:
    """On each exposure date (month end): the covered listings' exposures, weights and estimation universe.

    The estimation universe is the covered members with an industry; the weight is the square root of market value.
    """

    loadings: Mapping[pd.Timestamp, pd.DataFrame]
    weight: Mapping[pd.Timestamp, pd.Series]
    estu: Mapping[pd.Timestamp, pd.Series]
    industries: tuple[str, ...]
    styles: tuple[str, ...]
    dates: pd.DatetimeIndex = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dates", pd.DatetimeIndex(sorted(self.loadings)))

    @property
    def factors(self) -> list[str]:
        """Market, then the industries, then the styles - the order of every factor table."""
        return [MARKET, *self.industries, *self.styles]

    def latest(self, day: pd.Timestamp, *, strictly_before: bool = False) -> pd.Timestamp | None:
        """The exposure date in force on ``day`` (for a day's return: the last one strictly before it)."""
        k = int(self.dates.searchsorted(pd.Timestamp(day), side="left" if strictly_before else "right"))
        return None if k == 0 else pd.Timestamp(self.dates[k - 1])

    def on(self, day: pd.Timestamp) -> pd.DataFrame:
        """``listing_id x factor`` in force on ``day`` (empty before the first exposure date)."""
        e = self.latest(day)
        return self.loadings[e] if e is not None else pd.DataFrame(columns=self.factors)


def style_exposures(
    raw: Mapping[str, pd.DataFrame], inputs: RiskInputs, styles: Mapping[str, Sequence[str]] = STYLES
) -> Exposures:
    """The exposures on ``inputs.dates`` from the raw substyles (:func:`~.risk_styles.substyles`)."""
    listings = inputs.returns.columns
    industry = inputs.industry.reindex(listings)
    names = tuple(sorted(str(n) for n in industry.dropna().unique()))
    codes = industry.map({n: i for i, n in enumerate(names)}).fillna(-1).astype(int).to_numpy()
    dates = inputs.dates
    members = inputs.members.reindex(index=dates, columns=listings).fillna(False).astype(bool)
    company = inputs.company_value.reindex(index=dates, columns=listings)
    mid = mid_cap(company, members)
    loadings, weights, estus = {}, {}, {}
    for d in dates:
        value = inputs.listing_value.loc[d].reindex(listings).to_numpy(dtype=float)
        covered = members.loc[d].to_numpy() & (value > 0)
        if covered.sum() < 50:
            continue
        weight = np.sqrt(np.where(covered, value, 0.0))
        estu = covered & (codes >= 0)
        cv = company.loc[d].to_numpy(dtype=float)
        log_value = np.log(np.where(cv > 0, cv, np.nan))
        cols: dict[str, np.ndarray] = {MARKET: np.ones(len(listings))}
        for i, name in enumerate(names):
            cols[name] = (codes == i).astype(float)
        for style, subs in styles.items():
            parts = [standardise(raw[s].loc[d].to_numpy(dtype=float), weight, estu) for s in subs if s in raw]
            stack = np.vstack(parts) if parts else np.full((1, len(listings)), np.nan)
            known = (~np.isnan(stack)).sum(axis=0)
            average = np.where(known > 0, np.nansum(stack, axis=0) / np.maximum(known, 1), np.nan)
            cols[style] = fill_missing(standardise(average, weight, estu), log_value, codes, covered)
        cols[MID_CAP] = np.nan_to_num(mid.loc[d].to_numpy(dtype=float))
        frame = pd.DataFrame(cols, index=listings)[covered]
        ts = pd.Timestamp(d)
        loadings[ts] = frame.fillna(0.0)
        weights[ts] = pd.Series(weight[covered], index=frame.index)
        estus[ts] = pd.Series(estu[covered], index=frame.index)
    return Exposures(loadings, weights, estus, names, (*styles, MID_CAP))


# ---------------------------------------------------------------- factor returns


@dataclass(frozen=True)
class Estimation:
    """Daily factor returns, every covered share's specific return, and how well each day's regression fitted."""

    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    r2: pd.Series
    tstats: pd.DataFrame


def constrained_regression(
    x: np.ndarray, r: np.ndarray, w: np.ndarray, industries: slice
) -> tuple[np.ndarray, np.ndarray, float]:
    """One day's weighted regression with the industry returns' weighted average fixed at zero (equation 10).

    Every share has 1 in the market and 1 in one industry, so those columns are collinear; the constraint
    ``sum_j c_j f_j = 0`` (``c_j`` the industry's total weight) is substituted in - the largest industry's return is
    the minus the weighted sum of the others' - and the reduced problem solved by least squares. An industry with no
    share today returns zero. Returns the factor returns, their t-statistics and the weighted R squared.
    """
    ind = x[:, industries]
    c = (w[:, None] * ind).sum(axis=0)
    active = np.flatnonzero(c > 0)
    others = active[active != active[np.argmax(c[active])]] if len(active) else active
    basis = np.zeros((ind.shape[1], len(others)))
    basis[others, np.arange(len(others))] = 1.0
    if len(active):
        ref = active[np.argmax(c[active])]
        basis[ref, :] = -c[others] / c[ref]
    before, after = x[:, : industries.start], x[:, industries.stop :]
    z = np.hstack([before, ind @ basis, after])
    sw = np.sqrt(w)
    theta = np.linalg.lstsq(z * sw[:, None], r * sw, rcond=None)[0]
    resid = r - z @ theta
    k_before, k_ind = before.shape[1], len(others)
    # the factor returns are a linear map of the reduced coefficients: f = M theta
    m = np.zeros((x.shape[1], z.shape[1]))
    m[:k_before, :k_before] = np.eye(k_before)
    m[industries, k_before : k_before + k_ind] = basis
    m[industries.stop :, k_before + k_ind :] = np.eye(after.shape[1])
    f = m @ theta
    dof = max(len(r) - z.shape[1], 1)
    s2 = float((w * resid**2).sum() / dof)
    cov = m @ (s2 * np.linalg.pinv((z * w[:, None]).T @ z)) @ m.T
    with np.errstate(invalid="ignore", divide="ignore"):
        t = f / np.sqrt(np.diag(cov))
    centred = r - np.average(r, weights=w)
    r2 = 1.0 - float((w * resid**2).sum() / (w * centred**2).sum())
    return f, t, r2


def estimate_factor_returns(returns: pd.DataFrame, exposures: Exposures, *, clip: float = 0.25) -> Estimation:
    """The daily cross-sectional regressions (equations 9-10) over every day after the first exposure date."""
    factors = exposures.factors
    industries = slice(1, 1 + len(exposures.industries))
    days = pd.DatetimeIndex(returns.index[returns.index > exposures.dates[0]])
    column = {lid: j for j, lid in enumerate(returns.columns)}
    values = returns.loc[days].to_numpy(dtype=float)
    fr = np.full((len(days), len(factors)), np.nan)
    ts = np.full_like(fr, np.nan)
    r2 = np.full(len(days), np.nan)
    spec = np.full(values.shape, np.nan)
    prepared: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for t, d in enumerate(days):
        e = exposures.latest(d, strictly_before=True)
        if e is None:
            continue
        if e not in prepared:
            frame = exposures.loadings[e]
            prepared[e] = (
                np.array([column[i] for i in frame.index]),
                frame[factors].to_numpy(dtype=float),
                exposures.weight[e].to_numpy(dtype=float),
                exposures.estu[e].to_numpy(dtype=bool),
            )
        idx, x, w, estu = prepared[e]
        r = values[t, idx]
        valid = ~np.isnan(r)
        fit = valid & estu & (w > 0)
        if fit.sum() < 3 * len(factors):
            continue
        f, tt, rr = constrained_regression(x[fit], np.clip(r[fit], -clip, clip), w[fit], industries)
        fr[t], ts[t], r2[t] = f, tt, rr
        spec[t, idx[valid]] = r[valid] - x[valid] @ f
    return Estimation(
        factor_returns=pd.DataFrame(fr, index=days, columns=factors).dropna(how="all"),
        specific_returns=pd.DataFrame(spec, index=days, columns=returns.columns),
        r2=pd.Series(r2, index=days).dropna(),
        tstats=pd.DataFrame(ts, index=days, columns=factors).dropna(how="all"),
    )


# ---------------------------------------------------------------- factor covariance


def _nearest_psd(m: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """``m`` with negative eigenvalues raised to ``floor``: a covariance must give no portfolio a negative variance."""
    vals, vecs = np.linalg.eigh((m + m.T) / 2)
    if vals.min() >= floor:
        return m
    out: np.ndarray = (vecs * np.maximum(vals, floor)) @ vecs.T
    return out


def _snapshots(index: pd.DatetimeIndex, dates: Sequence[pd.Timestamp]) -> dict[int, list[pd.Timestamp]]:
    """``row -> dates`` whose latest row on or before them is that row."""
    out: dict[int, list[pd.Timestamp]] = {}
    for d in dates:
        p = int(index.searchsorted(pd.Timestamp(d), side="right")) - 1
        if p >= 0:
            out.setdefault(p, []).append(pd.Timestamp(d))
    return out


def ewma_covariance(
    factor_returns: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    *,
    var_half_life: float = 250.0,
    corr_half_life: float = 500.0,
    nw_lags: int = 1,
    min_obs: int = 250,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """BFRE 1.0's factor covariance on each of ``dates``, in daily units (Table 3: expanding window).

    Volatilities from returns weighted to halve every ``var_half_life`` days, correlations every
    ``corr_half_life``; the mean is taken as zero. Each lag-``l`` cross term up to ``nw_lags`` is added with
    Newey-West's weight ``1 - l / (nw_lags + 1)``, so serial correlation in daily returns reaches the forecast.
    """
    x = factor_returns.fillna(0.0).to_numpy(dtype=float)
    n, k = x.shape
    lams = (0.5 ** (1 / var_half_life), 0.5 ** (1 / corr_half_life))
    sums = [np.zeros((nw_lags + 1, k, k)) for _ in lams]
    totals = [0.0 for _ in lams]
    want = _snapshots(pd.DatetimeIndex(factor_returns.index), dates)
    lag_weights = [1 - lag / (nw_lags + 1) for lag in range(1, nw_lags + 1)]
    out = {}
    for p in range(n):
        for i, lam in enumerate(lams):
            s = sums[i]
            s *= lam
            s[0] += np.outer(x[p], x[p])
            for lag in range(1, nw_lags + 1):
                if p - lag >= 0:
                    s[lag] += np.outer(x[p], x[p - lag])
            totals[i] = lam * totals[i] + 1.0
        if p not in want or p + 1 < min_obs:
            continue
        cov_v, cov_c = (
            (s[0] + sum(wl * (s[lag] + s[lag].T) for lag, wl in enumerate(lag_weights, start=1))) / tot
            for s, tot in zip(sums, totals, strict=True)
        )
        vol = np.sqrt(np.maximum(np.diag(cov_v), 0.0))
        d = np.sqrt(np.maximum(np.diag(cov_c), 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(cov_c / np.outer(d, d))
        np.fill_diagonal(corr, 1.0)
        cov = _nearest_psd(corr * np.outer(vol, vol))
        for day in want[p]:
            out[day] = pd.DataFrame(cov, index=factor_returns.columns, columns=factor_returns.columns)
    return out


def factor_gamma(
    factor_returns: pd.DataFrame, covariance: Mapping[pd.Timestamp, pd.DataFrame], *, horizon: int = 21
) -> pd.Series:
    """Gamma on each date (equation 61): the least-squares scale from forecast to realised factor volatilities.

    Realised: the root of the summed squared daily returns over the last ``horizon`` days; forecast: the matrix's
    volatility over ``horizon`` days. Over 1 when the past month ran hotter than the forecast.
    """
    realised = (factor_returns.fillna(0.0) ** 2).rolling(horizon, min_periods=horizon).sum().pow(0.5)
    out = {}
    for d, cov in covariance.items():
        now = realised.loc[:d]
        if now.empty:
            continue
        r = now.iloc[-1].reindex(cov.index).to_numpy(dtype=float)
        f = np.sqrt(horizon * np.diag(cov.to_numpy()))
        ok = ~np.isnan(r) & (f > 0)
        out[d] = float((r[ok] * f[ok]).sum() / (f[ok] ** 2).sum()) if ok.any() else np.nan
    return pd.Series(out, dtype=float).sort_index()


# ---------------------------------------------------------------- specific risk


def clean_specific(specific: pd.DataFrame, limit: float = 10.0) -> pd.DataFrame:
    """Specific returns clipped at ``limit`` robust standard deviations of the day's cross-section.

    A takeover's 40% jump is one event, not a volatility: clipped, it still counts as a large move without setting
    the share's specific risk for the next year and a half on its own.
    """
    median = specific.median(axis=1)
    scale = 1.4826 * specific.sub(median, axis=0).abs().median(axis=1)
    bound = (limit * scale).to_numpy()[:, None]
    return pd.DataFrame(
        np.clip(specific.to_numpy(dtype=float), -bound, bound), index=specific.index, columns=specific.columns
    )


def ewma_specific_risk(
    specific: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    *,
    half_life: float = 125.0,
    window: int = 375,
    nw_lags: int = 1,
    min_obs: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each share's daily specific volatility on each of ``dates`` (Table 4), and how many days it rests on.

    The squared specific returns of the last ``window`` days, weighted to halve every ``half_life`` days, with
    Newey-West lag terms; a negative serial correlation may lower the variance to no less than a quarter of the
    plain one. Missing with fewer than ``min_obs`` days.
    """
    u = specific.to_numpy(dtype=float)
    present = (~np.isnan(u)).astype(float)
    u0 = np.nan_to_num(u)
    squares = u0**2
    lagged = [np.vstack([np.zeros((lag, u.shape[1])), u0[lag:] * u0[:-lag]]) for lag in range(1, nw_lags + 1)]
    index = pd.DatetimeIndex(list(dates))
    ends = np.asarray(pd.DatetimeIndex(specific.index).searchsorted(index, side="right"))
    full = decay_weights(window, half_life)
    vol = np.full((len(ends), u.shape[1]), np.nan)
    nobs = np.zeros_like(vol)
    for j, end in enumerate(ends.tolist()):
        lo = max(0, end - window)
        w = full[window - (end - lo) :]
        weight = w @ present[lo:end]
        plain = w @ squares[lo:end]
        cross = sum((1 - lag / (nw_lags + 1)) * 2 * (w @ lagged[lag - 1][lo:end]) for lag in range(1, nw_lags + 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.maximum(plain + cross, plain / 4) / weight
        count = present[lo:end].sum(axis=0)
        vol[j] = np.where(count >= min_obs, np.sqrt(var), np.nan)
        nobs[j] = count
    return (
        pd.DataFrame(vol, index=index, columns=specific.columns),
        pd.DataFrame(nobs, index=index, columns=specific.columns),
    )


def blend_cross_sectional(
    vol: pd.DataFrame, nobs: pd.DataFrame, exposures: Exposures, *, full_obs: int = 250
) -> pd.DataFrame:
    """Specific volatility for every covered share: its own history's, leaning on its peers' while that is short.

    The peers' estimate is a fit of log volatility on log market value and industry among the estimation universe's
    shares with a full history; a share's own estimate weighs ``days / full_obs`` (up to 1), the peers' the rest -
    an IPO starts from what shares like it carry (BFRE's cross-sectional overlay).
    """
    out = pd.DataFrame(np.nan, index=vol.index, columns=vol.columns)
    for d in vol.index:
        e = exposures.latest(d)
        if e is None:
            continue
        frame = exposures.loadings[e]
        ids = frame.index
        own = vol.loc[d].reindex(ids).to_numpy(dtype=float)
        days = nobs.loc[d].reindex(ids).fillna(0.0).to_numpy(dtype=float)
        design = np.column_stack(
            [
                np.ones(len(ids)),
                2 * np.log(exposures.weight[e].to_numpy(dtype=float)),
                frame[list(exposures.industries)],
            ]
        )
        known = exposures.estu[e].to_numpy(dtype=bool) & (days >= full_obs) & (own > 0)
        if known.sum() <= design.shape[1] + 10:
            out.loc[d, ids] = own
            continue
        coef = np.linalg.lstsq(design[known], np.log(own[known]), rcond=None)[0]
        peers = np.exp(design @ coef)
        a = np.clip(days / full_obs, 0.0, 1.0)
        out.loc[d, ids] = np.where(np.isnan(own), peers, a * own + (1 - a) * peers)
    return out


def specific_gamma(specific: pd.DataFrame, vol: pd.DataFrame, exposures: Exposures, *, horizon: int = 21) -> pd.Series:
    """Specific-risk gamma on each date (equation 62), over the estimation universe's shares."""
    realised = ((specific**2).rolling(horizon, min_periods=horizon * 2 // 3).mean() * horizon).pow(0.5)
    out = {}
    for d in vol.index:
        e = exposures.latest(d)
        if e is None:
            continue
        ids = exposures.estu[e].index[exposures.estu[e].to_numpy(dtype=bool)]
        now = realised.loc[:d]
        if now.empty:
            continue
        r = now.iloc[-1].reindex(ids).to_numpy(dtype=float)
        f = vol.loc[d].reindex(ids).to_numpy(dtype=float) * np.sqrt(horizon)
        ok = ~np.isnan(r) & ~np.isnan(f)
        out[d] = float((r[ok] * f[ok]).sum() / (f[ok] ** 2).sum()) if ok.sum() > 10 else np.nan
    return pd.Series(out, dtype=float).sort_index()


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last day of each month in ``index``."""
    return rebalance_dates(index, "ME")


def structural_specific_risk(
    vol_gamma: pd.DataFrame,
    specific: pd.DataFrame,
    exposures: Exposures,
    *,
    styles: Sequence[str] = ("Size", "Volatility", "Liquidity", "Momentum"),
    horizon: int = 21,
    months: int = 24,
    min_months: int = 12,
    bounds: tuple[float, float] = (0.5, 2.0),
) -> pd.DataFrame:
    """BFRE 2.0's structural specific risk (equations 65-66) on ``vol_gamma``'s dates, daily units.

    At each month end, a panel regression over the last ``months`` months: each share's realised specific volatility
    over a month, on its style exposures and its gamma-scaled forecast at the month's start, and their products - how
    a share's characteristics bend the common gamma. A forecast uses the latest month end's coefficients, the day's
    exposures and its gamma-scaled forecast; it stays within ``bounds`` of the gamma-scaled one (our guard against a
    fit that strays for an unusual share). Missing until ``min_months`` months are in the panel. ``vol_gamma`` must
    include every month end it is to learn from.
    """
    ends = month_ends(pd.DatetimeIndex(specific.index))
    ends = ends[ends.isin(vol_gamma.index)]
    squares = (specific**2).fillna(0.0)
    counted = specific.notna()
    rows: list[pd.DataFrame] = []
    coefs: dict[pd.Timestamp, np.ndarray] = {}
    for start, end in itertools.pairwise(ends):
        e = exposures.latest(start)
        if e is None:
            continue
        window = (squares.index > start) & (squares.index <= end)
        days = counted.loc[window].sum()
        realised = (squares.loc[window].sum() * horizon / days.where(days >= 10)).pow(0.5)
        estu = exposures.estu[e]
        ids = estu.index[estu.to_numpy(dtype=bool)]
        frame = pd.DataFrame(
            {
                "y": realised.reindex(ids),
                "sigma": vol_gamma.loc[start].reindex(ids) * np.sqrt(horizon),
                **{s: exposures.loadings[e].loc[ids, s] for s in styles},
            }
        ).dropna()
        rows.append(frame.assign(month=end))
        recent = pd.concat(rows[-months:])
        if len(rows) >= min_months:
            coefs[pd.Timestamp(end)] = np.linalg.lstsq(_structural_design(recent, styles), recent["y"], rcond=None)[0]
    out = pd.DataFrame(np.nan, index=vol_gamma.index, columns=vol_gamma.columns)
    fitted = pd.DatetimeIndex(sorted(coefs))
    for d in vol_gamma.index:
        k = fitted.searchsorted(d, side="right")
        e = exposures.latest(d)
        if k == 0 or e is None:
            continue
        ids = exposures.loadings[e].index
        base = vol_gamma.loc[d].reindex(ids) * np.sqrt(horizon)
        frame = pd.DataFrame({"sigma": base, **{s: exposures.loadings[e][s] for s in styles}})
        fit = _structural_design(frame, styles) @ coefs[fitted[k - 1]]
        b = base.to_numpy(dtype=float)
        lo, hi = bounds
        out.loc[d, ids] = np.clip(fit, lo * b, hi * b) / np.sqrt(horizon)
    return out


def _structural_design(frame: pd.DataFrame, styles: Sequence[str]) -> np.ndarray:
    sigma = frame["sigma"].to_numpy(dtype=float)
    x = frame[list(styles)].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(frame)), x, sigma, x * sigma[:, None]])


# ---------------------------------------------------------------- the model


@dataclass(frozen=True)
class RiskModelConfig:
    """The parameters of BFRE 2.0 Short Horizon (Tables 3, 4 and 23) and the choices made here."""

    clip: float = 0.25
    var_half_life: float = 250.0
    corr_half_life: float = 500.0
    factor_nw_lags: int = 1
    min_factor_days: int = 250
    specific_half_life: float = 125.0
    specific_window: int = 375
    specific_nw_lags: int = 1
    min_specific_days: int = 63
    full_specific_days: int = 250
    horizon: int = 21
    structural_months: int = 24
    structural_styles: tuple[str, ...] = ("Size", "Volatility", "Liquidity", "Momentum")


@dataclass(frozen=True)
class RiskModel:
    """Forecasts on the forecast dates, in daily units: BFRE 1.0 (exponentially weighted) and 2.0 (adapted)."""

    exposures: Exposures
    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    r2: pd.Series
    tstats: pd.DataFrame
    covariance: Mapping[pd.Timestamp, pd.DataFrame]
    covariance_ewma: Mapping[pd.Timestamp, pd.DataFrame]
    specific_risk: pd.DataFrame
    specific_risk_ewma: pd.DataFrame
    gamma: pd.DataFrame
    config: RiskModelConfig

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The days the model has a forecast for."""
        return pd.DatetimeIndex(sorted(self.covariance))

    def on(self, day: pd.Timestamp, version: str = "2.0") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """The exposures, factor covariance and specific volatility in force on ``day`` (a forecast date)."""
        day = pd.Timestamp(day)
        if day not in self.covariance:
            raise KeyError(f"no forecast on {day:%Y-%m-%d}: forecasts run {self.dates[0]:%Y-%m-%d} onwards")
        cov, spec = (
            (self.covariance[day], self.specific_risk)
            if version == "2.0"
            else (self.covariance_ewma[day], self.specific_risk_ewma)
        )
        loadings = self.exposures.on(day)
        return loadings, cov, pd.Series(spec.loc[day]).reindex(loadings.index)


def build_risk_model(
    returns: pd.DataFrame,
    exposures: Exposures,
    forecast_dates: Sequence[pd.Timestamp],
    config: RiskModelConfig | None = None,
) -> RiskModel:
    """Estimate the factor returns and make the forecasts on ``forecast_dates`` (those with enough history)."""
    c = config or RiskModelConfig()
    est = estimate_factor_returns(returns, exposures, clip=c.clip)
    cov1 = ewma_covariance(
        est.factor_returns,
        forecast_dates,
        var_half_life=c.var_half_life,
        corr_half_life=c.corr_half_life,
        nw_lags=c.factor_nw_lags,
        min_obs=c.min_factor_days,
    )
    dates = pd.DatetimeIndex(sorted(cov1))
    gamma_f = factor_gamma(est.factor_returns, cov1, horizon=c.horizon)
    scale = 0.5 + 0.5 * gamma_f.reindex(dates).fillna(1.0) ** 2
    cov2 = {d: m * float(scale[d]) for d, m in cov1.items()}

    specific = clean_specific(est.specific_returns)
    first = dates[0] if len(dates) else pd.Timestamp.max
    ends = month_ends(pd.DatetimeIndex(specific.index))
    every = dates.union(ends[ends >= first - pd.DateOffset(months=c.structural_months + 1)])
    vol, nobs = ewma_specific_risk(
        specific,
        list(every),
        half_life=c.specific_half_life,
        window=c.specific_window,
        nw_lags=c.specific_nw_lags,
        min_obs=c.min_specific_days,
    )
    vol = blend_cross_sectional(vol, nobs, exposures, full_obs=c.full_specific_days)
    gamma_s = specific_gamma(specific, vol, exposures, horizon=c.horizon)
    vol_gamma = vol.mul(gamma_s.reindex(vol.index).fillna(1.0), axis=0)
    structural = structural_specific_risk(
        vol_gamma,
        specific,
        exposures,
        styles=[x for x in c.structural_styles if x in exposures.styles],
        horizon=c.horizon,
        months=c.structural_months,
    )
    adapted = 0.5 * vol + 0.5 * structural.fillna(vol_gamma)
    return RiskModel(
        exposures=exposures,
        factor_returns=est.factor_returns,
        specific_returns=est.specific_returns,
        r2=est.r2,
        tstats=est.tstats,
        covariance=cov2,
        covariance_ewma=cov1,
        specific_risk=adapted.reindex(dates),
        specific_risk_ewma=vol.reindex(dates),
        gamma=pd.DataFrame({"factor": gamma_f, "specific": gamma_s.reindex(dates)}).reindex(dates),
        config=c,
    )


# ---------------------------------------------------------------- diagnostics


def factor_summary(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Per factor: yearly return and volatility, their ratio, correlation with the market, monthly autocorrelation.

    BFRE's Table 2. A style's return is that of a portfolio with one unit of the style and none of any other factor,
    so its volatility is the style's own risk and a steady return is a premium the market paid for it.
    """
    monthly = factor_returns.resample("ME").sum()
    out = pd.DataFrame(
        {
            "yearly return": factor_returns.mean() * 252,
            "yearly volatility": factor_returns.std() * np.sqrt(252),
            "correlation with market": factor_returns.corr()[MARKET],
            "monthly autocorrelation": monthly.apply(lambda s: s.autocorr()),
        }
    )
    out.insert(2, "return / volatility", out["yearly return"] / out["yearly volatility"])
    return out


def significance(tstats: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """Per factor: the share of days its return was significant (``|t| > threshold``) - BFRE's Figure 8.

    Above about 10% a factor explains returns more often than chance would (5% for ``|t| > 2``).
    """
    out: pd.Series = (tstats.abs() > threshold).mean().sort_values(ascending=False)
    return out


def style_panels(exposures: Exposures) -> dict[str, pd.DataFrame]:
    """``style -> date x listing_id``: each style's exposures on the exposure dates (for comparing with signals)."""
    out = {}
    for style in exposures.styles:
        rows = {d: exposures.loadings[d][style] for d in exposures.dates if style in exposures.loadings[d]}
        out[style] = pd.DataFrame(rows).T.sort_index()
    return out
