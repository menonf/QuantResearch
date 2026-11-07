"""Factor scores: ``date x listing_id`` panels where higher means more attractive. Pure functions.

Every score is made comparable across dates by :func:`cross_sectional_z`: within each date, only the members of the
universe are ranked, and ranks become z-scores (robust to outliers, no look-ahead - nothing crosses dates).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd


def cross_sectional_z(raw: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per date: percentile ranks of the members' values mapped to z-scores (mean 0, std ~1); non-members NaN."""
    values = raw.where(members) if members is not None else raw
    pct = values.rank(axis=1, pct=True)
    n = values.notna().sum(axis=1)
    # centre the percentiles in (0, 1): (rank - 0.5) / n
    centred = (pct.mul(n, axis=0) - 0.5).div(n, axis=0)
    return pd.DataFrame(_norm_ppf(centred.to_numpy(dtype=float)), index=centred.index, columns=centred.columns)


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's approximation, |error| < 1.2e-9) - avoids a SciPy dependency."""
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    out = np.full(p.shape, np.nan)
    lo, hi = 0.02425, 1 - 0.02425
    m = ~np.isnan(p)
    pl = m & (p < lo)
    ph = m & (p > hi)
    pm = m & ~pl & ~ph
    q = np.sqrt(-2 * np.log(p[pl]))
    out[pl] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    q = p[pm] - 0.5
    r = q * q
    out[pm] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
    q = np.sqrt(-2 * np.log(1 - p[ph]))
    out[ph] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    return out


def momentum(adjusted: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Return from ``lookback`` to ``skip`` trading days ago (classic 12-1 momentum); needs split-adjusted prices."""
    out: pd.DataFrame = adjusted.shift(skip) / adjusted.shift(lookback) - 1.0
    return out


def monthly_returns(returns: pd.DataFrame, min_days: int = 10) -> pd.DataFrame:
    """Daily returns compounded to calendar months (``month x listing_id``).

    A month with fewer than ``min_days`` returns is missing, not flat.
    """
    months = pd.DatetimeIndex(returns.index).to_period("M")
    grown = (1.0 + returns.fillna(0.0)).groupby(months).prod() - 1.0
    return grown.where(returns.notna().groupby(months).sum() >= min_days)


def residual_momentum(
    returns: pd.DataFrame,
    market: pd.Series,
    dates: pd.DatetimeIndex,
    *,
    window: int = 36,
    min_months: int = 12,
    formation: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """On each of ``dates``: the share's own recent return, the market taken out, per unit of its own noise.

    The return is over the last ``formation`` months but the latest ``skip``. Each share's monthly returns over up to
    ``window`` months (at least ``min_months``) are regressed on the ``market``'s; the score is the sum of the
    residuals over the formation months divided by the residuals' standard deviation. Plain momentum ranks partly on
    beta - after a rising year the high-beta shares lead, and they fall hardest when the market turns (2009, 2020); the
    residual ranks on what the company did itself (Blitz, Huij and Martens, 2011). ``returns`` and ``market`` are
    daily; the month of a date counts as its latest month.
    """
    r = monthly_returns(returns)
    m = monthly_returns(market.to_frame("m"))["m"].reindex(r.index)
    out = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
    months = r.index
    for d in dates:
        k = months.get_indexer(pd.PeriodIndex([pd.Timestamp(d).to_period("M")]))[0]
        if k < 0:
            continue
        y = r.iloc[max(0, k - window + 1) : k + 1].to_numpy(dtype=float)
        x = m.iloc[max(0, k - window + 1) : k + 1].to_numpy(dtype=float)[:, None]
        valid = ~np.isnan(y) & ~np.isnan(x)
        xs = np.where(valid, x, np.nan)
        ys = np.where(valid, y, np.nan)
        n = valid.sum(axis=0)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # a share with no months yet: all-missing columns
            mx, my = np.nanmean(xs, axis=0), np.nanmean(ys, axis=0)
            beta = np.nanmean((xs - mx) * (ys - my), axis=0) / np.nanmean((xs - mx) ** 2, axis=0)
            resid = ys - (my - beta * mx) - beta * xs
            recent = resid[-formation:-skip] if skip else resid[-formation:]
            score = np.nansum(recent, axis=0) / np.nanstd(resid, axis=0, ddof=1)
        enough = (n >= min_months) & (np.sum(~np.isnan(recent), axis=0) >= (formation - skip) * 2 // 3)
        out.loc[d] = np.where(enough, score, np.nan)
    return out


def dividend_yield(returns: pd.DataFrame, adjusted: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """The trailing year's dividends per unit of price: the sum of each day's total return less its price return.

    The difference between the two returns is the day's dividend over the previous price, so no dividend table is
    needed; a share with half a year of prices or less has no yield yet.
    """
    price = adjusted.pct_change(fill_method=None)
    out: pd.DataFrame = (returns - price).rolling(window, min_periods=window // 2).sum()
    return out


def low_volatility(returns: pd.DataFrame, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    """Minus the annualised volatility of daily returns over ``window`` days (calmer = higher)."""
    vol: pd.DataFrame = returns.rolling(window, min_periods=min_periods).std() * np.sqrt(252)
    return -vol


def _ratio(ratios: pd.DataFrame, name: str) -> pd.DataFrame:
    """One ratio as ``date x listing_id``."""
    return ratios[name].unstack("listing_id")  # noqa: PD010 - a (date, listing) series to a date x listing panel


def _positive_inverse(panel: pd.DataFrame) -> pd.DataFrame:
    """``1 / x`` where ``x > 0`` (a multiple on losses is not cheap)."""
    return (1.0 / panel).where(panel > 0)


def value(ratios: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean z-score of the yields: EBIT/EV, earnings (1/PE), book (1/PB) and sales (1/PS) to price."""
    parts = [
        _ratio(ratios, "EBIT/EV"),
        _positive_inverse(_ratio(ratios, "P/E")),
        _positive_inverse(_ratio(ratios, "P/B")),
        _positive_inverse(_ratio(ratios, "P/S")),
    ]
    return combine({str(i): cross_sectional_z(p, members) for i, p in enumerate(parts)})


def _return_on_equity(ratios: pd.DataFrame) -> pd.DataFrame:
    """RoE where book equity is positive: a loss over negative equity is a positive RoE, not a good business."""
    return _ratio(ratios, "RoE").where(_ratio(ratios, "Book Equity/TL") > 0)


def _return_on_capital(ratios: pd.DataFrame) -> pd.DataFrame:
    """ROC where the capital (net working capital + fixed assets) is positive.

    ROC and EBIT/TA share the numerator and total assets are positive, so the capital is positive exactly when the
    two have the same sign (both 0 when EBIT is 0).
    """
    roc = _ratio(ratios, "ROC")
    return roc.where(np.sign(roc) == np.sign(_ratio(ratios, "EBIT/TA")))


def quality(ratios: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean z-score of return on equity, return on capital, gross profit to assets and low leverage.

    Gross profit to assets (Novy-Marx), not gross margin: a margin compares a grocer with a software house, while
    profit per dollar of assets measures how productive the business is whatever it sells.
    """
    parts = {
        "roe": _return_on_equity(ratios),
        "roc": _return_on_capital(ratios),
        "gross_profitability": _ratio(ratios, "Gross Profit/TA"),
        "leverage": -_ratio(ratios, "Debt/Equity").where(_ratio(ratios, "Debt/Equity") >= 0),
    }
    return combine({k: cross_sectional_z(v, members) for k, v in parts.items()})


def cash_flow_yield(ratios: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Z-score of free cash flow per dollar of enterprise value (operating cash flow less capital spending).

    Cash a business throws off after keeping itself going is harder to dress up than earnings, and a loss-maker that
    burns cash is ranked low rather than left out (unlike the earnings yield in :func:`value`).
    """
    return cross_sectional_z(_ratio(ratios, "FCF/EV"), members)


def accruals(ratios: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Z-score of minus the accruals: net income less operating cash flow, per dollar of assets (less = higher).

    Profit that has not arrived as cash (receivables, inventory, estimates) has tended to be followed by lower returns
    (Sloan, 1996): earnings made of cash persist, earnings made of accruals reverse.
    """
    return cross_sectional_z(-_ratio(ratios, "Accruals/TA"), members)


def net_issuance(share_change: pd.DataFrame) -> pd.DataFrame:
    """Minus the change in the share count over a year (a company buying back its shares = higher).

    Companies issuing shares have tended to do worse than those buying them back; the change is a log, so a
    doubling and a halving are equally far from zero.
    """
    return -share_change


def magic_formula(ratios: pd.DataFrame, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Greenblatt: rank by earnings yield (EBIT/EV) plus rank by return on capital, equally weighted."""
    ey = _ratio(ratios, "EBIT/EV")
    roc = _return_on_capital(ratios).where(ey.notna())
    return combine({"ey": cross_sectional_z(ey, members), "roc": cross_sectional_z(roc, members)})


def combine(scores: Mapping[str, pd.DataFrame], weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Weighted mean of score panels, ignoring a missing score for a name (equal weights by default)."""
    names = list(scores)
    w = {k: (weights or {}).get(k, 1.0) for k in names}
    aligned = [scores[k] for k in names]
    index = aligned[0].index
    columns = aligned[0].columns
    for s in aligned[1:]:
        index, columns = index.union(s.index), columns.union(s.columns)
    num = pd.DataFrame(0.0, index=index, columns=columns)
    den = pd.DataFrame(0.0, index=index, columns=columns)
    for k in names:
        s = scores[k].reindex(index=index, columns=columns)
        num += s.fillna(0.0) * w[k]
        den += s.notna() * w[k]
    return num / den.where(den > 0)
