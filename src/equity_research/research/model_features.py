"""Features made by fitting a small model to each share's (or the market's) own history, walk-forward.

    garch = garch_volatility(data.returns, data.members, dates)      # GARCH(1,1) volatility forecast, refitted yearly
    har = har_volatility(data.returns, data.members, dates)          # HAR forecast of next month's volatility
    ffd = frac_diff(np.log(data.adjusted), d=0.4).reindex(dates)     # a price with memory but no trend to speak of
    trend = kalman_trend(np.log(data.adjusted))                       # the filtered level and slope of the log price
    brk = break_statistic(data.returns)                               # how unusual the latest run of returns is
    regime = market_regime(market, dates)                             # chance the market is in its turbulent state

*Machine Learning for Trading* (chapter 9) derives features from models fitted to each series. Every one here is known
at the close of its date: a model's parameters are estimated only on years already over (refitted each January), and
the series is then filtered forward through the year with those parameters fixed.

1. **GARCH(1,1)** (Bollerslev, 1986) - tomorrow's variance is a weighted mix of a long-run level, today's squared
   surprise and today's variance: ``s2[t+1] = omega + alpha e[t]^2 + beta s2[t]``. Fitted to each member's last three
   years of daily returns (``arch``), it forecasts volatility that reacts to shocks and decays back.
2. **HAR** (Corsi, 2009) - next month's average squared return regressed on the latest day's, week's and month's:
   volatility is persistent at all three horizons. One regression pooled across members, refitted each January on the
   month ends whose next month was over.
3. **Fractional differencing** (Lopez de Prado, 2018, chapter 5) - differencing a price once (taking returns) makes it
   stationary but erases its memory; differencing by a fraction ``d`` (weights ``w_k = -w_{k-1} (d - k + 1) / k``,
   cut where they fall below a threshold) keeps much of the memory while removing most of the trend.
4. **Kalman trend** - the local linear trend model (the log price is a level plus noise; the level grows by a slope
   that itself drifts). Its steady-state Kalman filter is Holt's linear smoothing with fixed gains, run here across all
   shares at once: the filtered slope (the trend, a year) and the price's gap from its filtered level.
5. **Break statistic** - the largest standardised move of the log price over the past 5 to 63 days,
   ``(y[t] - y[t-k]) / (sigma sqrt(k))``, signed: a CUSUM-type measure (Chu, Stinchcombe and White, 1996) of whether
   the latest run of returns is unusual for the share.
6. **Market regime** - a two-state Markov switching model of the market's daily return, the states differing in mean
   and variance (Hamilton, 1989; ``statsmodels``), refitted each January; the feature is the filtered probability of
   the turbulent (higher-variance) state - the same for every company on a date.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Years of daily returns each GARCH fit uses.
GARCH_YEARS = 3
#: Holt gains of the Kalman trend: the level adapts over about three months, the slope over about six.
LEVEL_GAIN, SLOPE_GAIN = 2 / (63 + 1), 2 / (126 + 1)
#: Look-backs (days) of the break statistic.
BREAK_LAGS = (5, 10, 21, 42, 63)
#: Trading days a year.
YEAR = 252


def _fit_garch(returns: np.ndarray) -> tuple[float, float, float, float, float]:
    """``(mu, omega, alpha, beta, next variance)`` of a GARCH(1,1) fitted to percent returns; NaNs if it fails.

    The next variance is the forecast for the day after the last return: the first day the fit is used on.
    """
    from arch import arch_model  # noqa: PLC0415 - fitted only where GARCH features are built

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        p = fit.params
        if fit.convergence_flag != 0 or p["alpha[1]"] + p["beta[1]"] >= 1:
            return (np.nan,) * 5
        return (
            float(p["mu"]),
            float(p["omega"]),
            float(p["alpha[1]"]),
            float(p["beta[1]"]),
            float(np.asarray(fit.forecast(horizon=1, reindex=False).variance)[-1, 0]),
        )
    except (ValueError, np.linalg.LinAlgError):
        return (np.nan,) * 5


def garch_volatility(
    returns: pd.DataFrame, members: pd.DataFrame, dates: pd.DatetimeIndex, *, years: int = GARCH_YEARS, n_jobs: int = -1
) -> pd.DataFrame:
    """``date x listing_id`` on ``dates``: the annualised one-day-ahead GARCH(1,1) volatility forecast.

    Each January, every listing that is a member on one of that year's ``dates`` and has at least a year of returns in
    the ``years`` before is fitted; its variance is then filtered day by day through the year with those parameters.
    """
    from sklearn.utils.parallel import Parallel, delayed  # noqa: PLC0415 - joblib, as scikit-learn runs it

    days = pd.DatetimeIndex(returns.index)
    pct = returns * 100
    out = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
    for year in sorted(set(pd.DatetimeIndex(dates).year)):
        start = pd.Timestamp(year=year, month=1, day=1)
        history = pct[(days < start) & (days >= start - pd.DateOffset(years=years))]
        in_year = pct[(days >= start) & (days < start + pd.DateOffset(years=1))]
        year_dates = dates[pd.DatetimeIndex(dates).year == year]
        if history.empty or in_year.empty:
            continue
        wanted = members.reindex(index=year_dates, columns=returns.columns, fill_value=False).astype(bool).any()
        names = [n for n in returns.columns[wanted.to_numpy()] if history[n].notna().sum() >= YEAR]
        fits = Parallel(n_jobs=n_jobs)(delayed(_fit_garch)(history[n].dropna().to_numpy()) for n in names)
        params = pd.DataFrame(fits, index=names, columns=["mu", "omega", "alpha", "beta", "s2"])
        mu, omega, alpha, beta, s2 = (params[c].to_numpy() for c in params.columns)
        variance = np.full((len(in_year), len(names)), np.nan)
        x = in_year[names].to_numpy(dtype=float)
        for i in range(len(in_year)):
            shock = np.where(np.isfinite(x[i]), x[i] - mu, 0.0)  # no return that day: no news
            s2 = omega + alpha * shock**2 + beta * s2  # s2 was today's variance; now tomorrow's, known at the close
            variance[i] = s2
        vol = pd.DataFrame(np.sqrt(variance * YEAR) / 100, index=in_year.index, columns=names)
        out.loc[year_dates, names] = vol.reindex(year_dates).to_numpy()
    return out


def har_volatility(returns: pd.DataFrame, members: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """``date x listing_id`` on ``dates``: the annualised HAR forecast of the next month's volatility.

    The regression, pooled over members and ``dates`` whose next 21 days had ended before the year began, is refitted
    each January; a year with no such history has no forecast.
    """
    sq = returns.pow(2)
    day = sq.reindex(dates)
    week = sq.rolling(5, min_periods=3).mean().reindex(dates)
    month = sq.rolling(21, min_periods=10).mean().reindex(dates)
    future = sq[::-1].rolling(21, min_periods=10).mean()[::-1].shift(-1).reindex(dates)  # the next 21 days
    days = pd.DatetimeIndex(returns.index)
    last = days.searchsorted(dates) + 21  # the row of each date's 21st day after
    ends = pd.Series(days[np.minimum(last, len(days) - 1)], index=dates).where(last < len(days), pd.Timestamp.max)
    on = members.reindex(index=dates, columns=returns.columns, fill_value=False).astype(bool)
    x = np.stack([day.to_numpy(), week.to_numpy(), month.to_numpy()], axis=-1)
    y = future.to_numpy()
    out = np.full(day.shape, np.nan)
    years = pd.DatetimeIndex(dates).year
    for year in sorted(set(years)):
        start = pd.Timestamp(year=year, month=1, day=1)
        train = (ends < start).to_numpy()[:, None] & on.to_numpy() & np.isfinite(y) & np.isfinite(x).all(axis=-1)
        if train.sum() < 1000:
            continue
        design = np.column_stack([np.ones(train.sum()), x[train]])
        coef, *_ = np.linalg.lstsq(design, y[train], rcond=None)
        rows = years == year
        forecast = coef[0] + x[rows] @ coef[1:]
        out[rows] = np.sqrt(np.clip(forecast, 0.0, None) * YEAR)
    return pd.DataFrame(out, index=dates, columns=returns.columns)


def ffd_weights(d: float, threshold: float = 1e-3) -> np.ndarray:
    """The fixed-width fractional-difference weights: ``w_0 = 1, w_k = -w_{k-1} (d - k + 1) / k``.

    The weights stop at the first whose size is below ``threshold``.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights)


def frac_diff(series: pd.DataFrame, d: float = 0.4, threshold: float = 1e-3) -> pd.DataFrame:
    """``series`` fractionally differenced by ``d`` with fixed-width weights: ``sum_k w_k series[t-k]``.

    Missing where any value of the window is missing (a gap is not filled with a guess).
    """
    w = ffd_weights(d, threshold)
    x = series.to_numpy(dtype=float)
    out = np.zeros_like(x)
    for k, wk in enumerate(w):
        shifted = np.full_like(x, np.nan)
        shifted[k:] = x[: len(x) - k] if k else x
        out = out + wk * shifted
    return pd.DataFrame(out, index=series.index, columns=series.columns)


def kalman_trend(log_price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(slope a year, gap)``: the steady-state Kalman (Holt) filter of a local linear trend in ``log_price``.

    The gap is the log price less its filtered level. A missing price carries the prediction forward.
    """
    y = log_price.to_numpy(dtype=float)
    n, m = y.shape
    level = np.full(m, np.nan)
    slope = np.zeros(m)
    slopes = np.full((n, m), np.nan)
    gaps = np.full((n, m), np.nan)
    for t in range(n):
        seen = np.isfinite(y[t])
        new = seen & np.isnan(level)
        level[new], slope[new] = y[t, new], 0.0
        predicted = level + slope
        update = seen & ~new
        new_level = np.where(update, LEVEL_GAIN * y[t] + (1 - LEVEL_GAIN) * predicted, predicted)
        slope = np.where(update, SLOPE_GAIN * (new_level - level) + (1 - SLOPE_GAIN) * slope, slope)
        level = np.where(new, level, new_level)
        slopes[t] = np.where(seen, slope * YEAR, np.nan)
        gaps[t] = np.where(seen, y[t] - level, np.nan)
    return (
        pd.DataFrame(slopes, index=log_price.index, columns=log_price.columns),
        pd.DataFrame(gaps, index=log_price.index, columns=log_price.columns),
    )


def break_statistic(returns: pd.DataFrame, lags: Sequence[int] = BREAK_LAGS, vol_days: int = YEAR) -> pd.DataFrame:
    """The signed largest standardised move ``(y[t] - y[t-k]) / (sigma sqrt(k))`` over ``k`` in ``lags``.

    ``y`` is the log total-return index and ``sigma`` the daily volatility over the ``vol_days`` before the move.
    """
    growth = pd.DataFrame(np.log1p(returns.fillna(0.0).to_numpy()), index=returns.index, columns=returns.columns)
    y = growth.cumsum().where(returns.notna())
    best = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for k in lags:
        sigma = returns.shift(k).rolling(vol_days, min_periods=vol_days // 2).std()
        z = (y - y.shift(k)) / (sigma * np.sqrt(k))
        better = best.isna() | (z.abs() > best.abs())
        best = best.where(~better, z)
    return best


def market_regime(market: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """On ``dates``: the filtered probability that the market is in its higher-variance state (walk-forward).

    Each January a two-state Markov switching model of daily returns (switching mean and variance) is fitted on every
    earlier day; the year's days are then filtered with those parameters. Years with under two years of history, or
    whose fit fails, are missing.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression  # noqa: PLC0415

    r = market.dropna() * 100
    out = pd.Series(np.nan, index=dates, name="turbulent")
    for year in sorted(set(pd.DatetimeIndex(dates).year)):
        start = pd.Timestamp(year=year, month=1, day=1)
        past = r[r.index < start]
        if len(past) < 2 * YEAR:
            continue
        upto = r[r.index < start + pd.DateOffset(years=1)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = MarkovRegression(past.to_numpy(), k_regimes=2, switching_variance=True).fit(disp=False)
                params = fitted.params
                filtered = MarkovRegression(upto.to_numpy(), k_regimes=2, switching_variance=True).filter(params)
        except (ValueError, np.linalg.LinAlgError):
            continue
        variances = [params[i] for i, name in enumerate(fitted.model.param_names) if name.startswith("sigma2")]
        turbulent = int(np.argmax(variances))
        probs = pd.Series(np.asarray(filtered.filtered_marginal_probabilities)[:, turbulent], index=upto.index)
        year_dates = dates[pd.DatetimeIndex(dates).year == year]
        out.loc[year_dates] = probs.reindex(year_dates, method="ffill").to_numpy()
    return out
