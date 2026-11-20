"""What a model is asked to predict: labels on daily ``date x instrument`` panels. Pure functions.

    fwd = forward_returns(data.returns, 21)                       # each day: the return over the next 21 days
    month = period_returns(data.returns, data.rebalances)         # each decision date: the return to the next one
    tb = triple_barrier(data.returns, horizon=21, width=1.0)      # which of three barriers each day's trade hit first
    ts = trend_scan(data.returns, spans=range(5, 22))             # the sign and strength of the coming trend

Labels look forward: a label on day ``t`` uses returns after ``t``, so a model may only be fitted on labels whose
window has ended before the day it predicts (``validation.py`` purges the rest). Conventions shared by every label:

* **Total returns** - ``returns`` are daily total returns (dividends in); a price path is their compounded index.
* **A listing that stops trading keeps what it made** - a missing day inside a window counts as no change, as in the
  backtest, so a delisting is not dropped from what is learned. A day on which the listing has no return of its own
  (before it lists, after it delists) has no label.
* **The window must exist** - a label whose window runs past the last day of ``returns`` is missing, not short.

The three families follow the book (*Machine Learning for Trading*, chapter 7) and the papers it cites:

1. **Forward returns** - the return over a fixed number of trading days (1, 5, 21) or to the next decision date.
2. **Triple barrier** (Lopez de Prado, 2018, chapter 3) - a trade opened on day ``t`` closes at the first of: the
   return reaching ``+width`` times the expected move (profit), reaching ``-width`` times it (loss), or ``horizon``
   days passing. The expected move is the trailing daily volatility times the square root of the horizon, so barriers
   are as wide for a volatile share as its usual swings. Label +1, -1, or the sign of the return at the deadline.
3. **Trend scanning** (Lopez de Prado, 2019) - for each span ``L`` the slope of the log price over the next ``L``
   days is fitted by least squares; the span whose slope has the largest absolute t-statistic is kept, and the label
   is that t-statistic (its sign is the trend's direction, its size the trend's clarity).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The book's daily horizons (``fwd_ret_1d``, ``fwd_ret_5d``, ``fwd_ret_21d``).
HORIZONS = (1, 5, 21)
#: Trailing days of the volatility that sizes the triple barrier.
VOL_DAYS = 63
#: The largest t-statistic a trend scan reports (a straight line has none: its residuals are zero).
MAX_T = 100.0


def _active(returns: pd.DataFrame) -> np.ndarray:
    """Whether each listing has a return of its own on each day."""
    return returns.notna().to_numpy()


def _log_growth(returns: pd.DataFrame) -> np.ndarray:
    """Daily log growth, a missing day counting as no change."""
    return np.log1p(returns.fillna(0.0).to_numpy(dtype=float))


def forward_returns(returns: pd.DataFrame, days: int) -> pd.DataFrame:
    """``date x instrument``: the total return from each day's close to the close ``days`` trading days later."""
    if days < 1:
        raise ValueError("days must be at least 1")
    growth = np.vstack([_log_growth(returns), np.zeros((1, returns.shape[1]))])
    level = np.cumsum(growth[::-1], axis=0)[::-1]  # level[i] = sum of growth from row i to the end
    n = len(returns)
    out = np.full(returns.shape, np.nan)
    if n > days:
        # the window of day t is rows t+1 .. t+days
        out[: n - days] = np.expm1(level[1 : n - days + 1] - level[days + 1 : n + 1])
    out[~_active(returns)] = np.nan
    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def period_returns(returns: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """``date x instrument`` on ``dates``: the total return from each date's close to the next date's close.

    The last date has no next one, so its label is missing; so is a listing with no return of its own on the date.
    """
    days = pd.DatetimeIndex(returns.index)
    growth = pd.DataFrame(_log_growth(returns), index=days, columns=returns.columns).cumsum()
    at = growth.reindex(dates, method="ffill")
    out = pd.DataFrame(np.expm1((at.shift(-1) - at).to_numpy()), index=at.index, columns=at.columns)
    active = returns.notna().reindex(dates, fill_value=False).to_numpy()
    return out.where(active)


@dataclass(frozen=True)
class TripleBarrier:
    """Triple-barrier labels: each a ``date x instrument`` panel on the days of ``returns``."""

    label: pd.DataFrame  # +1 profit barrier first, -1 loss barrier first, else the sign of the return at the deadline
    ret: pd.DataFrame  # the return when the trade closed
    days: pd.DataFrame  # trading days the trade was open
    barrier: pd.DataFrame  # the barrier's distance (a return) on the day the trade opened
    touch: pd.DataFrame  # how it closed: +1 the profit barrier, -1 the loss barrier, 0 the deadline


def triple_barrier(
    returns: pd.DataFrame, *, horizon: int = 21, width: float = 1.0, vol_days: int = VOL_DAYS
) -> TripleBarrier:
    """The triple-barrier label of a trade opened at each day's close (module docstring, item 2).

    The barrier is ``width`` times the trailing ``vol_days`` volatility of daily returns (known at the close of the
    day, so no look-ahead) times the square root of ``horizon``. A trade whose deadline lies past the data, or opened
    before ``vol_days // 2`` returns exist, has no label.
    """
    growth = _log_growth(returns)
    n, m = growth.shape
    sigma = returns.rolling(vol_days, min_periods=vol_days // 2).std().to_numpy(dtype=float)
    up = np.log1p(width * sigma * np.sqrt(horizon))
    down = np.log1p(np.maximum(-width * sigma * np.sqrt(horizon), -0.999))
    label = np.full((n, m), np.nan)
    ret = np.full((n, m), np.nan)
    held = np.full((n, m), np.nan)
    touch = np.full((n, m), np.nan)
    open_ = np.isfinite(sigma) & _active(returns)
    open_[max(n - horizon, 0) :] = False  # the deadline lies past the data
    path = np.zeros((n, m))
    for h in range(1, horizon + 1):
        path[: n - h] += growth[h:]  # day t's trade, h days on
        undecided = open_ & np.isnan(label)
        hit_up = undecided & (path >= up)
        hit_down = undecided & (path <= down) & ~hit_up
        for hit, sign in ((hit_up, 1.0), (hit_down, -1.0)):
            label[hit], ret[hit], held[hit], touch[hit] = sign, np.expm1(path[hit]), h, sign
    deadline = open_ & np.isnan(label)
    label[deadline] = np.sign(path[deadline])
    ret[deadline] = np.expm1(path[deadline])
    held[deadline] = horizon
    touch[deadline] = 0.0

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=returns.index, columns=returns.columns)

    barrier = np.where(open_, np.expm1(up), np.nan)
    return TripleBarrier(frame(label), frame(ret), frame(held), frame(barrier), frame(touch))


@dataclass(frozen=True)
class TrendScan:
    """Trend-scanning labels: each a ``date x instrument`` panel on the days of ``returns``."""

    t_value: pd.DataFrame  # the t-statistic of the clearest forward trend (its sign: up or down)
    span: pd.DataFrame  # the number of days of that trend


def trend_scan(returns: pd.DataFrame, spans: Iterable[int] = range(5, 22)) -> TrendScan:
    """The clearest trend of the log total-return index over the next ``L`` days, ``L`` in ``spans``.

    For each span, the least-squares slope of ``log index[t .. t+L]`` on ``0 .. L`` and its t-statistic, computed for
    every day at once from running sums; the span with the largest absolute t-statistic wins. A window running past the
    data, or a listing with no return of its own on the day, has no label. A perfectly straight path has no residual
    error, so its t-statistic is capped at :data:`MAX_T`.
    """
    y = np.cumsum(_log_growth(returns), axis=0)  # the log total-return index
    n, m = y.shape
    rows = np.arange(n, dtype=float)[:, None]
    c1 = np.vstack([np.zeros((1, m)), np.cumsum(y, axis=0)])  # running sums: window [t, t+L] = c[t+L+1] - c[t]
    c2 = np.vstack([np.zeros((1, m)), np.cumsum(y * y, axis=0)])
    cj = np.vstack([np.zeros((1, m)), np.cumsum(rows * y, axis=0)])
    best_t = np.full((n, m), np.nan)
    best_span = np.full((n, m), np.nan)
    for span in spans:
        k = span + 1  # points in the window
        if n < k or span < 2:
            continue
        last = n - span  # windows starting at rows 0 .. last-1
        start = rows[:last]
        sy = c1[span + 1 : span + 1 + last] - c1[:last]
        syy = c2[span + 1 : span + 1 + last] - c2[:last]
        sjy = cj[span + 1 : span + 1 + last] - cj[:last]
        sxy = sjy - start * sy  # x = j - t runs 0 .. span
        sx = span * k / 2.0
        sxx = span * k * (2 * span + 1) / 6.0
        vx = sxx - sx * sx / k
        slope = (sxy - sx * sy / k) / vx
        resid = np.maximum(syy - sy * sy / k - slope * slope * vx, 0.0)
        se = np.sqrt(resid / (k - 2) / vx)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, slope / se, np.sign(slope) * MAX_T)
        t = np.clip(t, -MAX_T, MAX_T)
        current = best_t[:last]
        better = np.isnan(current) | (np.abs(t) > np.abs(current))
        best_t[:last] = np.where(better, t, current)
        best_span[:last] = np.where(better, span, best_span[:last])
    inactive = ~_active(returns)
    best_t[inactive] = np.nan
    best_span[inactive] = np.nan
    return TrendScan(
        pd.DataFrame(best_t, index=returns.index, columns=returns.columns),
        pd.DataFrame(best_span, index=returns.index, columns=returns.columns),
    )


def label_summary(panels: Mapping[str, pd.DataFrame], members: pd.DataFrame) -> pd.DataFrame:
    """One row per label: the share of member-days it exists, and its mean, spread and tails over members."""
    rows = {}
    for name, panel in panels.items():
        on = members.reindex(index=panel.index, columns=panel.columns, fill_value=False).astype(bool).to_numpy()
        values = panel.to_numpy(dtype=float)[on]
        known = values[np.isfinite(values)]
        rows[name] = {
            "coverage": known.size / max(values.size, 1),
            "mean": float(known.mean()) if known.size else np.nan,
            "sd": float(known.std()) if known.size else np.nan,
            "1st percentile": float(np.percentile(known, 1)) if known.size else np.nan,
            "99th percentile": float(np.percentile(known, 99)) if known.size else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def barrier_summary(tb: TripleBarrier, members: pd.DataFrame) -> pd.Series:
    """How the trades of member-days closed: the share at each barrier and at the deadline, and the days open."""
    on = members.reindex(index=tb.label.index, columns=tb.label.columns, fill_value=False).astype(bool).to_numpy()
    touch, days = tb.touch.to_numpy()[on], tb.days.to_numpy()[on]
    known = np.isfinite(touch)
    touch, days = touch[known], days[known]
    return pd.Series(
        {
            "profit barrier": float((touch == 1).mean()),
            "loss barrier": float((touch == -1).mean()),
            "deadline": float((touch == 0).mean()),
            "days open": float(days.mean()),
        }
    )


def trimmed_excess(returns: pd.DataFrame, members: pd.DataFrame, trim: float = 0.01) -> pd.DataFrame:
    """``date x instrument``: each member's return clipped to the date's ``trim`` quantiles, less their mean.

    The excess over the average member keeps what a portfolio is paid (the mean, soaring few included) while the
    clipping stops one extreme company from dominating a least-squares fit; nothing is common to the date. Only
    members count, for the quantiles and for the mean; non-members are missing.
    """
    on = members.reindex(index=returns.index, columns=returns.columns, fill_value=False).astype(bool)
    known = returns.where(on)
    low, high = known.quantile(trim, axis=1), known.quantile(1 - trim, axis=1)
    clipped = known.clip(lower=low, upper=high, axis=0)
    return clipped.sub(clipped.mean(axis=1), axis=0)
