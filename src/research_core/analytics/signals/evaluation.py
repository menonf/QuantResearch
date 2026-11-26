"""How good is a signal before any model or portfolio uses it: IC, its inference, decay, quantiles, turnover, triage.

    ic = ic_series(score, fwd, members)                              # per date: rank correlation with what followed
    summary = ic_summary(ic, lags=20)                                # mean, spread, share positive, Newey-West t
    decay = ic_decay(score, data.returns, members, horizons=(1, 5, 21, 63))
    q = quantile_returns(score, fwd, members)                        # per date: the mean outcome of each fifth
    ledger = triage({"momentum": m, "value": v}, fwd, members, lags=20)   # one row per signal, with a verdict

*Machine Learning for Trading* (chapter 7) judges a signal before modelling it, and so do these functions. Every one
takes a ``date x instrument`` score, an outcome of the same shape (a label from ``labels.py``) and the members on
each date; only members with both count.

1. **Information coefficient** - per date, the correlation across members between score and outcome: Spearman's
   (ranks, the default) or Pearson's (values). Missing where fewer than :data:`MIN_NAMES` members count.
2. **Inference** - neighbouring dates' outcomes overlap when the label is longer than the step between dates, so
   their ICs are correlated and the plain t-statistic overstates the evidence. :func:`newey_west_t` uses Bartlett
   weights over ``lags`` dates (the overlap: a 21-day label on daily dates needs 20).
3. **Decay** - the IC of the same score against labels of growing length: how long its information lasts.
4. **Quantiles** - per date, the mean outcome of each fifth (or tenth) of members by score; the top-minus-bottom
   spread is what a long-short book would earn before costs, the top's excess what a long-only one could.
5. **Turnover** - the share of the top quantile replaced from one date to the next, the rank autocorrelation of the
   score, and where each quantile's members are a date later (:func:`quantile_transitions`): a signal that reshuffles
   every day costs more to trade than it may earn.
6. **Many signals at once** - testing many signals finds some by luck. :func:`benjamini_hochberg` keeps the expected
   share of false discoveries among those kept at or below ``q`` (Benjamini and Hochberg, 1995); :func:`triage`
   lists every signal with its IC, t-statistic, p-value, verdict and trading measures - the ledger the book keeps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd

from research_core.analytics.signals.labels import forward_returns

#: A date's IC needs at least this many members with both a score and an outcome.
MIN_NAMES = 30


def _both(
    score: pd.DataFrame, outcome: pd.DataFrame, members: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``score`` and ``outcome`` on the score's dates and columns, kept where a member has both."""
    y = outcome.reindex(index=score.index, columns=score.columns)
    keep = score.notna() & y.notna()
    if members is not None:
        keep &= members.reindex(index=score.index, columns=score.columns, fill_value=False).astype(bool)
    return score.where(keep), y.where(keep), keep


def ic_series(
    score: pd.DataFrame,
    outcome: pd.DataFrame,
    members: pd.DataFrame | None = None,
    *,
    method: str = "spearman",
    min_names: int = MIN_NAMES,
) -> pd.Series:
    """Per date: the correlation across members of ``score`` with ``outcome`` (``"spearman"`` or ``"pearson"``)."""
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', not {method!r}")
    s, y, keep = _both(score, outcome, members)
    if method == "spearman":
        s, y = s.rank(axis=1), y.rank(axis=1)
    ic = s.corrwith(y, axis=1)
    return ic.where(keep.sum(axis=1) >= min_names).rename("ic")


def newey_west_t(values: pd.Series, lags: int) -> float:
    """The t-statistic of the mean of ``values`` with Newey-West (Bartlett) standard errors over ``lags`` lags."""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    variance = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(e[lag:] @ e[:-lag]) / n
    return float(x.mean() / math.sqrt(variance / n)) if variance > 0 else float("nan")


def p_value(t: float) -> float:
    """The two-sided p-value of a t-statistic (normal approximation: these series run to hundreds of dates)."""
    return float("nan") if math.isnan(t) else 2 * (1 - NormalDist().cdf(abs(t)))


def ic_summary(ic: pd.Series | pd.DataFrame, lags: int) -> pd.DataFrame:
    """Per IC series: mean, standard deviation, their ratio, share of dates positive, Newey-West t, dates."""
    frame = ic.to_frame() if isinstance(ic, pd.Series) else ic
    rows = {}
    for name in frame.columns:
        x = frame[name].dropna()
        rows[name] = {
            "mean": x.mean(),
            "sd": x.std(),
            "mean / sd": x.mean() / x.std() if x.std() > 0 else float("nan"),
            "positive": float((x > 0).mean()) if len(x) else float("nan"),
            "t_nw": newey_west_t(x, lags),
            "dates": len(x),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def overlap_lags(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, horizon: int) -> int:
    """How many neighbouring ``dates`` an ``horizon``-day label overlaps: the Newey-West lags it needs."""
    positions = pd.DatetimeIndex(calendar).searchsorted(dates)
    step = float(np.median(np.diff(positions))) if len(positions) > 1 else 1.0
    return max(math.ceil(horizon / max(step, 1.0)) - 1, 0)


def ic_decay(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    members: pd.DataFrame | None = None,
    *,
    horizons: Sequence[int] = (1, 5, 21, 63),
    method: str = "spearman",
) -> pd.DataFrame:
    """Per horizon (trading days): the mean IC of ``score`` against the forward return over it, and its t-statistic."""
    dates = pd.DatetimeIndex(score.index)
    rows = {}
    for h in horizons:
        fwd = forward_returns(returns, h).reindex(dates)
        ic = ic_series(score, fwd, members, method=method)
        lags = overlap_lags(dates, pd.DatetimeIndex(returns.index), h)
        rows[h] = {"mean_ic": ic.mean(), "t_nw": newey_west_t(ic, lags)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("horizon")


def quantile_returns(
    score: pd.DataFrame, outcome: pd.DataFrame, members: pd.DataFrame | None = None, *, quantiles: int = 5
) -> pd.DataFrame:
    """``date x quantile`` (1 lowest score): the mean outcome of each quantile of members by score."""
    s, y, keep = _both(score, outcome, members)
    pct = s.rank(axis=1, pct=True)
    bucket = np.ceil(pct * quantiles).clip(1, quantiles)
    out = {q: y.where(bucket == q).mean(axis=1) for q in range(1, quantiles + 1)}
    return pd.DataFrame(out).where(keep.sum(axis=1) >= MIN_NAMES, axis=0).rename_axis(columns="quantile")


def quantile_spread(table: pd.DataFrame) -> pd.Series:
    """Per date: the top quantile's mean outcome less the bottom's (:func:`quantile_returns`)."""
    return (table[table.columns.max()] - table[table.columns.min()]).rename("spread")


def top_turnover(score: pd.DataFrame, members: pd.DataFrame | None = None, *, top: float = 0.2) -> pd.Series:
    """Per date: the share of the top ``top`` of members by score that was not in it on the previous date."""
    s = score if members is None else score.where(members.reindex_like(score).fillna(False).astype(bool))
    pct = s.rank(axis=1, pct=True)
    held = (pct > 1 - top).to_numpy()
    out = np.full(len(s), np.nan)
    for i in range(1, len(s)):
        now = held[i].sum()
        out[i] = (held[i] & ~held[i - 1]).sum() / now if now else np.nan
    return pd.Series(out, index=score.index, name="turnover")


def quantile_transitions(
    score: pd.DataFrame, members: pd.DataFrame | None = None, *, quantiles: int = 5, horizon: int = 1
) -> pd.DataFrame:
    """``from x to``: of the members in each quantile on a date, the share in each quantile ``horizon`` dates on.

    Rows sum to 1, over the members scored on both dates (one that left in between counts in neither). A heavy
    diagonal is a persistent score - its books change little from one rebalance to the next, so they cost little to
    hold; a flat table is a score that reshuffles every date. Quantile 1 holds the lowest scores, as in
    :func:`quantile_returns`.
    """
    s = score if members is None else score.where(members.reindex_like(score).fillna(False).astype(bool))
    bucket = np.ceil(s.rank(axis=1, pct=True) * quantiles).clip(1, quantiles)
    cells = np.asarray(bucket, dtype=float)
    now, later = cells[:-horizon], cells[horizon:]
    both = ~np.isnan(now) & ~np.isnan(later)
    table = pd.crosstab(now[both].astype(int), later[both].astype(int), normalize="index")
    levels = range(1, quantiles + 1)
    table = table.reindex(index=levels, columns=levels, fill_value=0.0)
    return table.rename_axis(index="from", columns=f"to (+{horizon})")


def rank_autocorrelation(score: pd.DataFrame, members: pd.DataFrame | None = None, *, lag: int = 1) -> pd.Series:
    """Per date: the Spearman correlation of the members' scores with their scores ``lag`` dates earlier."""
    s = score if members is None else score.where(members.reindex_like(score).fillna(False).astype(bool))
    ranks = s.rank(axis=1)
    earlier = s.shift(lag)
    both = ranks.notna() & earlier.notna()
    return ranks.where(both).corrwith(earlier.where(both).rank(axis=1), axis=1).rename("rank_autocorrelation")


def benjamini_hochberg(p_values: pd.Series, q: float = 0.10) -> pd.Series:
    """Which hypotheses are discoveries at false-discovery rate ``q`` (Benjamini-Hochberg step-up)."""
    p = p_values.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(False, index=p_values.index, name="discovery")
    passed = p.to_numpy() <= q * np.arange(1, m + 1) / m
    keep = int(np.max(np.nonzero(passed)[0])) + 1 if passed.any() else 0
    out = pd.Series(False, index=p_values.index, name="discovery")
    out.loc[p.index[:keep]] = True
    return out


def triage(
    scores: Mapping[str, pd.DataFrame],
    outcome: pd.DataFrame,
    members: pd.DataFrame | None = None,
    *,
    lags: int,
    q: float = 0.10,
    top: float = 0.2,
) -> pd.DataFrame:
    """One row per signal: IC and its inference, the false-discovery verdict and how costly it is to trade.

    Columns: ``mean_ic``, ``t_nw``, ``p_value``, ``discovery`` (Benjamini-Hochberg at ``q`` across all the signals
    given), ``spread`` (mean top-minus-bottom fifth outcome), ``turnover`` (mean share of the top ``top`` replaced
    each date) and ``rank_autocorrelation`` (mean, lag 1). Sorted by the absolute t-statistic.
    """
    rows = {}
    for name, score in scores.items():
        ic = ic_series(score, outcome, members)
        t = newey_west_t(ic, lags)
        rows[name] = {
            "mean_ic": ic.mean(),
            "t_nw": t,
            "p_value": p_value(t),
            "spread": quantile_spread(quantile_returns(score, outcome, members)).mean(),
            "turnover": top_turnover(score, members, top=top).mean(),
            "rank_autocorrelation": rank_autocorrelation(score, members).mean(),
        }
    ledger = pd.DataFrame.from_dict(rows, orient="index")
    ledger.insert(3, "discovery", benjamini_hochberg(ledger["p_value"], q))
    return ledger.reindex(ledger["t_nw"].abs().sort_values(ascending=False).index).rename_axis("signal")


def move_against_cost(
    returns: pd.DataFrame,
    cost_bps: pd.DataFrame,
    dates: pd.DatetimeIndex,
    members: pd.DataFrame | None = None,
    *,
    horizons: Sequence[int] = (1, 5, 21),
) -> pd.DataFrame:
    """Per horizon: the typical size of a member's move over it against the cost of a round trip, on ``dates``.

    ``median move`` is the median absolute forward return; ``round trip`` the median of twice the one-way cost; their
    ratio says how many round trips one typical move pays for. A signal must predict a fair part of the move, so a
    horizon whose ratio is near 1 leaves nothing to earn.
    """
    on = None
    if members is not None:
        on = members.reindex(index=dates, columns=returns.columns, fill_value=False).astype(bool).to_numpy()
    cost = cost_bps.reindex(index=dates, columns=returns.columns).to_numpy(dtype=float) * 2 / 10_000
    rows = {}
    for h in horizons:
        move = np.abs(forward_returns(returns, h).reindex(dates).to_numpy(dtype=float))
        keep = np.isfinite(move) & np.isfinite(cost) & (on if on is not None else True)
        rows[h] = {"median move": float(np.median(move[keep])), "round trip": float(np.median(cost[keep]))}
    out = pd.DataFrame.from_dict(rows, orient="index").rename_axis("horizon")
    out["move / cost"] = out["median move"] / out["round trip"]
    return out


def required_ic(information_ratio: float, names: float, periods_per_year: float) -> float:
    """The IC the Fundamental Law of Active Management needs for ``information_ratio``.

    Grinold's law, IR ~ IC x sqrt(breadth), with breadth the independent bets a year: ``names`` x ``periods_per_year``.
    Bets on overlapping horizons or correlated companies are fewer than that, so this is the least IC that could do.
    """
    return information_ratio / math.sqrt(names * periods_per_year)


def rank_correlations(
    rows: Mapping[str, pd.DataFrame],
    columns: Mapping[str, pd.DataFrame] | None = None,
    members: pd.DataFrame | None = None,
    *,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """``row signal x column signal``: their Spearman correlation across members, averaged over the dates they share.

    How much two signals rank companies alike: near 1, one adds little to the other. ``columns`` defaults to
    ``rows`` (a correlation matrix).
    """
    columns = rows if columns is None else columns
    out: dict[str, dict[str, float]] = {
        a: {b: float(ic_series(x, y, members, min_names=min_names).mean()) for b, y in columns.items()}
        for a, x in rows.items()
    }
    return pd.DataFrame(out).T


def ic_flat_zero(
    score: pd.DataFrame,
    outcome: pd.DataFrame,
    members: pd.DataFrame | None = None,
    *,
    method: str = "spearman",
    min_names: int = MIN_NAMES,
) -> pd.Series:
    """:func:`ic_series`, with 0 on a date where the score is one value for every member that has an outcome.

    A forecast that ranks nobody has no skill, and scoring it 0 rather than missing stops it being judged only on the
    dates where it did rank - which flatters, when settings are tuned, a penalty strong enough to zero every weight.
    """
    ic = ic_series(score, outcome, members, method=method, min_names=min_names)
    s, _, keep = _both(score, outcome, members)
    level = (s.max(axis=1) == s.min(axis=1)) & (keep.sum(axis=1) >= min_names)
    return ic.mask(level & ic.isna(), 0.0)
