"""Rebuild a bond index's monthly return from its constituents, and say how much of it could be rebuilt.

    rebuilt = index_returns(data.weights, data.total_returns)       # per month: return, covered, leavers, bonds
    official = official_returns(data.official)                      # the provider's own, from its month-end levels
    tracking(rebuilt["return"], official, periods_per_year=12)      # research_core's tracking statistics

The index's return over month *t* is the sum, over the bonds it held, of the weight set at the rebalance of *t-1*
times the bond's return over *t*. The files give every bond's weight, but a return only for the bonds still in the
index after the rebalance of *t*: one that left at *t* (matured, called, upgraded, defaulted) has its last month's
return in no file. So each month reports

* ``covered`` - the share of the held weight whose return is known;
* ``return`` - the weighted return of that share, scaled up to the whole (the leavers assumed to have earned the
  same as the rest - the neutral guess; a month of defaults makes it too kind);
* ``leavers`` - the share of weight that left without a return, the size of what that guess is about.

A month whose previous month end has no file cannot be rebuilt and is left out. Pure functions over ``date x
instrument`` panels.
"""

from __future__ import annotations

import pandas as pd


def consecutive(dates: pd.DatetimeIndex) -> pd.Series:
    """Per date: whether the row before it is the month end just before (a missing file breaks the chain)."""
    months = pd.Series(dates.to_period("M"), index=dates)
    return (months - months.shift(1)).map(lambda gap: getattr(gap, "n", None) == 1).fillna(False).astype(bool)


def held_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Per month end *t*: the weights the index held over the month ending *t* (those set at *t-1*)."""
    held = weights.shift(1)
    return held[consecutive(pd.DatetimeIndex(weights.index))]


def index_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """``[return, covered, leavers, bonds]`` per month (see the module docstring)."""
    held = held_weights(weights).fillna(0.0)
    earned = returns.reindex(index=held.index, columns=held.columns)
    known = earned.notna() & (held > 0)
    total = held.sum(axis=1)
    covered_weight = held.where(known, 0.0).sum(axis=1)
    return pd.DataFrame(
        {
            "return": (held * earned.fillna(0.0)).where(known, 0.0).sum(axis=1) / covered_weight,
            "covered": covered_weight / total,
            "leavers": 1.0 - covered_weight / total,
            "bonds": (held > 0).sum(axis=1),
        }
    )


def official_returns(levels: pd.Series) -> pd.Series:
    """The provider's monthly return from its month-end levels (only where the month before is known too)."""
    levels = levels.sort_index()
    change = levels / levels.shift(1) - 1.0
    return change[consecutive(pd.DatetimeIndex(levels.index))].rename("official")


def weighted_average(weights: pd.DataFrame, values: pd.DataFrame) -> pd.Series:
    """Per month end: the index's weighted average of a per-bond measure, over the members that have it."""
    aligned = values.reindex(index=weights.index, columns=weights.columns)
    w = weights.where(aligned.notna(), 0.0).fillna(0.0)
    return (w * aligned.fillna(0.0)).sum(axis=1) / w.sum(axis=1)


def weight_by_group(weights: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """``date x group``: the index's weight in each group (a rating, a sector) on each month end."""
    aligned = groups.reindex(index=weights.index, columns=weights.columns)
    cells = aligned.to_numpy().ravel()
    names = sorted({str(c) for c in cells if pd.notna(c)})
    held = weights.fillna(0.0)
    return pd.DataFrame({name: held.where(aligned == name, 0.0).sum(axis=1) for name in names}).rename_axis(
        index="date"
    )
