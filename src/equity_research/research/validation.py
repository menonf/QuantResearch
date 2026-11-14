"""Which dates a model may learn from when it is tested on others: walk-forward folds, purged and embargoed.

    ends = label_ends(dates, data.returns.index, 21)          # when each date's 21-day label is known
    folds = walk_forward(dates, ends, train_years=10)         # test each year; train on the ten before it, purged
    inner = purged_kfold(folds[0].train, ends, k=5)           # tuning inside one fold's training dates only
    table = describe(folds)                                    # the folds as a table

A label looks forward, so a training date whose label window reaches into the test period has seen part of what is
being tested: its label was not known on the first test day. *Machine Learning for Trading* (chapter 6) and
Lopez de Prado (2018, chapter 7) remove such dates:

1. **Label ends** - each decision date is mapped to the day its label is complete (:func:`label_ends`): ``h`` trading
   days later for an ``h``-day label, the next decision date for a period label.
2. **Walk-forward folds** (:func:`walk_forward`) - each calendar year from ``first_test`` is a test set; its training
   set is every earlier date (``train_years=None``, an expanding window) or those of the ``train_years`` before it,
   **purged** of dates whose label ends on or after the test's first day. Nothing after a test year trains for it.
3. **Purged k-fold** (:func:`purged_kfold`) - for tuning inside a training set: consecutive blocks of dates, each in
   turn the validation block; training dates whose label window overlaps the block's are purged, and those in an
   **embargo** just after the block are dropped too, because serial correlation lets a date right after the block
   carry information about it.

Folds hold decision dates only; the rows of a date (its companies) go wherever the date goes, so one company's
neighbour on the same day is never on the other side of the split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One split: the dates fitted on, the dates tested, and the training dates purged to keep them apart."""

    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purged: pd.DatetimeIndex


def label_ends(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """``date -> day its label is complete``: ``horizon`` trading days of ``calendar`` after each date.

    A date whose label would end beyond ``calendar`` ends at ``Timestamp.max`` (never known within the data).
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    position = cal.searchsorted(dates)
    target = position + horizon
    ends = [cal[int(i)] if i < len(cal) else pd.Timestamp.max for i in target]
    return pd.Series(ends, index=dates, name="label_end")


def next_date_ends(dates: pd.DatetimeIndex) -> pd.Series:
    """``date -> the next decision date``: when a period label (to the next decision date) is complete."""
    d = pd.DatetimeIndex(dates).sort_values()
    return pd.Series([*d[1:], pd.Timestamp.max], index=d, name="label_end")


def walk_forward(
    dates: pd.DatetimeIndex,
    ends: pd.Series,
    *,
    train_years: int | None = 10,
    first_test: int | None = None,
    last_test: int | None = None,
    min_train: int = 1,
) -> list[Fold]:
    """One fold per calendar year from ``first_test`` to ``last_test``: test on the year, train before it, purged.

    ``train_years=None`` trains on every earlier date (expanding); otherwise on the ``train_years`` calendar years
    before the test year. A year with fewer than ``min_train`` training dates has no fold.
    """
    d = pd.DatetimeIndex(dates).sort_values()
    years = sorted(set(d.year))
    first = first_test if first_test is not None else years[0] + (train_years or 1)
    last = last_test if last_test is not None else years[-1]
    end_of = ends.reindex(d)
    folds = []
    for year in range(first, last + 1):
        test = d[d.year == year]
        if test.empty:
            continue
        start = test.min()
        before = d[d < start]
        if train_years is not None:
            before = before[before.year >= year - train_years]
        known = end_of.reindex(before).to_numpy() < start.to_datetime64()
        train, purged = before[known], before[~known]
        if len(train) >= min_train:
            folds.append(Fold(train, test, purged))
    return folds


def purged_kfold(dates: pd.DatetimeIndex, ends: pd.Series, *, k: int = 5, embargo: int = 0) -> list[Fold]:
    """``k`` consecutive blocks of ``dates``, each the test once; training purged and embargoed around it.

    A training date is purged when its label window ``[date, label end]`` overlaps the block's ``[first date, last
    label end]``; the ``embargo`` dates right after the block are dropped as well.
    """
    d = pd.DatetimeIndex(dates).sort_values()
    if k < 2 or k > len(d):
        raise ValueError(f"k must be between 2 and the number of dates ({len(d)})")
    end_of = ends.reindex(d).to_numpy()
    blocks = np.array_split(np.arange(len(d)), k)
    folds = []
    for block in blocks:
        test = d[block]
        lo, hi = test.min().to_datetime64(), end_of[block].max()
        overlap = (d.to_numpy() <= hi) & (end_of >= lo)
        after = np.zeros(len(d), dtype=bool)
        after[block[-1] + 1 : block[-1] + 1 + embargo] = True
        out = np.zeros(len(d), dtype=bool)
        out[block] = True
        train_mask = ~out & ~overlap & ~after
        folds.append(Fold(d[train_mask], test, d[~out & (overlap | after)]))
    return folds


def describe(folds: list[Fold]) -> pd.DataFrame:
    """One row per fold: training and test spans, their sizes and how many training dates were purged."""
    rows = [
        {
            "train from": f.train.min(),
            "train to": f.train.max(),
            "train dates": len(f.train),
            "purged": len(f.purged),
            "test from": f.test.min(),
            "test to": f.test.max(),
            "test dates": len(f.test),
        }
        for f in folds
    ]
    return pd.DataFrame(rows, index=pd.RangeIndex(1, len(folds) + 1, name="fold"))
