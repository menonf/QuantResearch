"""A learned value score: a model of next year's return fitted only on what was known. Pure functions over panels.

    features = book_features(data.ratios)                          # (date, listing_id) x the book's 18 ratios
    target = forward_returns(data.returns, data.rebalances)        # date x listing_id, the year after each date
    score = learned_value(data.ratios, data.returns, data.members, data.rebalances)

Following Damon Lee, *Build Your Own AI Investor* (2nd ed., chapters 4-6):

1. **Features** - the 18 accounting ratios the book uses (valuation, returns, leverage, liquidity and Altman's four),
   each clipped to the book's bounds (:data:`BOOK_CLIPS`) so a near-zero denominator does not dominate.
2. **Target** - the share's total return over the year after the decision date.
3. **Model** - gradient-boosted regression trees with the book's settings (100 trees, depth 10, learning rate 0.1),
   as scikit-learn's histogram version: the same model family, many times faster, and it takes a missing ratio as
   missing (the book sets missing accounting items to zero).
4. **Distress filter** - the book picks only companies with Altman's Z'' (1995) above 2; here a company at or below
   it scores below every other.

One change is not optional. The book trains on a random half of all company-years and tests on the other half, so
the model learns from years after the ones it then "predicts". Here the model is **walk-forward**: refitted each
January on the decision dates whose year-ahead return was known by then (an expanding window), and used for that
year's dates only. Training uses quarter-end dates: the year-ahead returns of neighbouring months overlap almost
entirely, so the months between add little but time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from equity_research.data.ratios import lookup

#: The book's ratios and the bounds it clips each to (``fixXRatios``), by our ratio names.
BOOK_CLIPS: dict[str, tuple[float, float]] = {
    "EV/EBIT": (-500.0, 500.0),
    "Op. In./(NWC+FA)": (-5.0, 5.0),
    "P/E": (-1000.0, 1000.0),
    "P/B": (-50.0, 100.0),
    "P/S": (0.0, 500.0),
    "Op. In./Interest Expense": (-600.0, 600.0),
    "Working Capital Ratio": (0.0, 30.0),
    "RoE": (-5.0, 5.0),
    "ROCE": (-2.0, 2.0),
    "Debt/Equity": (0.0, 100.0),
    "Debt Ratio": (0.0, 50.0),
    "Cash Ratio": (0.0, 30.0),
    "Asset Turnover": (0.0, 500.0),
    "Gross Profit Margin": (0.0, 1.0),
    "(CA-CL)/TA": (-1.5, 2.0),
    "RE/TA": (-20.0, 2.0),
    "EBIT/TA": (-2.0, 1.0),
    "Book Equity/TL": (-2.0, 20.0),
}
#: Altman's Z'' (1995) at or below this is the distress zone the book leaves out.
DISTRESS_Z = 2.0
#: Months whose decision dates are training rows (quarter ends), and the fewest training dates a model needs.
TRAIN_MONTHS = (3, 6, 9, 12)
MIN_TRAIN_DATES = 8
HORIZON = pd.DateOffset(years=1)


def book_features(ratios: pd.DataFrame) -> pd.DataFrame:
    """``(date, listing_id)`` x the book's 18 ratios, clipped to :data:`BOOK_CLIPS`; a missing ratio stays missing."""
    out = ratios.reindex(columns=list(BOOK_CLIPS)).astype(float)
    for name, (low, high) in BOOK_CLIPS.items():
        out[name] = out[name].clip(low, high)
    return out


def altman_z(ratios: pd.DataFrame) -> pd.Series:
    """Altman's Z'' (1995) per ``(date, listing_id)``: 3.25 + 6.51 (CA-CL)/TA + 3.26 RE/TA + 6.72 EBIT/TA + 1.05 BE/TL.

    Low is close to default: below 1.1 distress, above 2.6 safe.
    """
    return (
        3.25
        + 6.51 * ratios["(CA-CL)/TA"]
        + 3.26 * ratios["RE/TA"]
        + 6.72 * ratios["EBIT/TA"]
        + 1.05 * ratios["Book Equity/TL"]
    )


def forward_returns(returns: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """``date x listing_id``: the total return from each date's close to the last trading day a year later.

    Missing where the year has not ended within ``returns``. A share that stops trading keeps what it had made (a
    missing day counts as no change, as in the backtest), so a delisting is not dropped from what is learned.
    """
    days = pd.DatetimeIndex(returns.index)
    grown = pd.DataFrame(
        np.log1p(returns.fillna(0.0).to_numpy(dtype=float)), index=days, columns=returns.columns
    ).cumsum()
    start = grown.reindex(dates)
    ends = pd.DatetimeIndex([d + HORIZON for d in dates])
    position = days.searchsorted(ends, side="right") - 1  # the last trading day on or before a year later
    known = ends <= days.max()
    end = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
    end.loc[dates[known]] = grown.iloc[position[known]].to_numpy()
    return (end - start).apply(np.expm1)


def default_model() -> Any:
    """The book's gradient-boosted trees: 100 trees of depth 10, learning rate 0.1, a fixed seed."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415 - only research that learns needs it

    return HistGradientBoostingRegressor(max_iter=100, max_depth=10, learning_rate=0.1, random_state=42)


def walk_forward(
    features: pd.DataFrame,
    target: pd.DataFrame,
    members: pd.DataFrame,
    dates: pd.DatetimeIndex,
    model: Callable[[], Any] = default_model,
) -> pd.DataFrame:
    """``date x listing_id`` predictions on ``dates``, each from a model that saw only years already over.

    For each calendar year a model is fitted on the training dates (:data:`TRAIN_MONTHS`) whose year-ahead return had
    ended before the year's first decision date, and predicts that year's members; a year with fewer than
    :data:`MIN_TRAIN_DATES` such dates has no predictions. ``features`` is ``(date, listing_id)`` x features,
    ``target`` and ``members`` are ``date x listing_id``.
    """
    names = list(features.columns)
    rows = features.reset_index()
    rows["y"] = lookup(target, rows["date"], rows["listing_id"])
    rows = rows[lookup(members.astype(float), rows["date"], rows["listing_id"]) == 1.0]  # members on the day only
    out = pd.DataFrame(np.nan, index=dates, columns=members.columns)
    years = pd.Series(dates, index=dates).groupby(dates.year)
    for _, in_year in years:
        first = in_year.min()
        ended = rows["date"] + HORIZON < first  # the year-ahead return was known before this year began
        train = rows[ended & rows["date"].dt.month.isin(TRAIN_MONTHS) & rows["y"].notna()]
        if train["date"].nunique() < MIN_TRAIN_DATES:
            continue
        fitted = model().fit(train[names].to_numpy(dtype=float), train["y"].to_numpy(dtype=float))
        now = rows[rows["date"].isin(in_year)]
        if now.empty:
            continue
        predicted = pd.Series(
            fitted.predict(now[names].to_numpy(dtype=float)),
            index=pd.MultiIndex.from_frame(now[["date", "listing_id"]]),
        )
        wide = predicted.unstack("listing_id")  # noqa: PD010 - long to wide
        out.loc[wide.index, wide.columns] = wide.to_numpy()
    return out


def learned_value(
    ratios: pd.DataFrame,
    returns: pd.DataFrame,
    members: pd.DataFrame,
    dates: pd.DatetimeIndex,
    model: Callable[[], Any] = default_model,
) -> pd.DataFrame:
    """``date x listing_id``: the walk-forward predicted year-ahead return; a company in distress ranks last.

    The book's whole method (see the module docstring); higher is more attractive, like every factor score.
    """
    features = book_features(ratios)
    predicted = walk_forward(features, forward_returns(returns, dates), members, dates, model)
    return distress_last(predicted, ratios)


def distress_last(score: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    """``score`` with every company in Altman's distress zone (Z'' at or below :data:`DISTRESS_Z`) set below all others.

    ``score`` is ``date x listing_id``; a missing score stays missing (no model, no rank to lower).
    """
    z = altman_z(ratios).unstack("listing_id").reindex(index=score.index, columns=score.columns)  # noqa: PD010
    return score.where(~(z <= DISTRESS_Z) | score.isna(), -np.inf)
