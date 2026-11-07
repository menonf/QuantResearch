"""Choosing a learner, third version: version 2 with a target and a yardstick that measure what a portfolio earns.

    features = v2.ranked_features(data)                                   # as version 2: 26 percentiles
    returns = ml.forward_returns(data.returns, data.rebalances)           # what each year paid
    target = trimmed_excess(returns, data.members)                        # learned: the trimmed excess return
    predictions = mc.candidate_predictions(features, target, data.members, data.rebalances, candidates=CANDIDATES)
    spread = top_spreads(predictions, target, data.members)               # judged: what the top 30 % earned
    table = mc.selection_table(spread, dates)                             # chosen each January, as before
    score = flat_to_index(score, data.members, data.market_cap)           # no ranking -> the largest companies

Notebook 5.2 found that version 2's models ranked companies better than chance (rank ICs of +0.04 to +0.06) without
their portfolios earning more. Its percentile target taught them the *typical* outcome: the most volatile fifth of
members had a median year 7.4 points below the typical member's, but a mean 2.2 points above the average member's, and
a portfolio is paid the mean. Three things change, written down before any result of this version was seen; the
features, the settings and the walk-forward training stay version 2's:

1. **Target** (:func:`trimmed_excess`) - the year-ahead return less the date's average member's, after clipping each
   date's returns to their 1st and 99th percentiles (:data:`TRIM`). The average is kept - a soaring few count for
   what they earned - while one +8,000 % year no longer outweighs a thousand companies; and nothing is common to the
   date.
2. **Yardstick** (:func:`top_spread`) - each month, the average target of the model's top :data:`TOP` of members (the
   fraction the portfolio buys) less the members' average: what the portfolio's entry group earned over the year that
   followed. The choice each January uses it; the rank IC is still reported, to compare with versions 1 and 2.
3. **A flat forecast** - a model that predicts one value for every member (the elastic net in version 2, when
   cross-validation found nothing to fit) has no ranking: its yardstick and its IC that month are 0, not missing, so
   the comparison between candidates goes on; and its portfolio holds the largest companies by market value
   (:func:`flat_to_index`), close to the index, instead of cash.

Portfolios are registered in a new family (:data:`FAMILY`); the notebook also deflates by all three versions' trials.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from equity_research.research import ml
from equity_research.research import model_choice as mc
from equity_research.research import model_choice_v2 as v2
from equity_research.research.model_choice import Candidate

#: The registry family of every portfolio of version 3.
FAMILY = "ml_model_choice_v3_lo"
#: Each date's returns are clipped to these lower and upper quantiles before the average is taken off.
TRIM = 0.01
#: The fraction of members the yardstick takes from the top: the portfolio's entry line.
TOP = 0.30
#: Version 2's learners and settings, unchanged: only the target and the yardstick differ.
CANDIDATES: dict[str, Candidate] = dict(v2.CANDIDATES)

PROTOCOL: dict[str, Any] = {
    "version": 3,
    "candidates": {name: c.describe() for name, c in CANDIDATES.items()},
    "features": v2.PROTOCOL["features"],
    "target": f"year-ahead total return clipped to each date's members' {TRIM:.0%}-{1 - TRIM:.0%} quantiles, "
    "less the clipped members' mean",
    "training": v2.PROTOCOL["training"],
    "criterion": f"mean monthly target of the top {TOP:.0%} of members by prediction less the members' mean, out of "
    "sample; a flat forecast scores 0",
    "choice": f"each January the highest mean criterion so far; {mc.FALLBACK} until {mc.MIN_VALIDATION_DATES} common "
    "dates",
    "flat": "a date on which a score is one value for every member holds the largest companies by market value",
    "distress": f"Altman Z'' <= {ml.DISTRESS_Z} ranks last",
}


def trimmed_excess(returns: pd.DataFrame, members: pd.DataFrame, trim: float = TRIM) -> pd.DataFrame:
    """``date x listing_id``: each member's return clipped to the date's ``trim`` quantiles, less their mean.

    Only members count, for the quantiles and for the mean; non-members are missing.
    """
    on = members.reindex(index=returns.index, columns=returns.columns, fill_value=False).astype(bool)
    known = returns.where(on)
    low, high = known.quantile(trim, axis=1), known.quantile(1 - trim, axis=1)
    clipped = known.clip(lower=low, upper=high, axis=0)
    return clipped.sub(clipped.mean(axis=1), axis=0)


def flat(predictions: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Per date: whether the members' predictions are one value for all of them (a forecast that ranks nobody)."""
    on = members.reindex(index=predictions.index, columns=predictions.columns, fill_value=False).astype(bool)
    values = predictions.where(on)
    count = values.notna().sum(axis=1)
    return (count > 0) & (values.max(axis=1) == values.min(axis=1))


def top_spread(predictions: pd.DataFrame, outcome: pd.DataFrame, members: pd.DataFrame, top: float = TOP) -> pd.Series:
    """Per date: the average ``outcome`` of the top ``top`` of members by prediction, less all such members' average.

    Members with both a prediction and an outcome count. 0 where the forecast is flat (:func:`flat`); missing where
    fewer than :data:`model_choice.MIN_NAMES` count.
    """
    on = members.reindex(index=predictions.index, columns=predictions.columns, fill_value=False).astype(bool)
    y = outcome.reindex(index=predictions.index, columns=predictions.columns)
    both = on & predictions.notna() & y.notna()
    pct = predictions.where(both).rank(axis=1, pct=True)
    chosen = pct > 1 - top
    spread = y.where(chosen).mean(axis=1) - y.where(both).mean(axis=1)
    spread = spread.where(~flat(predictions.where(both), on), 0.0)
    return spread.where(both.sum(axis=1) >= mc.MIN_NAMES).rename("top_spread")


def top_spreads(predictions: Mapping[str, pd.DataFrame], outcome: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """``date x candidate``: each candidate's :func:`top_spread`."""
    return pd.DataFrame({name: top_spread(p, outcome, members) for name, p in predictions.items()})


def information_coefficients(
    predictions: Mapping[str, pd.DataFrame], returns: pd.DataFrame, members: pd.DataFrame
) -> pd.DataFrame:
    """``date x candidate``: :func:`model_choice.rank_ic`, 0 instead of missing where a forecast is flat."""
    out = mc.information_coefficients(predictions, returns, members)
    for name, p in predictions.items():
        level = flat(p, members).reindex(out.index, fill_value=False) & out[name].isna()
        out.loc[level, name] = 0.0
    return out


def flat_to_index(score: pd.DataFrame, members: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """``score``, with market value in place of a flat date's: a forecast that ranks nobody holds the largest companies.

    A long-only top-30 % book weighted by market value then holds most of the index's value, where a flat score would
    select nobody and hold cash.
    """
    level = flat(score, members)
    out = score.copy()
    if level.any():
        days = score.index[level.to_numpy()]
        out.loc[days] = market_cap.reindex(index=days, columns=score.columns).to_numpy(dtype=float)
    return out


def quintile_outcomes(feature: pd.DataFrame, outcome: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """By quintile of a percentile ``feature`` (1 lowest): the mean and the median of ``outcome`` over members.

    For seeing what a target rewards: a quintile whose median is low but whose mean is high is one a percentile target
    ranks low and a portfolio is paid for.
    """
    on = members.reindex(index=outcome.index, columns=outcome.columns, fill_value=False).astype(bool)
    f = feature.reindex(index=outcome.index, columns=outcome.columns).where(on)
    quintile = np.ceil(f.to_numpy(dtype=float) * 5).clip(1, 5)
    frame = pd.DataFrame({"quintile": quintile.ravel(), "y": outcome.where(on).to_numpy(dtype=float).ravel()}).dropna()
    return frame.groupby("quintile")["y"].agg(["mean", "median"]).rename_axis("quintile")


def leanings(predictions: Mapping[str, pd.DataFrame], panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """``candidate x feature``: what each model mostly follows.

    The average over dates of the rank correlation across companies between a model's predictions and a feature.
    """
    out = {}
    for name, p in predictions.items():
        ranks = p.rank(axis=1)
        out[name] = {
            f: ranks.corrwith(panel.reindex(index=p.index, columns=p.columns).rank(axis=1), axis=1).mean()
            for f, panel in panels.items()
        }
    return pd.DataFrame(out).T


def feature_panel(features: pd.DataFrame, name: str) -> pd.DataFrame:
    """One feature of a ``(date, listing_id)`` frame as ``date x listing_id``."""
    return features[name].unstack("listing_id")  # noqa: PD010 - long to wide
