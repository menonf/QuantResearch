"""Choosing a learner, second version: :mod:`model_choice`'s protocol with the learning task repaired (notebook 5.2).

    features = ranked_features(data)                                  # (date, listing_id) x 26 percentile ranks
    returns = ml.forward_returns(data.returns, data.rebalances)       # what each year paid: the IC is judged on it
    target = ranked_target(returns, data.members)                     # what is learned: each year's percentile
    predictions = mc.candidate_predictions(features, target, data.members, data.rebalances, candidates=CANDIDATES)
    ic = mc.information_coefficients(predictions, returns, data.members)   # then the choice exactly as version 1

Notebook 5.1 found that none of the book's eight learners ranked S&P 1500 companies better than chance, and a
diagnosis found four reasons. Each is changed here, and every setting was written down before any result of this
version was seen:

1. **Target** - year-ahead returns run from -77 % to +153 % between their 1st and 99th percentiles (the largest
   +8,364 %), so a least-squares fit is driven by a few extreme companies, and part of what it learns is which year
   was good - which a ranking cannot use. Here the target is each company's percentile among the date's members
   (:func:`ranked_target`): bounded, the same average every date, and the quantity the rank IC judges.
2. **Features** - ratios drift (the median price-to-sales rose from 1.15 in 2008 to 2.08 in 2024), so a split
   learned on old levels means something else later. Every feature is a percentile among the date's members.
3. **Flexibility** - trees of depth 10 and 20 memorise noise when the signal is this weak. The settings
   (:data:`CANDIDATES`) follow what works on stock returns (Gu, Kelly and Xiu, 2020): shallow trees, leaves of at
   least :data:`MIN_LEAF` company-dates, many averaged trees, slow boosting of depth-2 trees, a wide neighbourhood, a
   soft support-vector fit. The elastic net's penalty has no scale-free fixed value, so it is chosen inside each
   training window on five consecutive blocks of dates - never on the year predicted.
4. **Information** - the book's 18 ratios are mostly valuation and balance-sheet levels. :data:`EXTRA_FEATURES` adds
   price trend and risk (12-1 month momentum, the last month's return, a year's volatility), size, share issuance,
   the latest earnings surprise and growth of earnings and of sales per share.

The fast versions also make the study quick: histogram boosting for the book's exact boosting, a gentler SVR.
Portfolios are registered in a new family (:data:`FAMILY`). This version was designed after version 1's results, so
it is another try at the same question: the notebook deflates its Sharpe ratios by both families' trials as well.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from equity_research.research import factors as fx
from equity_research.research import ml
from equity_research.research.data import UniverseData
from equity_research.research.model_choice import FALLBACK, MIN_VALIDATION_DATES, SEED, Candidate

#: The registry family of every portfolio of version 2.
FAMILY = "ml_model_choice_v2_lo"
#: The fewest training company-dates a tree leaf may hold: about 5 % of the first training window (8 quarter-ends of
#: some 1,400 members), so no leaf rests on a handful of companies.
MIN_LEAF = 500
#: Features beyond the book's 18 ratios: price trend and risk, size, issuance, surprise and growth.
EXTRA_FEATURES = (
    "momentum_12_1",
    "return_1m",
    "volatility_1y",
    "size",
    "share_change",
    "earnings_surprise",
    "eps_growth",
    "sales_growth",
)
_TREE = (("min_samples_leaf", MIN_LEAF), ("random_state", SEED))

#: The eight learners again, with settings for a weak signal, fixed before any result of this version.
CANDIDATES: dict[str, Candidate] = {
    "linear": Candidate("Linear regression", "LinearRegression"),
    "elastic_net": Candidate("Elastic net", "ElasticNetCV", (("l1_ratio", (0.1, 0.5, 0.9, 1.0)), ("cv", 5))),
    "knn": Candidate("k-nearest neighbours", "KNeighborsRegressor", (("n_neighbors", 500),)),
    "svr": Candidate(
        "Support-vector regression",
        "SVR",
        (("kernel", "rbf"), ("C", 1.0), ("gamma", "scale"), ("epsilon", 0.1)),
        sample=10_000,
    ),
    "decision_tree": Candidate("Decision tree", "DecisionTreeRegressor", (("max_depth", 4), *_TREE)),
    "random_forest": Candidate(
        "Random forest",
        "RandomForestRegressor",
        (("n_estimators", 300), ("max_depth", 6), ("max_features", 0.33), *_TREE),
    ),
    "extra_trees": Candidate(
        "Extra trees",
        "ExtraTreesRegressor",
        (("n_estimators", 300), ("max_depth", 6), ("max_features", 0.33), *_TREE),
    ),
    "gradient_boosting": Candidate(
        "Gradient-boosted trees",
        "HistGradientBoostingRegressor",
        (
            ("max_iter", 300),
            ("learning_rate", 0.05),
            ("max_depth", 2),
            ("l2_regularization", 1.0),
            ("early_stopping", False),
            *_TREE,
        ),
    ),
}

#: Version 2's protocol: what changed from :data:`model_choice.PROTOCOL`, and what did not.
PROTOCOL: dict[str, Any] = {
    "version": 2,
    "candidates": {name: c.describe() for name, c in CANDIDATES.items()},
    "features": f"ml.BOOK_CLIPS + {list(EXTRA_FEATURES)}, each a percentile among the date's members; "
    "missing -> training median",
    "target": "percentile among the date's members of the total return over the year after each month end",
    "training": f"walk-forward, refitted each January, quarter ends, at least {ml.MIN_TRAIN_DATES} dates",
    "criterion": "mean monthly rank IC over members against the raw year-ahead return, out of sample",
    "choice": f"each January the highest mean IC so far; {FALLBACK} until {MIN_VALIDATION_DATES} common dates",
    "distress": f"Altman Z'' <= {ml.DISTRESS_Z} ranks last",
}


def member_ranks(panel: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """``date x listing_id``: each value's percentile among the date's members (missing off members or unknown).

    A percentile means the same thing every year - "cheaper than 80 % of the index" - where a level (a P/S of 1.5)
    drifts with the market.
    """
    on = members.reindex(index=panel.index, columns=panel.columns, fill_value=False).astype(bool)
    return panel.where(on).rank(axis=1, pct=True)


def ranked_target(returns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """``date x listing_id``: each member's percentile by its year-ahead return - the target version 2 learns.

    Bounded (no company's +8,000 % year dominates a least-squares fit) and centred the same every date (nothing to
    learn about which year was good), it is what the rank IC judges a ranking by.
    """
    return member_ranks(returns, members)


def extra_panels(data: UniverseData) -> dict[str, pd.DataFrame]:
    """:data:`EXTRA_FEATURES` as raw ``rebalance date x listing_id`` panels (each known on its date)."""
    raw = {
        "momentum_12_1": fx.momentum(data.adjusted),
        "return_1m": data.adjusted / data.adjusted.shift(21) - 1.0,
        "volatility_1y": -fx.low_volatility(data.returns),
        "size": data.market_cap,
        "share_change": data.share_change,
        "earnings_surprise": data.earnings_surprise,
        "eps_growth": data.eps_growth,
        "sales_growth": data.sales_growth,
    }
    return {name: raw[name].reindex(index=data.rebalances, columns=data.members.columns) for name in EXTRA_FEATURES}


def ranked_features(data: UniverseData) -> pd.DataFrame:
    """``(date, listing_id) x feature`` on rebalance dates, members only, each a percentile among the date's members.

    The features are the book's 18 ratios (clipped as in :func:`ml.book_features`) and :data:`EXTRA_FEATURES`.
    """
    members = data.members.reindex(index=data.rebalances, columns=data.members.columns, fill_value=False).astype(bool)
    book = ml.book_features(data.ratios)
    panels = {
        name: book[name].unstack("listing_id").reindex(index=data.rebalances, columns=members.columns)  # noqa: PD010
        for name in book.columns
    } | extra_panels(data)
    rows, cols = np.nonzero(members.to_numpy())
    index = pd.MultiIndex.from_arrays([members.index[rows], members.columns[cols]], names=["date", "listing_id"])
    ranks = {name: member_ranks(panel, members).to_numpy(dtype=float)[rows, cols] for name, panel in panels.items()}
    return pd.DataFrame(ranks, index=index)


def date_share(panel: pd.DataFrame, members: pd.DataFrame) -> float:
    """The share of ``panel``'s variance (over members) that is between dates rather than across companies on a date.

    For a target or a model's predictions: what is common to all companies on a date is the market's year, which a
    ranking of companies cannot use.
    """
    on = members.reindex(index=panel.index, columns=panel.columns, fill_value=False).astype(bool)
    values = panel.where(on).to_numpy(dtype=float)
    known = np.isfinite(values)
    counts = known.sum(axis=1)
    means = np.where(counts > 0, np.where(known, values, 0.0).sum(axis=1) / np.maximum(counts, 1), np.nan)
    between = np.broadcast_to(means[:, None], values.shape)[known]
    total = values[known]
    return float(np.var(between) / np.var(total)) if total.size > 1 and np.var(total) > 0 else float("nan")
