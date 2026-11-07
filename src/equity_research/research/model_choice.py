"""Choosing among the book's eight learners without hindsight: candidates fixed in advance, chosen walk-forward.

    predictions = candidate_predictions(features, target, members, dates)   # name -> date x listing, walk-forward
    ic = information_coefficients(predictions, target, members)             # date x name: rank IC of each date
    table = selection_table(ic, dates)                                      # each January: what was known, chosen
    score = chosen_score(predictions, table["chosen"], members)            # each year from the model chosen then
    result = run_score(data, config, score, protocol=PROTOCOL, engine=engine)

Damon Lee's *Build Your Own AI Investor* (2nd ed.) tries eight scikit-learn regressors, tunes each on random
train/test splits of all company-years (chapter 4), backtests the survivors on many random 50/50 splits (chapter 6)
and keeps the one whose backtests looked best (chapter 7: gradient-boosted trees, k-nearest neighbours a close
second). Every step of that choice saw the years it was judged on. Here:

1. **Pre-registered candidates** (:data:`CANDIDATES`) - the book's eight models with the settings its chapter 6
   trains them with, fixed before any result on our data was seen, never tuned here. Missing ratios take the training
   set's median (the book set missing accounting items to zero); the models the book power-transforms are
   power-transformed. Support-vector regression is fitted on a random 10,000 training rows (the book's backtests
   trained on about 7,800; its cost grows with the square of the rows).
2. **Walk-forward predictions** - each candidate through :func:`ml.walk_forward`: refitted every January on the
   quarter-end dates whose year-ahead return was already known, predicting only that year.
3. **A pre-registered criterion** - the rank information coefficient (IC): each month, the Spearman correlation
   between a model's predictions and the year-ahead returns that followed, across the index's members.
4. **Walk-forward choice** - each January, the candidate with the highest mean IC over the dates whose returns were
   already known (an expanding window) is used for that year. Until :data:`MIN_VALIDATION_DATES` such dates exist
   there is no evidence to choose by, so the year uses the average of all eight models' ranks (:func:`ensemble`).
5. **Every portfolio registered** - each candidate's own portfolio, the walk-forward choice and the ensemble are runs
   of one new family (:data:`FAMILY`), so the deflated Sharpe ratio counts every variant looked at.

The predictions are kept under ``data/<db>/research/ml_candidates/`` keyed by the candidate's settings and a digest
of the inputs: the book's exact gradient boosting takes about two minutes a fit, and a rerun on the same data reuses
them.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from equity_research.research import factors as fx
from equity_research.research import ml
from equity_research.research.backtest import run_backtest
from equity_research.research.data import UniverseData
from equity_research.research.registry import config_hash, fence
from equity_research.research.strategy import StrategyConfig, StrategyResult, finish_run, strategy_book
from equity_research.settings import get_settings

#: The registry family of every portfolio this study runs (new, so no existing trial count moves).
FAMILY = "ml_model_choice_lo"
#: The score used while no candidate has enough out-of-sample evidence to be chosen.
FALLBACK = "ensemble"
#: The fewest out-of-sample monthly ICs (common to all candidates) a choice is made on: one year's.
MIN_VALIDATION_DATES = 12
#: A date's IC needs at least this many members with both a prediction and a known return.
MIN_NAMES = 30
#: Lags of the Newey-West variance: monthly dates whose year-ahead returns overlap by up to eleven months.
HAC_LAGS = 11
SEED = 42


@dataclass(frozen=True)
class Candidate:
    """One pre-registered learner: a scikit-learn regressor, its settings and its preprocessing."""

    label: str
    estimator: str
    settings: tuple[tuple[str, Any], ...] = ()
    power: bool = False  # Yeo-Johnson power transform and standardisation first, as the book does for this model
    sample: int | None = None  # fit on at most this many random training rows

    def build(self) -> Any:
        """A fresh, unfitted pipeline: median imputation, the power transform if any, the regressor."""
        from sklearn.impute import SimpleImputer  # noqa: PLC0415 - only research that learns needs scikit-learn
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import PowerTransformer  # noqa: PLC0415

        steps: list[Any] = [SimpleImputer(strategy="median")]
        if self.power:
            steps.append(PowerTransformer())
        steps.append(_estimator(self.estimator)(**dict(self.settings), **_speed(self.estimator)))
        model = make_pipeline(*steps)
        return model if self.sample is None else Sampled(model, self.sample)

    def describe(self) -> dict[str, Any]:
        """What identifies the candidate in the registry and in its stored predictions."""
        return {"estimator": self.estimator, **dict(self.settings), "power": self.power, "sample": self.sample}


#: The book's eight regressors with the settings its chapter 6 ``train...Model`` functions use.
CANDIDATES: dict[str, Candidate] = {
    "linear": Candidate("Linear regression", "LinearRegression", power=True),
    "elastic_net": Candidate("Elastic net", "ElasticNet", (("l1_ratio", 0.00001),), power=True),
    "knn": Candidate("k-nearest neighbours", "KNeighborsRegressor", (("n_neighbors", 40),), power=True),
    "svr": Candidate(
        "Support-vector regression",
        "SVR",
        (("kernel", "rbf"), ("C", 100.0), ("gamma", 0.1), ("epsilon", 0.1)),
        power=True,
        sample=10_000,
    ),
    "decision_tree": Candidate("Decision tree", "DecisionTreeRegressor", (("max_depth", 20), ("random_state", SEED))),
    "random_forest": Candidate("Random forest", "RandomForestRegressor", (("max_depth", 10), ("random_state", SEED))),
    "extra_trees": Candidate("Extra trees", "ExtraTreesRegressor", (("max_depth", 10), ("random_state", SEED))),
    "gradient_boosting": Candidate(
        "Gradient-boosted trees",
        "GradientBoostingRegressor",
        (("n_estimators", 100), ("learning_rate", 0.1), ("max_depth", 10), ("random_state", SEED)),
    ),
}

#: The whole protocol, written down before the first result: it goes into every registered run's config.
PROTOCOL: dict[str, Any] = {
    "candidates": {name: c.describe() for name, c in CANDIDATES.items()},
    "features": "ml.BOOK_CLIPS, clipped; missing -> training median",
    "target": "total return over the year after each month end",
    "training": f"walk-forward, refitted each January, quarter ends, at least {ml.MIN_TRAIN_DATES} dates",
    "criterion": "mean monthly rank IC over members, out of sample",
    "choice": f"each January the highest mean IC so far; {FALLBACK} until {MIN_VALIDATION_DATES} common dates",
    "distress": f"Altman Z'' <= {ml.DISTRESS_Z} ranks last",
}


def _estimator(name: str) -> type[Any]:
    """The scikit-learn class named ``name`` (imported here: only research that learns needs it)."""
    from sklearn import ensemble, linear_model, neighbors, svm, tree  # noqa: PLC0415

    for module in (linear_model, neighbors, svm, tree, ensemble):
        if hasattr(module, name):
            return getattr(module, name)  # type: ignore[no-any-return]
    raise KeyError(f"no scikit-learn regressor {name!r}")


def _speed(name: str) -> dict[str, Any]:
    """Settings that change only how fast a model fits, never what it fits (so not part of its identity)."""
    return {"n_jobs": -1} if name in ("RandomForestRegressor", "ExtraTreesRegressor") else {}


class Sampled:
    """A model fitted on at most ``rows`` training rows drawn at random (with a fixed seed) - for slow learners."""

    def __init__(self, model: Any, rows: int, seed: int = SEED) -> None:
        self.model = model
        self.rows = rows
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> Sampled:
        """Fit on all rows, or on ``rows`` of them chosen at random."""
        if len(x) > self.rows:
            keep = np.random.default_rng(self.seed).choice(len(x), self.rows, replace=False)
            x, y = x[keep], y[keep]
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """The fitted model's predictions."""
        return np.asarray(self.model.predict(x), dtype=float)


def predict_candidate(
    features: pd.DataFrame,
    target: pd.DataFrame,
    members: pd.DataFrame,
    dates: pd.DatetimeIndex,
    name: str,
    *,
    candidates: Mapping[str, Candidate] = CANDIDATES,
) -> pd.DataFrame:
    """``date x listing_id``: candidate ``name``'s walk-forward predictions (:func:`ml.walk_forward`)."""
    return ml.walk_forward(features, target, members, dates, candidates[name].build)


def inputs_digest(features: pd.DataFrame, target: pd.DataFrame, members: pd.DataFrame) -> str:
    """A digest of what the models learn from: equal digests mean equal predictions."""
    h = hashlib.sha256()
    for frame in (features, target, members.astype(float)):
        h.update(str(list(frame.columns)).encode())
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def predictions_path(
    name: str, digest: str, root: Path | None = None, candidates: Mapping[str, Candidate] = CANDIDATES
) -> Path:
    """Where candidate ``name``'s predictions for inputs ``digest`` are kept."""
    key = config_hash(
        {"candidate": candidates[name].describe(), "inputs": digest, "train": (ml.TRAIN_MONTHS, ml.MIN_TRAIN_DATES)}
    )
    return (root or get_settings().data_dir / "research") / "ml_candidates" / f"{name}-{key}.parquet"


def candidate_predictions(
    features: pd.DataFrame,
    target: pd.DataFrame,
    members: pd.DataFrame,
    dates: pd.DatetimeIndex,
    names: Iterable[str] | None = None,
    *,
    candidates: Mapping[str, Candidate] = CANDIDATES,
    root: Path | None = None,
    progress: Callable[[str, float, bool], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Each candidate's walk-forward predictions, read back when already stored for the same inputs.

    ``names`` are some of ``candidates`` (all by default) - :data:`CANDIDATES` unless another study's are given.
    ``progress(name, seconds, stored)`` is told after each candidate how long it took and whether it was read back.
    """
    # The database returns rows in no fixed order, and a tree's tie-breaks between equally good splits follow row
    # order: sorted inputs make both the digest and the fitted trees the same from one load to the next.
    # Each load also differs in the last bits of some values (sums added in another order), which deep trees turn into
    # different splits: rounded to single precision (seven digits, far finer than any ratio means) they agree.
    features = features.sort_index().sort_index(axis=1).astype("float32").astype(float)
    target = target.sort_index().sort_index(axis=1).astype("float32").astype(float)
    members = members.sort_index().sort_index(axis=1)
    digest = inputs_digest(features, target, members)
    out: dict[str, pd.DataFrame] = {}
    for name in candidates if names is None else names:
        path = predictions_path(name, digest, root, candidates)
        started = time.perf_counter()
        stored = path.exists()
        if stored:
            long = pd.read_parquet(path)
            wide = long.pivot_table(index="date", columns="listing_id", values="prediction")
            out[name] = wide.reindex(index=dates, columns=members.columns)
        else:
            out[name] = predict_candidate(features, target, members, dates, name, candidates=candidates)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = out[name].rename_axis(index="date", columns="listing_id").reset_index()
            frame.melt(id_vars="date", value_name="prediction").dropna().to_parquet(path)
        if progress is not None:
            progress(name, time.perf_counter() - started, stored)
    return out


def rank_ic(predictions: pd.DataFrame, target: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Per date: the Spearman correlation across members between the predictions and the returns that followed.

    Missing where fewer than :data:`MIN_NAMES` members have both (no model yet, or the year not yet over).
    """
    known = members.reindex(index=predictions.index, columns=predictions.columns, fill_value=False).astype(bool)
    y = target.reindex(index=predictions.index, columns=predictions.columns)
    both = known & predictions.notna() & y.notna()
    p = predictions.where(both).rank(axis=1)
    r = y.where(both).rank(axis=1)
    ic = p.corrwith(r, axis=1)  # Pearson correlation of ranks = Spearman's
    return ic.where(both.sum(axis=1) >= MIN_NAMES).rename("ic")


def information_coefficients(
    predictions: Mapping[str, pd.DataFrame], target: pd.DataFrame, members: pd.DataFrame
) -> pd.DataFrame:
    """``date x candidate``: each candidate's rank IC (:func:`rank_ic`)."""
    return pd.DataFrame({name: rank_ic(p, target, members) for name, p in predictions.items()})


def newey_west_t(values: pd.Series, lags: int = HAC_LAGS) -> float:
    """The t-statistic of the mean of ``values`` with Newey-West (Bartlett) standard errors.

    Overlapping year-ahead returns make neighbouring months' ICs share eleven of their twelve months, so the plain
    t-statistic (which assumes independent months) overstates the evidence by about the square root of the overlap.
    """
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    variance = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(e[lag:] @ e[:-lag]) / n
    return float(x.mean() / math.sqrt(variance / n)) if variance > 0 else float("nan")


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """Per candidate: mean IC, its spread, the share of months positive and the Newey-West t-statistic."""
    return pd.DataFrame(
        {
            "mean_ic": ic.mean(),
            "ic_sd": ic.std(),
            "positive": (ic > 0).sum() / ic.notna().sum(),
            "t_nw": pd.Series({name: newey_west_t(ic[name]) for name in ic.columns}),
            "months": ic.notna().sum(),
        }
    )


def selection_table(
    ic: pd.DataFrame, dates: pd.DatetimeIndex, *, min_dates: int = MIN_VALIDATION_DATES
) -> pd.DataFrame:
    """Per year of ``dates``: each candidate's mean IC as known that January, how many dates it rests on, the choice.

    An IC is known once its year-ahead return has ended before the year's first decision date - the rule
    :func:`ml.walk_forward` trains by. Only dates where every candidate has an IC count, so all are judged on the same
    months. With fewer than ``min_dates`` of them the year's choice is :data:`FALLBACK`.
    """
    rows = {}
    for year in sorted(set(dates.year)):
        first = dates[dates.year == year].min()
        common = ic[pd.DatetimeIndex(ic.index) + ml.HORIZON < first].dropna()
        means = common.mean()
        chosen = str(means.idxmax()) if len(common) >= min_dates else FALLBACK
        rows[int(year)] = {**means.to_dict(), "dates": len(common), "chosen": chosen}
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("year")
    return table[[*ic.columns, "dates", "chosen"]]


def ensemble(predictions: Mapping[str, pd.DataFrame], members: pd.DataFrame) -> pd.DataFrame:
    """``date x listing_id``: the average across candidates of each member's percentile rank by prediction.

    Ranks, not predictions, are averaged so a model with widely spread predictions does not outvote the rest.
    """
    ranks = [
        p.where(members.reindex(index=p.index, columns=p.columns, fill_value=False).astype(bool)).rank(axis=1, pct=True)
        for p in predictions.values()
    ]
    stacked = np.stack([r.to_numpy(dtype=float) for r in ranks])
    count = np.isfinite(stacked).sum(axis=0)
    mean = np.where(count > 0, np.nansum(stacked, axis=0) / np.maximum(count, 1), np.nan)
    return pd.DataFrame(mean, index=ranks[0].index, columns=ranks[0].columns)


def chosen_score(predictions: Mapping[str, pd.DataFrame], chosen: pd.Series, members: pd.DataFrame) -> pd.DataFrame:
    """``date x listing_id``: each year's predictions from the candidate chosen that January (``chosen``: year -> name).

    Scales differ between models, so the result is only meaningful within a date - which is how scores are used.
    """
    first = next(iter(predictions.values()))
    out = pd.DataFrame(np.nan, index=first.index, columns=first.columns)
    pooled = ensemble(predictions, members)
    for year, name in chosen.items():
        source = pooled if name == FALLBACK else predictions[str(name)]
        rows = out.index[pd.DatetimeIndex(out.index).year == year]
        out.loc[rows] = source.loc[rows].to_numpy()
    return out


def run_score(
    data: UniverseData,
    config: StrategyConfig,
    score: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    engine: Engine | None = None,
) -> StrategyResult:
    """:func:`strategy.run_strategy`'s steps for a score already built on data already loaded.

    Fence, rank ``score`` among each date's members, build ``config``'s book from ``config.start``, backtest with the
    universe's costs, and (with ``engine``) register in ``config.family`` with ``protocol`` in the config - so the
    learned scores, which take long to build, are built once for every portfolio of the study. ``data`` must be
    ``config``'s universe, loaded from :data:`strategy.TRAIN_FROM` to ``config.end``.
    """
    holdout = fence(config.end, open_holdout=config.open_holdout)
    members = data.members.reindex(data.rebalances)
    z = fx.cross_sectional_z(score.reindex(index=data.rebalances, columns=data.members.columns), members)
    dates = data.rebalances[data.rebalances >= pd.Timestamp(config.start)]
    book = strategy_book(data, config, z, dates)
    returns = data.returns[data.returns.index <= pd.Timestamp(config.end)]
    result = run_backtest(
        book, returns, cost_bps=data.cost_bps, borrow_bps=data.borrow_bps, buy_tax_bps=data.buy_tax_bps
    )
    metrics, run_id = finish_run(
        result,
        data.benchmark,
        asdict(config) | {"score": dict(protocol)},
        family=config.family,
        engine=engine,
        holdout=holdout,
    )
    return StrategyResult(config, z, book, result, metrics, run_id)


def toy_fits(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, candidates: Mapping[str, Candidate] = CANDIDATES
) -> pd.DataFrame:
    """``grid x candidate``: each candidate, with its registered settings, fitted to one feature ``x`` -> ``y``.

    For seeing how differently the eight shapes are drawn through the same points; not part of the study.
    """
    xs, gs = x.reshape(-1, 1), grid.reshape(-1, 1)
    return pd.DataFrame({name: c.build().fit(xs, y).predict(gs) for name, c in candidates.items()}, index=grid)


def toy_data(n: int = 400, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)``: a smooth, non-linear relation buried in fat-tailed noise - a cartoon of a ratio and a return."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n)
    y = 0.5 * np.sin(1.2 * x) + 0.1 * x + 0.35 * rng.standard_t(4, n)
    return x, y
