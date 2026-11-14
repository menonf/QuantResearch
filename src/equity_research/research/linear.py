"""The linear baseline (docs/ML4T_PLAN.md, P2): penalised regressions fitted per fold, tuned inside each fold.

    x = centred_ranks(features, members)                         # each feature a centred percentile; missing -> 0
    y = labels.trimmed_excess(monthly, members)                  # next month's return over the average member
    run = walk_forward(x, y, folds, ends, MODELS["ridge"])       # out-of-sample scores, chosen penalty, coefficients
    stable = coefficient_stability(run.coefficients)             # does each feature keep its sign from fold to fold
    bands = conformal(x, y, folds, ends, MODELS["ridge"], ...)    # how wide an honest 90 % interval is
    yardstick = composite({"issuance": -share_change, "ROCE": roce}, members)

Every later model must beat a linear one, and a linear one must beat a plain average of the features that survived
the triage (notebook 6.2) - otherwise the fitting adds nothing. The steps, fixed before any result:

1. **Inputs** - each feature becomes its percentile among the date's members, less one half (:func:`centred_ranks`):
   every feature on the same scale, an outlier no further out than the next company, and a missing value at the
   median (0), so it neither helps nor hurts.
2. **Models** (:data:`MODELS`) - ridge (an L2 penalty: every coefficient shrunk towards 0), lasso (L1: weak ones set
   to exactly 0), elastic net (half each), and logistic regression of whether a member beats the average member (its
   probability is the score). Each has a grid of penalty strengths written down in advance.
3. **Nested tuning** (:func:`walk_forward`) - inside each walk-forward fold's training dates, the grid is scored by
   purged k-fold cross-validation (``validation.purged_kfold``) on the validation blocks' mean rank IC; the best
   penalty is refitted on all the fold's training dates and predicts its test year. The test year is never seen
   while choosing.
4. **Coefficient stability** - the coefficients of each fold side by side: a feature whose sign flips from one fold to
   the next is not a reliable part of the model, whatever its average.
5. **Conformal intervals** (Vovk et al., 2005; split conformal) - fitted on the training dates but the last year, the
   model's absolute errors on that year give the quantile ``q`` for which ``prediction +/- q`` covered 90 % of
   outcomes; the test year shows whether it still does, and how wide such an honest interval is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equity_research.research import evaluation as ev
from equity_research.research import store
from equity_research.research.validation import Fold, purged_kfold

SEED = 42
#: The registry family of P2's searches and out-of-sample results (new, so no existing trial count moves).
FAMILY = "ml4t_linear_monthly"


@dataclass(frozen=True)
class LinearModel:
    """One penalised linear model and the grid of its penalty, fixed in advance."""

    label: str
    estimator: str  # a scikit-learn class in sklearn.linear_model
    parameter: str  # the penalty's keyword
    grid: tuple[float, ...]
    fixed: tuple[tuple[str, Any], ...] = ()
    classifier: bool = False  # fitted on whether the target is above 0; scored by the probability

    def build(self, value: float) -> Any:
        """A fresh estimator with the penalty ``value``."""
        from sklearn import linear_model  # noqa: PLC0415 - only research that learns needs scikit-learn

        return getattr(linear_model, self.estimator)(**{self.parameter: value}, **dict(self.fixed))

    def describe(self) -> dict[str, Any]:
        """What identifies the model in the registry."""
        return {"estimator": self.estimator, "parameter": self.parameter, "grid": list(self.grid), **dict(self.fixed)}


#: The four linear models of P2 and their grids. Inputs lie in [-0.5, 0.5] and the target is a monthly excess return
#: (a standard deviation of about 0.1), so the grids span from barely penalised to everything shrunk to nothing.
MODELS: dict[str, LinearModel] = {
    "ridge": LinearModel("Ridge", "Ridge", "alpha", tuple(10.0**k for k in range(0, 7))),
    "lasso": LinearModel("Lasso", "Lasso", "alpha", tuple(10.0**k for k in range(-7, -2)), (("max_iter", 5000),)),
    "elastic_net": LinearModel(
        "Elastic net",
        "ElasticNet",
        "alpha",
        tuple(10.0**k for k in range(-7, -2)),
        (("l1_ratio", 0.5), ("max_iter", 5000)),
    ),
    "logistic": LinearModel(
        "Logistic regression",
        "LogisticRegression",
        "C",
        tuple(10.0**k for k in range(-4, 1)),
        (("max_iter", 1000),),
        classifier=True,
    ),
}


def centred_ranks(features: pd.DataFrame) -> pd.DataFrame:
    """``(date, listing_id) x feature``: each value's percentile among its date's rows, less 0.5; missing -> 0.

    ``features`` holds the members only (as ``features.stack`` makes it), so the percentiles are among the members.
    """
    ranks = features.groupby(level="date").rank(pct=True) - 0.5
    return ranks.fillna(0.0)


def design_rows(
    x: pd.DataFrame, y: pd.DataFrame, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    """The design rows of ``dates`` that have a target: features, target and their ``(date, listing_id)``."""
    part = x[x.index.get_level_values("date").isin(dates)]
    index = pd.MultiIndex.from_product([y.index, y.columns], names=["date", "listing_id"])
    target = pd.Series(y.to_numpy(dtype=float).ravel(), index=index, name="y").dropna()
    joined = part.join(target, how="inner")
    rows = pd.MultiIndex.from_frame(joined.index.to_frame(index=False))
    return joined.drop(columns="y").to_numpy(dtype=float), joined["y"].to_numpy(dtype=float), rows


def _fit(model: LinearModel, value: float, x: np.ndarray, y: np.ndarray) -> Any:
    target = (y > 0).astype(int) if model.classifier else y
    return model.build(value).fit(x, target)


def _score(fitted: Any, model: LinearModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(fitted.predict_proba(x)[:, 1] if model.classifier else fitted.predict(x), dtype=float)


def to_wide(values: np.ndarray, index: pd.MultiIndex) -> pd.DataFrame:
    """``date x listing_id`` from values on ``(date, listing_id)`` rows."""
    return pd.Series(values, index=index).unstack("listing_id")  # noqa: PD010 - long to wide


def _coefficients(fitted: Any) -> np.ndarray:
    return np.ravel(fitted.coef_)


def inner_scores(
    model: LinearModel,
    x: pd.DataFrame,
    y: pd.DataFrame,
    train: pd.DatetimeIndex,
    ends: pd.Series,
    *,
    k: int = 5,
    embargo: int = 1,
) -> pd.Series:
    """``penalty -> mean validation rank IC`` over purged k-fold blocks of ``train``: the nested tuning of one fold."""
    scores: dict[float, list[float]] = {value: [] for value in model.grid}
    for block in purged_kfold(train, ends, k=k, embargo=embargo):
        xt, yt, _ = design_rows(x, y, block.train)
        xv, yv, iv = design_rows(x, y, block.test)
        truth = to_wide(yv, iv)
        for value in model.grid:
            predicted = to_wide(_score(_fit(model, value, xt, yt), model, xv), iv)
            scores[value].append(float(ev.ic_flat_zero(predicted, truth).mean()))
    return pd.Series({value: float(np.nanmean(s)) for value, s in scores.items()}, name="validation_ic")


@dataclass(frozen=True)
class WalkForward:
    """A model's walk-forward run: test-year scores, the penalty each fold chose, coefficients and the tuning grid."""

    scores: pd.DataFrame  # date x listing_id over the test years
    chosen: pd.Series  # fold's first test year -> penalty
    coefficients: pd.DataFrame  # fold's first test year x feature
    tuning: pd.DataFrame  # fold's first test year x penalty: mean validation IC
    features: list[str] = field(default_factory=list)


def walk_forward(
    x: pd.DataFrame,
    y: pd.DataFrame,
    folds: Sequence[Fold],
    ends: pd.Series,
    model: LinearModel,
    *,
    k: int = 5,
    embargo: int = 1,
) -> WalkForward:
    """Each fold: tune on its training dates (:func:`inner_scores`), refit the best penalty on them, score its test."""
    scores, chosen, coefficients, tuning = [], {}, {}, {}
    for fold in folds:
        year = int(fold.test.min().year)
        grid = inner_scores(model, x, y, fold.train, ends, k=k, embargo=embargo)
        best = float(grid.index.to_numpy(dtype=float)[int(np.nanargmax(grid.to_numpy()))])
        xt, yt, _ = design_rows(x, y, fold.train)
        fitted = _fit(model, best, xt, yt)
        part = x[x.index.get_level_values("date").isin(fold.test)]
        scores.append(to_wide(_score(fitted, model, part.to_numpy(dtype=float)), part.index))  # type: ignore[arg-type]
        chosen[year], coefficients[year], tuning[year] = best, _coefficients(fitted), grid
    return WalkForward(
        scores=pd.concat(scores).sort_index(),
        chosen=pd.Series(chosen, name=model.parameter),
        coefficients=pd.DataFrame(coefficients, index=list(x.columns)).T,
        tuning=pd.DataFrame(tuning).T,
        features=list(x.columns),
    )


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Per feature: how its coefficient varies across folds, sorted by the absolute mean.

    The mean and spread, how often it keeps the mean's sign, and how often it is exactly 0 (dropped by an L1 penalty).
    """
    mean, sd = coefficients.mean(), coefficients.std()
    same = (np.sign(coefficients) == np.sign(mean)).mean()
    out = pd.DataFrame(
        {"mean": mean, "sd": sd, "same sign": same, "zero": (coefficients == 0).mean(), "folds": len(coefficients)}
    )
    return out.reindex(out["mean"].abs().sort_values(ascending=False).index)


def conformal_quantile(errors: np.ndarray, coverage: float = 0.9) -> float:
    """The split-conformal quantile: the ``ceil((n + 1) coverage) / n`` empirical quantile of absolute errors."""
    e = np.sort(np.abs(errors[np.isfinite(errors)]))
    n = len(e)
    if n == 0:
        return float("nan")
    rank = min(math.ceil((n + 1) * coverage), n)
    return float(e[rank - 1])


def conformal(
    x: pd.DataFrame,
    y: pd.DataFrame,
    folds: Sequence[Fold],
    ends: pd.Series,
    model: LinearModel,
    *,
    penalties: Mapping[int, float],
    coverage: float = 0.9,
    calibration_months: int = 12,
) -> pd.DataFrame:
    """Per fold: split-conformal intervals for a regression ``model`` with the fold's chosen penalty.

    The model is fitted on the fold's training dates but the last ``calibration_months`` (purged of labels that reach
    them), its absolute errors on those months give the half-width; the test year then reports the share of outcomes
    inside ``prediction +/- half-width`` and how the half-width compares with the spread of the predictions.
    """
    if model.classifier:
        raise ValueError("conformal intervals here are for regression models")
    rows = {}
    for fold in folds:
        year = int(fold.test.min().year)
        calibration = fold.train[-calibration_months:]
        start = calibration.min().to_datetime64()
        proper = fold.train[ends.reindex(fold.train).to_numpy() < start]
        fitted = _fit(model, penalties[year], *design_rows(x, y, proper)[:2])
        xc, yc, _ = design_rows(x, y, calibration)
        half = conformal_quantile(yc - _score(fitted, model, xc), coverage)
        xs, ys, _ = design_rows(x, y, fold.test)
        predicted = _score(fitted, model, xs)
        rows[year] = {
            "half-width": half,
            "coverage": float(np.mean(np.abs(ys - predicted) <= half)),
            "prediction sd": float(np.std(predicted)),
            "outcome sd": float(np.std(ys)),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("test year")


def composite(signals: Mapping[str, pd.DataFrame], members: pd.DataFrame | None = None) -> pd.DataFrame:
    """``date x listing_id``: the equal-weighted average of each signal's centred percentile among the members.

    The yardstick of P2: what the triage's survivors give with no fitting at all. A signal whose IC is negative is
    passed negated, so every part points the same way.
    """
    parts = []
    for panel in signals.values():
        on = panel if members is None else panel.where(members.reindex_like(panel).fillna(False).astype(bool))
        parts.append(on.rank(axis=1, pct=True) - 0.5)
    stacked = np.stack([p.to_numpy(dtype=float) for p in parts])
    count = np.isfinite(stacked).sum(axis=0)
    mean = np.where(count > 0, np.nansum(stacked, axis=0) / np.maximum(count, 1), np.nan)
    first = parts[0]
    return pd.DataFrame(mean, index=first.index, columns=first.columns)


def ic_difference(a: pd.Series, b: pd.Series, lags: int = 0) -> dict[str, float]:
    """The mean of ``a - b`` (two IC series on the same dates) and its Newey-West t-statistic: is one better?"""
    d = (a - b).dropna()
    return {"mean difference": float(d.mean()), "t_nw": ev.newey_west_t(d, lags), "dates": len(d)}


def stored_walk_forward(
    x: pd.DataFrame,
    y: pd.DataFrame,
    folds: Sequence[Fold],
    ends: pd.Series,
    model: LinearModel,
    *,
    name: str,
    config: Mapping[str, Any],
    snapshot: str | None,
    root: Path | None = None,
    k: int = 5,
    embargo: int = 1,
) -> WalkForward:
    """:func:`walk_forward`, read back from the artifact store when this ``config`` and ``snapshot`` already ran."""
    key = dict(config) | {"model": model.describe(), "k": k, "embargo": embargo}
    parts = ("scores", "chosen", "coefficients", "tuning")
    stored = [store.load("linear", f"{name}-{part}", key, snapshot=snapshot, root=root) for part in parts]
    found = [frame for frame in stored if frame is not None]
    if len(found) == len(parts):
        scores, chosen, coefficients, tuning = found
        tuning.columns = [float(c) for c in tuning.columns]
        return WalkForward(
            scores=scores,
            chosen=chosen.iloc[:, 0].rename(model.parameter),
            coefficients=coefficients,
            tuning=tuning,
            features=list(coefficients.columns),
        )
    run = walk_forward(x, y, folds, ends, model, k=k, embargo=embargo)
    frames = (run.scores, run.chosen.to_frame(), run.coefficients, run.tuning)
    for part, frame in zip(parts, frames, strict=True):
        store.save(frame, "linear", f"{name}-{part}", key, snapshot=snapshot, root=root)
    return run
