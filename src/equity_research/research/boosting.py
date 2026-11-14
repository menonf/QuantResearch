"""Gradient boosting (docs/ML4T_PLAN.md, P3): LightGBM per fold, a grid then an Optuna search, TreeSHAP, intervals.

    run = walk_forward(x, y, folds, ends, grid())             # the 3 x 3 grid, chosen inside each fold
    trials, best = optuna_search(x, y, folds[0].train, ends, loss="l2", n_trials=40)   # tuned before 2015 only
    fixed = walk_forward(x, y, folds, ends, {"optuna": best})  # those settings refitted each fold, not retuned
    importance = run.shap                                       # mean |SHAP| of each feature, fold by fold

The same inputs, target and folds as the linear baseline (``linear.py``), so the two compare like for like. Fixed
before any result:

1. **Losses** (:data:`LOSSES`) - squared error (the mean); Huber (squared near 0, linear beyond ``HUBER_DELTA``, so an
   extreme month pulls less); and LambdaRank (Burges, 2010), which learns the *order* of the companies on each date
   directly - each date a group, each company labelled with its quintile of the target.
2. **Leaf profiles** (:data:`PROFILES`) - small, medium and large trees: (4 leaves, at least 1,000 company-months a
   leaf), (16, 500), (64, 200). With the shared settings in :data:`BASE`: 300 trees at a learning rate of 0.03, 70 % of
   rows and of features per tree, an L2 penalty of 1 on leaf values.
3. **Grid, nested** - the nine (loss, profile) pairs are scored inside each walk-forward fold by purged k-fold
   cross-validation on mean rank IC, as the linear models' penalties are; the best is refitted on the fold.
4. **Optuna, once** - a Bayesian search (tree-structured Parzen estimator) over the leaf count, leaf size, learning
   rate, number of trees, row and feature fractions and the L2 penalty, on the first fold's training window only
   (before the first test year), every trial scored by the same purged cross-validation and returned for registering.
   Its best settings are then refitted in every fold without retuning: tuning ten windows would cost ten searches.
5. **TreeSHAP** (Lundberg et al., 2020) - each test prediction split exactly into one contribution per feature
   (LightGBM's ``pred_contrib``); the mean absolute contribution is the feature's importance in that fold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equity_research.research import evaluation as ev
from equity_research.research import store
from equity_research.research.linear import conformal_quantile, design_rows, to_wide
from equity_research.research.validation import Fold, purged_kfold

SEED = 42
#: The registry family of P3's searches and out-of-sample results.
FAMILY = "ml4t_gbm_monthly"
#: Huber's switch from squared to linear error: about half a month's typical excess return.
HUBER_DELTA = 0.05
#: Settings every configuration shares unless it says otherwise.
BASE: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
#: The losses of the grid.
LOSSES = ("l2", "huber", "rank")
#: Leaf profiles: (leaves a tree, least company-months a leaf).
PROFILES = {"small": (4, 1000), "medium": (16, 500), "large": (64, 200)}
#: Quantiles of the target that become LambdaRank's relevance labels.
RANK_LEVELS = 5


@dataclass(frozen=True)
class BoostConfig:
    """One LightGBM configuration: a loss and its settings (on top of :data:`BASE`)."""

    loss: str
    settings: tuple[tuple[str, Any], ...] = ()

    def params(self) -> dict[str, Any]:
        """Every LightGBM setting of this configuration."""
        out = BASE | dict(self.settings)
        if self.loss == "huber":
            out |= {"objective": "huber", "alpha": HUBER_DELTA}
        elif self.loss == "l2":
            out |= {"objective": "regression"}
        elif self.loss == "rank":
            out |= {"objective": "lambdarank"}
        else:
            raise ValueError(f"unknown loss {self.loss!r}")
        return out

    def describe(self) -> dict[str, Any]:
        """What identifies the configuration in the registry and the artifact store."""
        return {"loss": self.loss, **dict(self.settings)}


def grid() -> dict[str, BoostConfig]:
    """The nine (loss, leaf profile) configurations, named ``loss/profile``."""
    return {
        f"{loss}/{name}": BoostConfig(loss, (("num_leaves", leaves), ("min_child_samples", least)))
        for loss in LOSSES
        for name, (leaves, least) in PROFILES.items()
    }


def _groups(index: pd.MultiIndex) -> np.ndarray:
    """Rows per date, in the order the rows come (they are sorted by date)."""
    dates = index.get_level_values("date")
    _, counts = np.unique(dates.to_numpy(), return_counts=True)
    return counts


def _relevance(y: np.ndarray, index: pd.MultiIndex) -> np.ndarray:
    """Each row's quintile of the target among its date's rows: 0 (lowest) to RANK_LEVELS - 1."""
    pct = pd.Series(y, index=index).groupby(level="date").rank(pct=True).to_numpy(dtype=float)
    levels: np.ndarray = np.clip(np.ceil(pct * RANK_LEVELS).astype(int) - 1, 0, RANK_LEVELS - 1)
    return levels


def fit(config: BoostConfig, x: np.ndarray, y: np.ndarray, index: pd.MultiIndex) -> Any:
    """A fitted LightGBM model (a ranker for the rank loss, which learns from each date as a group)."""
    import lightgbm as lgb  # noqa: PLC0415 - the optional ml group

    if config.loss == "rank":
        return lgb.LGBMRanker(**config.params()).fit(x, _relevance(y, index), group=_groups(index))
    return lgb.LGBMRegressor(**config.params()).fit(x, y)


def predict(model: Any, x: np.ndarray) -> np.ndarray:
    """The model's scores."""
    return np.asarray(model.predict(x), dtype=float)


def contributions(model: Any, x: np.ndarray) -> np.ndarray:
    """TreeSHAP: each row's prediction split into one contribution per feature (the bias column dropped)."""
    values: np.ndarray = np.asarray(model.predict(x, pred_contrib=True), dtype=float)
    return values[:, :-1]


def validation_ic(
    config: BoostConfig,
    x: pd.DataFrame,
    y: pd.DataFrame,
    train: pd.DatetimeIndex,
    ends: pd.Series,
    *,
    k: int,
    embargo: int,
) -> float:
    """The mean rank IC of ``config`` over the validation blocks of a purged k-fold split of ``train``."""
    ics = []
    for block in purged_kfold(train, ends, k=k, embargo=embargo):
        xt, yt, it = design_rows(x, y, block.train)
        xv, yv, iv = design_rows(x, y, block.test)
        predicted = to_wide(predict(fit(config, xt, yt, it), xv), iv)
        ics.append(float(ev.ic_flat_zero(predicted, to_wide(yv, iv)).mean()))
    return float(np.nanmean(ics))


@dataclass(frozen=True)
class BoostRun:
    """A walk-forward run: test-year scores, the configuration each fold chose, its tuning and SHAP importance."""

    scores: pd.DataFrame  # date x listing_id over the test years
    chosen: pd.Series  # fold's first test year -> configuration name
    tuning: pd.DataFrame  # fold's first test year x configuration: mean validation IC (empty when not tuned)
    shap: pd.DataFrame  # fold's first test year x feature: mean |SHAP| on the test year


def walk_forward(
    x: pd.DataFrame,
    y: pd.DataFrame,
    folds: Sequence[Fold],
    ends: pd.Series,
    configs: Mapping[str, BoostConfig],
    *,
    k: int = 5,
    embargo: int = 1,
) -> BoostRun:
    """Each fold: the best of ``configs`` (chosen by purged k-fold if there are several), refitted, on its test year.

    The best is scored on its training dates by purged k-fold cross-validation, refitted on all of them, and scores
    its test year; the test year's mean absolute SHAP contributions are kept too.
    """
    scores, chosen, tuning, shap = [], {}, {}, {}
    for fold in folds:
        year = int(fold.test.min().year)
        if len(configs) > 1:
            grid_ic = {
                name: validation_ic(c, x, y, fold.train, ends, k=k, embargo=embargo) for name, c in configs.items()
            }
            tuning[year] = grid_ic
            best = max(grid_ic, key=lambda n: -math.inf if math.isnan(grid_ic[n]) else grid_ic[n])
        else:
            best = next(iter(configs))
        xt, yt, it = design_rows(x, y, fold.train)
        model = fit(configs[best], xt, yt, it)
        part = x[x.index.get_level_values("date").isin(fold.test)]
        xs = part.to_numpy(dtype=float)
        index = pd.MultiIndex.from_frame(part.index.to_frame(index=False))
        scores.append(to_wide(predict(model, xs), index))
        shap[year] = np.abs(contributions(model, xs)).mean(axis=0)
        chosen[year] = best
    return BoostRun(
        scores=pd.concat(scores).sort_index(),
        chosen=pd.Series(chosen, name="config"),
        tuning=pd.DataFrame(tuning).T,
        shap=pd.DataFrame(shap, index=list(x.columns)).T,
    )


def optuna_search(
    x: pd.DataFrame,
    y: pd.DataFrame,
    train: pd.DatetimeIndex,
    ends: pd.Series,
    *,
    loss: str,
    n_trials: int = 40,
    k: int = 5,
    embargo: int = 1,
) -> tuple[pd.DataFrame, BoostConfig]:
    """A TPE search of ``loss``'s settings on ``train`` only: every trial (settings and validation IC), and the best.

    The search space: leaves 2-64, least rows a leaf 100-3,000, learning rate 0.005-0.1 (all on log scales), trees
    100-800, row and feature fractions 0.3-1, the L2 penalty 0.001-100 (log). Seeded, so a rerun repeats it.
    """
    import optuna  # noqa: PLC0415 - the optional ml group

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        settings = (
            ("num_leaves", trial.suggest_int("num_leaves", 2, 64, log=True)),
            ("min_child_samples", trial.suggest_int("min_child_samples", 100, 3000, log=True)),
            ("learning_rate", trial.suggest_float("learning_rate", 0.005, 0.1, log=True)),
            ("n_estimators", trial.suggest_int("n_estimators", 100, 800, step=50)),
            ("subsample", trial.suggest_float("subsample", 0.3, 1.0)),
            ("colsample_bytree", trial.suggest_float("colsample_bytree", 0.3, 1.0)),
            ("reg_lambda", trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True)),
        )
        value = validation_ic(BoostConfig(loss, settings), x, y, train, ends, k=k, embargo=embargo)
        return -1.0 if math.isnan(value) else value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    trials = pd.DataFrame([{**t.params, "validation_ic": t.value} for t in study.trials])
    best = BoostConfig(loss, tuple(sorted(study.best_params.items())))
    return trials, best


def stored_walk_forward(
    x: pd.DataFrame,
    y: pd.DataFrame,
    folds: Sequence[Fold],
    ends: pd.Series,
    configs: Mapping[str, BoostConfig],
    *,
    name: str,
    config: Mapping[str, Any],
    snapshot: str | None,
    root: Path | None = None,
    k: int = 5,
    embargo: int = 1,
) -> BoostRun:
    """:func:`walk_forward`, read back from the artifact store when this ``config`` and ``snapshot`` already ran."""
    key = dict(config) | {"configs": {n: c.describe() for n, c in configs.items()}, "k": k, "embargo": embargo}
    parts = ("scores", "chosen", "tuning", "shap")
    stored = [store.load("gbm", f"{name}-{part}", key, snapshot=snapshot, root=root) for part in parts]
    found = [frame for frame in stored if frame is not None]
    if len(found) == len(parts):
        scores, chosen, tuning, shap = found
        return BoostRun(scores=scores, chosen=chosen.iloc[:, 0].rename("config"), tuning=tuning, shap=shap)
    run = walk_forward(x, y, folds, ends, configs, k=k, embargo=embargo)
    for part, frame in zip(parts, (run.scores, run.chosen.to_frame(), run.tuning, run.shap), strict=True):
        store.save(frame, "gbm", f"{name}-{part}", key, snapshot=snapshot, root=root)
    return run


def stored_search(
    x: pd.DataFrame,
    y: pd.DataFrame,
    train: pd.DatetimeIndex,
    ends: pd.Series,
    *,
    loss: str,
    n_trials: int,
    config: Mapping[str, Any],
    snapshot: str | None,
    root: Path | None = None,
) -> tuple[pd.DataFrame, BoostConfig]:
    """:func:`optuna_search`, read back from the artifact store when it already ran on this data."""
    key = dict(config) | {"loss": loss, "n_trials": n_trials, "train": [str(train.min()), str(train.max())]}
    trials = store.load("gbm", "optuna-trials", key, snapshot=snapshot, root=root)
    if trials is None:
        trials, _ = optuna_search(x, y, train, ends, loss=loss, n_trials=n_trials)
        store.save(trials, "gbm", "optuna-trials", key, snapshot=snapshot, root=root)
    row = trials.iloc[int(np.nanargmax(trials["validation_ic"].to_numpy()))].drop("validation_ic")
    integers = {"num_leaves", "min_child_samples", "n_estimators"}
    settings = tuple(sorted((str(k), int(v) if k in integers else float(v)) for k, v in row.items()))
    return trials, BoostConfig(loss, settings)


def conformal(
    x: pd.DataFrame,
    y: pd.DataFrame,
    folds: Sequence[Fold],
    ends: pd.Series,
    config: BoostConfig,
    *,
    coverage: float = 0.9,
    calibration_months: int = 12,
) -> pd.DataFrame:
    """Split-conformal intervals for a regression ``config``, as :func:`linear.conformal` gives them.

    Per fold: the half-width from the last training year's errors, and the test year's coverage.
    """
    if config.loss == "rank":
        raise ValueError("a ranking model predicts no return, so it has no interval")
    rows = {}
    for fold in folds:
        year = int(fold.test.min().year)
        calibration = fold.train[-calibration_months:]
        proper = fold.train[ends.reindex(fold.train).to_numpy() < calibration.min().to_datetime64()]
        xp, yp, ip = design_rows(x, y, proper)
        model = fit(config, xp, yp, ip)
        xc, yc, _ = design_rows(x, y, calibration)
        half = conformal_quantile(yc - predict(model, xc), coverage)
        xs, ys, _ = design_rows(x, y, fold.test)
        predicted = predict(model, xs)
        rows[year] = {
            "half-width": half,
            "coverage": float(np.mean(np.abs(ys - predicted) <= half)),
            "prediction sd": float(np.std(predicted)),
            "outcome sd": float(np.std(ys)),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("test year")
