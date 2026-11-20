"""Causal ML (docs/ML4T_PLAN.md, P7): double machine learning of a characteristic's effect on returns.

    est = dml(x, y, treatment="mom_12_1", controls=[...], blocks=purged_kfold(dates, ends, k=5))
    naive = ols_effect(x, y, "mom_12_1")                       # no controls
    linear = ols_effect(x, y, "mom_12_1", controls=[...])       # linear controls
    zoo = factor_zoo(x, y, ["share_change", "ROCE", ...], blocks)   # each as treatment, the rest as controls, with FDR

Does a characteristic *move* returns, or does it only travel with things that do? *Machine Learning for Trading*
(chapter 15) answers with double machine learning (Chernozhukov et al., 2018) in the partially linear model

    y = theta d + g(X) + e,      d = m(X) + v,

with ``y`` next month's return over the average member, ``d`` the characteristic (a centred percentile, so ``theta``
is the return gap between the lowest and the highest company) and ``X`` the other characteristics (the confounders).

1. **Nuisances** - ``l(X) = E[y | X]`` and ``m(X) = E[d | X]`` are fitted by a flexible learner (gradient-boosted
   trees), so a curved or interacting confounder is still controlled for.
2. **Cross-fitting** - the dates are cut into blocks (``validation.purged_kfold``: consecutive, purged of
   overlapping labels); each block's nuisances come from models fitted on the other blocks. A model never predicts
   the rows it was fitted on, which would bias ``theta`` towards the overfit.
3. **Orthogonal estimate** - ``theta = sum(v_hat e_hat) / sum(v_hat^2)`` with the residuals
   ``e_hat = y - l_hat(X)`` and ``v_hat = d - m_hat(X)``: errors in the nuisances enter only as their product, so they
   do not bias ``theta`` to first order (Neyman orthogonality).
4. **Inference** - companies on one date share shocks, so the standard error is clustered by date: the scores
   ``psi = (e_hat - theta v_hat) v_hat`` are summed within each date before their variance is taken.

"Causal" holds only if no confounder is missing from ``X`` - an assumption, not a result. What DML removes is the bias
that fitting flexible controls would otherwise add.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from research_core.analytics.signals.evaluation import benjamini_hochberg, p_value
from research_core.analytics.signals.validation import Fold

SEED = 42


def _learner(kind: str) -> Any:
    """A fresh nuisance learner: ``boosting`` (scikit-learn's histogram gradient boosting) or ``linear`` (OLS)."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415

    if kind == "boosting":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=500, random_state=SEED)  # fmt: skip
    if kind == "linear":
        return LinearRegression()
    raise ValueError(f"unknown learner {kind!r}")


def _frame(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """``x`` with the target column ``__y`` joined, rows without a target dropped."""
    index = pd.MultiIndex.from_product([y.index, y.columns], names=["date", "listing_id"])
    target = pd.Series(y.to_numpy(dtype=float).ravel(), index=index, name="__y").dropna()
    return x.join(target, how="inner")


def _clustered(psi: np.ndarray, dates: np.ndarray, denominator: float) -> float:
    """The standard error of theta from scores ``psi``, clustered by date."""
    sums = pd.Series(psi).groupby(dates).sum().to_numpy()
    return float(math.sqrt(float((sums**2).sum())) / denominator)


def dml(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    treatment: str,
    controls: Sequence[str],
    blocks: Sequence[Fold],
    learner: str = "boosting",
) -> dict[str, float]:
    """The double-ML estimate of ``treatment``'s effect on ``y`` (``date x listing_id``) given ``controls``.

    ``blocks`` are the cross-fitting folds (each a block of test dates and the dates its nuisances are fitted on).
    Returns ``theta``, its date-clustered standard error, t-statistic, p-value and the number of rows used.
    """
    data = _frame(x[[treatment, *controls]], y)
    dates = data.index.get_level_values("date")
    e_hat = np.full(len(data), np.nan)
    v_hat = np.full(len(data), np.nan)
    for block in blocks:
        fit_rows = dates.isin(block.train)
        out_rows = dates.isin(block.test)
        xc_fit, xc_out = data.loc[fit_rows, list(controls)].to_numpy(), data.loc[out_rows, list(controls)].to_numpy()
        e_hat[out_rows] = data.loc[out_rows, "__y"].to_numpy() - _learner(learner).fit(
            xc_fit, data.loc[fit_rows, "__y"].to_numpy()).predict(xc_out)  # fmt: skip
        v_hat[out_rows] = data.loc[out_rows, treatment].to_numpy() - _learner(learner).fit(
            xc_fit, data.loc[fit_rows, treatment].to_numpy()).predict(xc_out)  # fmt: skip
    used = np.isfinite(e_hat) & np.isfinite(v_hat)
    e, v, d = e_hat[used], v_hat[used], dates.to_numpy()[used]
    denominator = float(v @ v)
    theta = float(v @ e) / denominator
    se = _clustered((e - theta * v) * v, d, denominator)
    t = theta / se
    return {"theta": theta, "se": se, "t": t, "p_value": p_value(t), "rows": int(used.sum())}


def ols_effect(x: pd.DataFrame, y: pd.DataFrame, treatment: str, controls: Sequence[str] = ()) -> dict[str, float]:
    """The OLS coefficient of ``treatment`` (with linear ``controls`` and a constant), date-clustered standard error."""
    data = _frame(x[[treatment, *controls]], y)
    dates = data.index.get_level_values("date").to_numpy()
    design = np.column_stack([np.ones(len(data)), data[[treatment, *controls]].to_numpy()])
    target = data["__y"].to_numpy()
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    # the treatment's coefficient is the regression on the treatment's own residual given the rest (Frisch-Waugh-Lovell)
    rest = np.delete(design, 1, axis=1)
    d_resid = design[:, 1] - rest @ np.linalg.lstsq(rest, design[:, 1], rcond=None)[0]
    denominator = float(d_resid @ d_resid)
    se = _clustered(residual * d_resid, dates, denominator)
    theta = float(coef[1])
    return {"theta": theta, "se": se, "t": theta / se, "p_value": p_value(theta / se), "rows": len(data)}


def factor_zoo(
    x: pd.DataFrame,
    y: pd.DataFrame,
    candidates: Sequence[str],
    blocks: Sequence[Fold],
    *,
    exclude: Sequence[str] = (),
    learner: str = "boosting",
    q: float = 0.10,
) -> pd.DataFrame:
    """Each candidate as the treatment, every other feature (but ``exclude``) as a control: DML, then FDR control.

    A characteristic that keeps an effect once all the others are controlled for flexibly adds something of its own;
    one whose effect disappears was standing in for others.
    """
    rows = {}
    for name in candidates:
        controls = [c for c in x.columns if c != name and c not in exclude]
        rows[name] = dml(x, y, treatment=name, controls=controls, blocks=blocks, learner=learner)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["discovery"] = benjamini_hochberg(table["p_value"], q)
    return table.rename_axis("characteristic")
