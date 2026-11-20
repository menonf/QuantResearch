from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.ml import linear as ln
from research_core.analytics.signals import evaluation as ev
from research_core.analytics.signals import validation as va

MONTHS = pd.date_range("2010-01-31", "2017-12-31", freq="ME")
NAMES = list(range(1, 61))


def _panel(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five features of 60 companies; next month's excess return is 0.05 x the first, plus noise."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([MONTHS, NAMES], names=["date", "listing_id"])
    raw = pd.DataFrame(rng.normal(size=(len(index), 5)), index=index, columns=["signal", "a", "b", "c", "d"])
    raw.iloc[::11, 2] = np.nan  # gaps in a feature
    x = ln.centred_ranks(raw)
    y = (0.05 * x["signal"] + rng.normal(0, 0.03, len(x))).unstack("listing_id")  # noqa: PD010
    return x, y


def _folds(y: pd.DataFrame) -> tuple[list[va.Fold], pd.Series]:
    ends = va.next_date_ends(pd.DatetimeIndex(y.index))
    return va.walk_forward(pd.DatetimeIndex(y.index), ends, train_years=4, first_test=2015), ends


def test_centred_ranks_put_every_feature_on_one_scale() -> None:
    x, _ = _panel()
    assert x.to_numpy().min() >= -0.5 and x.to_numpy().max() <= 0.5
    assert np.isfinite(x.to_numpy()).all()  # a missing value sits at the median
    one_date = x.xs(MONTHS[0], level="date")["signal"]
    assert one_date.max() == pytest.approx(0.5) and one_date.min() == pytest.approx(1 / 60 - 0.5)


def test_walk_forward_ridge_learns_the_signal_out_of_sample() -> None:
    x, y = _panel()
    folds, ends = _folds(y)
    run = ln.walk_forward(x, y, folds, ends, ln.MODELS["ridge"], k=3)
    assert list(run.chosen.index) == [2015, 2016, 2017]
    assert set(run.chosen) <= set(ln.MODELS["ridge"].grid)
    assert run.scores.index.min().year == 2015  # scores only in test years
    ic = ev.ic_series(run.scores, y)
    # the best possible: corr(0.05 x, 0.05 x + noise) = 0.0144 / sqrt(0.0144^2 + 0.03^2) = 0.43
    assert ic.mean() > 0.38
    stable = ln.coefficient_stability(run.coefficients)
    assert stable.index[0] == "signal" and stable.loc["signal", "same sign"] == 1.0
    assert run.tuning.shape == (3, len(ln.MODELS["ridge"].grid))


def test_lasso_and_logistic_run_and_point_the_right_way() -> None:
    x, y = _panel(1)
    folds, ends = _folds(y)
    lasso = ln.walk_forward(x, y, folds, ends, ln.MODELS["lasso"], k=3)
    assert (lasso.coefficients["signal"] > 0).all()
    logistic = ln.walk_forward(x, y, folds, ends, ln.MODELS["logistic"], k=3)
    s = logistic.scores.to_numpy()
    assert np.nanmin(s) > 0 and np.nanmax(s) < 1  # probabilities
    assert ev.ic_series(logistic.scores, y).mean() > 0.3


def test_conformal_quantile_and_coverage() -> None:
    assert ln.conformal_quantile(np.arange(1.0, 101.0), 0.9) == 91.0  # ceil(101 x 0.9) = 91st smallest
    x, y = _panel(2)
    folds, ends = _folds(y)
    bands = ln.conformal(x, y, folds, ends, ln.MODELS["ridge"], penalties={2015: 1.0, 2016: 1.0, 2017: 1.0})
    assert bands["coverage"].between(0.85, 0.95).all()  # exchangeable noise: 90 % covered, about
    assert (bands["half-width"] > bands["prediction sd"]).all()  # honest intervals are wider than the predictions vary
    with pytest.raises(ValueError, match="regression"):
        ln.conformal(x, y, folds, ends, ln.MODELS["logistic"], penalties={})


def test_composite_and_ic_difference() -> None:
    dates = MONTHS[:2]
    a = pd.DataFrame(np.tile(np.arange(60.0), (2, 1)), index=dates, columns=NAMES)
    combined = ln.composite({"a": a, "minus_a": -a})
    assert np.allclose(combined.to_numpy(), 1 / 120)  # opposite signals cancel to the middle (ranks 1/60 .. 1)
    same = ln.composite({"a": a, "b": a * 3})
    assert same.iloc[0].is_monotonic_increasing
    members = pd.DataFrame(True, index=dates, columns=NAMES)
    members[1] = False
    assert ln.composite({"a": a}, members)[1].isna().all()
    out = ln.ic_difference(pd.Series([0.05, 0.04, 0.06]), pd.Series([0.01, 0.0, 0.02]))
    assert out["mean difference"] == pytest.approx(0.04) and out["dates"] == 3


def test_a_stored_run_is_read_back(tmp_path: Path) -> None:
    x, y = _panel(3)
    folds, ends = _folds(y)
    config = {"study": "test"}
    first = ln.stored_walk_forward(
        x, y, folds, ends, ln.MODELS["ridge"], name="ridge", config=config, snapshot="s", root=tmp_path, k=3
    )
    again = ln.stored_walk_forward(
        x.iloc[:0], y, folds, ends, ln.MODELS["ridge"], name="ridge", config=config, snapshot="s", root=tmp_path, k=3
    )
    pd.testing.assert_frame_equal(
        first.scores, again.scores, check_freq=False, check_column_type=False, check_names=False
    )
    assert first.chosen.tolist() == again.chosen.tolist()
    assert np.allclose(again.tuning.columns.to_numpy(dtype=float), ln.MODELS["ridge"].grid)
