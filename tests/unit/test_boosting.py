from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.research import boosting as gb
from equity_research.research import evaluation as ev
from equity_research.research import linear as ln
from equity_research.research import validation as va

MONTHS = pd.date_range("2010-01-31", "2017-12-31", freq="ME")
NAMES = list(range(1, 61))


def _panel(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four features of 60 companies; next month's excess return rises with the first, in a curve."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([MONTHS, NAMES], names=["date", "listing_id"])
    raw = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index, columns=["signal", "a", "b", "c"])
    x = ln.centred_ranks(raw)
    s = x["signal"]
    y = (0.05 * s + 0.2 * (s**2 - 1 / 12) + rng.normal(0, 0.03, len(x))).unstack("listing_id")  # noqa: PD010
    return x, y


def _folds(y: pd.DataFrame) -> tuple[list[va.Fold], pd.Series]:
    ends = va.next_date_ends(pd.DatetimeIndex(y.index))
    return va.walk_forward(pd.DatetimeIndex(y.index), ends, train_years=4, first_test=2015), ends


def test_the_grid_is_three_losses_by_three_leaf_profiles() -> None:
    configs = gb.grid()
    assert len(configs) == 9 and set(configs) == {f"{loss}/{p}" for loss in gb.LOSSES for p in gb.PROFILES}
    huber = configs["huber/small"].params()
    assert huber["objective"] == "huber" and huber["alpha"] == gb.HUBER_DELTA and huber["num_leaves"] == 4
    assert configs["rank/large"].params()["objective"] == "lambdarank"
    with pytest.raises(ValueError, match="unknown loss"):
        gb.BoostConfig("hinge").params()


def test_rank_labels_are_quintiles_within_each_date() -> None:
    index = pd.MultiIndex.from_product([MONTHS[:2], range(10)], names=["date", "listing_id"])
    y = np.tile(np.arange(10.0), 2)
    assert gb._relevance(y, index).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2
    assert gb._groups(index).tolist() == [10, 10]


def test_walk_forward_learns_a_curve_and_shap_finds_its_feature() -> None:
    x, y = _panel()
    folds, ends = _folds(y)
    configs = {"l2/small": gb.grid()["l2/small"], "rank/small": gb.grid()["rank/small"]}
    run = gb.walk_forward(x, y, folds, ends, configs, k=3)
    assert set(run.chosen) <= set(configs) and run.tuning.shape == (3, 2)
    assert ev.ic_series(run.scores, y).mean() > 0.3  # a curve a straight line would miss half of
    assert (run.shap.idxmax(axis=1) == "signal").all()
    single = gb.walk_forward(x, y, folds[:1], ends, {"only": gb.grid()["l2/medium"]}, k=3)
    assert single.tuning.empty and single.chosen.tolist() == ["only"]


def test_optuna_searches_on_the_training_window_and_is_stored(tmp_path: Path) -> None:
    x, y = _panel(1)
    folds, ends = _folds(y)
    trials, best = gb.stored_search(
        x, y, folds[0].train, ends, loss="l2", n_trials=3, config={"t": 1}, snapshot="s", root=tmp_path
    )
    assert len(trials) == 3 and "validation_ic" in trials and best.loss == "l2"
    assert dict(best.settings)["num_leaves"] == int(dict(best.settings)["num_leaves"])
    again, best_again = gb.stored_search(
        x.iloc[:0], y, folds[0].train, ends, loss="l2", n_trials=3, config={"t": 1}, snapshot="s", root=tmp_path
    )  # read back: the empty inputs are never used
    assert best_again == best and len(again) == 3


def test_a_stored_run_is_read_back_and_intervals_cover(tmp_path: Path) -> None:
    x, y = _panel(2)
    folds, ends = _folds(y)
    configs = {"l2/small": gb.grid()["l2/small"]}
    first = gb.stored_walk_forward(x, y, folds, ends, configs, name="g", config={}, snapshot="s", root=tmp_path, k=3)
    again = gb.stored_walk_forward(
        x.iloc[:0], y, folds, ends, configs, name="g", config={}, snapshot="s", root=tmp_path, k=3
    )
    assert first.chosen.tolist() == again.chosen.tolist()
    assert np.allclose(first.shap.to_numpy(), again.shap.to_numpy())
    bands = gb.conformal(x, y, folds, ends, gb.grid()["l2/small"])
    assert bands["coverage"].between(0.83, 0.97).all()
    with pytest.raises(ValueError, match="ranking"):
        gb.conformal(x, y, folds, ends, gb.grid()["rank/small"])
