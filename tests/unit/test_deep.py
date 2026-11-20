import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from equity_research.ml import deep as dp
from equity_research.ml import linear as ln
from research_core.analytics.signals import evaluation as ev
from research_core.analytics.signals import validation as va

NAMES = list(range(1, 81))


def test_week_ends_and_weekly_ranks() -> None:
    days = pd.bdate_range("2020-01-06", "2020-01-17")
    weeks = dp.week_ends(days)
    assert list(weeks) == [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-17")]
    r = pd.DataFrame(0.0, index=days, columns=[1, 2, 3])
    r.loc["2020-01-13":, 3] = 0.01  # name 3 rises in the second week
    members = pd.DataFrame(True, index=days, columns=[1, 2, 3])
    members[2] = False
    ranks = dp.weekly_ranks(r, members, weeks)
    last = ranks.to_numpy()[-1]
    assert last[2] > last[0] and ranks[2].eq(0).all()


def _weekly(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Weekly ranks of 80 names over 2010-2016; next week's target is 0.02 x this week's rank, plus noise."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-08", "2016-12-30", freq="W-FRI")
    past = pd.DataFrame(rng.uniform(-0.5, 0.5, (len(weeks), len(NAMES))), index=weeks, columns=NAMES)
    target = 0.02 * past + rng.normal(0, 0.01, past.shape)
    members = pd.DataFrame(True, index=weeks, columns=NAMES)
    return past, target, members


def test_sequences_hold_the_lookback_oldest_first() -> None:
    past, target, members = _weekly()
    x, y, index = dp.sequences(past, pd.DatetimeIndex(past.index[[10, 60]]), members, target)
    assert x.shape == (80, dp.LOOKBACK)  # the 11th week has no full lookback: only the 61st counts
    assert np.allclose(x[0], past[1].iloc[60 - dp.LOOKBACK + 1 : 61].to_numpy())
    assert y[0] == pytest.approx(target.iloc[60, 0]) and index[0] == (past.index[60], 1)


def test_every_model_builds_and_learns_last_weeks_rank() -> None:
    past, target, members = _weekly(1)
    x, y, _ = dp.sequences(past, pd.DatetimeIndex(past.index[60:250]), members, target)
    xv, yv, _ = dp.sequences(past, pd.DatetimeIndex(past.index[250:300]), members, target)
    for name in dp.SEQUENCE_MODELS:
        model = dp.train(dp.build(name), x, y, xv, yv, epochs=15, patience=3, batch=1024, lr=3e-3)
        predicted = dp.predict(model, xv)
        assert predicted.shape == yv.shape
        assert np.corrcoef(predicted, xv[:, -1])[0, 1] > 0.8, name
    with pytest.raises(ValueError, match="unknown model"):
        dp.build("transformer")


def test_walk_forward_sequences_is_out_of_sample_and_repeatable() -> None:
    past, target, members = _weekly(2)

    def run_once() -> dp.DeepRun:
        return dp.walk_forward_sequences(
            "nlinear",
            past,
            target,
            members,
            first_test=2016,
            last_test=2016,
            max_rows=5000,
            epochs=8,
            batch=512,
            lr=3e-3,
        )

    run = run_once()
    assert run.scores.index.min().year == 2016 and run.rows.loc[2016] == 5000
    assert ev.ic_series(run.scores, target, min_names=30).mean() > 0.4
    again = run_once()
    assert np.allclose(run.scores.to_numpy(), again.scores.to_numpy())  # seeded and deterministic


def test_mlp_ensemble_on_features() -> None:
    rng = np.random.default_rng(3)
    months = pd.date_range("2010-01-31", "2016-12-31", freq="ME")
    index = pd.MultiIndex.from_product([months, NAMES], names=["date", "listing_id"])
    x = ln.centred_ranks(pd.DataFrame(rng.normal(size=(len(index), 3)), index=index, columns=["s", "a", "b"]))
    y = (0.05 * x["s"] + rng.normal(0, 0.02, len(x))).unstack("listing_id")  # noqa: PD010
    ends = va.next_date_ends(months)
    folds = va.walk_forward(months, ends, train_years=4, first_test=2016)
    run = dp.walk_forward_mlp(x, y, folds, members_count=2, epochs=10, batch=512, lr=3e-3)
    assert ev.ic_series(run.scores, y).mean() > 0.3
