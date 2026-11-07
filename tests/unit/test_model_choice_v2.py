import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.research import ml
from equity_research.research import model_choice as mc
from equity_research.research import model_choice_v2 as v2
from equity_research.research.data import UniverseData
from equity_research.research.runs import newest_by_score
from equity_research.research.strategy import deflated_active

NAMES = list(range(1, 41))


def _universe() -> UniverseData:
    days = pd.bdate_range("2019-01-01", "2021-12-31")
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (len(days), len(NAMES))), index=days, columns=NAMES)
    adjusted = 100 * (1 + returns).cumprod()
    members = pd.DataFrame(True, index=days, columns=NAMES)
    members[40] = False  # never a member: never a row, never ranked
    zero = pd.DataFrame(0.0, index=days, columns=NAMES)
    rebalances = pd.DatetimeIndex(returns.groupby(days.to_period("M")).apply(lambda m: m.index.max()))
    panel = pd.DataFrame(rng.normal(size=(len(rebalances), len(NAMES))), index=rebalances, columns=NAMES)
    index = pd.MultiIndex.from_product([rebalances, NAMES], names=["date", "listing_id"])
    ratios = pd.DataFrame(rng.normal(size=(len(index), len(ml.BOOK_CLIPS))), index=index, columns=list(ml.BOOK_CLIPS))
    return UniverseData(
        universe="TEST", close=adjusted, adjusted=adjusted, returns=returns, members=members,
        segment=pd.DataFrame("large", index=days, columns=NAMES), cost_bps=zero, borrow_bps=zero, buy_tax_bps=zero,
        rebalances=rebalances, ratios=ratios, benchmark=returns.mean(axis=1), market_cap=panel.abs() + 1,
        share_change=panel, earnings_surprise=panel, eps_growth=panel, sales_growth=panel,
    )  # fmt: skip


def test_member_ranks_are_percentiles_among_the_dates_members_only() -> None:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.DataFrame([[10.0, 1.0, 5.0, 1000.0], [1.0, 2.0, np.nan, 3.0]], index=dates, columns=[1, 2, 3, 4])
    members = pd.DataFrame(True, index=dates, columns=[1, 2, 3, 4])
    members[4] = False  # a non-member's extreme value neither ranks nor moves the others' ranks
    ranks = v2.member_ranks(panel, members)
    assert ranks.iloc[0].tolist()[:3] == pytest.approx([1.0, 1 / 3, 2 / 3])
    assert ranks[4].isna().all() and bool(ranks.isna().iloc[1, 2])  # missing stays missing
    assert ranks.iloc[1].tolist()[:2] == pytest.approx([0.5, 1.0])


def test_the_ranked_target_is_bounded_whatever_the_outliers() -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.DataFrame(np.tile([0.05, -0.2, 0.1, 83.6], (3, 1)), index=dates, columns=[1, 2, 3, 4])
    target = v2.ranked_target(returns, pd.DataFrame(True, index=dates, columns=[1, 2, 3, 4]))
    assert target[4].tolist() == [1.0, 1.0, 1.0]  # +8,360 % counts as the best, not as 400 times the next
    assert v2.date_share(target, pd.DataFrame(True, index=dates, columns=[1, 2, 3, 4])) == pytest.approx(0.0)


def test_date_share_separates_the_years_level_from_the_ranking() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    members = pd.DataFrame(True, index=dates, columns=[1, 2, 3])
    level = pd.DataFrame(np.repeat([[0.1], [0.3], [-0.2], [0.0]], 3, axis=1), index=dates, columns=[1, 2, 3])
    spread = pd.DataFrame(np.tile([-1.0, 0.0, 1.0], (4, 1)), index=dates, columns=[1, 2, 3])
    assert v2.date_share(level, members) == pytest.approx(1.0)  # only which month: nothing to rank
    assert v2.date_share(spread, members) == pytest.approx(0.0)  # only the ranking
    assert 0 < v2.date_share(level + spread, members) < 1


def test_ranked_features_are_the_books_ratios_and_the_extras_as_percentiles_of_members() -> None:
    data = _universe()
    features = v2.ranked_features(data)
    assert list(features.columns) == [*ml.BOOK_CLIPS, *v2.EXTRA_FEATURES]
    assert 40 not in features.index.get_level_values("listing_id")  # members only
    assert set(features.index.get_level_values("date")) == set(data.rebalances)
    values = features.to_numpy()
    assert np.nanmin(values) > 0 and np.nanmax(values) == 1.0
    # momentum needs a year of prices: missing at first, ranked once it exists
    assert features["momentum_12_1"].loc[data.rebalances[0]].isna().all()
    assert features["momentum_12_1"].loc[data.rebalances[-1]].notna().all()
    last = data.rebalances[-1]
    caps = data.market_cap.loc[last, NAMES[:-1]]
    size = features["size"].xs(last, level="date")
    assert size.to_numpy() == pytest.approx(caps.rank(pct=True).to_numpy())  # size from market values


def test_every_version_2_candidate_fits_and_predicts_with_its_registered_settings() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(1_200, 4))
    x[::7, 1] = np.nan
    y = x[:, 0] + rng.normal(0, 0.1, 1_200)
    for name, candidate in v2.CANDIDATES.items():
        predicted = candidate.build().fit(x, y).predict(x[:5])
        assert predicted.shape == (5,) and np.isfinite(predicted).all(), name
    assert set(v2.PROTOCOL["candidates"]) == set(v2.CANDIDATES) == set(mc.CANDIDATES)  # the same eight learners
    assert v2.FAMILY != mc.FAMILY  # a new family: version 1's trial count does not move


def test_version_2_predictions_are_stored_apart_from_version_1s(tmp_path: Path) -> None:
    assert mc.predictions_path("knn", "d", tmp_path) != mc.predictions_path("knn", "d", tmp_path, v2.CANDIDATES)
    months = pd.date_range("2006-01-31", "2011-12-31", freq="ME")
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.uniform(size=(len(months), len(NAMES))), index=months, columns=NAMES)
    long = x.rename_axis(index="date", columns="listing_id").reset_index().melt(id_vars="date", value_name="f")
    features = long.set_index(["date", "listing_id"])
    members = pd.DataFrame(True, index=months, columns=NAMES)
    target = v2.ranked_target(x + rng.normal(0, 0.05, x.shape), members)
    out = mc.candidate_predictions(
        features, target, members, months, ["linear"], candidates=v2.CANDIDATES, root=tmp_path
    )
    assert mc.information_coefficients(out, target, members)["linear"].dropna().min() > 0.8


def test_deflating_by_more_trials_lowers_the_probability() -> None:
    days = pd.bdate_range("2015-01-01", "2020-12-31")
    rng = np.random.default_rng(4)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    net = benchmark + rng.normal(0.0001, 0.002, len(days))
    ten, twenty = deflated_active(net, benchmark, 0.4, 10), deflated_active(net, benchmark, 0.4, 20)
    assert 0 < twenty < ten < 1


def test_newest_by_score_shows_each_learners_last_run() -> None:
    def row(run_id: int, score: str, cagr: float) -> dict[str, object]:
        config = {"factors": [[f"ml:{score}", 1.0]], "enter": 0.3}
        return {"run_id": run_id, "config_json": json.dumps(config), "metrics_json": json.dumps({"cagr": cagr})}

    rows = pd.DataFrame([row(130, "knn", 0.14), row(131, "svr", 0.12), row(140, "knn", 0.15)])
    table = newest_by_score(rows)
    assert table.loc["knn", "run_id"] == 140 and table.loc["knn", "cagr"] == 0.15
    assert table.loc["svr", "run_id"] == 131
    assert newest_by_score(rows.iloc[:0]).empty
