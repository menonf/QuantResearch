"""The research-run registry on a scratch database: what counts as another trial."""

from pathlib import Path

import pytest
from conftest import scratch_database

from equity_research.research.registry import record_run, trial_count, trials_with
from equity_research.research.runs import score_metrics

pytestmark = pytest.mark.integration


def test_running_a_registered_configuration_again_is_not_another_trial(tmp_path: Path) -> None:
    with scratch_database("equity_research_test_registry", tmp_path) as engine:
        first = {"factors": "value", "enter": 0.1}
        assert trials_with(engine, "idea", first) == 1
        record_run(engine, first, {"sharpe": 0.5}, family="idea")
        assert trials_with(engine, "idea", first) == 1  # the same configuration, run again
        assert trials_with(engine, "idea", {**first, "enter": 0.2}) == 2  # a new variant
        record_run(engine, first, {"sharpe": 0.5}, family="idea")
        assert trial_count(engine, "idea") == 1
        assert trials_with(engine, "another idea", first) == 1  # trials are counted per family


def test_score_metrics_shows_each_scores_newest_run(tmp_path: Path) -> None:
    with scratch_database("equity_research_test_registry", tmp_path) as engine:
        knn, svr = {"factors": [["ml:knn", 1.0]]}, {"factors": [["ml:svr", 1.0]]}
        record_run(engine, knn, {"cagr": 0.14}, family="study")
        record_run(engine, svr, {"cagr": 0.12}, family="study")
        newest = record_run(engine, knn, {"cagr": 0.15}, family="study")  # the same learner, run again
        record_run(engine, knn, {"cagr": 0.99}, family="another study")
        table = score_metrics(engine, "study")
        assert sorted(table.index) == ["knn", "svr"]
        assert table.loc["knn", "run_id"] == newest and table.loc["knn", "cagr"] == 0.15
        assert score_metrics(engine, "no such study").empty
