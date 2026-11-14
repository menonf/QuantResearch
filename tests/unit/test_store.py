from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.research import panels, regression, store


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame(np.arange(6.0).reshape(3, 2), index=dates, columns=pd.Index([101, 202], dtype="int64"))


def test_an_artifact_is_read_back_with_its_listing_columns_and_its_sidecar(tmp_path: Path) -> None:
    frame = _panel()
    path = store.save(frame, "labels", "fwd_ret_21d", {"horizon": 21}, snapshot="2026-11-01T00:00:00", root=tmp_path)
    assert path.parent == tmp_path / "labels" and path.name.startswith("fwd_ret_21d-")
    back = store.load("labels", "fwd_ret_21d", {"horizon": 21}, snapshot="2026-11-01T00:00:00", root=tmp_path)
    assert back is not None
    pd.testing.assert_frame_equal(back, frame, check_freq=False, check_column_type=False)
    assert pd.api.types.is_integer_dtype(back.columns)  # listing ids come back as integers
    meta = store.sidecar(path)
    assert meta["config"] == {"horizon": 21} and meta["data_snapshot"] == "2026-11-01T00:00:00" and meta["rows"] == 3


def test_another_config_or_data_snapshot_is_another_artifact(tmp_path: Path) -> None:
    store.save(_panel(), "labels", "x", {"horizon": 21}, snapshot="a", root=tmp_path)
    assert store.load("labels", "x", {"horizon": 5}, snapshot="a", root=tmp_path) is None
    assert store.load("labels", "x", {"horizon": 21}, snapshot="b", root=tmp_path) is None  # new data: rebuilt


def test_a_new_returns_method_is_another_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store.save(_panel(), "labels", "x", {"horizon": 21}, snapshot="a", root=tmp_path)
    monkeypatch.setattr(store, "RETURNS_VERSION", panels.RETURNS_VERSION + 1)
    assert store.load("labels", "x", {"horizon": 21}, snapshot="a", root=tmp_path) is None  # rebuilt, not reused


def test_cached_builds_once(tmp_path: Path) -> None:
    calls: list[int] = []

    def build() -> pd.DataFrame:
        calls.append(1)
        return _panel()

    first = store.cached("features", "f", {"a": 1}, build, root=tmp_path)
    again = store.cached("features", "f", {"a": 1}, build, root=tmp_path)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, again, check_freq=False, check_column_type=False)


def test_text_columns_stay_text(tmp_path: Path) -> None:
    frame = pd.DataFrame({"cagr": [0.1], "vol": [0.2]}, index=["a"])
    store.save(frame, "regression", "t", {}, root=tmp_path)
    back = store.load("regression", "t", {}, root=tmp_path)
    assert back is not None and list(back.columns) == ["cagr", "vol"]


def test_the_regression_check_lists_only_numbers_that_moved() -> None:
    before = pd.DataFrame(
        {
            "part": ["strategy"] * 3 + ["risk_model"],
            "row": ["multi_factor_lo", "multi_factor_lo", "garp_sp500", "Market"],
            "measure": ["cagr", "vol", "cagr", "yearly return"],
            "value": [0.131, 0.18, 0.12, np.nan],
        }
    )
    after = before.copy()
    after.loc[0, "value"] = 0.131 + 1e-13  # floating-point noise: not a change
    assert regression.compare(before, after).empty
    after.loc[2, "value"] = 0.125  # a real change
    moved = regression.compare(before, after)
    assert moved[["row", "measure"]].to_numpy().tolist() == [["garp_sp500", "cagr"]]
    fewer = after.iloc[:3]  # a number that is no longer measured is a difference too
    assert "yearly return" in set(regression.compare(before, fewer)["measure"])


def test_long_numbers_name_their_part_row_and_measure() -> None:
    frame = pd.DataFrame({"cagr": [0.1, 0.2]}, index=pd.Index(["a", "b"], name="family"))
    long = regression._long(frame, "strategy")
    assert long.to_numpy().tolist() == [["strategy", "a", "cagr", 0.1], ["strategy", "b", "cagr", 0.2]]
