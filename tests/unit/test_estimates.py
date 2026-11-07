import datetime as dt
import threading
import time
from typing import Any, cast

import pandas as pd
import pytest
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, PendingRollbackError

from equity_research import ops
from equity_research.ingest.vendors.refinitiv import fields as f
from equity_research.ingest.vendors.refinitiv import normalize as norm
from equity_research.ingest.vendors.refinitiv.client import LsegClient, RequestShapeError, RunawayRequestError
from equity_research.ops import KeepAlive
from equity_research.pipelines import load_estimates as load_estimates_module
from equity_research.pipelines.load_estimates import last_month_end, live_first, month_ends


def test_consensus_reads_any_measures_columns() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["AAPL.OQ", "AAPL.OQ", "OLD.N^B24"],
            "Revenue - Mean": [240.9e9, 241.0e9, None],  # no analyst left: no row
            "Calc Date": ["2016-03-31", "2016-04-30", "2016-03-31"],
            "Period End Date": ["2017-09-30", "2017-09-30", "2016-12-31"],
            "Revenue - Median": [240.0e9, 241.0e9, None],
            "Revenue - Standard Deviation": [5e9, 4e9, None],
            "Revenue - Number of Included Estimates": [39, 40, None],
        }
    )
    out = norm.consensus(raw)
    assert out["ric"].tolist() == ["AAPL.OQ", "AAPL.OQ"]
    assert out["as_of_date"].tolist() == [pd.Timestamp("2016-03-31"), pd.Timestamp("2016-04-30")]
    assert out["estimates"].tolist() == [39, 40] and out["mean"].iloc[0] == 240.9e9
    with pytest.raises(norm.MissingColumnsError, match="ending ' - Median'"):
        norm.consensus(raw.drop(columns="Revenue - Median"))


def test_actuals_keep_when_they_were_reported() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["AAPL.OQ"],
            "Earnings Per Share - Actual": [2.305],
            "Period End Date": ["2015-09-30"],
            "Report Date": ["2015-10-27 16:30:00"],
        }
    )
    out = norm.actuals(raw).iloc[0]
    assert out["value"] == 2.305 and out["announced_at"] == pd.Timestamp("2015-10-27 16:30")


def test_revisions_are_dated_by_the_day_asked() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["AAPL.OQ", "OLD.N^E11"],
            "Number of Analysts Revising Up": [1, None],
            "Number of Analysts Revising Down": [4, None],  # gone by then: no row
        }
    )
    out = norm.revisions(raw, "2015-06-30")
    assert out["ric"].tolist() == ["AAPL.OQ"]
    assert (out["up"].tolist(), out["down"].tolist()) == ([1], [4])
    assert out["as_of_date"].iloc[0] == pd.Timestamp("2015-06-30")


def test_per_share_figures_go_back_into_the_shares_of_their_own_date() -> None:
    # LSEG, asked in 2026, quotes Apple's 2015 consensus in post-2020 shares (2.12); in 2015's shares it was 4x that
    rows = pd.DataFrame(
        {
            "listing_id": [1, 1, 2],
            "as_of_date": pd.to_datetime(["2015-01-31", "2021-01-31", "2015-01-31"]),
            "mean": [2.12, 3.0, 5.0],
        }
    )
    splits = pd.DataFrame(
        {
            "listing_id": [1, 1, 1],
            "ex_date": pd.to_datetime(["2014-06-09", "2020-08-31", "2027-01-04"]),
            "ratio": [7.0, 4.0, 2.0],  # before the date; between the date and the fetch; after the fetch
        }
    )
    out = norm.in_shares_of_their_date(rows, ["mean"], "as_of_date", splits, pd.Timestamp("2026-10-30"))
    assert out["mean"].tolist() == pytest.approx([8.48, 3.0, 5.0])


def test_month_ends_and_a_renamed_listing_keeps_its_live_rics_answer() -> None:
    assert month_ends("2024-01-15", "2024-03-31") == ["2024-01-31", "2024-02-29", "2024-03-31"]
    rows = pd.DataFrame({"ric": ["CCO.N^A24", "CCC.N"], "listing_id": [7, 7], "mean": [1.0, 2.0]})
    assert live_first(rows, ["listing_id"])["ric"].tolist() == ["CCC.N"]


def test_the_revision_fields_are_never_asked_as_a_history() -> None:
    # documented in the fields module and kept by the load: a monthly history of these never answers
    assert f.REVISION_FIELDS == ["TR.NumEstRevisingUp", "TR.NumEstRevisingDown"]
    assert "Frq" not in f.REVISION_COLUMNS and set(f.REVISION_PERIODS) == {"FY1", "FY2"}


class _Counter:
    """Workspace's request count, falling while a runaway call re-sends itself."""

    def __init__(self) -> None:
        self.left = 1000

    def allowance(self) -> dict[str, int]:
        return {"daily_request_remaining": self.left}


def _guarded(monkeypatch: pytest.MonkeyPatch, counter: _Counter) -> LsegClient:
    client = LsegClient(pause_seconds=0.0, watch_seconds=0.01, max_retries=3)
    monkeypatch.setattr(client, "allowance", counter.allowance)
    monkeypatch.setattr(client, "close", lambda: None)
    return client


def test_a_call_that_keeps_costing_without_an_answer_is_stopped_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter, release, calls = _Counter(), threading.Event(), []

    def runaway() -> Any:
        calls.append(1)
        while not release.wait(0.005):
            counter.left -= 1  # the library re-sending, a request each time
        return pd.DataFrame()

    client = _guarded(monkeypatch, counter)
    with pytest.raises(RunawayRequestError, match="no answer"):
        client._call(runaway, "revisions as a history")
    release.set()
    assert len(calls) == 1  # asking again would only run away again


def test_a_call_that_answers_or_fails_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _guarded(monkeypatch, _Counter())
    assert client._call(lambda: pd.DataFrame({"a": [1]}), "fine")["a"].tolist() == [1]

    def refused() -> Any:
        raise RuntimeError("Unable to resolve all requested fields in ['TR.NOPE']")

    with pytest.raises(RequestShapeError, match=r"TR.NOPE"):
        client._call(refused, "unknown field")


def test_the_load_ends_at_the_last_closed_month_and_a_refresh_waits_for_the_next() -> None:
    assert last_month_end(dt.date(2026, 10, 30)) == dt.date(2026, 9, 30)
    assert last_month_end(dt.date(2026, 11, 1)) == dt.date(2026, 10, 31)
    assert last_month_end(dt.date(2026, 10, 31)) == dt.date(2026, 9, 30)  # the day itself has not closed yet


def test_a_refresh_asks_only_the_month_ends_it_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[dict[str, Any]] = []

    def load(engine: Engine, **kw: Any) -> dict[str, Any]:
        asked.append(kw)
        return {"complete": True}

    monkeypatch.setattr(load_estimates_module, "load_estimates", load)
    monkeypatch.setattr(load_estimates_module, "newest_estimate", lambda engine, market: dt.date(2026, 9, 30))
    quiet = load_estimates_module.refresh_estimates(cast(Engine, None), today=dt.date(2026, 10, 30))
    assert asked == [] and quiet["complete"] and "up_to_date" in quiet  # September is held; October not closed
    load_estimates_module.refresh_estimates(cast(Engine, None), today=dt.date(2026, 11, 3))
    assert asked[-1]["since"] == dt.date(2026, 10, 1) and asked[-1]["end"] == "2026-10-31"
    monkeypatch.setattr(load_estimates_module, "newest_estimate", lambda engine, market: None)
    load_estimates_module.refresh_estimates(cast(Engine, None), today=dt.date(2026, 11, 3))
    assert asked[-1]["since"] is None  # nothing held yet: the whole history
    load_estimates_module.refresh_estimates(cast(Engine, None), today=dt.date(2026, 11, 3), market="UK")
    assert asked[-1]["market"] == "UK"  # each market refreshes from its own newest consensus


class _LockSession:
    """A connection that counts the keep-alive's touches; ``fail_after`` touches it is gone."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.touches = 0
        self.fail_after = fail_after

    def exec_driver_sql(self, sql: str) -> Any:
        if self.fail_after is not None and self.touches >= self.fail_after:
            raise DBAPIError("SELECT 1", None, Exception("connection forcibly closed"))
        self.touches += 1
        return self

    def scalar(self) -> int:
        return 1

    def commit(self) -> None:
        return None


def test_the_writer_locks_session_is_kept_alive_and_a_loss_is_noticed() -> None:
    session = _LockSession()
    alive = KeepAlive(cast(Connection, session), every=0.01)
    deadline = time.time() + 2
    while session.touches < 3 and time.time() < deadline:
        time.sleep(0.01)
    alive.stop()
    assert session.touches >= 3 and not alive.lost
    gone = KeepAlive(cast(Connection, _LockSession(fail_after=1)), every=0.01)
    deadline = time.time() + 2
    while not gone.lost and time.time() < deadline:
        time.sleep(0.01)
    gone.stop()
    assert gone.lost


class _DeadLock(_LockSession):
    """A lock connection the server dropped: every statement fails the way SQLAlchemy fails them."""

    closed = False

    def exec_driver_sql(self, sql: str) -> Any:
        raise PendingRollbackError("Can't reconnect until invalid transaction is rolled back.")

    def close(self) -> None:
        self.closed = True


def test_a_lock_lost_with_its_session_is_not_given_back() -> None:
    dead = _DeadLock()
    ops._release(cast(Connection, dead), lost=True)  # used to raise PendingRollbackError after the job's work was done
    assert dead.closed
