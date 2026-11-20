from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from research_core.data import ops
from research_core.data.raw_store import get_batch, land_frame, prune, read_frame

pytestmark = pytest.mark.integration


def test_landing_the_same_frame_twice_returns_the_same_batch(conn: Connection, tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    first = land_frame(conn, vendor="sec", dataset="test_frame", source="unit test", frame=frame, raw_dir=tmp_path)
    second = land_frame(conn, vendor="sec", dataset="test_frame", source="unit test", frame=frame, raw_dir=tmp_path)
    assert first.is_new and not second.is_new
    assert first.batch_id == second.batch_id
    assert first.path.exists() and first.path.is_relative_to(tmp_path / "sec" / "test_frame")
    pd.testing.assert_frame_equal(read_frame(get_batch(conn, first.batch_id, raw_dir=tmp_path)), frame)


def test_a_second_writer_is_refused(engine: Engine) -> None:
    with ops.job("test_lock_holder", engine) as run:
        with pytest.raises(ops.JobLockedError, match="test_lock_holder"), ops.job("test_second_writer", engine):
            pass
        run.finish(rows=0)
    with engine.begin() as c:  # the test's own job rows
        status = c.execute(sa.text("SELECT status FROM ops.job_run WHERE job = 'test_lock_holder'")).scalars().all()
        c.execute(sa.text("DELETE FROM ops.job_run WHERE job IN ('test_lock_holder', 'test_second_writer')"))
    assert status == ["succeeded"]


def test_the_next_job_of_the_same_process_gets_the_lock(engine: Engine) -> None:
    for name in ("test_first_job", "test_next_job"):  # a refresh runs its loads one after another
        with ops.job(name, engine) as run:
            run.finish(rows=0)
    with engine.begin() as c:
        status = (
            c.execute(
                sa.text(
                    "SELECT status FROM ops.job_run WHERE job IN ('test_first_job', 'test_next_job') ORDER BY run_id"
                )
            )
            .scalars()
            .all()
        )
        c.execute(sa.text("DELETE FROM ops.job_run WHERE job IN ('test_first_job', 'test_next_job')"))
    assert status == ["succeeded", "succeeded"]


def test_pruning_keeps_the_newest_files_and_every_batch_row(conn: Connection, tmp_path: Path) -> None:
    batches = [
        land_frame(
            conn,
            vendor="sec",
            dataset="test_bulk",
            source=f"copy {i}",
            frame=pd.DataFrame({"x": [i]}),
            raw_dir=tmp_path,
        )
        for i in range(3)
    ]
    policy = {("sec", "test_bulk"): 2}
    planned = prune(conn, retention=policy, raw_dir=tmp_path, dry_run=True)
    assert [p.batch_id for p in planned] == [batches[0].batch_id] and batches[0].path.exists()  # a dry run: nothing
    removed = prune(conn, retention=policy, raw_dir=tmp_path)
    assert [p.batch_id for p in removed] == [batches[0].batch_id]
    assert not batches[0].path.exists() and batches[1].path.exists() and batches[2].path.exists()
    status = conn.exec_driver_sql("SELECT status FROM ingest.batch WHERE batch_id = ?", (batches[0].batch_id,)).scalar()
    assert status == "superseded"  # the row stays: rows loaded from it still name their delivery
    assert prune(conn, retention=policy, raw_dir=tmp_path) == []  # nothing left to do
