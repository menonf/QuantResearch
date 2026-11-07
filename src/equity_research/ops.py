"""Job runs and the writer lock.

    with job("load_lseg", engine) as run:
        ...
        run.finish(rows=n, detail={"prices": result})

Every pipeline run is a row in ``ops.job_run``. Writers take an exclusive application lock (``sp_getapplock`` on
:data:`LOCK_NAME`), held by a dedicated session for the duration of the job, so two loads never write at once - the
second fails immediately with :class:`JobLockedError`. A crashed process releases the lock when its session closes.

The lock's session would sit idle for the whole job, and Azure drops an idle connection - the lock with it (seen in an
overnight estimates load, 2026-10-31) - so a background thread touches it every :data:`KEEPALIVE_SECONDS`.
"""

from __future__ import annotations

import getpass
import json
import logging
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError

from equity_research.db import schema
from equity_research.raw.store import code_version

log = logging.getLogger(__name__)

LOCK_NAME = "equity_research_writer"
#: How often the lock's session is touched while a job runs (well inside Azure's idle-connection timeout).
KEEPALIVE_SECONDS = 240.0


class JobLockedError(RuntimeError):
    """Another writer holds the lock."""


@dataclass
class JobRun:
    """The run being recorded."""

    run_id: int
    job: str
    rows: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def finish(self, rows: int | None = None, detail: dict[str, Any] | None = None) -> None:
        """Record the outcome reported when the job succeeds."""
        self.rows = rows
        if detail:
            self.detail.update(detail)


def _acquire(engine: Engine, job: str) -> Connection:
    conn = engine.connect()
    result = conn.exec_driver_sql(
        "DECLARE @r INT; EXEC @r = sp_getapplock @Resource = ?, @LockMode = 'Exclusive', @LockOwner = 'Session', "
        "@LockTimeout = 0; SELECT @r",
        (LOCK_NAME,),
    ).scalar()
    conn.commit()
    if result is None or int(result) < 0:
        holder = conn.exec_driver_sql(
            "SELECT TOP 1 job FROM ops.job_run WHERE status = 'running' ORDER BY started_at DESC"
        ).scalar()
        conn.close()
        raise JobLockedError(f"{job}: another writer holds the lock (running: {holder or 'unknown'})")
    return conn


class KeepAlive:
    """Touch ``conn`` every ``every`` seconds until stopped, so the server does not close it as idle.

    The thread is the connection's only user while it runs (the job writes through other connections). A session
    lost all the same is logged when it happens: the job carries on, but no longer holds the writer lock.
    """

    def __init__(self, conn: Connection, every: float = KEEPALIVE_SECONDS) -> None:
        self._conn = conn
        self._every = every
        self._stop = threading.Event()
        self.lost = False
        self._thread = threading.Thread(target=self._run, name="writer lock keep-alive", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            try:
                self._conn.exec_driver_sql("SELECT 1").scalar()
                self._conn.commit()  # ends the read's transaction; the lock belongs to the session and stays
            except DBAPIError as exc:
                self.lost = True
                log.warning("the writer lock's session was lost mid-job: %s", str(exc.orig)[:160])
                return

    def stop(self) -> None:
        """Stop touching the connection (before it is released)."""
        self._stop.set()
        self._thread.join(timeout=self._every + 5.0)


@contextmanager
def job(name: str, engine: Engine, *, exclusive: bool = True) -> Iterator[JobRun]:
    """Record a run of ``name``; with ``exclusive`` hold the writer lock while it runs."""
    lock = _acquire(engine, name) if exclusive else None
    alive = KeepAlive(lock) if lock is not None else None
    try:
        with engine.begin() as conn:
            run_id = conn.execute(
                sa.insert(schema.job_run)
                .values(
                    job=name[:128],
                    status="running",
                    host=socket.gethostname(),
                    user_name=getpass.getuser(),
                    code_version=code_version(),
                )
                .returning(schema.job_run.c.run_id)
            ).scalar_one()
        run = JobRun(int(run_id), name)
        try:
            yield run
        except BaseException as exc:
            _close(engine, run, "failed", {"error": f"{type(exc).__name__}: {exc}"[:4000], **run.detail})
            raise
        _close(engine, run, "succeeded", run.detail)
    finally:
        if alive is not None:
            alive.stop()
        if lock is not None:
            _release(lock, lost=alive is not None and alive.lost)


def _release(lock: Connection, *, lost: bool = False) -> None:
    """Give the lock back, then the connection.

    Closing alone returns the connection to the pool with its session - and the session's lock - still open, which
    locked the next job of the same process out. A session the keep-alive found ``lost`` took the lock with it: there
    is nothing to give back, and the dead connection refuses any statement (SQLAlchemy's ``PendingRollbackError``,
    which ended an 11-hour load that had finished its work, 2026-10-31).
    """
    if lost:
        log.warning("the writer lock ended with its session earlier in the job: nothing to give back")
        _discard(lock)
        return
    try:
        lock.exec_driver_sql("EXEC sp_releaseapplock @Resource = ?, @LockOwner = 'Session'", (LOCK_NAME,))
        lock.commit()
    except DBAPIError as exc:
        if not exc.connection_invalidated:
            raise
        # the server closed the session (Azure drops a connection left idle through an hour-long load): the lock
        # ended with it, so there is nothing to give back - and the job's work, already recorded, must not fail now
        log.warning("the writer lock's session was already closed by the server: %s", str(exc.orig)[:160])
    finally:
        _discard(lock)


def _discard(lock: Connection) -> None:
    """Close the lock's connection; one the server already dropped may complain - it is gone either way."""
    try:
        lock.close()
    except sa.exc.SQLAlchemyError as exc:
        log.warning("closing the writer lock's connection: %s", str(exc)[:160])


def _close(engine: Engine, run: JobRun, status: str, detail: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.update(schema.job_run)
            .where(schema.job_run.c.run_id == run.run_id)
            .values(
                status=status,
                finished_at=sa.func.sysutcdatetime(),
                rows=run.rows,
                detail=json.dumps(detail, default=str) if detail else None,
            )
        )
