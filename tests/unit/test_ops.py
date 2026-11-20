"""The writer lock is given back - or found already gone with its session - without failing a finished job."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError

from research_core.data import ops


class _Lock:
    """A lock session whose release fails with ``error`` (None: succeeds)."""

    def __init__(self, error: DBAPIError | None) -> None:
        self.error = error
        self.closed = False

    def exec_driver_sql(self, sql: str, params: Any) -> None:
        if self.error is not None:
            raise self.error

    def commit(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _error(invalidated: bool) -> DBAPIError:
    return DBAPIError(
        "EXEC sp_releaseapplock", None, Exception("(10054) forcibly closed"), connection_invalidated=invalidated
    )


def test_a_session_the_server_closed_has_no_lock_left_to_release() -> None:
    lock = _Lock(_error(invalidated=True))
    ops._release(cast(Connection, lock))
    assert lock.closed


def test_any_other_release_error_is_raised() -> None:
    lock = _Lock(_error(invalidated=False))
    with pytest.raises(DBAPIError):
        ops._release(cast(Connection, lock))
    assert lock.closed
