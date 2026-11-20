"""The engine waits for a paused serverless database instead of failing on its first slow login."""

from __future__ import annotations

import contextlib
import time
from typing import Any, cast
from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from research_core.data.db import engine as db_engine
from research_core.settings import DatabaseSettings

AZURE = DatabaseSettings("example.database.windows.net", "equity_research", "user", "secret", "ODBC Driver 18")


class _Conn:
    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, statement: object) -> None:
        return None


class _Engine:
    """Fails the first ``failures`` connections with ``message``, then connects."""

    def __init__(self, failures: int, message: str) -> None:
        self.failures = failures
        self.message = message
        self.attempts = 0

    def connect(self) -> _Conn:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise DBAPIError("SELECT 1", None, Exception(self.message))
        return _Conn()


def _patch(monkeypatch: pytest.MonkeyPatch, fake: _Engine) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def create_engine(url: sa.URL, **kwargs: Any) -> _Engine:
        seen.update(kwargs, url=url)
        return fake

    monkeypatch.setattr(sa, "create_engine", create_engine)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return seen


def test_login_timeout_is_given_to_the_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, _Engine(0, ""))
    db_engine.make_engine(AZURE)
    assert seen["connect_args"] == {"timeout": db_engine.LOGIN_TIMEOUT}
    assert "LoginTimeout" not in seen["url"].query  # the ODBC driver rejects it as a connection-string attribute


def test_a_login_timed_out_while_resuming_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Engine(2, "[08001] TCP Provider: Timeout error [258].  (258) (SQLDriverConnect)")
    _patch(monkeypatch, fake)
    db_engine.make_engine(AZURE)
    assert fake.attempts == 3


def test_other_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Engine(1, "[28000] Login failed for user (18456)")
    _patch(monkeypatch, fake)
    with pytest.raises(DBAPIError):
        db_engine.make_engine(AZURE)
    assert fake.attempts == 1


class _Waking:
    """An engine whose database answers 'not currently available' a few times, as a resuming one does."""

    def __init__(self, failures: int, code: str = "40613") -> None:
        self.failures, self.code, self.calls = failures, code, 0
        self.url = sa.make_url("mssql+pyodbc://server/equity_research")

    def connect(self) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise DBAPIError("SELECT 1", None, Exception(f"not currently available ({self.code})"))
        return contextlib.nullcontext(mock.Mock())


def test_wake_waits_for_a_resuming_database_and_raises_anything_else(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    resuming = _Waking(2)
    db_engine.wake(cast(Engine, resuming))
    assert resuming.calls == 3
    with pytest.raises(DBAPIError):
        db_engine.wake(cast(Engine, _Waking(1, code="18456")))  # a failed login is not a pause
    with pytest.raises(DBAPIError):
        db_engine.wake(cast(Engine, _Waking(9)), attempts=3)
