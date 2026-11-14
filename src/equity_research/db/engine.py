"""Engine for the system of record (Azure SQL or SQL Server via ODBC)."""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from functools import lru_cache

import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from equity_research.settings import DatabaseSettings, get_settings

log = logging.getLogger(__name__)

#: A paused serverless database answers the first connection with "not currently available" (40613) while it
#: resumes, which takes up to a minute - or keeps the login waiting until it times out (258, "delay in login
#: response").
RESUME_ERROR_CODES = ("40613", "40197", "40501", "49918", "(258)")
#: Seconds a login may wait. The ODBC driver ignores a timeout in the connection string; pyodbc's ``timeout`` sets it.
LOGIN_TIMEOUT = 60


def connection_url(db: DatabaseSettings) -> sa.URL:
    """SQLAlchemy URL for ``db`` (SQL authentication when a user is configured, else Windows authentication)."""
    query = {"driver": db.driver, "Encrypt": "yes", "TrustServerCertificate": "no"}
    if not db.uses_sql_auth:
        query.update({"Trusted_Connection": "yes", "Encrypt": "no", "TrustServerCertificate": "yes"})
    return sa.URL.create(
        "mssql+pyodbc",
        username=db.user if db.uses_sql_auth else None,
        password=db.password if db.uses_sql_auth else None,
        host=db.server,
        port=1433 if "." in db.server else None,
        database=db.name,
        query=query,
    )


def make_engine(db: DatabaseSettings, *, wake_attempts: int = 6) -> Engine:
    """An engine with fast bulk inserts; waits for a paused serverless database to resume."""
    engine = sa.create_engine(
        connection_url(db),
        connect_args={"timeout": LOGIN_TIMEOUT},
        fast_executemany=True,
        pool_pre_ping=True,
        pool_recycle=1800,
        use_insertmanyvalues=False,
    )
    wake(engine, attempts=wake_attempts)
    return engine


def wake(engine: Engine, *, attempts: int = 6) -> None:
    """Connect once, waiting for a paused serverless database to resume - before writing after a long pause.

    An engine waits when it is made; a load that then does hours of work without the database (downloads, model
    scoring) finds it paused again, and its next connection must wait the same way.
    """
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            return
        except DBAPIError as exc:
            if attempt == attempts or not any(code in str(exc.orig) for code in RESUME_ERROR_CODES):
                raise
            log.info("database is resuming - retry %d/%d in 20s", attempt, attempts)
            time.sleep(20)


def create_database(db: DatabaseSettings) -> bool:
    """Create the database ``db`` names when the server does not have it yet; True when it was created.

    For a fresh local SQL Server database. An Azure SQL database is created in the portal (its pricing tier is
    chosen there), so on Azure this only confirms that it exists.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", db.name):
        raise ValueError(f"database name {db.name!r}: use letters, digits and _ only")
    master = sa.create_engine(connection_url(dataclasses.replace(db, name="master")), isolation_level="AUTOCOMMIT")
    try:
        with master.connect() as conn:
            if conn.execute(sa.text("SELECT DB_ID(:n)"), {"n": db.name}).scalar() is not None:
                return False
            conn.execute(sa.text(f"CREATE DATABASE [{db.name}]"))
            return True
    finally:
        master.dispose()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """The process engine for the configured database."""
    return make_engine(get_settings().database)


def describe(url: sa.URL) -> str:
    """``server/database`` without credentials (for logs)."""
    return f"{url.host}/{url.database}"


__all__ = ["connection_url", "describe", "get_engine", "make_engine"]
