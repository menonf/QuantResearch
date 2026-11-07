import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from equity_research.db import engine as db_engine
from equity_research.db import migrate
from equity_research.db.engine import get_engine
from equity_research.settings import get_settings


@pytest.fixture(scope="session")
def engine() -> Engine:
    """The configured database - never the shared Azure one unless asked for (EQR_ALLOW_AZURE_TESTS=1)."""
    server = get_settings().database.server
    if "database.windows.net" in server and os.environ.get("EQR_ALLOW_AZURE_TESTS") != "1":
        pytest.exit(
            f"integration tests would write to the shared database on {server}: point EQR_DB_SERVER / EQR_DB_NAME at a "
            "local one, or set EQR_ALLOW_AZURE_TESTS=1",
            returncode=2,
        )
    return get_engine()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    """A connection whose work is rolled back at the end of the test."""
    connection = engine.connect()
    tx = connection.begin()
    try:
        yield connection
    finally:
        tx.rollback()
        connection.close()


@contextmanager
def scratch_database(name: str, data_dir: Path) -> Iterator[Engine]:
    """A new, migrated database next to the configured one (never on Azure), dropped afterwards."""
    settings = get_settings()
    if "database.windows.net" in settings.database.server:
        pytest.skip("the scratch database is created on a local SQL Server only")
    mp = pytest.MonkeyPatch()
    mp.setenv("EQR_DB_SERVER", settings.database.server)
    mp.setenv("EQR_DB_NAME", name)
    mp.setenv("EQR_DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    db_engine.get_engine.cache_clear()
    try:
        db_engine.create_database(get_settings().database)
        migrate.upgrade_to_latest()
        yield db_engine.get_engine()
    finally:
        db_engine.get_engine().dispose()
        master = sa.create_engine(
            db_engine.connection_url(
                get_settings().database.__class__(**{**get_settings().database.__dict__, "name": "master"})
            ),
            isolation_level="AUTOCOMMIT",
        )
        with master.connect() as c:
            c.execute(sa.text(f"ALTER DATABASE [{name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE"))
            c.execute(sa.text(f"DROP DATABASE [{name}]"))
        master.dispose()
        mp.undo()
        get_settings.cache_clear()
        db_engine.get_engine.cache_clear()
