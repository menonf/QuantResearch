"""Configuration: environment variables first, then the OS keyring; never stored in the repository.

| Setting | Environment variable | Keyring key (service ``equity_research``) | Default |
|---|---|---|---|
| database server | ``EQR_DB_SERVER`` | ``db_server`` | - |
| database name | ``EQR_DB_NAME`` | ``db_name`` | ``equity_research`` |
| database user / password | ``EQR_DB_USER`` / ``EQR_DB_PASSWORD`` | ``db_user`` / ``db_password`` | - |
| ODBC driver | ``EQR_ODBC_DRIVER`` | - | ``ODBC Driver 18 for SQL Server`` |
| data folder (raw zone, cache) | ``EQR_DATA_DIR`` | - | ``<repository>/data/<database name>`` |
| SEC contact (User-Agent) | ``EQR_SEC_USER_AGENT`` | ``sec_user_agent`` | - |
| Tiingo API key | ``EQR_TIINGO_API_KEY`` | ``tiingo_api_key`` | - |
| SimFin API key | ``EQR_SIMFIN_API_KEY`` | ``simfin_api_key`` | - |
| start of the research hold-out | ``EQR_HOLDOUT_START`` | - | ``2025-01-01`` |

When ``EQR_DB_SERVER`` is set, the database name, user and password come from the environment only: the keyring
holds the shared database's login, never a local one's (no user means Windows authentication). Vendor API keys
belong to the person, not to a database, so they come from the keyring whichever database is used.

Each database has its own data folder by default: the raw zone holds the files its ``ingest.batch`` rows point to,
and the cache mirrors that database only - two databases never share either.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import keyring

KEYRING_SERVICE = "equity_research"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _get(env: str, key: str | None = None, default: str | None = None) -> str | None:
    value = os.environ.get(env)
    if value:
        return value
    if key is not None:
        stored = keyring.get_password(KEYRING_SERVICE, key)
        if stored:
            return str(stored)
    return default


class SettingMissingError(RuntimeError):
    """A required setting is in neither the environment nor the keyring."""


@dataclass(frozen=True)
class DatabaseSettings:
    """Connection settings for the system of record (Azure SQL / SQL Server)."""

    server: str
    name: str
    user: str | None
    password: str | None
    driver: str

    @property
    def uses_sql_auth(self) -> bool:
        """True when a user and password are configured (Azure SQL); else Windows authentication."""
        return bool(self.user and self.password)


@dataclass(frozen=True)
class Settings:
    """All settings of one process."""

    database: DatabaseSettings
    data_dir: Path
    sec_user_agent: str | None
    holdout_start: str = "2025-01-01"

    @property
    def raw_dir(self) -> Path:
        """The raw zone: immutable vendor responses."""
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        """The local research caches (Parquet / DuckDB): one folder per asset class, ``cache/<asset class>``."""
        return self.data_dir / "cache"


def _required(env: str, key: str, what: str) -> str:
    value = _get(env, key)
    if not value:
        raise SettingMissingError(
            f"{what} is not configured: set {env} or keyring.set_password({KEYRING_SERVICE!r}, {key!r}, ...)"
        )
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process settings (read once)."""
    # credentials belong to their server: a server named in the environment (a local copy, a fresh database) never
    # picks up the keyring's login for the shared one - without EQR_DB_USER it uses Windows authentication
    local = bool(os.environ.get("EQR_DB_SERVER"))
    database = DatabaseSettings(
        server=_required("EQR_DB_SERVER", "db_server", "the database server"),
        name=_get("EQR_DB_NAME", None if local else "db_name", "equity_research") or "equity_research",
        user=_get("EQR_DB_USER", None if local else "db_user"),
        password=_get("EQR_DB_PASSWORD", None if local else "db_password"),
        driver=_get("EQR_ODBC_DRIVER", default="ODBC Driver 18 for SQL Server") or "ODBC Driver 18 for SQL Server",
    )
    data_dir = Path(_get("EQR_DATA_DIR") or REPO_ROOT / "data" / database.name)
    return Settings(
        database=database,
        data_dir=data_dir,
        sec_user_agent=_get("EQR_SEC_USER_AGENT", "sec_user_agent"),
        holdout_start=_get("EQR_HOLDOUT_START", default="2025-01-01") or "2025-01-01",
    )


def api_key(vendor: str) -> str:
    """A vendor's API key, or :class:`SettingMissingError` saying where to put it (never printed, never in a URL)."""
    return _required(f"EQR_{vendor.upper()}_API_KEY", f"{vendor}_api_key", f"the {vendor} API key")
