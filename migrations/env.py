"""Alembic environment: the engine and metadata come from the package (no URL in alembic.ini)."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from equity_research.db.engine import get_engine
from equity_research.db.schema import SCHEMAS, metadata

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    """Only the project's schemas are managed (dbo holds just alembic_version)."""
    if type_ == "schema":
        return name in SCHEMAS
    return True


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        include_schemas=True,
        include_name=include_name,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """On the connection a caller handed over (``config.attributes["connection"]``), else the configured database."""
    given = config.attributes.get("connection")
    if given is not None:
        _run(given)
        return
    with get_engine().connect() as connection:
        _run(connection)


if context.is_offline_mode():
    raise SystemExit("offline migrations are not supported - run against the database")
run_migrations_online()
