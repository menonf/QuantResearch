"""Create or update the database's tables from Python (what ``uv run alembic upgrade head`` does)."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy.engine import Engine

from research_core.data.db.engine import get_engine
from research_core.settings import REPO_ROOT


def upgrade_to_latest(engine: Engine | None = None) -> str:
    """Apply every migration not yet applied (to ``engine``, else the configured database); returns its revision."""
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.attributes["configure_logger"] = False  # keep the caller's logging (a notebook's, say)
    target = engine or get_engine()
    with target.begin() as conn:
        config.attributes["connection"] = conn
        command.upgrade(config, "head")
    with target.connect() as conn:
        return str(MigrationContext.configure(conn).get_current_revision())


def head_revision() -> str:
    """The newest migration's revision (what ``upgrade_to_latest`` brings a database to)."""
    from alembic.script import ScriptDirectory  # noqa: PLC0415

    head = ScriptDirectory.from_config(Config(str(REPO_ROOT / "alembic.ini"))).get_current_head()
    return str(head)
