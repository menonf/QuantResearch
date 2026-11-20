"""A database built by the migrations is exactly the code's tables, with the reference rows every database holds."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from conftest import scratch_database

from research_core.data.db import schema

pytestmark = pytest.mark.integration


def test_the_migrated_database_is_laid_out_as_the_code_is(tmp_path: Path) -> None:
    with scratch_database("equity_research_test_migrations", tmp_path) as engine, engine.connect() as conn:
        schemas = schema.all_schemas()
        context = MigrationContext.configure(
            conn,
            opts={
                "include_schemas": True,
                "compare_type": True,
                "include_name": lambda name, kind, _: name in schemas if kind == "schema" else True,
            },
        )
        assert compare_metadata(context, schema.metadata) == []

        def rows(sql: str) -> list[tuple[object, ...]]:
            return [tuple(r) for r in conn.execute(sa.text(sql)).all()]

        assert rows("SELECT asset_class FROM core.asset_class ORDER BY 1") == [
            (a,) for a in schema.load_asset_classes()
        ]
        assert rows("SELECT vendor_id, vendor_code FROM core.vendor ORDER BY 1") == sorted(
            (vid, code) for code, (vid, _) in schema.VENDORS.items()
        )
        # every instrument type belongs to a registered asset class, or to all of them (NULL)
        owners = {r[0] for r in rows("SELECT DISTINCT asset_class FROM core.instrument_type")}
        assert owners <= {None, *schema.load_asset_classes()}
        assert rows("SELECT instrument_type FROM core.instrument_type WHERE instrument_type IN ('bond', 'equity')")
        assert rows("SELECT id_type FROM core.id_type WHERE id_type IN ('ISIN', 'ICE', 'CIK')")
