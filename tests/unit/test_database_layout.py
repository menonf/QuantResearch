"""The database is laid out as the code is (docs/ARCHITECTURE.md sections 3 and 4), checked from the declarations.

* ``research_core`` declares the schemas ``core``, ``ingest``, ``ops`` and ``research``, and nothing else;
* each asset-class package is registered under its own name and declares one schema, named after it;
* whatever walks the whole database - a copy, an issuer merge - sees every asset class's tables, whichever command
  runs it.
"""

import importlib

import sqlalchemy as sa

from research_core.data import identity
from research_core.data.db import copy, schema

PACKAGES = {"equity": "equity_research", "fixed_income": "fixed_income_research"}


def _declared(module: object) -> set[str]:
    return {f"{t.schema}.{t.name}" for t in vars(module).values() if isinstance(t, sa.Table)}


def test_every_asset_class_is_registered_under_its_own_name() -> None:
    found = schema.load_asset_classes()
    assert set(found) == set(PACKAGES)
    for name, module in found.items():
        assert module.__name__ == f"{PACKAGES[name]}.data.schema"
        assert name == importlib.import_module(PACKAGES[name]).ASSET_CLASS


def test_the_core_declares_only_its_own_schemas() -> None:
    assert {name.split(".")[0] for name in _declared(schema)} == set(schema.CORE_SCHEMAS)


def test_every_table_belongs_to_the_package_that_owns_its_schema() -> None:
    found = schema.load_asset_classes()
    owned = {name: _declared(module) for name, module in found.items()}
    for name, tables in owned.items():
        assert tables, f"{name} declares no table"
        assert {t.split(".")[0] for t in tables} == {name}
    every = _declared(schema).union(*owned.values())
    assert every == set(schema.metadata.tables), "a table is declared outside the schema modules"
    assert schema.all_schemas() == (*schema.CORE_SCHEMAS, *sorted(PACKAGES))


def test_a_copy_and_an_issuer_merge_see_every_asset_class() -> None:
    copied = {t.fullname for t in copy._tables()}
    assert {"core.listing", "equity.price_bar", "fixed_income.bond_analytics"} <= copied
    of_issuer = {f"{t.fullname}.{c}" for t, c in identity._children(schema.issuer)}
    assert {"equity.financial_fact.issuer_id", "fixed_income.bond_analytics.issuer_id"} <= of_issuer
    of_listing = {f"{t.fullname}.{c}" for t, c in identity._children(schema.listing)}
    assert {"equity.price_bar.listing_id", "fixed_income.index_constituent.listing_id"} <= of_listing
