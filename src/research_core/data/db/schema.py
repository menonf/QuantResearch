"""The canonical data model (docs/ARCHITECTURE.md section 3) as SQLAlchemy Core tables: the shared part.

The database is laid out as the code is. This package owns four schemas:

* ``core`` - the asset classes, the vendors and the security master (issuers, listings, identifiers);
* ``ingest`` - the raw files landed; ``ops`` - job runs and quality checks;
* ``research`` - the run registry and the strategies' rule sets, each row naming its asset class.

Each asset-class package owns one schema named after it (``equity``, ``fixed_income``) and declares its tables on this
module's ``metadata``. It registers itself under the entry-point group :data:`ASSET_CLASS_ENTRY_POINTS`
(``pyproject.toml``), so :func:`load_asset_classes` gives whatever walks the whole database - the migrations, a
database copy, an issuer merge - every table without this package importing any asset class.

Conventions:

* every canonical row carries ``batch_id`` (provenance: the raw file it came from) and ``loaded_at`` (when we knew it);
* vendors are referred to by ``vendor_id``, a small fixed number (:data:`VENDORS`) - the same in every database, so
  tables copy between databases unchanged;
* facts that the world learns at a time carry ``published_at`` (when it could be known);
* identifier values use a case-sensitive binary collation - vendor codes such as ``ATH_pe`` and ``ATH_PE`` differ;
* dates are ``DATE``, instants ``DATETIME2`` in UTC.
"""

from __future__ import annotations

import datetime as dt
from functools import cache
from importlib.metadata import entry_points
from types import ModuleType
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.dialects import mssql

#: The schemas this package owns; every other schema is an asset class's, named after it.
CORE_SCHEMAS = ("core", "ingest", "ops", "research")

#: Where each asset-class package registers its schema module: ``<asset class> = "<package>.data.schema"``.
ASSET_CLASS_ENTRY_POINTS = "research_core.asset_classes"

metadata = sa.MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

CODE = sa.String(64, collation="Latin1_General_100_BIN2")  # identifier values: case-sensitive
UTC_NOW = sa.text("SYSUTCDATETIME()")


def instant() -> sa.types.TypeEngine[dt.datetime]:
    """``DATETIME2(0)``: an instant in UTC to the second."""
    return cast("sa.types.TypeEngine[dt.datetime]", mssql.DATETIME2(precision=0))  # type: ignore[no-untyped-call]


#: ``{vendor_code: (vendor_id, name)}`` - fixed ids, seeded by the first migration; a new vendor is a new migration.
VENDORS: dict[str, tuple[int, str]] = {
    "refinitiv": (1, "LSEG / Refinitiv"),
    "sec": (2, "SEC EDGAR (XBRL company facts)"),
    "yahoo": (3, "Yahoo Finance"),
    "financedatabase": (4, "FinanceDatabase (open source, JerBouma)"),
    "norgate": (5, "Norgate Data"),
    "tiingo": (6, "Tiingo"),
    "simfin": (7, "SimFin"),
    "sraf": (8, "Notre Dame SRAF (Loughran-McDonald dictionary)"),
    "repository": (9, "Corrections kept in this repository"),
    "ice": (10, "ICE Data Indices (bond index constituent files)"),
    "fred": (11, "FRED, Federal Reserve Bank of St. Louis"),
}


def vendor_id(code: str) -> int:
    """The ``vendor_id`` of a vendor code (``"sec"`` -> 2)."""
    try:
        return VENDORS[code][0]
    except KeyError:
        raise KeyError(f"unknown vendor {code!r}; known: {sorted(VENDORS)}") from None


def vendor_code(vendor_id_: int) -> str:
    """The code of a ``vendor_id`` (2 -> ``"sec"``)."""
    for code, (vid, _) in VENDORS.items():
        if vid == vendor_id_:
            return code
    raise KeyError(f"unknown vendor_id {vendor_id_}")


def one_of(column: str, values: tuple[str, ...]) -> str:
    """A CHECK constraint's condition: ``column`` holds one of ``values``."""
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


@cache
def load_asset_classes() -> dict[str, ModuleType]:
    """``{asset class: its schema module}``, every registered one imported, so ``metadata`` holds every table.

    Each asset class's tables must all be in the schema named after it: a table elsewhere is refused here, which keeps
    the database's layout the code's.
    """
    found: dict[str, ModuleType] = {}
    for point in sorted(entry_points(group=ASSET_CLASS_ENTRY_POINTS), key=lambda p: p.name):
        module = point.load()
        stray = sorted(
            f"{t.schema}.{t.name}"
            for t in vars(module).values()
            if isinstance(t, sa.Table) and t.metadata is metadata and t.schema != point.name
        )
        if stray:
            raise RuntimeError(f"asset class {point.name!r} declares tables outside its schema: {stray}")
        found[point.name] = module
    if not found:
        raise RuntimeError(
            f"no asset class is registered under {ASSET_CLASS_ENTRY_POINTS!r}: is the project installed?"
        )
    return found


def all_schemas() -> tuple[str, ...]:
    """Every schema of the database: this package's, then one per registered asset class."""
    return (*CORE_SCHEMAS, *load_asset_classes())


def vendor_column(*, primary_key: bool = False) -> sa.Column[Any]:
    """``vendor_id``: who supplied the row."""
    return sa.Column(
        "vendor_id", sa.SmallInteger, sa.ForeignKey("core.vendor.vendor_id"), primary_key=primary_key, nullable=False
    )


def asset_class_column() -> sa.Column[Any]:
    """``asset_class``: which asset class a run or a rule set belongs to (``equity``, ``fixed_income``)."""
    return sa.Column("asset_class", sa.String(16), sa.ForeignKey("core.asset_class.asset_class"), nullable=False)


def provenance_columns() -> list[sa.Column[Any]]:
    """``loaded_at`` (when we knew it) and ``batch_id`` (the raw file it came from): on every canonical row."""
    return [
        sa.Column("loaded_at", instant(), nullable=False, server_default=UTC_NOW),
        sa.Column("batch_id", sa.Integer, sa.ForeignKey("ingest.batch.batch_id"), nullable=True),
    ]


# ---------------------------------------------------------------- ingest / ops / research

batch = sa.Table(
    "batch",
    metadata,
    sa.Column("batch_id", sa.Integer, sa.Identity(), primary_key=True),
    vendor_column(),
    sa.Column("dataset", sa.String(64), nullable=False),
    sa.Column("source", sa.Unicode(1024), nullable=False),  # the request: URL, query, or source description
    sa.Column("raw_path", sa.Unicode(512), nullable=False),  # relative to the raw zone
    sa.Column("checksum", sa.CHAR(64), nullable=False),  # sha256 of the raw file
    sa.Column("bytes", sa.BigInteger, nullable=False),
    sa.Column("rows", sa.Integer, nullable=True),
    sa.Column("fetched_at", instant(), nullable=False),
    sa.Column("registered_at", instant(), nullable=False, server_default=UTC_NOW),
    sa.Column("status", sa.String(16), nullable=False, server_default="landed"),
    sa.Column("detail", sa.UnicodeText, nullable=True),
    sa.Column("code_version", sa.String(40), nullable=True),
    sa.CheckConstraint(one_of("status", ("landed", "normalized", "failed", "superseded")), name="status"),
    sa.UniqueConstraint("checksum", "dataset"),
    schema="ingest",
)

job_run = sa.Table(
    "job_run",
    metadata,
    sa.Column("run_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("job", sa.String(128), nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("started_at", instant(), nullable=False, server_default=UTC_NOW),
    sa.Column("finished_at", instant(), nullable=True),
    sa.Column("rows", sa.BigInteger, nullable=True),
    sa.Column("detail", sa.UnicodeText, nullable=True),
    sa.Column("host", sa.String(128), nullable=True),
    sa.Column("user_name", sa.String(128), nullable=True),
    sa.Column("code_version", sa.String(40), nullable=True),
    sa.CheckConstraint(one_of("status", ("running", "succeeded", "failed")), name="status"),
    schema="ops",
)

check_result = sa.Table(
    "check_result",
    metadata,
    sa.Column("check_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("run_id", sa.Integer, sa.ForeignKey("ops.job_run.run_id"), nullable=True),
    sa.Column("check_name", sa.String(128), nullable=False),
    sa.Column("status", sa.String(8), nullable=False),
    sa.Column("value", sa.Unicode(512), nullable=True),
    sa.Column("threshold", sa.Unicode(512), nullable=True),
    sa.Column("checked_at", instant(), nullable=False, server_default=UTC_NOW),
    sa.CheckConstraint(one_of("status", ("pass", "warn", "fail")), name="status"),
    schema="ops",
)

research_run = sa.Table(
    "run",
    metadata,
    sa.Column("run_id", sa.Integer, sa.Identity(), primary_key=True),
    asset_class_column(),  # an equity backtest or a bond one
    sa.Column("family", sa.String(64), nullable=False),  # the idea being tested; its runs are its trials
    sa.Column("config_hash", sa.CHAR(16), nullable=False),
    sa.Column("config_json", sa.UnicodeText, nullable=False),
    sa.Column("code_version", sa.String(40), nullable=True),
    sa.Column("data_snapshot", instant(), nullable=True),  # max loaded_at of the data used
    sa.Column("user_name", sa.String(128), nullable=True),
    sa.Column("started_at", instant(), nullable=False, server_default=UTC_NOW),
    sa.Column("finished_at", instant(), nullable=True),
    sa.Column("holdout_opened", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("metrics_json", sa.UnicodeText, nullable=True),
    sa.Column("notes", sa.UnicodeText, nullable=True),
    sa.Index(None, "config_hash"),
    sa.Index(None, "asset_class", "family"),
    schema="research",
)

#: The kinds of rule a set may hold: a factor's cleaning limits and exclusions (Barings' QGARP limits), a weight
#: in a composite, a selection threshold. A new kind is a new migration.
RULE_TYPES = ("lower_limit", "upper_limit", "exclude_financials", "invert", "weight", "threshold", "setting")

# A strategy's rules, shared by everyone: a named set and its rules. A set is never changed once loaded (a run names
# the set it used, so the set must still say what it said then) - a changed rule is a new set, under a new code.
rule_set = sa.Table(
    "rule_set",
    metadata,
    sa.Column("rule_set_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("code", sa.String(64), nullable=False, unique=True),  # what a run's settings name: "gems-2026-11"
    asset_class_column(),
    sa.Column("strategy", sa.String(64), nullable=False),  # the family of runs it belongs to: "barings_qgarp"
    sa.Column("description", sa.Unicode(400), nullable=False),
    sa.Column("source", sa.Unicode(400), nullable=False),  # where the rules came from
    sa.Column("checksum", sa.CHAR(64), nullable=False),  # sha256 of its rules: a set is loaded once, unchanged
    sa.Column("loaded_at", instant(), nullable=False, server_default=UTC_NOW),
    schema="research",
)

strategy_rule = sa.Table(
    "strategy_rule",
    metadata,
    sa.Column("rule_set_id", sa.Integer, sa.ForeignKey("research.rule_set.rule_set_id"), primary_key=True),
    sa.Column("subject", sa.String(128), primary_key=True),  # what the rule is about: a factor, a composite, "*"
    sa.Column("rule_type", sa.String(32), primary_key=True),
    sa.Column("value", sa.Float, nullable=False),
    sa.CheckConstraint("rule_type IN (" + ", ".join(f"'{t}'" for t in RULE_TYPES) + ")", name="rule_type"),
    schema="research",
)

# ---------------------------------------------------------------- core
# The security master every asset class shares. What can be listed and how it can be identified are rows, not
# constraints: an asset class added later (crypto, FX, futures) inserts its instrument types and identifier codes in
# the migration that creates its schema, and no table here changes.

# The asset classes, one per asset-class package and schema: what a run or a rule set is about.
asset_class = sa.Table(
    "asset_class",
    metadata,
    sa.Column("asset_class", sa.String(16), primary_key=True),  # its schema's name: equity, fixed_income
    sa.Column("name", sa.Unicode(64), nullable=False),
    schema="core",
)

# What a listing can be (equity, etf, index, fund, bond, fx...) and the asset class whose package loads and studies it
# - NULL for a type every asset class uses (a currency's exchange rate).
instrument_type = sa.Table(
    "instrument_type",
    metadata,
    sa.Column("instrument_type", sa.String(16), primary_key=True),
    sa.Column("asset_class", sa.String(16), sa.ForeignKey("core.asset_class.asset_class"), nullable=True),
    sa.Column("description", sa.Unicode(256), nullable=False),
    schema="core",
)

# The kinds of code an issuer or a listing is known by (TICKER, RIC, ISIN, CIK, ICE...), each with what it is.
id_type = sa.Table(
    "id_type",
    metadata,
    sa.Column("id_type", sa.String(16), primary_key=True),
    sa.Column("description", sa.Unicode(256), nullable=False),
    schema="core",
)

vendor = sa.Table(
    "vendor",
    metadata,
    sa.Column("vendor_id", sa.SmallInteger, primary_key=True, autoincrement=False),
    sa.Column("vendor_code", sa.String(32), nullable=False, unique=True),
    sa.Column("name", sa.Unicode(128), nullable=False),
    schema="core",
)

#: The reference tables the migrations fill, alike in every database: a copy compares them and never copies them.
SEEDED = (asset_class, vendor, instrument_type, id_type)

issuer = sa.Table(
    "issuer",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("name", sa.Unicode(256), nullable=False),
    # where the company is registered (Linde: IE) - not where its shares trade, which is the listing's exchange
    sa.Column("domicile_country", sa.CHAR(2), nullable=True),
    # where it is incorporated (Glencore: JE, run from CH) - UK stamp duty follows it
    sa.Column("incorporation_country", sa.CHAR(2), nullable=True),
    *provenance_columns(),
    schema="core",
)


listing = sa.Table(
    "listing",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.Identity(), primary_key=True),
    # NULL: no issuer - an index level, an exchange rate (later a future, a crypto asset)
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), nullable=True),
    sa.Column("name", sa.Unicode(256), nullable=False),
    sa.Column("instrument_type", sa.String(16), sa.ForeignKey("core.instrument_type.instrument_type"), nullable=False),
    sa.Column("share_class", sa.String(16), nullable=True),
    sa.Column("exchange_mic", sa.String(8), nullable=True),
    # what it is priced in: ISO 4217 (USD), a minor unit (GBX), or a crypto asset's code (USDT)
    sa.Column("currency", sa.String(8), nullable=True),
    sa.Column("listed_on", sa.Date, nullable=True),
    sa.Column("delisted_on", sa.Date, nullable=True),
    *provenance_columns(),
    sa.Index(None, "issuer_id"),
    sa.Index(None, "instrument_type"),
    schema="core",
)

identifier = sa.Table(
    "identifier",
    metadata,
    sa.Column("identifier_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("entity_type", sa.String(8), nullable=False),
    sa.Column("entity_id", sa.Integer, nullable=False),
    sa.Column("id_type", sa.String(16), sa.ForeignKey("core.id_type.id_type"), nullable=False),
    sa.Column("value", CODE, nullable=False),
    sa.Column("valid_from", sa.Date, nullable=False, server_default=sa.text("'1900-01-01'")),
    sa.Column("valid_to", sa.Date, nullable=True),  # exclusive; NULL = still valid
    # where the code is unique: a ticker among one country's exchanges (US, GB) - later a venue's own symbols; '' for
    # codes unique everywhere
    sa.Column("market", sa.String(16), nullable=False, server_default=""),
    vendor_column(),  # who told us
    *provenance_columns(),
    sa.CheckConstraint(one_of("entity_type", ("issuer", "listing")), name="entity_type"),
    sa.UniqueConstraint("id_type", "value", "valid_from", "entity_type", "market"),
    sa.Index(None, "entity_type", "entity_id"),
    schema="core",
)
