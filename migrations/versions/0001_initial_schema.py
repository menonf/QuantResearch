"""Initial schema: the database laid out as the code is (docs/ARCHITECTURE.md sections 3 and 4).

* ``core``, ``ingest``, ``ops``, ``research`` - ``research_core``'s: the asset classes, vendors and security master
  (issuers, listings, identifiers, what each can be), the raw files landed, job runs and checks, the run registry;
* ``equity`` - ``equity_research``'s: sectors, indices and universes, prices, actions, share counts, filings, facts,
  filing text, estimates;
* ``fixed_income`` - ``fixed_income_research``'s: bond indices, their levels and constituents, bond analytics, the
  provider's issuer classification.

The reference rows every database holds alike - vendors (fixed ids, so tables copy between databases unchanged), asset
classes, instrument types and identifier codes - are seeded here. An asset class added later creates its schema and
inserts its own rows (``core.asset_class``, ``core.instrument_type``, ``core.id_type``) in its migration; the shared
tables do not change. This history replaces the first one (``migrations/legacy_equity_research``).

Revision ID: 0001
Revises:
Create Date: 2026-11-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMAS = ("core", "ingest", "ops", "research", "equity", "fixed_income")

ASSET_CLASSES = [
    {"asset_class": "equity", "name": "Equities"},
    {"asset_class": "fixed_income", "name": "Fixed income"},
]
VENDORS = [
    {"vendor_id": 1, "vendor_code": "refinitiv", "name": "LSEG / Refinitiv"},
    {"vendor_id": 2, "vendor_code": "sec", "name": "SEC EDGAR (XBRL company facts)"},
    {"vendor_id": 3, "vendor_code": "yahoo", "name": "Yahoo Finance"},
    {"vendor_id": 4, "vendor_code": "financedatabase", "name": "FinanceDatabase (open source, JerBouma)"},
    {"vendor_id": 5, "vendor_code": "norgate", "name": "Norgate Data"},
    {"vendor_id": 6, "vendor_code": "tiingo", "name": "Tiingo"},
    {"vendor_id": 7, "vendor_code": "simfin", "name": "SimFin"},
    {"vendor_id": 8, "vendor_code": "sraf", "name": "Notre Dame SRAF (Loughran-McDonald dictionary)"},
    {"vendor_id": 9, "vendor_code": "repository", "name": "Corrections kept in this repository"},
    {"vendor_id": 10, "vendor_code": "ice", "name": "ICE Data Indices (bond index constituent files)"},
    {"vendor_id": 11, "vendor_code": "fred", "name": "FRED, Federal Reserve Bank of St. Louis"},
]
INSTRUMENT_TYPES = [
    {"instrument_type": "equity", "asset_class": "equity", "description": "a share class of a company"},
    {"instrument_type": "etf", "asset_class": "equity", "description": "an exchange-traded fund, traded as a share"},
    {
        "instrument_type": "fund",
        "asset_class": "equity",
        "description": "a closed-end fund (a UK investment trust): an index member traded as a share, whose accounts "
        "are a portfolio's - research leaves it out",
    },
    {"instrument_type": "index", "asset_class": "equity", "description": "an equity index's level (no issuer)"},
    {"instrument_type": "bond", "asset_class": "fixed_income", "description": "one bond issue of its issuer"},
    {
        "instrument_type": "fx",
        "asset_class": None,
        "description": "one unit of the listing's currency priced in US dollars: an exchange rate, for any asset class",
    },
]
ID_TYPES = [
    {"id_type": "TICKER", "description": "an exchange ticker, unique among one country's exchanges (its market)"},
    {"id_type": "RIC", "description": "LSEG's instrument code; a retired one keeps its ^ suffix (BAC.N^I98)"},
    {"id_type": "CUSIP", "description": "the CUSIP of a US or Canadian security"},
    {"id_type": "ISIN", "description": "the International Securities Identification Number"},
    {"id_type": "CIK", "description": "the SEC's number for a filer (an issuer's)"},
    {"id_type": "SEDOL", "description": "the London Stock Exchange's security code"},
    {"id_type": "FIGI", "description": "the Financial Instrument Global Identifier"},
    {"id_type": "PERMID", "description": "LSEG's permanent identifier of an organisation"},
    {"id_type": "YAHOO", "description": "a Yahoo Finance symbol (^GSPC is the S&P 500 index there)"},
    {"id_type": "NORGATE", "description": "Norgate Data's permanent asset id (its symbols change on delisting)"},
    {"id_type": "SIMFIN", "description": "SimFin's company id (an issuer's)"},
    {
        "id_type": "ICE",
        "description": "the ticker ICE's bond indices file an issuer and its subsidiaries under (an issuer's)",
    },
]


def _seed() -> None:
    """The reference rows every database holds alike."""
    for name, rows in (
        ("asset_class", ASSET_CLASSES),
        ("vendor", VENDORS),
        ("instrument_type", INSTRUMENT_TYPES),
        ("id_type", ID_TYPES),
    ):
        table = sa.table(name, *(sa.column(c) for c in rows[0]), schema="core")
        op.bulk_insert(table, rows)


def upgrade() -> None:
    for schema in SCHEMAS:
        op.execute(f"IF SCHEMA_ID('{schema}') IS NULL EXEC('CREATE SCHEMA [{schema}]')")
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "asset_class",
        sa.Column("asset_class", sa.String(length=16), nullable=False),
        sa.Column("name", sa.Unicode(length=64), nullable=False),
        sa.PrimaryKeyConstraint("asset_class", name=op.f("pk_asset_class")),
        schema="core",
    )
    op.create_table(
        "id_type",
        sa.Column("id_type", sa.String(length=16), nullable=False),
        sa.Column("description", sa.Unicode(length=256), nullable=False),
        sa.PrimaryKeyConstraint("id_type", name=op.f("pk_id_type")),
        schema="core",
    )
    op.create_table(
        "vendor",
        sa.Column("vendor_id", sa.SmallInteger(), autoincrement=False, nullable=False),
        sa.Column("vendor_code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.Unicode(length=128), nullable=False),
        sa.PrimaryKeyConstraint("vendor_id", name=op.f("pk_vendor")),
        sa.UniqueConstraint("vendor_code", name=op.f("uq_vendor_vendor_code")),
        schema="core",
    )
    op.create_table(
        "index_definition",
        sa.Column("index_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("code", sa.String(length=16), nullable=False),
        sa.Column("name", sa.Unicode(length=128), nullable=False),
        sa.Column("provider", sa.Unicode(length=64), nullable=True),
        sa.Column("currency", sa.CHAR(length=3), nullable=True),
        sa.PrimaryKeyConstraint("index_id", name=op.f("pk_index_definition")),
        sa.UniqueConstraint("code", name=op.f("uq_index_definition_code")),
        schema="fixed_income",
    )
    op.create_table(
        "job_run",
        sa.Column("run_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("job", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "started_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("finished_at", mssql.DATETIME2(precision=0), nullable=True),
        sa.Column("rows", sa.BigInteger(), nullable=True),
        sa.Column("detail", sa.UnicodeText(), nullable=True),
        sa.Column("host", sa.String(length=128), nullable=True),
        sa.Column("user_name", sa.String(length=128), nullable=True),
        sa.Column("code_version", sa.String(length=40), nullable=True),
        sa.CheckConstraint("status IN ('running', 'succeeded', 'failed')", name=op.f("ck_job_run_status")),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_job_run")),
        schema="ops",
    )
    op.create_table(
        "instrument_type",
        sa.Column("instrument_type", sa.String(length=16), nullable=False),
        sa.Column("asset_class", sa.String(length=16), nullable=True),
        sa.Column("description", sa.Unicode(length=256), nullable=False),
        sa.ForeignKeyConstraint(
            ["asset_class"], ["core.asset_class.asset_class"], name=op.f("fk_instrument_type_asset_class_asset_class")
        ),
        sa.PrimaryKeyConstraint("instrument_type", name=op.f("pk_instrument_type")),
        schema="core",
    )
    op.create_table(
        "batch",
        sa.Column("batch_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("source", sa.Unicode(length=1024), nullable=False),
        sa.Column("raw_path", sa.Unicode(length=512), nullable=False),
        sa.Column("checksum", sa.CHAR(length=64), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=True),
        sa.Column("fetched_at", mssql.DATETIME2(precision=0), nullable=False),
        sa.Column(
            "registered_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("status", sa.String(length=16), server_default="landed", nullable=False),
        sa.Column("detail", sa.UnicodeText(), nullable=True),
        sa.Column("code_version", sa.String(length=40), nullable=True),
        sa.CheckConstraint("status IN ('landed', 'normalized', 'failed', 'superseded')", name=op.f("ck_batch_status")),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_batch_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("batch_id", name=op.f("pk_batch")),
        sa.UniqueConstraint("checksum", "dataset", name=op.f("uq_batch_checksum_dataset")),
        schema="ingest",
    )
    op.create_table(
        "check_result",
        sa.Column("check_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("check_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("value", sa.Unicode(length=512), nullable=True),
        sa.Column("threshold", sa.Unicode(length=512), nullable=True),
        sa.Column(
            "checked_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.CheckConstraint("status IN ('pass', 'warn', 'fail')", name=op.f("ck_check_result_status")),
        sa.ForeignKeyConstraint(["run_id"], ["ops.job_run.run_id"], name=op.f("fk_check_result_run_id_job_run")),
        sa.PrimaryKeyConstraint("check_id", name=op.f("pk_check_result")),
        schema="ops",
    )
    op.create_table(
        "rule_set",
        sa.Column("rule_set_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("asset_class", sa.String(length=16), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Unicode(length=400), nullable=False),
        sa.Column("source", sa.Unicode(length=400), nullable=False),
        sa.Column("checksum", sa.CHAR(length=64), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["asset_class"], ["core.asset_class.asset_class"], name=op.f("fk_rule_set_asset_class_asset_class")
        ),
        sa.PrimaryKeyConstraint("rule_set_id", name=op.f("pk_rule_set")),
        sa.UniqueConstraint("code", name=op.f("uq_rule_set_code")),
        schema="research",
    )
    op.create_table(
        "run",
        sa.Column("run_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("asset_class", sa.String(length=16), nullable=False),
        sa.Column("family", sa.String(length=64), nullable=False),
        sa.Column("config_hash", sa.CHAR(length=16), nullable=False),
        sa.Column("config_json", sa.UnicodeText(), nullable=False),
        sa.Column("code_version", sa.String(length=40), nullable=True),
        sa.Column("data_snapshot", mssql.DATETIME2(precision=0), nullable=True),
        sa.Column("user_name", sa.String(length=128), nullable=True),
        sa.Column(
            "started_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("finished_at", mssql.DATETIME2(precision=0), nullable=True),
        sa.Column("holdout_opened", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("metrics_json", sa.UnicodeText(), nullable=True),
        sa.Column("notes", sa.UnicodeText(), nullable=True),
        sa.ForeignKeyConstraint(
            ["asset_class"], ["core.asset_class.asset_class"], name=op.f("fk_run_asset_class_asset_class")
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_run")),
        schema="research",
    )
    op.create_index(
        op.f("ix_run_asset_class_family"), "run", ["asset_class", "family"], unique=False, schema="research"
    )
    op.create_index(op.f("ix_run_config_hash"), "run", ["config_hash"], unique=False, schema="research")
    op.create_table(
        "identifier",
        sa.Column("identifier_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("entity_type", sa.String(length=8), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("id_type", sa.String(length=16), nullable=False),
        sa.Column("value", sa.String(length=64, collation="Latin1_General_100_BIN2"), nullable=False),
        sa.Column("valid_from", sa.Date(), server_default=sa.text("'1900-01-01'"), nullable=False),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column("market", sa.String(length=16), server_default="", nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint("entity_type IN ('issuer', 'listing')", name=op.f("ck_identifier_entity_type")),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_identifier_batch_id_batch")),
        sa.ForeignKeyConstraint(["id_type"], ["core.id_type.id_type"], name=op.f("fk_identifier_id_type_id_type")),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_identifier_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("identifier_id", name=op.f("pk_identifier")),
        sa.UniqueConstraint(
            "id_type",
            "value",
            "valid_from",
            "entity_type",
            "market",
            name=op.f("uq_identifier_id_type_value_valid_from_entity_type_market"),
        ),
        schema="core",
    )
    op.create_index(
        op.f("ix_identifier_entity_type_entity_id"),
        "identifier",
        ["entity_type", "entity_id"],
        unique=False,
        schema="core",
    )
    op.create_table(
        "issuer",
        sa.Column("issuer_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("name", sa.Unicode(length=256), nullable=False),
        sa.Column("domicile_country", sa.CHAR(length=2), nullable=True),
        sa.Column("incorporation_country", sa.CHAR(length=2), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_issuer_batch_id_batch")),
        sa.PrimaryKeyConstraint("issuer_id", name=op.f("pk_issuer")),
        schema="core",
    )
    op.create_table(
        "index_level",
        sa.Column("index_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("total_return_level", sa.Float(), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_index_level_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["index_id"],
            ["fixed_income.index_definition.index_id"],
            name=op.f("fk_index_level_index_id_index_definition"),
        ),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_index_level_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("index_id", "vendor_id", "as_of_date", name=op.f("pk_index_level")),
        schema="fixed_income",
    )
    op.create_table(
        "strategy_rule",
        sa.Column("rule_set_id", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=128), nullable=False),
        sa.Column("rule_type", sa.String(length=32), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "rule_type IN ('lower_limit', 'upper_limit', 'exclude_financials', 'invert', 'weight', 'threshold', "
            "'setting')",
            name=op.f("ck_strategy_rule_rule_type"),
        ),
        sa.ForeignKeyConstraint(
            ["rule_set_id"], ["research.rule_set.rule_set_id"], name=op.f("fk_strategy_rule_rule_set_id_rule_set")
        ),
        sa.PrimaryKeyConstraint("rule_set_id", "subject", "rule_type", name=op.f("pk_strategy_rule")),
        schema="research",
    )
    op.create_table(
        "listing",
        sa.Column("listing_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("issuer_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.Unicode(length=256), nullable=False),
        sa.Column("instrument_type", sa.String(length=16), nullable=False),
        sa.Column("share_class", sa.String(length=16), nullable=True),
        sa.Column("exchange_mic", sa.String(length=8), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=True),
        sa.Column("listed_on", sa.Date(), nullable=True),
        sa.Column("delisted_on", sa.Date(), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_listing_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["instrument_type"],
            ["core.instrument_type.instrument_type"],
            name=op.f("fk_listing_instrument_type_instrument_type"),
        ),
        sa.ForeignKeyConstraint(["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_listing_issuer_id_issuer")),
        sa.PrimaryKeyConstraint("listing_id", name=op.f("pk_listing")),
        schema="core",
    )
    op.create_index(op.f("ix_listing_instrument_type"), "listing", ["instrument_type"], unique=False, schema="core")
    op.create_index(op.f("ix_listing_issuer_id"), "listing", ["issuer_id"], unique=False, schema="core")
    op.create_table(
        "classification",
        sa.Column("issuer_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("scheme", sa.String(length=16), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column("sector", sa.Unicode(length=64), nullable=True),
        sa.Column("industry_group", sa.Unicode(length=96), nullable=True),
        sa.Column("industry", sa.Unicode(length=96), nullable=True),
        sa.Column("sub_industry", sa.Unicode(length=128), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_classification_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_classification_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_classification_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("issuer_id", "vendor_id", "scheme", "valid_from", name=op.f("pk_classification")),
        schema="equity",
    )
    op.create_table(
        "document",
        sa.Column("issuer_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("filing_ref", sa.String(length=32), nullable=False),
        sa.Column("cik", sa.Integer(), nullable=False),
        sa.Column("form", sa.String(length=16), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=True),
        sa.Column("published_at", mssql.DATETIME2(precision=0), nullable=False),
        sa.Column("items", sa.String(length=96), nullable=True),
        sa.Column("primary_document", sa.String(length=160), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "form IN ('10-K', '10-K405', '10-KT', '10-K/A', '10-K405/A', '10-KT/A', '10-Q', '10-Q/A', '8-K', '8-K/A')",
            name=op.f("ck_document_form"),
        ),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_document_batch_id_batch")),
        sa.ForeignKeyConstraint(["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_document_issuer_id_issuer")),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_document_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("issuer_id", "vendor_id", "filing_ref", name=op.f("pk_document")),
        schema="equity",
    )
    op.create_index(
        op.f("ix_document_issuer_id_published_at"),
        "document",
        ["issuer_id", "published_at"],
        unique=False,
        schema="equity",
    )
    op.create_table(
        "filing",
        sa.Column("issuer_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("filing_ref", sa.String(length=32), nullable=False),
        sa.Column("form", sa.String(length=16), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("fiscal_year", sa.SmallInteger(), nullable=True),
        sa.Column("fiscal_period", sa.String(length=4), nullable=True),
        sa.Column("published_at", mssql.DATETIME2(precision=0), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_filing_batch_id_batch")),
        sa.ForeignKeyConstraint(["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_filing_issuer_id_issuer")),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_filing_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("issuer_id", "vendor_id", "filing_ref", name=op.f("pk_filing")),
        schema="equity",
    )
    op.create_index(
        op.f("ix_filing_issuer_id_period_end"), "filing", ["issuer_id", "period_end"], unique=False, schema="equity"
    )
    op.create_table(
        "financial_fact",
        sa.Column("issuer_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("item", sa.String(length=48), nullable=False),
        sa.Column("period_type", sa.String(length=3), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("published_at", mssql.DATETIME2(precision=0), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=8), nullable=False),
        sa.Column("filing_ref", sa.String(length=32), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint("period_type IN ('I', 'Q', 'YTD', 'FY')", name=op.f("ck_financial_fact_period_type")),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_financial_fact_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_financial_fact_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_financial_fact_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint(
            "issuer_id",
            "vendor_id",
            "item",
            "period_type",
            "period_end",
            "published_at",
            name=op.f("pk_financial_fact"),
        ),
        schema="equity",
    )
    op.create_table(
        "text_measure",
        sa.Column("issuer_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("filing_ref", sa.String(length=32), nullable=False),
        sa.Column("section", sa.String(length=8), nullable=False),
        sa.Column("measure", sa.String(length=24), nullable=False),
        sa.Column("method", sa.String(length=24), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint("section IN ('all', 'mdna', 'risk')", name=op.f("ck_text_measure_section")),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_text_measure_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_text_measure_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_text_measure_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint(
            "issuer_id", "vendor_id", "filing_ref", "section", "measure", name=op.f("pk_text_measure")
        ),
        schema="equity",
    )
    op.create_table(
        "issuer_classification",
        sa.Column("issuer_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("sector", sa.Unicode(length=64), nullable=True),
        sa.Column("industry_group", sa.Unicode(length=96), nullable=True),
        sa.Column("industry", sa.Unicode(length=96), nullable=True),
        sa.Column("sub_industry", sa.Unicode(length=128), nullable=True),
        sa.Column("country", sa.CHAR(length=2), nullable=True),
        sa.Column("region", sa.Unicode(length=32), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_issuer_classification_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_issuer_classification_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_issuer_classification_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("issuer_id", "vendor_id", "as_of_date", name=op.f("pk_issuer_classification")),
        schema="fixed_income",
    )
    op.create_table(
        "corporate_action",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("action_type", sa.String(length=16), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=True),
        sa.Column("ratio", sa.Float(), nullable=True),
        sa.Column("currency", sa.CHAR(length=3), nullable=True),
        sa.Column("published_at", mssql.DATETIME2(precision=0), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "action_type IN ('cash_dividend', 'stock_dividend', 'split', 'spinoff')",
            name=op.f("ck_corporate_action_action_type"),
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_corporate_action_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_corporate_action_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_corporate_action_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "action_type", "ex_date", name=op.f("pk_corporate_action")),
        schema="equity",
    )
    op.create_table(
        "estimate",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("measure", sa.String(length=8), nullable=False),
        sa.Column("period", sa.String(length=3), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("fiscal_period_end", sa.Date(), nullable=True),
        sa.Column("mean", sa.Float(), nullable=False),
        sa.Column("median", sa.Float(), nullable=True),
        sa.Column("std_dev", sa.Float(), nullable=True),
        sa.Column("estimates", sa.Integer(), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "measure IN ('eps', 'revenue', 'ebit', 'dps', 'cfps', 'roe')", name=op.f("ck_estimate_measure")
        ),
        sa.CheckConstraint("period IN ('FY1', 'FY2', 'FY3')", name=op.f("ck_estimate_period")),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_estimate_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_estimate_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_estimate_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "measure", "period", "as_of_date", name=op.f("pk_estimate")),
        schema="equity",
    )
    op.create_table(
        "estimate_actual",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("measure", sa.String(length=8), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("announced_at", mssql.DATETIME2(precision=0), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "measure IN ('eps', 'revenue', 'ebit', 'dps', 'cfps', 'roe')", name=op.f("ck_estimate_actual_measure")
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_estimate_actual_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_estimate_actual_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_estimate_actual_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "measure", "period_end", name=op.f("pk_estimate_actual")),
        schema="equity",
    )
    op.create_table(
        "estimate_revision",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("measure", sa.String(length=8), nullable=False),
        sa.Column("period", sa.String(length=3), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("up", sa.Integer(), nullable=True),
        sa.Column("down", sa.Integer(), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "measure IN ('eps', 'revenue', 'ebit', 'dps', 'cfps', 'roe')", name=op.f("ck_estimate_revision_measure")
        ),
        sa.CheckConstraint("period IN ('FY1', 'FY2', 'FY3')", name=op.f("ck_estimate_revision_period")),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_estimate_revision_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_estimate_revision_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_estimate_revision_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint(
            "listing_id", "vendor_id", "measure", "period", "as_of_date", name=op.f("pk_estimate_revision")
        ),
        schema="equity",
    )
    op.create_table(
        "index_definition",
        sa.Column("index_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.Unicode(length=128), nullable=False),
        sa.Column("provider", sa.Unicode(length=64), nullable=True),
        sa.Column("currency", sa.CHAR(length=3), nullable=True),
        sa.Column("country", sa.CHAR(length=2), nullable=True),
        sa.Column("level_listing_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["level_listing_id"], ["core.listing.listing_id"], name=op.f("fk_index_definition_level_listing_id_listing")
        ),
        sa.PrimaryKeyConstraint("index_id", name=op.f("pk_index_definition")),
        sa.UniqueConstraint("code", name=op.f("uq_index_definition_code")),
        schema="equity",
    )
    op.create_table(
        "price_bar",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float(), nullable=True),
        sa.Column("high", sa.Float(), nullable=True),
        sa.Column("low", sa.Float(), nullable=True),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("currency", sa.CHAR(length=3), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_price_bar_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_price_bar_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_price_bar_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "trade_date", name=op.f("pk_price_bar")),
        schema="equity",
    )
    op.create_table(
        "share_count",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("shares", sa.Float(), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.CheckConstraint("kind IN ('float', 'outstanding')", name=op.f("ck_share_count_kind")),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_share_count_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_share_count_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_share_count_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "kind", "as_of_date", name=op.f("pk_share_count")),
        schema="equity",
    )
    op.create_table(
        "universe",
        sa.Column("universe_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.Unicode(length=128), nullable=False),
        sa.Column("benchmark_listing_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["benchmark_listing_id"], ["core.listing.listing_id"], name=op.f("fk_universe_benchmark_listing_id_listing")
        ),
        sa.PrimaryKeyConstraint("universe_id", name=op.f("pk_universe")),
        sa.UniqueConstraint("code", name=op.f("uq_universe_code")),
        schema="equity",
    )
    op.create_table(
        "bond_analytics",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("issuer_id", sa.Integer(), nullable=True),
        sa.Column("coupon_pct", sa.Float(), nullable=True),
        sa.Column("maturity_date", sa.Date(), nullable=True),
        sa.Column("rating", sa.String(length=8), nullable=True),
        sa.Column("subordination", sa.Unicode(length=32), nullable=True),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("accrued_interest", sa.Float(), nullable=True),
        sa.Column("oas_bp", sa.Float(), nullable=True),
        sa.Column("asset_swap_spread_bp", sa.Float(), nullable=True),
        sa.Column("effective_yield_pct", sa.Float(), nullable=True),
        sa.Column("yield_to_worst_pct", sa.Float(), nullable=True),
        sa.Column("yield_to_maturity_pct", sa.Float(), nullable=True),
        sa.Column("total_return_mtd_pct", sa.Float(), nullable=True),
        sa.Column("excess_return_mtd_pct", sa.Float(), nullable=True),
        sa.Column("effective_duration", sa.Float(), nullable=True),
        sa.Column("modified_duration_to_worst", sa.Float(), nullable=True),
        sa.Column("modified_duration", sa.Float(), nullable=True),
        sa.Column("effective_convexity", sa.Float(), nullable=True),
        sa.Column("convexity_to_worst", sa.Float(), nullable=True),
        sa.Column("convexity", sa.Float(), nullable=True),
        sa.Column("spread_duration", sa.Float(), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_bond_analytics_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["core.issuer.issuer_id"], name=op.f("fk_bond_analytics_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_bond_analytics_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_bond_analytics_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "as_of_date", name=op.f("pk_bond_analytics")),
        schema="fixed_income",
    )
    op.create_index(
        op.f("ix_bond_analytics_as_of_date"), "bond_analytics", ["as_of_date"], unique=False, schema="fixed_income"
    )
    op.create_table(
        "index_constituent",
        sa.Column("index_id", sa.Integer(), nullable=False),
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column("weight_pct", sa.Float(), nullable=False),
        sa.Column("prior_weight_pct", sa.Float(), nullable=True),
        sa.Column("face_value", sa.Float(), nullable=True),
        sa.Column("cash", sa.Float(), nullable=True),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_index_constituent_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["index_id"],
            ["fixed_income.index_definition.index_id"],
            name=op.f("fk_index_constituent_index_id_index_definition"),
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_index_constituent_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_index_constituent_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("index_id", "listing_id", "as_of_date", name=op.f("pk_index_constituent")),
        schema="fixed_income",
    )
    op.create_index(
        op.f("ix_index_constituent_as_of_date"),
        "index_constituent",
        ["as_of_date"],
        unique=False,
        schema="fixed_income",
    )
    op.create_index(
        op.f("ix_index_constituent_listing_id"),
        "index_constituent",
        ["listing_id"],
        unique=False,
        schema="fixed_income",
    )
    op.create_table(
        "index_membership",
        sa.Column("index_id", sa.Integer(), nullable=False),
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("vendor_id", sa.SmallInteger(), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_index_membership_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["index_id"],
            ["equity.index_definition.index_id"],
            name=op.f("fk_index_membership_index_id_index_definition"),
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["core.listing.listing_id"], name=op.f("fk_index_membership_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["core.vendor.vendor_id"], name=op.f("fk_index_membership_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("index_id", "listing_id", "start_date", name=op.f("pk_index_membership")),
        schema="equity",
    )
    op.create_index(
        op.f("ix_index_membership_listing_id"), "index_membership", ["listing_id"], unique=False, schema="equity"
    )
    op.create_table(
        "universe_component",
        sa.Column("universe_id", sa.Integer(), nullable=False),
        sa.Column("index_id", sa.Integer(), nullable=False),
        sa.Column("segment", sa.String(length=16), nullable=False),
        sa.Column("cost_bps", sa.Float(), nullable=False),
        sa.Column("borrow_bps", sa.Float(), nullable=True),
        sa.Column("priority", sa.Integer(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(
            ["index_id"],
            ["equity.index_definition.index_id"],
            name=op.f("fk_universe_component_index_id_index_definition"),
        ),
        sa.ForeignKeyConstraint(
            ["universe_id"], ["equity.universe.universe_id"], name=op.f("fk_universe_component_universe_id_universe")
        ),
        sa.PrimaryKeyConstraint("universe_id", "index_id", name=op.f("pk_universe_component")),
        schema="equity",
    )
    # ### end Alembic commands ###
    _seed()


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("universe_component", schema="equity")
    op.drop_index(op.f("ix_index_membership_listing_id"), table_name="index_membership", schema="equity")
    op.drop_table("index_membership", schema="equity")
    op.drop_index(op.f("ix_index_constituent_listing_id"), table_name="index_constituent", schema="fixed_income")
    op.drop_index(op.f("ix_index_constituent_as_of_date"), table_name="index_constituent", schema="fixed_income")
    op.drop_table("index_constituent", schema="fixed_income")
    op.drop_index(op.f("ix_bond_analytics_as_of_date"), table_name="bond_analytics", schema="fixed_income")
    op.drop_table("bond_analytics", schema="fixed_income")
    op.drop_table("universe", schema="equity")
    op.drop_table("share_count", schema="equity")
    op.drop_table("price_bar", schema="equity")
    op.drop_table("index_definition", schema="equity")
    op.drop_table("estimate_revision", schema="equity")
    op.drop_table("estimate_actual", schema="equity")
    op.drop_table("estimate", schema="equity")
    op.drop_table("corporate_action", schema="equity")
    op.drop_table("issuer_classification", schema="fixed_income")
    op.drop_table("text_measure", schema="equity")
    op.drop_table("financial_fact", schema="equity")
    op.drop_index(op.f("ix_filing_issuer_id_period_end"), table_name="filing", schema="equity")
    op.drop_table("filing", schema="equity")
    op.drop_index(op.f("ix_document_issuer_id_published_at"), table_name="document", schema="equity")
    op.drop_table("document", schema="equity")
    op.drop_table("classification", schema="equity")
    op.drop_index(op.f("ix_listing_issuer_id"), table_name="listing", schema="core")
    op.drop_index(op.f("ix_listing_instrument_type"), table_name="listing", schema="core")
    op.drop_table("listing", schema="core")
    op.drop_table("strategy_rule", schema="research")
    op.drop_table("index_level", schema="fixed_income")
    op.drop_table("issuer", schema="core")
    op.drop_index(op.f("ix_identifier_entity_type_entity_id"), table_name="identifier", schema="core")
    op.drop_table("identifier", schema="core")
    op.drop_index(op.f("ix_run_config_hash"), table_name="run", schema="research")
    op.drop_index(op.f("ix_run_asset_class_family"), table_name="run", schema="research")
    op.drop_table("run", schema="research")
    op.drop_table("rule_set", schema="research")
    op.drop_table("check_result", schema="ops")
    op.drop_table("batch", schema="ingest")
    op.drop_table("instrument_type", schema="core")
    op.drop_table("job_run", schema="ops")
    op.drop_table("index_definition", schema="fixed_income")
    op.drop_table("vendor", schema="core")
    op.drop_table("id_type", schema="core")
    op.drop_table("asset_class", schema="core")
    # ### end Alembic commands ###
    for schema in reversed(SCHEMAS):
        op.execute(f"IF SCHEMA_ID('{schema}') IS NOT NULL EXEC('DROP SCHEMA [{schema}]')")
