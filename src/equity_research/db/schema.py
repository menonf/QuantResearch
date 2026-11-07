"""The canonical data model (docs/ARCHITECTURE.md section 3) as SQLAlchemy Core tables.

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
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.dialects import mssql

SCHEMAS = ("ref", "market", "fund", "ingest", "ops", "research")

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


#: fund = a closed-end fund (a UK investment trust): it trades as a share in an index, its accounts are a portfolio's
#: fx = one unit of the listing's currency, priced in US dollars (an exchange rate, for dividends declared abroad)
INSTRUMENT_TYPES = ("equity", "etf", "index", "fund", "fx")
#: YAHOO = a Yahoo Finance symbol (^GSPC is the S&P 500 index there); RIC = LSEG's code, a retired one with its
#: ``^`` suffix (BAC.N^I98); NORGATE = Norgate Data's permanent asset id (its symbols change on delisting);
#: SIMFIN = SimFin's company id (an issuer's, not a listing's)
ID_TYPES = ("TICKER", "RIC", "CUSIP", "ISIN", "CIK", "SEDOL", "FIGI", "PERMID", "YAHOO", "NORGATE", "SIMFIN")
ACTION_TYPES = ("cash_dividend", "stock_dividend", "split", "spinoff")
#: I instant, Q ~3 months, YTD other durations (6 / 9 months), FY ~12 months
PERIOD_TYPES = ("I", "Q", "YTD", "FY")
#: Analysts' forecast measures (consensus estimates) and the fiscal years forecast (FY1 = the current one)
ESTIMATE_MEASURES = ("eps", "revenue", "ebit", "dps", "cfps", "roe")
FORECAST_PERIODS = ("FY1", "FY2", "FY3")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _vendor(*, primary_key: bool = False) -> sa.Column[Any]:
    """``vendor_id``: who supplied the row."""
    return sa.Column(
        "vendor_id", sa.SmallInteger, sa.ForeignKey("ref.vendor.vendor_id"), primary_key=primary_key, nullable=False
    )


def _provenance() -> list[sa.Column[Any]]:
    return [
        sa.Column("loaded_at", instant(), nullable=False, server_default=UTC_NOW),
        sa.Column("batch_id", sa.Integer, sa.ForeignKey("ingest.batch.batch_id"), nullable=True),
    ]


# ---------------------------------------------------------------- ingest / ops / research

batch = sa.Table(
    "batch",
    metadata,
    sa.Column("batch_id", sa.Integer, sa.Identity(), primary_key=True),
    _vendor(),
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
    sa.CheckConstraint(_in("status", ("landed", "normalized", "failed", "superseded")), name="status"),
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
    sa.CheckConstraint(_in("status", ("running", "succeeded", "failed")), name="status"),
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
    sa.CheckConstraint(_in("status", ("pass", "warn", "fail")), name="status"),
    schema="ops",
)

research_run = sa.Table(
    "run",
    metadata,
    sa.Column("run_id", sa.Integer, sa.Identity(), primary_key=True),
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
    sa.Index(None, "family"),
    schema="research",
)

# ---------------------------------------------------------------- ref

vendor = sa.Table(
    "vendor",
    metadata,
    sa.Column("vendor_id", sa.SmallInteger, primary_key=True, autoincrement=False),
    sa.Column("vendor_code", sa.String(32), nullable=False, unique=True),
    sa.Column("name", sa.Unicode(128), nullable=False),
    schema="ref",
)

issuer = sa.Table(
    "issuer",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("name", sa.Unicode(256), nullable=False),
    # where the company is registered (Linde: IE) - not where its shares trade, which is the listing's exchange
    sa.Column("domicile_country", sa.CHAR(2), nullable=True),
    # where it is incorporated (Glencore: JE, run from CH) - UK stamp duty follows it
    sa.Column("incorporation_country", sa.CHAR(2), nullable=True),
    *_provenance(),
    schema="ref",
)

classification = sa.Table(
    "classification",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("ref.issuer.issuer_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("scheme", sa.String(16), primary_key=True),  # GICS
    sa.Column("valid_from", sa.Date, primary_key=True),  # the first day we saw it - history builds from there
    sa.Column("valid_to", sa.Date, nullable=True),  # exclusive; NULL = still the classification
    sa.Column("sector", sa.Unicode(64), nullable=True),
    sa.Column("industry_group", sa.Unicode(96), nullable=True),
    sa.Column("industry", sa.Unicode(96), nullable=True),
    sa.Column("sub_industry", sa.Unicode(128), nullable=True),
    *_provenance(),
    schema="ref",
)

listing = sa.Table(
    "listing",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("ref.issuer.issuer_id"), nullable=True),  # NULL: index levels
    sa.Column("name", sa.Unicode(256), nullable=False),
    sa.Column("instrument_type", sa.String(16), nullable=False),
    sa.Column("share_class", sa.String(16), nullable=True),
    sa.Column("exchange_mic", sa.String(8), nullable=True),
    sa.Column("currency", sa.CHAR(3), nullable=True),
    sa.Column("listed_on", sa.Date, nullable=True),
    sa.Column("delisted_on", sa.Date, nullable=True),
    *_provenance(),
    sa.CheckConstraint(_in("instrument_type", INSTRUMENT_TYPES), name="instrument_type"),
    sa.Index(None, "issuer_id"),
    schema="ref",
)

identifier = sa.Table(
    "identifier",
    metadata,
    sa.Column("identifier_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("entity_type", sa.String(8), nullable=False),
    sa.Column("entity_id", sa.Integer, nullable=False),
    sa.Column("id_type", sa.String(8), nullable=False),
    sa.Column("value", CODE, nullable=False),
    sa.Column("valid_from", sa.Date, nullable=False, server_default=sa.text("'1900-01-01'")),
    sa.Column("valid_to", sa.Date, nullable=True),  # exclusive; NULL = still valid
    # where the code is unique: a ticker among one country's exchanges (US, GB); '' for codes unique everywhere
    sa.Column("market", sa.String(2), nullable=False, server_default=""),
    _vendor(),  # who told us
    *_provenance(),
    sa.CheckConstraint(_in("entity_type", ("issuer", "listing")), name="entity_type"),
    sa.CheckConstraint(_in("id_type", ID_TYPES), name="id_type"),
    sa.UniqueConstraint("id_type", "value", "valid_from", "entity_type", "market"),
    sa.Index(None, "entity_type", "entity_id"),
    schema="ref",
)

index_definition = sa.Table(
    "index_definition",
    metadata,
    sa.Column("index_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("code", sa.String(32), nullable=False, unique=True),
    sa.Column("name", sa.Unicode(128), nullable=False),
    sa.Column("provider", sa.Unicode(64), nullable=True),
    sa.Column("currency", sa.CHAR(3), nullable=True),
    sa.Column("country", sa.CHAR(2), nullable=True),
    sa.Column("level_listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), nullable=True),
    schema="ref",
)

index_membership = sa.Table(
    "index_membership",
    metadata,
    sa.Column("index_id", sa.Integer, sa.ForeignKey("ref.index_definition.index_id"), primary_key=True),
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), primary_key=True),
    sa.Column("start_date", sa.Date, primary_key=True),
    sa.Column("end_date", sa.Date, nullable=True),  # exclusive; NULL = still a member
    _vendor(),
    *_provenance(),
    sa.Index(None, "listing_id"),
    schema="ref",
)

universe = sa.Table(
    "universe",
    metadata,
    sa.Column("universe_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("code", sa.String(32), nullable=False, unique=True),
    sa.Column("name", sa.Unicode(128), nullable=False),
    sa.Column("benchmark_listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), nullable=True),
    schema="ref",
)

universe_component = sa.Table(
    "universe_component",
    metadata,
    sa.Column("universe_id", sa.Integer, sa.ForeignKey("ref.universe.universe_id"), primary_key=True),
    sa.Column("index_id", sa.Integer, sa.ForeignKey("ref.index_definition.index_id"), primary_key=True),
    sa.Column("segment", sa.String(16), nullable=False),
    sa.Column("cost_bps", sa.Float, nullable=False),
    sa.Column("borrow_bps", sa.Float, nullable=True),
    sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
    schema="ref",
)

# ---------------------------------------------------------------- market

price_bar = sa.Table(
    "price_bar",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("trade_date", sa.Date, primary_key=True),
    sa.Column("open", sa.Float, nullable=True),
    sa.Column("high", sa.Float, nullable=True),
    sa.Column("low", sa.Float, nullable=True),
    sa.Column("close", sa.Float, nullable=False),  # unadjusted
    sa.Column("volume", sa.BigInteger, nullable=True),
    sa.Column("currency", sa.CHAR(3), nullable=True),
    *_provenance(),
    schema="market",
)

corporate_action = sa.Table(
    "corporate_action",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("action_type", sa.String(16), primary_key=True),
    sa.Column("ex_date", sa.Date, primary_key=True),
    sa.Column("amount", sa.Float, nullable=True),  # cash per share (dividends)
    sa.Column("ratio", sa.Float, nullable=True),  # new shares per old share (splits: 2.0 = 2-for-1)
    sa.Column("currency", sa.CHAR(3), nullable=True),
    sa.Column("published_at", instant(), nullable=True),
    *_provenance(),
    sa.CheckConstraint(_in("action_type", ACTION_TYPES), name="action_type"),
    schema="market",
)

share_count = sa.Table(
    "share_count",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("kind", sa.String(16), primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),
    sa.Column("shares", sa.Float, nullable=False),
    *_provenance(),
    sa.CheckConstraint(_in("kind", ("float", "outstanding")), name="kind"),
    schema="market",
)

# ---------------------------------------------------------------- fund

filing = sa.Table(
    "filing",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("ref.issuer.issuer_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("filing_ref", sa.String(32), primary_key=True),  # SEC accession number
    sa.Column("form", sa.String(16), nullable=False),
    sa.Column("period_end", sa.Date, nullable=False),
    sa.Column("fiscal_year", sa.SmallInteger, nullable=True),
    sa.Column("fiscal_period", sa.String(4), nullable=True),
    sa.Column("published_at", instant(), nullable=False),
    *_provenance(),
    sa.Index(None, "issuer_id", "period_end"),
    schema="fund",
)

financial_fact = sa.Table(
    "financial_fact",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("ref.issuer.issuer_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("item", sa.String(48), primary_key=True),
    sa.Column("period_type", sa.String(3), primary_key=True),
    sa.Column("period_end", sa.Date, primary_key=True),
    sa.Column("published_at", instant(), primary_key=True),
    sa.Column("period_start", sa.Date, nullable=True),
    sa.Column("value", sa.Float, nullable=False),
    sa.Column("unit", sa.String(8), nullable=False),
    sa.Column("filing_ref", sa.String(32), nullable=True),
    *_provenance(),
    sa.CheckConstraint(_in("period_type", PERIOD_TYPES), name="period_type"),
    schema="fund",
)


# ---------------------------------------------------------------- fund: analysts' estimates
# Per listing, not per company: a per-share forecast belongs to a share class (BRK.A and BRK.B differ). Per-share
# figures (eps, dps, cfps) are in the shares of their own date - a later split never rewrites them, like prices.

estimate = sa.Table(
    "estimate",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("measure", sa.String(8), primary_key=True),
    sa.Column("period", sa.String(3), primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),  # the consensus as it stood on this day
    sa.Column("fiscal_period_end", sa.Date, nullable=True),
    sa.Column("mean", sa.Float, nullable=False),
    sa.Column("median", sa.Float, nullable=True),
    sa.Column("std_dev", sa.Float, nullable=True),
    sa.Column("estimates", sa.Integer, nullable=True),  # how many analysts' estimates the consensus includes
    *_provenance(),
    sa.CheckConstraint(_in("measure", ESTIMATE_MEASURES), name="measure"),
    sa.CheckConstraint(_in("period", FORECAST_PERIODS), name="period"),
    schema="fund",
)
estimate_revision = sa.Table(
    "estimate_revision",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("measure", sa.String(8), primary_key=True),
    sa.Column("period", sa.String(3), primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),
    sa.Column("up", sa.Integer, nullable=True),  # analysts who raised their estimate recently
    sa.Column("down", sa.Integer, nullable=True),  # and who lowered it
    *_provenance(),
    sa.CheckConstraint(_in("measure", ESTIMATE_MEASURES), name="measure"),
    sa.CheckConstraint(_in("period", FORECAST_PERIODS), name="period"),
    schema="fund",
)
estimate_actual = sa.Table(
    "estimate_actual",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("ref.listing.listing_id"), primary_key=True),
    _vendor(primary_key=True),
    sa.Column("measure", sa.String(8), primary_key=True),
    sa.Column("period_end", sa.Date, primary_key=True),  # the fiscal year's end
    sa.Column("value", sa.Float, nullable=False),  # as the analysts' service counts it, comparable to the estimates
    sa.Column("announced_at", instant(), nullable=True),
    *_provenance(),
    sa.CheckConstraint(_in("measure", ESTIMATE_MEASURES), name="measure"),
    schema="fund",
)
