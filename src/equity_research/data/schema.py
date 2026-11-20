"""The equity tables of the canonical data model (docs/ARCHITECTURE.md section 3), on the shared metadata.

Sectors, indices and universes; prices, corporate actions and share counts; filings, accounting facts, filing text
and analysts' estimates. The tables every asset class shares - vendors, issuers, listings, identifiers, batches,
job runs, the run registry - are in :mod:`research_core.data.db.schema`, whose conventions these follow.

Every table is in the schema ``equity``, this package's own. Importing this module registers the tables on that
module's ``metadata``; the package registers the module as the asset class ``equity`` (``pyproject.toml``), so
whatever walks the whole database finds it through :func:`research_core.data.db.schema.load_asset_classes`.
"""

from __future__ import annotations

import sqlalchemy as sa

from research_core.data.db.schema import (
    instant,
    metadata,
    one_of,
    provenance_columns,
    vendor_column,
)

ACTION_TYPES = ("cash_dividend", "stock_dividend", "split", "spinoff")


#: I instant, Q ~3 months, YTD other durations (6 / 9 months), FY ~12 months
PERIOD_TYPES = ("I", "Q", "YTD", "FY")


#: Analysts' forecast measures (consensus estimates) and the fiscal years forecast (FY1 = the current one)
ESTIMATE_MEASURES = ("eps", "revenue", "ebit", "dps", "cfps", "roe")


FORECAST_PERIODS = ("FY1", "FY2", "FY3")


classification = sa.Table(
    "classification",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("scheme", sa.String(16), primary_key=True),  # GICS
    sa.Column("valid_from", sa.Date, primary_key=True),  # the first day we saw it - history builds from there
    sa.Column("valid_to", sa.Date, nullable=True),  # exclusive; NULL = still the classification
    sa.Column("sector", sa.Unicode(64), nullable=True),
    sa.Column("industry_group", sa.Unicode(96), nullable=True),
    sa.Column("industry", sa.Unicode(96), nullable=True),
    sa.Column("sub_industry", sa.Unicode(128), nullable=True),
    *provenance_columns(),
    schema="equity",
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
    sa.Column("level_listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), nullable=True),
    schema="equity",
)


index_membership = sa.Table(
    "index_membership",
    metadata,
    sa.Column("index_id", sa.Integer, sa.ForeignKey("equity.index_definition.index_id"), primary_key=True),
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    sa.Column("start_date", sa.Date, primary_key=True),
    sa.Column("end_date", sa.Date, nullable=True),  # exclusive; NULL = still a member
    vendor_column(),
    *provenance_columns(),
    sa.Index(None, "listing_id"),
    schema="equity",
)


universe = sa.Table(
    "universe",
    metadata,
    sa.Column("universe_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("code", sa.String(32), nullable=False, unique=True),
    sa.Column("name", sa.Unicode(128), nullable=False),
    sa.Column("benchmark_listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), nullable=True),
    schema="equity",
)


universe_component = sa.Table(
    "universe_component",
    metadata,
    sa.Column("universe_id", sa.Integer, sa.ForeignKey("equity.universe.universe_id"), primary_key=True),
    sa.Column("index_id", sa.Integer, sa.ForeignKey("equity.index_definition.index_id"), primary_key=True),
    sa.Column("segment", sa.String(16), nullable=False),
    sa.Column("cost_bps", sa.Float, nullable=False),
    sa.Column("borrow_bps", sa.Float, nullable=True),
    sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
    schema="equity",
)


price_bar = sa.Table(
    "price_bar",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("trade_date", sa.Date, primary_key=True),
    sa.Column("open", sa.Float, nullable=True),
    sa.Column("high", sa.Float, nullable=True),
    sa.Column("low", sa.Float, nullable=True),
    sa.Column("close", sa.Float, nullable=False),  # unadjusted
    sa.Column("volume", sa.BigInteger, nullable=True),
    sa.Column("currency", sa.CHAR(3), nullable=True),
    *provenance_columns(),
    schema="equity",
)


corporate_action = sa.Table(
    "corporate_action",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("action_type", sa.String(16), primary_key=True),
    sa.Column("ex_date", sa.Date, primary_key=True),
    sa.Column("amount", sa.Float, nullable=True),  # cash per share (dividends)
    sa.Column("ratio", sa.Float, nullable=True),  # new shares per old share (splits: 2.0 = 2-for-1)
    sa.Column("currency", sa.CHAR(3), nullable=True),
    sa.Column("published_at", instant(), nullable=True),
    *provenance_columns(),
    sa.CheckConstraint(one_of("action_type", ACTION_TYPES), name="action_type"),
    schema="equity",
)


share_count = sa.Table(
    "share_count",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("kind", sa.String(16), primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),
    sa.Column("shares", sa.Float, nullable=False),
    *provenance_columns(),
    sa.CheckConstraint(one_of("kind", ("float", "outstanding")), name="kind"),
    schema="equity",
)


filing = sa.Table(
    "filing",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("filing_ref", sa.String(32), primary_key=True),  # SEC accession number
    sa.Column("form", sa.String(16), nullable=False),
    sa.Column("period_end", sa.Date, nullable=False),
    sa.Column("fiscal_year", sa.SmallInteger, nullable=True),
    sa.Column("fiscal_period", sa.String(4), nullable=True),
    sa.Column("published_at", instant(), nullable=False),
    *provenance_columns(),
    sa.Index(None, "issuer_id", "period_end"),
    schema="equity",
)


financial_fact = sa.Table(
    "financial_fact",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("item", sa.String(48), primary_key=True),
    sa.Column("period_type", sa.String(3), primary_key=True),
    sa.Column("period_end", sa.Date, primary_key=True),
    sa.Column("published_at", instant(), primary_key=True),
    sa.Column("period_start", sa.Date, nullable=True),
    sa.Column("value", sa.Float, nullable=False),
    sa.Column("unit", sa.String(8), nullable=False),
    sa.Column("filing_ref", sa.String(32), nullable=True),
    *provenance_columns(),
    sa.CheckConstraint(one_of("period_type", PERIOD_TYPES), name="period_type"),
    schema="equity",
)


DOCUMENT_FORMS = ("10-K", "10-K405", "10-KT", "10-K/A", "10-K405/A", "10-KT/A", "10-Q", "10-Q/A", "8-K", "8-K/A")


TEXT_SECTIONS = ("all", "mdna", "risk")


# ---------------------------------------------------------------- filing documents and their text
# Every filing the SEC's index lists for a company (annual and quarterly reports, current reports), with the instant
# EDGAR accepted it - when the world could read it - and, for an 8-K, the events it reports (item codes such as 2.02,
# results; 5.02, an officer leaving). The text itself stays in the raw zone (a 10-K is ~1.5 MB of HTML; forty
# thousand of them would fill the database); ``text_measure`` holds what research reads of it: word counts by
# dictionary category, similarity to the company's previous report, a language model's tone - each named by its
# ``method``, so a new dictionary or model re-measures from the raw zone.
document = sa.Table(
    "document",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("filing_ref", sa.String(32), primary_key=True),  # SEC accession number
    sa.Column("cik", sa.Integer, nullable=False),
    sa.Column("form", sa.String(16), nullable=False),
    sa.Column("filing_date", sa.Date, nullable=False),
    sa.Column("report_date", sa.Date, nullable=True),  # a report's period end, an 8-K's event date
    sa.Column("published_at", instant(), nullable=False),  # accepted by EDGAR (UTC)
    sa.Column("items", sa.String(96), nullable=True),  # an 8-K's item codes, comma-separated
    sa.Column("primary_document", sa.String(160), nullable=True),
    *provenance_columns(),
    sa.CheckConstraint(one_of("form", DOCUMENT_FORMS), name="form"),
    sa.Index(None, "issuer_id", "published_at"),
    schema="equity",
)


text_measure = sa.Table(
    "text_measure",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("filing_ref", sa.String(32), primary_key=True),
    sa.Column("section", sa.String(8), primary_key=True),
    sa.Column("measure", sa.String(24), primary_key=True),
    sa.Column("method", sa.String(24), nullable=False),
    sa.Column("value", sa.Float, nullable=False),
    *provenance_columns(),
    sa.CheckConstraint(one_of("section", TEXT_SECTIONS), name="section"),
    schema="equity",
)


# ---------------------------------------------------------------- analysts' estimates
# Per listing, not per company: a per-share forecast belongs to a share class (BRK.A and BRK.B differ). Per-share
# figures (eps, dps, cfps) are in the shares of their own date - a later split never rewrites them, like prices.
estimate = sa.Table(
    "estimate",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("measure", sa.String(8), primary_key=True),
    sa.Column("period", sa.String(3), primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),  # the consensus as it stood on this day
    sa.Column("fiscal_period_end", sa.Date, nullable=True),
    sa.Column("mean", sa.Float, nullable=False),
    sa.Column("median", sa.Float, nullable=True),
    sa.Column("std_dev", sa.Float, nullable=True),
    sa.Column("estimates", sa.Integer, nullable=True),  # how many analysts' estimates the consensus includes
    *provenance_columns(),
    sa.CheckConstraint(one_of("measure", ESTIMATE_MEASURES), name="measure"),
    sa.CheckConstraint(one_of("period", FORECAST_PERIODS), name="period"),
    schema="equity",
)


estimate_revision = sa.Table(
    "estimate_revision",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("measure", sa.String(8), primary_key=True),
    sa.Column("period", sa.String(3), primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),
    sa.Column("up", sa.Integer, nullable=True),  # analysts who raised their estimate recently
    sa.Column("down", sa.Integer, nullable=True),  # and who lowered it
    *provenance_columns(),
    sa.CheckConstraint(one_of("measure", ESTIMATE_MEASURES), name="measure"),
    sa.CheckConstraint(one_of("period", FORECAST_PERIODS), name="period"),
    schema="equity",
)


estimate_actual = sa.Table(
    "estimate_actual",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("measure", sa.String(8), primary_key=True),
    sa.Column("period_end", sa.Date, primary_key=True),  # the fiscal year's end
    sa.Column("value", sa.Float, nullable=False),  # as the analysts' service counts it, comparable to the estimates
    sa.Column("announced_at", instant(), nullable=True),
    *provenance_columns(),
    sa.CheckConstraint(one_of("measure", ESTIMATE_MEASURES), name="measure"),
    schema="equity",
)
