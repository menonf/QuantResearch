"""The fixed income tables of the canonical data model (docs/ARCHITECTURE.md section 3), on the shared metadata.

A bond is a ``core.listing`` (instrument type ``bond``) of a ``core.issuer``, with its ISIN and CUSIP in
``core.identifier`` - the tables every asset class shares (:mod:`research_core.data.db.schema`). These tables add what
an index provider reports about bonds, every month end:

* ``fixed_income.index_definition`` - an index (``H0A0``, ICE BofA US High Yield);
* ``fixed_income.index_level`` - its official total-return level, the yardstick a rebuilt index is held against;
* ``fixed_income.index_constituent`` - which bonds the index holds and at what weight;
* ``fixed_income.bond_analytics`` - what is true of the bond whatever index holds it: terms, rating, price, spreads,
  yields, durations, the month's returns;
* ``fixed_income.issuer_classification`` - what the provider says of the issuer, month by month: its four industry
  levels and its country.

Everything is **as reported** and keyed by the month end it is about, so files may arrive in any order and a file
loaded twice writes nothing. Units are the provider's and named in the column: ``_pct`` percent, ``_bp`` basis points;
durations in years, prices per 100 of face value, face values in millions of the bond's currency.

Every table is in the schema ``fixed_income``, this package's own. Importing this module registers the tables on the
shared ``metadata``; the package registers the module as the asset class ``fixed_income`` (``pyproject.toml``), so
whatever walks the whole database finds it through :func:`research_core.data.db.schema.load_asset_classes`.
"""

from __future__ import annotations

import sqlalchemy as sa

from research_core.data.db.schema import metadata, provenance_columns, vendor_column

index_definition = sa.Table(
    "index_definition",
    metadata,
    sa.Column("index_id", sa.Integer, sa.Identity(), primary_key=True),
    sa.Column("code", sa.String(16), nullable=False, unique=True),  # the provider's own code
    sa.Column("name", sa.Unicode(128), nullable=False),
    sa.Column("provider", sa.Unicode(64), nullable=True),
    sa.Column("currency", sa.CHAR(3), nullable=True),
    schema="fixed_income",
)

index_level = sa.Table(
    "index_level",
    metadata,
    sa.Column("index_id", sa.Integer, sa.ForeignKey("fixed_income.index_definition.index_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),
    sa.Column("total_return_level", sa.Float, nullable=False),
    *provenance_columns(),
    schema="fixed_income",
)

index_constituent = sa.Table(
    "index_constituent",
    metadata,
    sa.Column("index_id", sa.Integer, sa.ForeignKey("fixed_income.index_definition.index_id"), primary_key=True),
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    # the month end of the provider's file: the index is rebalanced on that day, so these are its members and
    # weights for the month that follows
    sa.Column("as_of_date", sa.Date, primary_key=True),
    vendor_column(),
    sa.Column("weight_pct", sa.Float, nullable=False),
    sa.Column("prior_weight_pct", sa.Float, nullable=True),  # its weight during the month just ended; NULL: new
    sa.Column("face_value", sa.Float, nullable=True),
    sa.Column("cash", sa.Float, nullable=True),
    *provenance_columns(),
    sa.Index(None, "listing_id"),
    sa.Index(None, "as_of_date"),
    schema="fixed_income",
)

bond_analytics = sa.Table(
    "bond_analytics",
    metadata,
    sa.Column("listing_id", sa.Integer, sa.ForeignKey("core.listing.listing_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),
    # the issuer the provider filed the bond under that month (a bond changes hands: its issuer is not one for ever)
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), nullable=True),
    sa.Column("coupon_pct", sa.Float, nullable=True),
    sa.Column("maturity_date", sa.Date, nullable=True),
    sa.Column("rating", sa.String(8), nullable=True),
    sa.Column("subordination", sa.Unicode(32), nullable=True),
    sa.Column("price", sa.Float, nullable=True),
    sa.Column("accrued_interest", sa.Float, nullable=True),
    sa.Column("oas_bp", sa.Float, nullable=True),
    sa.Column("asset_swap_spread_bp", sa.Float, nullable=True),
    sa.Column("effective_yield_pct", sa.Float, nullable=True),
    sa.Column("yield_to_worst_pct", sa.Float, nullable=True),
    sa.Column("yield_to_maturity_pct", sa.Float, nullable=True),
    sa.Column("total_return_mtd_pct", sa.Float, nullable=True),  # over the month ending as_of_date, in its currency
    sa.Column("excess_return_mtd_pct", sa.Float, nullable=True),  # over duration-matched government bonds
    sa.Column("effective_duration", sa.Float, nullable=True),
    sa.Column("modified_duration_to_worst", sa.Float, nullable=True),
    sa.Column("modified_duration", sa.Float, nullable=True),
    sa.Column("effective_convexity", sa.Float, nullable=True),
    sa.Column("convexity_to_worst", sa.Float, nullable=True),
    sa.Column("convexity", sa.Float, nullable=True),
    sa.Column("spread_duration", sa.Float, nullable=True),
    *provenance_columns(),
    sa.Index(None, "as_of_date"),
    schema="fixed_income",
)

issuer_classification = sa.Table(
    "issuer_classification",
    metadata,
    sa.Column("issuer_id", sa.Integer, sa.ForeignKey("core.issuer.issuer_id"), primary_key=True),
    vendor_column(primary_key=True),
    sa.Column("as_of_date", sa.Date, primary_key=True),
    # the provider's four industry levels, widest first (Corporate / Industrials / Basic Industry / Metals...)
    sa.Column("sector", sa.Unicode(64), nullable=True),
    sa.Column("industry_group", sa.Unicode(96), nullable=True),
    sa.Column("industry", sa.Unicode(96), nullable=True),
    sa.Column("sub_industry", sa.Unicode(128), nullable=True),
    sa.Column("country", sa.CHAR(2), nullable=True),
    sa.Column("region", sa.Unicode(32), nullable=True),
    *provenance_columns(),
    schema="fixed_income",
)
