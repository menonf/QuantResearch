"""Fixed income: bonds as listings, and what an index provider reports about them every month end.

* vendors 10 ``ice`` (ICE Data Indices' constituent files) and 11 ``fred`` (official index levels);
* ``ref.listing.instrument_type`` ``bond``; ``ref.identifier.id_type`` ``ICE`` - the ticker ICE files an issuer and
  its subsidiaries under;
* ``ref.bond_index``, ``market.bond_index_level``, ``market.bond_index_constituent``, ``market.bond_analytics`` and
  ``ref.bond_issuer_month`` (``fixed_income_research/data/schema.py``).

Nothing existing changes shape: the equity tables and every row in them are as they were.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TYPES = "ck_listing_instrument_type"
TYPES_BEFORE = "'equity', 'etf', 'index', 'fund', 'fx'"
CODES = "ck_identifier_id_type"
CODES_BEFORE = "'TICKER', 'RIC', 'CUSIP', 'ISIN', 'CIK', 'SEDOL', 'FIGI', 'PERMID', 'YAHOO', 'NORGATE', 'SIMFIN'"
BONDS = "(SELECT listing_id FROM ref.listing WHERE instrument_type = 'bond')"
BOND_ISSUERS = "(SELECT entity_id FROM ref.identifier WHERE entity_type = 'issuer' AND id_type = 'ICE')"


def _provenance() -> list[sa.Column[object]]:
    return [
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
    ]


def _vendor() -> sa.Column[object]:
    return sa.Column("vendor_id", sa.SmallInteger(), nullable=False)


def upgrade() -> None:
    op.execute(
        "INSERT INTO ref.vendor (vendor_id, vendor_code, name) VALUES "
        "(10, 'ice', 'ICE Data Indices (bond index constituent files)'), "
        "(11, 'fred', 'FRED, Federal Reserve Bank of St. Louis')"
    )
    op.execute(f"ALTER TABLE ref.listing DROP CONSTRAINT {TYPES}")
    op.execute(f"ALTER TABLE ref.listing ADD CONSTRAINT {TYPES} CHECK (instrument_type IN ({TYPES_BEFORE}, 'bond'))")
    op.execute(f"ALTER TABLE ref.identifier DROP CONSTRAINT {CODES}")
    op.execute(f"ALTER TABLE ref.identifier ADD CONSTRAINT {CODES} CHECK (id_type IN ({CODES_BEFORE}, 'ICE'))")
    op.create_table(
        "bond_index",
        sa.Column("index_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("code", sa.String(length=16), nullable=False),
        sa.Column("name", sa.Unicode(length=128), nullable=False),
        sa.Column("provider", sa.Unicode(length=64), nullable=True),
        sa.Column("currency", sa.CHAR(length=3), nullable=True),
        sa.PrimaryKeyConstraint("index_id", name=op.f("pk_bond_index")),
        sa.UniqueConstraint("code", name=op.f("uq_bond_index_code")),
        schema="ref",
    )
    op.create_table(
        "bond_index_level",
        sa.Column("index_id", sa.Integer(), nullable=False),
        _vendor(),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("total_return_level", sa.Float(), nullable=False),
        *_provenance(),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_bond_index_level_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["index_id"], ["ref.bond_index.index_id"], name=op.f("fk_bond_index_level_index_id_bond_index")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_bond_index_level_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("index_id", "vendor_id", "as_of_date", name=op.f("pk_bond_index_level")),
        schema="market",
    )
    op.create_table(
        "bond_index_constituent",
        sa.Column("index_id", sa.Integer(), nullable=False),
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        _vendor(),
        sa.Column("weight_pct", sa.Float(), nullable=False),
        sa.Column("prior_weight_pct", sa.Float(), nullable=True),
        sa.Column("face_value", sa.Float(), nullable=True),
        sa.Column("cash", sa.Float(), nullable=True),
        *_provenance(),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_bond_index_constituent_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["index_id"], ["ref.bond_index.index_id"], name=op.f("fk_bond_index_constituent_index_id_bond_index")
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["ref.listing.listing_id"], name=op.f("fk_bond_index_constituent_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_bond_index_constituent_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("index_id", "listing_id", "as_of_date", name=op.f("pk_bond_index_constituent")),
        schema="market",
    )
    op.create_index(
        op.f("ix_bond_index_constituent_listing_id"), "bond_index_constituent", ["listing_id"], schema="market"
    )
    op.create_index(
        op.f("ix_bond_index_constituent_as_of_date"), "bond_index_constituent", ["as_of_date"], schema="market"
    )
    op.create_table(
        "bond_analytics",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        _vendor(),
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
        *_provenance(),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_bond_analytics_batch_id_batch")),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["ref.issuer.issuer_id"], name=op.f("fk_bond_analytics_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["ref.listing.listing_id"], name=op.f("fk_bond_analytics_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_bond_analytics_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "as_of_date", name=op.f("pk_bond_analytics")),
        schema="market",
    )
    op.create_index(op.f("ix_bond_analytics_as_of_date"), "bond_analytics", ["as_of_date"], schema="market")
    op.create_table(
        "bond_issuer_month",
        sa.Column("issuer_id", sa.Integer(), nullable=False),
        _vendor(),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("sector", sa.Unicode(length=64), nullable=True),
        sa.Column("industry_group", sa.Unicode(length=96), nullable=True),
        sa.Column("industry", sa.Unicode(length=96), nullable=True),
        sa.Column("sub_industry", sa.Unicode(length=128), nullable=True),
        sa.Column("country", sa.CHAR(length=2), nullable=True),
        sa.Column("region", sa.Unicode(length=32), nullable=True),
        *_provenance(),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_bond_issuer_month_batch_id_batch")
        ),
        sa.ForeignKeyConstraint(
            ["issuer_id"], ["ref.issuer.issuer_id"], name=op.f("fk_bond_issuer_month_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_bond_issuer_month_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("issuer_id", "vendor_id", "as_of_date", name=op.f("pk_bond_issuer_month")),
        schema="ref",
    )


def downgrade() -> None:
    op.drop_table("bond_issuer_month", schema="ref")
    op.drop_table("bond_analytics", schema="market")
    op.drop_table("bond_index_constituent", schema="market")
    op.drop_table("bond_index_level", schema="market")
    op.drop_table("bond_index", schema="ref")
    # the bonds, the issuers only they brought, and their codes: nothing else knows the two new codes
    op.execute(f"DELETE FROM ref.identifier WHERE entity_type = 'listing' AND entity_id IN {BONDS}")
    op.execute("DELETE FROM ref.listing WHERE instrument_type = 'bond'")
    op.execute(
        f"DELETE FROM ref.issuer WHERE issuer_id IN {BOND_ISSUERS} "
        "AND NOT EXISTS (SELECT 1 FROM ref.listing l WHERE l.issuer_id = ref.issuer.issuer_id)"
    )
    op.execute("DELETE FROM ref.identifier WHERE id_type = 'ICE'")
    op.execute(f"ALTER TABLE ref.identifier DROP CONSTRAINT {CODES}")
    op.execute(f"ALTER TABLE ref.identifier ADD CONSTRAINT {CODES} CHECK (id_type IN ({CODES_BEFORE}))")
    op.execute(f"ALTER TABLE ref.listing DROP CONSTRAINT {TYPES}")
    op.execute(f"ALTER TABLE ref.listing ADD CONSTRAINT {TYPES} CHECK (instrument_type IN ({TYPES_BEFORE}))")
    op.execute("DELETE FROM ingest.batch WHERE vendor_id IN (10, 11)")
    op.execute("DELETE FROM ref.vendor WHERE vendor_id IN (10, 11)")
