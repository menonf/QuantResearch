"""Analysts' estimates: consensus snapshots, revision counts and reported actuals (I/B/E/S, through LSEG).

Per listing, since a per-share forecast belongs to a share class; per-share figures in the shares of their own date.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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
            ["listing_id"], ["ref.listing.listing_id"], name=op.f("fk_estimate_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_estimate_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "measure", "period", "as_of_date", name=op.f("pk_estimate")),
        schema="fund",
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
            ["listing_id"], ["ref.listing.listing_id"], name=op.f("fk_estimate_actual_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_estimate_actual_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("listing_id", "vendor_id", "measure", "period_end", name=op.f("pk_estimate_actual")),
        schema="fund",
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
            ["listing_id"], ["ref.listing.listing_id"], name=op.f("fk_estimate_revision_listing_id_listing")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_estimate_revision_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint(
            "listing_id", "vendor_id", "measure", "period", "as_of_date", name=op.f("pk_estimate_revision")
        ),
        schema="fund",
    )


def downgrade() -> None:
    op.drop_table("estimate_revision", schema="fund")
    op.drop_table("estimate_actual", schema="fund")
    op.drop_table("estimate", schema="fund")
