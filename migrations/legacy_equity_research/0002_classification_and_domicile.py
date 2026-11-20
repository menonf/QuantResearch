"""Sector classification with history, and the company's domicile kept apart from where it trades.

* ``ref.classification``: one row per company, vendor and scheme for each period a classification held; a change
  closes the old row and opens a new one. The sectors that were on ``ref.issuer`` move there, valid from the day
  they were loaded.
* ``ref.issuer.country`` held the country of the exchange (LSEG's "Exchange Country Code") under a name meaning the
  company's domicile; it becomes ``domicile_country`` and is emptied - the exchange is the listing's
  (``exchange_mic``), the domicile is loaded from its own field.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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
            ["issuer_id"], ["ref.issuer.issuer_id"], name=op.f("fk_classification_issuer_id_issuer")
        ),
        sa.ForeignKeyConstraint(
            ["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_classification_vendor_id_vendor")
        ),
        sa.PrimaryKeyConstraint("issuer_id", "vendor_id", "scheme", "valid_from", name=op.f("pk_classification")),
        schema="ref",
    )
    op.execute(
        "INSERT INTO ref.classification (issuer_id, vendor_id, scheme, valid_from, sector, industry_group, industry) "
        "SELECT issuer_id, classified_by, 'GICS', CAST(loaded_at AS DATE), sector, industry_group, industry "
        "FROM ref.issuer WHERE classified_by IS NOT NULL AND sector IS NOT NULL"
    )
    op.drop_constraint("fk_issuer_classified_by_vendor", "issuer", schema="ref", type_="foreignkey")
    for column in ("classified_by", "industry", "industry_group", "sector"):
        op.drop_column("issuer", column, schema="ref")
    op.execute("EXEC sp_rename 'ref.issuer.country', 'domicile_country', 'COLUMN'")
    op.execute("UPDATE ref.issuer SET domicile_country = NULL")


def downgrade() -> None:
    op.execute("EXEC sp_rename 'ref.issuer.domicile_country', 'country', 'COLUMN'")
    op.add_column("issuer", sa.Column("sector", sa.Unicode(length=64), nullable=True), schema="ref")
    op.add_column("issuer", sa.Column("industry_group", sa.Unicode(length=96), nullable=True), schema="ref")
    op.add_column("issuer", sa.Column("industry", sa.Unicode(length=96), nullable=True), schema="ref")
    op.add_column("issuer", sa.Column("classified_by", sa.SmallInteger(), nullable=True), schema="ref")
    op.create_foreign_key(
        "fk_issuer_classified_by_vendor",
        "issuer",
        "vendor",
        ["classified_by"],
        ["vendor_id"],
        source_schema="ref",
        referent_schema="ref",
    )
    op.execute(
        "UPDATE i SET sector = c.sector, industry_group = c.industry_group, industry = c.industry, "
        "classified_by = c.vendor_id FROM ref.issuer i JOIN ref.classification c ON c.issuer_id = i.issuer_id "
        "AND c.scheme = 'GICS' AND c.valid_to IS NULL"
    )
    op.drop_table("classification", schema="ref")
