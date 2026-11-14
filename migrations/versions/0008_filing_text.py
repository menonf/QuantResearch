"""Filing documents and measures of their text: ``fund.document``, ``fund.text_measure``; vendor 8 ``sraf``.

``fund.document`` lists every annual, quarterly and current report of a company with the instant EDGAR accepted it
and an 8-K's item codes; ``fund.text_measure`` holds what research reads of a document's text (the text stays in the
raw zone). ``sraf`` is Notre Dame's Software Repository for Accounting and Finance, which publishes the
Loughran-McDonald dictionary.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-08
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FORMS = "'10-K', '10-K405', '10-KT', '10-K/A', '10-K405/A', '10-KT/A', '10-Q', '10-Q/A', '8-K', '8-K/A'"


def _provenance() -> list[sa.Column[Any]]:
    return [
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.Column("batch_id", sa.Integer(), nullable=True),
    ]


def upgrade() -> None:
    op.execute(
        "INSERT INTO ref.vendor (vendor_id, vendor_code, name) "
        "VALUES (8, 'sraf', 'Notre Dame SRAF (Loughran-McDonald dictionary)')"
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
        *_provenance(),
        sa.CheckConstraint(f"form IN ({FORMS})", name=op.f("ck_document_form")),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_document_batch_id_batch")),
        sa.ForeignKeyConstraint(["issuer_id"], ["ref.issuer.issuer_id"], name=op.f("fk_document_issuer_id_issuer")),
        sa.ForeignKeyConstraint(["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_document_vendor_id_vendor")),
        sa.PrimaryKeyConstraint("issuer_id", "vendor_id", "filing_ref", name=op.f("pk_document")),
        schema="fund",
    )
    op.create_index(
        op.f("ix_document_issuer_id_published_at"),
        "document",
        ["issuer_id", "published_at"],
        unique=False,
        schema="fund",
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
        *_provenance(),
        sa.CheckConstraint("section IN ('all', 'mdna', 'risk')", name=op.f("ck_text_measure_section")),
        sa.ForeignKeyConstraint(["batch_id"], ["ingest.batch.batch_id"], name=op.f("fk_text_measure_batch_id_batch")),
        sa.ForeignKeyConstraint(["issuer_id"], ["ref.issuer.issuer_id"], name=op.f("fk_text_measure_issuer_id_issuer")),
        sa.ForeignKeyConstraint(["vendor_id"], ["ref.vendor.vendor_id"], name=op.f("fk_text_measure_vendor_id_vendor")),
        sa.PrimaryKeyConstraint(
            "issuer_id", "vendor_id", "filing_ref", "section", "measure", name=op.f("pk_text_measure")
        ),
        schema="fund",
    )


def downgrade() -> None:
    op.drop_table("text_measure", schema="fund")
    op.drop_index(op.f("ix_document_issuer_id_published_at"), table_name="document", schema="fund")
    op.drop_table("document", schema="fund")
    op.execute("DELETE FROM ingest.batch WHERE vendor_id = 8")
    op.execute("DELETE FROM ref.vendor WHERE vendor_id = 8")
