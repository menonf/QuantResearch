"""Vendor 9 ``repository``: facts this repository states itself where every vendor is wrong or silent.

The first are corporate actions LSEG never recorded - a new share class handed to holders (Alphabet's class C of
2014), a split it lacks (C.H. Robinson's 2-for-1 of 2005) - kept in ``ingest/corrections/corporate_actions.csv``,
landed in the raw zone like a vendor's answer and written to ``market.corporate_action`` under this vendor.

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-14
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "INSERT INTO ref.vendor (vendor_id, vendor_code, name) "
        "VALUES (9, 'repository', 'Corrections kept in this repository')"
    )


def downgrade() -> None:
    op.execute("DELETE FROM market.corporate_action WHERE vendor_id = 9")
    op.execute("DELETE FROM ingest.batch WHERE vendor_id = 9")
    op.execute("DELETE FROM ref.vendor WHERE vendor_id = 9")
