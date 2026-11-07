"""FinanceDatabase and Norgate Data as vendors, and Norgate's asset id as an identifier type.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CHECK = "ck_identifier_id_type"
BEFORE = "'TICKER', 'RIC', 'CUSIP', 'ISIN', 'CIK', 'SEDOL', 'FIGI', 'PERMID', 'YAHOO'"


def upgrade() -> None:
    op.execute(
        "INSERT INTO ref.vendor (vendor_id, vendor_code, name) VALUES "
        "(4, 'financedatabase', 'FinanceDatabase (open source, JerBouma)'), (5, 'norgate', 'Norgate Data')"
    )
    op.execute(f"ALTER TABLE ref.identifier DROP CONSTRAINT {CHECK}")
    op.execute(f"ALTER TABLE ref.identifier ADD CONSTRAINT {CHECK} CHECK (id_type IN ({BEFORE}, 'NORGATE'))")


def downgrade() -> None:
    op.execute("DELETE FROM ref.identifier WHERE id_type = 'NORGATE'")
    op.execute(f"ALTER TABLE ref.identifier DROP CONSTRAINT {CHECK}")
    op.execute(f"ALTER TABLE ref.identifier ADD CONSTRAINT {CHECK} CHECK (id_type IN ({BEFORE}))")
    for table in (
        "ref.identifier",
        "ref.classification",
        "ref.index_membership",
        "market.price_bar",
        "market.corporate_action",
        "market.share_count",
        "fund.financial_fact",
        "fund.filing",
        "ingest.batch",
    ):
        op.execute(f"DELETE FROM {table} WHERE vendor_id IN (4, 5)")
    op.execute("DELETE FROM ref.vendor WHERE vendor_id IN (4, 5)")
