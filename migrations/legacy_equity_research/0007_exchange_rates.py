"""Exchange rates as listings: ``ref.listing.instrument_type`` ``fx``.

An ``fx`` listing is one unit of its ``currency``, priced in US dollars, day by day: research turns a dividend declared
in another currency (Anglo American's dollars, Old Mutual's rand) into its universe's currency on its ex-date.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-01
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TYPES = "ck_listing_instrument_type"


def upgrade() -> None:
    op.execute(f"ALTER TABLE ref.listing DROP CONSTRAINT {TYPES}")
    op.execute(
        f"ALTER TABLE ref.listing ADD CONSTRAINT {TYPES} "
        "CHECK (instrument_type IN ('equity', 'etf', 'index', 'fund', 'fx'))"
    )


def downgrade() -> None:
    rates = "(SELECT listing_id FROM ref.listing WHERE instrument_type = 'fx')"
    op.execute(f"DELETE FROM market.price_bar WHERE listing_id IN {rates}")
    op.execute(f"DELETE FROM ref.identifier WHERE entity_type = 'listing' AND entity_id IN {rates}")
    op.execute("DELETE FROM ref.listing WHERE instrument_type = 'fx'")
    op.execute(f"ALTER TABLE ref.listing DROP CONSTRAINT {TYPES}")
    op.execute(
        f"ALTER TABLE ref.listing ADD CONSTRAINT {TYPES} CHECK (instrument_type IN ('equity', 'etf', 'index', 'fund'))"
    )
