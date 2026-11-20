"""Several markets: tickers unique within a market, where a company is incorporated, closed-end funds.

* ``ref.identifier.market`` - a ticker names a share only among one country's exchanges (London's AAL is Anglo
  American, Nasdaq's American Airlines), so it is part of the identifier's key: ``US``, ``GB``; ``''`` for codes the
  same everywhere (ISIN, CUSIP, RIC...) and tickers of listings whose exchange is not recorded. Existing tickers take
  their listing's market.
* ``ref.issuer.incorporation_country`` - where the company is incorporated (Glencore: Jersey), which decides UK stamp
  duty; ``domicile_country`` stays where it is run.
* ``ref.listing.instrument_type`` ``fund`` - a closed-end fund (a UK investment trust): it trades as a share in an
  index, but its accounts are a portfolio's.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_KEY = "uq_identifier_id_type_value_valid_from_entity_type"
NEW_KEY = "uq_identifier_id_type_value_valid_from_entity_type_market"
TYPES = "ck_listing_instrument_type"
US = "'XNYS', 'XNAS', 'XASE', 'ARCX', 'BATS'"


def upgrade() -> None:
    op.add_column("identifier", sa.Column("market", sa.String(2), nullable=False, server_default=""), schema="ref")
    op.execute(
        "UPDATE i SET market = CASE WHEN l.exchange_mic IN ("
        + US
        + ") THEN 'US' WHEN l.exchange_mic = 'XLON' THEN 'GB' ELSE '' END "
        "FROM ref.identifier i JOIN ref.listing l ON l.listing_id = i.entity_id "
        "WHERE i.entity_type = 'listing' AND i.id_type = 'TICKER'"
    )
    op.drop_constraint(OLD_KEY, "identifier", schema="ref", type_="unique")
    op.create_unique_constraint(
        NEW_KEY, "identifier", ["id_type", "value", "valid_from", "entity_type", "market"], schema="ref"
    )
    op.add_column("issuer", sa.Column("incorporation_country", sa.CHAR(2), nullable=True), schema="ref")
    op.execute(f"ALTER TABLE ref.listing DROP CONSTRAINT {TYPES}")
    op.execute(
        f"ALTER TABLE ref.listing ADD CONSTRAINT {TYPES} CHECK (instrument_type IN ('equity', 'etf', 'index', 'fund'))"
    )


def downgrade() -> None:
    op.execute("UPDATE ref.listing SET instrument_type = 'equity' WHERE instrument_type = 'fund'")
    op.execute(f"ALTER TABLE ref.listing DROP CONSTRAINT {TYPES}")
    op.execute(f"ALTER TABLE ref.listing ADD CONSTRAINT {TYPES} CHECK (instrument_type IN ('equity', 'etf', 'index'))")
    op.drop_column("issuer", "incorporation_country", schema="ref")
    # a ticker held in two markets cannot go back under the old key: the later market's copy is dropped
    op.execute(
        "DELETE i FROM ref.identifier i WHERE i.market <> '' AND EXISTS (SELECT 1 FROM ref.identifier j "
        "WHERE j.id_type = i.id_type AND j.value = i.value AND j.valid_from = i.valid_from "
        "AND j.entity_type = i.entity_type AND j.identifier_id < i.identifier_id)"
    )
    op.drop_constraint(NEW_KEY, "identifier", schema="ref", type_="unique")
    op.create_unique_constraint(OLD_KEY, "identifier", ["id_type", "value", "valid_from", "entity_type"], schema="ref")
    op.drop_column("identifier", "market", schema="ref", mssql_drop_default=True)
