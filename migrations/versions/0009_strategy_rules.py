"""Strategies' rules as shared tables: ``research.rule_set`` and ``research.strategy_rule``.

A rule set is a named collection of rules a strategy applies - the limits a factor's values must lie within
(Barings' ``settings.metric_rules`` is the first), and later a composite's weights or a selection's thresholds. A
registered run names the set it used, so a set is loaded once and never changed: a changed rule is a new set.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RULE_TYPES = "'lower_limit', 'upper_limit', 'exclude_financials', 'invert', 'weight', 'threshold', 'setting'"


def upgrade() -> None:
    op.create_table(
        "rule_set",
        sa.Column("rule_set_id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Unicode(length=400), nullable=False),
        sa.Column("source", sa.Unicode(length=400), nullable=False),
        sa.Column("checksum", sa.CHAR(length=64), nullable=False),
        sa.Column(
            "loaded_at", mssql.DATETIME2(precision=0), server_default=sa.text("SYSUTCDATETIME()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("rule_set_id", name=op.f("pk_rule_set")),
        sa.UniqueConstraint("code", name=op.f("uq_rule_set_code")),
        schema="research",
    )
    op.create_table(
        "strategy_rule",
        sa.Column("rule_set_id", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=128), nullable=False),
        sa.Column("rule_type", sa.String(length=32), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.CheckConstraint(f"rule_type IN ({RULE_TYPES})", name=op.f("ck_strategy_rule_rule_type")),
        sa.ForeignKeyConstraint(
            ["rule_set_id"], ["research.rule_set.rule_set_id"], name=op.f("fk_strategy_rule_rule_set_id_rule_set")
        ),
        sa.PrimaryKeyConstraint("rule_set_id", "subject", "rule_type", name=op.f("pk_strategy_rule")),
        schema="research",
    )


def downgrade() -> None:
    op.drop_table("strategy_rule", schema="research")
    op.drop_table("rule_set", schema="research")
