"""What the database holds: rows per table, and the newest data per source.

table_rows(engine)       # [schema, table, rows]
newest(engine)           # [what, vendor, newest] - the latest day of each kind of data
"""

from __future__ import annotations

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research.db.schema import SCHEMAS

NEWEST = """
SELECT n.what, v.vendor_code AS vendor, n.newest FROM (
    SELECT 'daily prices' AS what, vendor_id, MAX(trade_date) AS newest FROM market.price_bar GROUP BY vendor_id
    UNION ALL SELECT 'dividends and splits', vendor_id, MAX(ex_date) FROM market.corporate_action GROUP BY vendor_id
    UNION ALL SELECT 'share counts', vendor_id, MAX(as_of_date) FROM market.share_count GROUP BY vendor_id
    UNION ALL SELECT 'company reports (published)', vendor_id, CAST(MAX(published_at) AS DATE)
        FROM fund.financial_fact GROUP BY vendor_id
    UNION ALL SELECT 'index membership changes', vendor_id, MAX(start_date) FROM ref.index_membership
        GROUP BY vendor_id
    UNION ALL SELECT 'analysts'' consensus', vendor_id, MAX(as_of_date) FROM fund.estimate GROUP BY vendor_id
    UNION ALL SELECT 'estimate revisions', vendor_id, MAX(as_of_date) FROM fund.estimate_revision GROUP BY vendor_id
    UNION ALL SELECT 'reported actuals (estimates)', vendor_id, CAST(MAX(announced_at) AS DATE)
        FROM fund.estimate_actual GROUP BY vendor_id
    UNION ALL SELECT 'raw files landed', vendor_id, CAST(MAX(fetched_at) AS DATE) FROM ingest.batch GROUP BY vendor_id
) n JOIN ref.vendor v ON v.vendor_id = n.vendor_id
"""


def table_rows(engine: Engine) -> pd.DataFrame:
    """``[schema, table, rows]`` for the project's tables (from the storage statistics, so it is instant)."""
    sql = sa.text(
        "SELECT s.name AS [schema], t.name AS [table], SUM(p.rows) AS rows FROM sys.tables t "
        "JOIN sys.schemas s ON s.schema_id = t.schema_id "
        "JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1) "
        "GROUP BY s.name, t.name ORDER BY s.name, t.name"
    )
    with engine.connect() as conn:
        rows = pd.DataFrame(conn.execute(sql).all(), columns=["schema", "table", "rows"])
    return rows[rows["schema"].isin(SCHEMAS)].reset_index(drop=True)


def newest(engine: Engine) -> pd.DataFrame:
    """``[what, vendor, newest]``: the latest date of each kind of data, per vendor."""
    with engine.connect() as conn:
        out = pd.DataFrame(conn.execute(sa.text(NEWEST)).all(), columns=["what", "vendor", "newest"])
    return out.dropna(subset=["newest"]).sort_values(["what", "vendor"]).reset_index(drop=True)
