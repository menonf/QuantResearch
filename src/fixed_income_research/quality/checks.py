"""The fixed income data-quality checks, recorded in ``ops.check_result`` and used as gates.

    results = run_checks(engine, BOND_CHECKS)
    gate(results)                       # raises if any check failed

Each measures one number with SQL (:mod:`research_core.data.quality` runs and records them). They ask whether the
month-end files were loaded whole and join up: weights that sum to the index, bonds with their analytics, months
without gaps, and how much of the index leaves each month without a return for its last month.
"""

from __future__ import annotations

from research_core.data.quality import Check

_MONTHS = """
WITH months AS (
    SELECT DISTINCT index_id, as_of_date FROM fixed_income.index_constituent
), spans AS (
    SELECT index_id, DATEDIFF(month, MIN(as_of_date), MAX(as_of_date)) + 1 AS expected, COUNT(*) AS held
    FROM months GROUP BY index_id
)
"""
#: The weight (percent of the index) that is in one month's file and not in the next: it left at the rebalance, and
#: its return over that last month is in no file.
_LEAVERS = """
WITH months AS (
    SELECT index_id, as_of_date, LEAD(as_of_date) OVER (PARTITION BY index_id ORDER BY as_of_date) AS next_date
    FROM (SELECT DISTINCT index_id, as_of_date FROM fixed_income.index_constituent) d
), left_weight AS (
    SELECT c.index_id, c.as_of_date, SUM(CASE WHEN n.listing_id IS NULL THEN c.weight_pct ELSE 0 END) AS left_pct
    FROM fixed_income.index_constituent c
    JOIN months m ON m.index_id = c.index_id AND m.as_of_date = c.as_of_date
        AND DATEDIFF(month, m.as_of_date, m.next_date) = 1
    LEFT JOIN fixed_income.index_constituent n
        ON n.index_id = c.index_id AND n.listing_id = c.listing_id AND n.as_of_date = m.next_date
    GROUP BY c.index_id, c.as_of_date
)
"""

BOND_CHECKS: tuple[Check, ...] = (
    Check(
        name="bond_index_weights_sum_to_100",
        sql=(
            "SELECT CAST(SUM(CASE WHEN ABS(total - 100.0) <= 0.1 THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) "
            "FROM (SELECT index_id, as_of_date, SUM(weight_pct) AS total FROM fixed_income.index_constituent "
            "GROUP BY index_id, as_of_date) w"
        ),
        warn=1.0,
        fail=0.99,
        describe="Share of index months whose constituent weights sum to 100 % (a file loaded in part does not)",
        examples=(
            "SELECT TOP 50 i.code, w.as_of_date, w.total FROM (SELECT index_id, as_of_date, SUM(weight_pct) AS total "
            "FROM fixed_income.index_constituent GROUP BY index_id, as_of_date) w "
            "JOIN fixed_income.index_definition i ON i.index_id = w.index_id "
            "WHERE ABS(w.total - 100.0) > 0.1 ORDER BY w.as_of_date"
        ),
    ),
    Check(
        name="bond_index_missing_months",
        sql=_MONTHS + "SELECT CAST(COALESCE(SUM(expected - held), 0) AS FLOAT) FROM spans",
        warn=0,
        fail=24,
        higher_is_better=False,
        describe="Month ends between an index's first and last file that have no file (no return can be rebuilt)",
    ),
    Check(
        name="bond_constituents_without_analytics",
        sql=(
            "SELECT CAST(COUNT(*) AS FLOAT) FROM fixed_income.index_constituent c WHERE NOT EXISTS (SELECT 1 FROM "
            "fixed_income.bond_analytics a WHERE a.listing_id = c.listing_id AND a.as_of_date = c.as_of_date)"
        ),
        warn=0,
        fail=0,
        higher_is_better=False,
        describe="Index members with no analytics on the same month end",
    ),
    Check(
        name="bonds_without_issuer",
        sql="SELECT CAST(COUNT(*) AS FLOAT) FROM core.listing WHERE instrument_type = 'bond' AND issuer_id IS NULL",
        warn=0,
        fail=0,
        higher_is_better=False,
        describe="Bonds not linked to an issuer",
    ),
    Check(
        name="held_bonds_without_return",
        sql=(
            "SELECT CAST(SUM(CASE WHEN a.total_return_mtd_pct IS NULL THEN c.prior_weight_pct ELSE 0 END) AS FLOAT) "
            "/ NULLIF(SUM(c.prior_weight_pct), 0) FROM fixed_income.index_constituent c "
            "JOIN fixed_income.bond_analytics a ON a.listing_id = c.listing_id AND a.as_of_date = c.as_of_date "
            "WHERE c.prior_weight_pct IS NOT NULL"
        ),
        warn=0.005,
        fail=0.05,
        higher_is_better=False,
        describe="Share of the weight held through a month (and still in the index) that has no return for it",
    ),
    Check(
        name="bond_index_weight_leaving_without_return",
        sql=_LEAVERS + "SELECT CAST(AVG(left_pct) AS FLOAT) FROM left_weight",
        warn=3.0,
        fail=10.0,
        higher_is_better=False,
        describe=(
            "Average percent of an index that leaves at a month-end rebalance: the files hold the next month's "
            "members, so these bonds' last month of return is in no file"
        ),
        examples=_LEAVERS + "SELECT TOP 50 i.code, l.as_of_date, l.left_pct FROM left_weight l "
        "JOIN fixed_income.index_definition i ON i.index_id = l.index_id ORDER BY l.left_pct DESC",
    ),
    Check(
        name="bond_rows_without_provenance",
        sql=(
            "SELECT CAST((SELECT COUNT(*) FROM fixed_income.bond_analytics WHERE batch_id IS NULL) "
            "+ (SELECT COUNT(*) FROM fixed_income.index_constituent WHERE batch_id IS NULL) "
            "+ (SELECT COUNT(*) FROM fixed_income.issuer_classification WHERE batch_id IS NULL) "
            "+ (SELECT COUNT(*) FROM fixed_income.index_level WHERE batch_id IS NULL) AS FLOAT)"
        ),
        warn=0,
        fail=0,
        higher_is_better=False,
        describe="Fixed income rows that do not name the raw file they came from",
    ),
)
