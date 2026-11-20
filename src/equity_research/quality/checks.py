"""Data-quality checks, recorded in ``ops.check_result`` and used as gates.

    results = run_checks(engine, CORE_CHECKS, run_id=run.run_id)
    gate(results)                       # raises if any check failed: nothing downstream runs on bad data

A check measures one number with SQL and compares it with a warning and a failure threshold.
"""

from __future__ import annotations

from research_core.data.quality import Check

#: A trading day is one on which at least half the index members have a price: a vendor's index levels on an
#: exchange holiday (Refinitiv prints some) must not count every stock as missing that day.
#: Counted by sets, never member-day by member-day (tens of millions of them): the members on a day are a running
#: total of joins less leaves, the priced ones a join of the bars (one per listing and day, whichever vendors priced
#: it) with the spells, per index and only then per country, carried through the running total in one pass (the
#: spells joined to their indices first, or the priced counts joined back to the running total, and Azure's plan ran
#: for hours instead of seconds). Each
#: country's indices count on their own trading days - the days some of its members were priced: a London bank
#: holiday is not a day New York's shares went unpriced, nor Thanksgiving one London's did.
_MEMBER_DAYS_CTE = """
WITH bars AS (SELECT DISTINCT listing_id, trade_date FROM equity.price_bar),
priced_in AS (
    SELECT m.index_id, b.trade_date, COUNT(*) AS priced FROM bars b
    JOIN equity.index_membership m ON m.listing_id = b.listing_id AND b.trade_date >= m.start_date
         AND (m.end_date IS NULL OR b.trade_date < m.end_date)
    GROUP BY m.index_id, b.trade_date
),
priced AS (
    SELECT COALESCE(d.country, '') AS country, p.trade_date, SUM(p.priced) AS priced
    FROM priced_in p JOIN equity.index_definition d ON d.index_id = p.index_id
    GROUP BY COALESCE(d.country, ''), p.trade_date
),
moves AS (
    SELECT COALESCE(d.country, '') AS country, m.start_date AS day, 1 AS n
    FROM equity.index_membership m JOIN equity.index_definition d ON d.index_id = m.index_id
    UNION ALL SELECT COALESCE(d.country, ''), m.end_date, -1
    FROM equity.index_membership m JOIN equity.index_definition d ON d.index_id = m.index_id
    WHERE m.end_date IS NOT NULL
),
calendar AS (
    SELECT country, day, SUM(n) AS n, SUM(priced) AS priced, MAX(traded) AS traded
    FROM (
        SELECT country, day, n, 0 AS priced, 0 AS traded FROM moves
        UNION ALL SELECT country, trade_date, 0, priced, 1 FROM priced
    ) x
    GROUP BY country, day
),
running AS (
    SELECT country, day, traded, priced,
           SUM(n) OVER (PARTITION BY country ORDER BY day ROWS UNBOUNDED PRECEDING) AS members
    FROM calendar
),
by_day AS (SELECT country, day AS trade_date, members, priced FROM running WHERE traded = 1 AND members > 0),
trading_days AS (SELECT country, trade_date, members, priced FROM by_day WHERE priced >= 0.5 * members)
"""
MEMBER_DAYS = _MEMBER_DAYS_CTE + "SELECT CAST(SUM(priced) AS FLOAT) / NULLIF(SUM(members), 0) FROM trading_days"
#: every universe benchmark priced on every trading day of its country from its first price on (reporting needs it).
#: Hash joins only: left to choose, Azure loops over the benchmarks and counts the trading days again for each.
BENCHMARK_DAYS = (
    _MEMBER_DAYS_CTE
    + """,
benchmark_country AS (
    SELECT u.benchmark_listing_id AS listing_id, MIN(COALESCE(d.country, '')) AS country
    FROM equity.universe u JOIN equity.universe_component c ON c.universe_id = u.universe_id
    JOIN equity.index_definition d ON d.index_id = c.index_id
    WHERE u.benchmark_listing_id IS NOT NULL GROUP BY u.benchmark_listing_id
),
benchmark_bars AS (
    SELECT DISTINCT b.listing_id, b.country, p.trade_date
    FROM benchmark_country b JOIN equity.price_bar p ON p.listing_id = b.listing_id
),
benchmarks AS (
    SELECT listing_id, country, MIN(trade_date) AS first_day FROM benchmark_bars GROUP BY listing_id, country
),
expected AS (
    SELECT b.listing_id, t.trade_date FROM trading_days t
    JOIN benchmarks b ON b.country = t.country AND t.trade_date >= b.first_day
)
SELECT AVG(CASE WHEN x.trade_date IS NULL THEN 0.0 ELSE 1.0 END) FROM expected e
LEFT JOIN benchmark_bars x ON x.listing_id = e.listing_id AND x.trade_date = e.trade_date
OPTION (HASH JOIN)
"""
)

#: A share's price moving 2.5x or more in a day, up or down, that no split or spin-off explains: a missed split or
#: a bad price far more often than a real move. Equities only (index levels do not split).
_JUMPS = """
WITH p AS (
    SELECT b.listing_id, b.vendor_id, b.trade_date, b.[close], b.currency,
           LAG(b.[close]) OVER (PARTITION BY b.listing_id, b.vendor_id ORDER BY b.trade_date) AS prev_close,
           LAG(b.currency) OVER (PARTITION BY b.listing_id, b.vendor_id ORDER BY b.trade_date) AS prev_currency
    FROM equity.price_bar b
    JOIN core.listing l ON l.listing_id = b.listing_id AND l.instrument_type IN ('equity', 'fund')
),
r AS (
    SELECT listing_id, vendor_id, ex_date, EXP(SUM(LOG(ratio))) AS ratio FROM equity.corporate_action
    WHERE action_type IN ('split', 'spinoff') AND ratio > 0 GROUP BY listing_id, vendor_id, ex_date
),
jumps AS (
    SELECT p.listing_id, p.vendor_id, p.trade_date, p.prev_close, p.[close], r.ratio
    FROM p LEFT JOIN r ON r.listing_id = p.listing_id AND r.vendor_id = p.vendor_id AND r.ex_date = p.trade_date
    WHERE p.prev_close > 0 AND p.[close] > 0 AND ABS(LOG(p.[close] * COALESCE(r.ratio, 1.0) / p.prev_close)) > LOG(2.5)
      AND COALESCE(p.currency, '') = COALESCE(p.prev_currency, '')  -- a change of quote currency explains its move
)
"""
#: An index member's price halving in a day (a fall of 1.8x to 2.5x - below the 2.5x of the jumps above) with no
#: split or spin-off within 5 days, where something else says the shares changed: a sibling share line of the company
#: starting to trade in the 30 days before it or the 5 after (a new class handed to holders: Alphabet's class C of
#: 2014, Under Armour's of 2016 - trading "when issued" a week or two before), or the line's own count of shares
#: rising 1.7x or more across it (a missed split: C.H. Robinson's 2-for-1 of 2005). Research books such a day as a
#: loss of half the holding. A real collapse with new shares issued shows up too (Guaranty Financial, 2008).
_SUSPECT_HALVINGS = """
WITH p AS (
    SELECT b.listing_id, b.vendor_id, b.trade_date, b.[close],
           LAG(b.[close]) OVER (PARTITION BY b.listing_id, b.vendor_id ORDER BY b.trade_date) AS prev_close
    FROM equity.price_bar b
    JOIN core.listing l ON l.listing_id = b.listing_id AND l.instrument_type = 'equity'
),
halved AS (
    SELECT p.listing_id, p.vendor_id, p.trade_date, p.prev_close, p.[close], l.issuer_id
    FROM p JOIN core.listing l ON l.listing_id = p.listing_id
    WHERE p.[close] > 0 AND p.prev_close / p.[close] BETWEEN 1.8 AND 2.5
      AND EXISTS (SELECT 1 FROM equity.index_membership m WHERE m.listing_id = p.listing_id
                  AND p.trade_date >= m.start_date AND (m.end_date IS NULL OR p.trade_date < m.end_date))
      AND NOT EXISTS (SELECT 1 FROM equity.corporate_action a WHERE a.listing_id = p.listing_id
                      AND a.action_type IN ('split', 'spinoff') AND ABS(DATEDIFF(day, a.ex_date, p.trade_date)) <= 5)
),
first_px AS (SELECT listing_id, MIN(trade_date) AS first_trade FROM equity.price_bar GROUP BY listing_id),
suspect AS (
    SELECT h.*, sib.listing_id AS new_line_id, cb.shares AS shares_before, ca.shares AS shares_after
    FROM halved h
    OUTER APPLY (SELECT TOP 1 s.listing_id FROM core.listing s JOIN first_px f ON f.listing_id = s.listing_id
                 WHERE s.issuer_id = h.issuer_id AND s.listing_id <> h.listing_id
                   AND f.first_trade BETWEEN DATEADD(day, -30, h.trade_date) AND DATEADD(day, 5, h.trade_date)) sib
    OUTER APPLY (SELECT TOP 1 c.shares FROM equity.share_count c WHERE c.listing_id = h.listing_id
                   AND c.kind = 'outstanding' AND c.as_of_date < DATEADD(day, -3, h.trade_date)
                   AND c.as_of_date >= DATEADD(day, -100, h.trade_date) ORDER BY c.as_of_date DESC) cb
    OUTER APPLY (SELECT TOP 1 c.shares FROM equity.share_count c WHERE c.listing_id = h.listing_id
                   AND c.kind = 'outstanding' AND c.as_of_date > DATEADD(day, 3, h.trade_date)
                   AND c.as_of_date <= DATEADD(day, 100, h.trade_date) ORDER BY c.as_of_date) ca
    WHERE sib.listing_id IS NOT NULL OR ca.shares >= 1.7 * cb.shares
)
"""
#: A member (now or in the past) whose prices stop more than 45 days before its membership does: a leaver the
#: vendor's event list missed, or a listing whose prices are missing.
_MEMBERS_AFTER_LAST_PRICE = """
WITH last_px AS (SELECT listing_id, MAX(trade_date) AS last_day FROM equity.price_bar GROUP BY listing_id),
newest AS (SELECT MAX(trade_date) AS d FROM equity.price_bar),
stale AS (
    SELECT m.index_id, m.listing_id, m.start_date, m.end_date, p.last_day
    FROM equity.index_membership m CROSS JOIN newest n LEFT JOIN last_px p ON p.listing_id = m.listing_id
    WHERE p.last_day IS NULL OR p.last_day < DATEADD(day, -45, COALESCE(m.end_date, n.d))
)
"""


#: The canonical tables whose every row names the raw batch it came from.
PROVENANCED = (
    "core.listing",
    "core.identifier",
    "equity.index_membership",
    "equity.classification",
    "equity.price_bar",
    "equity.corporate_action",
    "equity.share_count",
    "equity.financial_fact",
    "equity.estimate",
    "equity.estimate_revision",
    "equity.estimate_actual",
    "equity.document",
    "equity.text_measure",
)
_OVERLAPPING_SPELLS = """
WITH overlaps AS (
    SELECT a.index_id, a.listing_id, a.vendor_id, a.start_date, a.end_date, b.start_date AS next_start
    FROM equity.index_membership a JOIN equity.index_membership b
      ON a.index_id = b.index_id AND a.listing_id = b.listing_id AND a.vendor_id = b.vendor_id
     AND a.start_date < b.start_date AND COALESCE(a.end_date, '9999-12-31') > b.start_date
)
"""
#: A market's indices never share a member (S&P 500 / 400 / 600, FTSE 100 / 250 / SmallCap): a listing in two at once
#: is a replay's inferred date left standing (``ingest.membership.disjoint`` settles them on the next universe replay).
_IN_TWO_INDICES = """
WITH spells AS (
    SELECT m.index_id, m.listing_id, m.vendor_id, m.start_date, m.end_date, d.country
    FROM equity.index_membership m JOIN equity.index_definition d ON d.index_id = m.index_id
),
twice AS (
    SELECT a.listing_id, a.vendor_id, a.index_id, a.start_date, a.end_date, b.index_id AS other_index_id,
           b.start_date AS other_start, b.end_date AS other_end
    FROM spells a JOIN spells b
      ON a.listing_id = b.listing_id AND a.vendor_id = b.vendor_id AND a.country = b.country
     AND a.index_id < b.index_id
     AND a.start_date < COALESCE(b.end_date, '9999-12-31') AND b.start_date < COALESCE(a.end_date, '9999-12-31')
)
"""
_BEFORE_FIRST_TRADE = """
WITH first_px AS (SELECT listing_id, MIN(trade_date) AS first_trade FROM equity.price_bar GROUP BY listing_id),
early AS (
    SELECT m.index_id, m.listing_id, m.vendor_id, m.start_date, f.first_trade
    FROM equity.index_membership m JOIN first_px f ON f.listing_id = m.listing_id
    WHERE m.start_date < DATEADD(day, -7, f.first_trade)
)
"""


CORE_CHECKS: tuple[Check, ...] = (
    Check(
        "member_day_price_coverage", MEMBER_DAYS, warn=0.999, fail=0.99, describe="share of member-days with a price"
    ),
    Check(
        "benchmark_day_coverage",
        BENCHMARK_DAYS,
        warn=0.999,
        fail=0.5,  # reporting only (betas and returns come from the members' prices): a gap warns, it does not stop
        describe="share of trading days each universe benchmark has a level, from its first one",
    ),
    Check(
        "unexplained_price_jumps",
        _JUMPS + "SELECT CAST(COUNT(*) AS FLOAT) FROM jumps",
        warn=0,
        fail=1e9,  # some moves are real (a bankruptcy, a buyout): each one is looked at, none stops a load
        higher_is_better=False,
        describe="days a share moved 2.5x or more that no split or spin-off explains",
        examples=_JUMPS + "SELECT TOP 50 j.*, l.name FROM jumps j JOIN core.listing l ON l.listing_id = j.listing_id "
        "ORDER BY ABS(LOG(j.[close] * COALESCE(j.ratio, 1.0) / j.prev_close)) DESC",
    ),
    Check(
        "suspect_halvings",
        _SUSPECT_HALVINGS + "SELECT CAST(COUNT(*) AS FLOAT) FROM suspect",
        warn=0,
        fail=1e9,  # some are real (a collapse with new shares issued): each one is looked at, none stops a load
        higher_is_better=False,
        describe="days a member's price halved, unexplained, as a new share line began or its share count doubled",
        examples=_SUSPECT_HALVINGS + "SELECT TOP 50 s.*, l.name FROM suspect s "
        "JOIN core.listing l ON l.listing_id = s.listing_id ORDER BY s.trade_date",
    ),
    Check(
        "members_after_last_price",
        _MEMBERS_AFTER_LAST_PRICE + "SELECT CAST(COUNT(*) AS FLOAT) FROM stale",
        warn=0,
        fail=1e9,
        higher_is_better=False,
        describe="membership spells that go on more than 45 days after the listing's last price (a missed leaver)",
        examples=_MEMBERS_AFTER_LAST_PRICE
        + "SELECT TOP 50 d.code, s.*, l.name FROM stale s JOIN core.listing l ON l.listing_id = s.listing_id "
        "JOIN equity.index_definition d ON d.index_id = s.index_id ORDER BY s.last_day",
    ),
    Check(
        "equity_listings_without_issuer",
        "SELECT CAST(COUNT(*) AS FLOAT) FROM core.listing WHERE instrument_type = 'equity' AND issuer_id IS NULL",
        warn=0,
        fail=1e9,
        higher_is_better=False,
        describe="equity listings no company is attached to",
    ),
    Check(
        "facts_published_before_period_end",
        "SELECT CAST(COUNT(*) AS FLOAT) FROM equity.financial_fact WHERE period_end > CAST(published_at AS DATE)",
        warn=0,
        fail=0,
        higher_is_better=False,
        describe="financial facts about a period that had not ended when they were published (impossible: bad data)",
        examples="SELECT TOP 50 f.*, i.name FROM equity.financial_fact f "
        "JOIN core.issuer i ON i.issuer_id = f.issuer_id "
        "WHERE f.period_end > CAST(f.published_at AS DATE) ORDER BY f.period_end DESC",
    ),
    Check(
        "overlapping_membership_spells",
        _OVERLAPPING_SPELLS + "SELECT CAST(COUNT(*) AS FLOAT) FROM overlaps",
        warn=0,
        fail=0,  # a share counted twice in an index: weights and member counts would be wrong
        higher_is_better=False,
        describe="membership spells of one listing in one index (one vendor) that overlap",
        examples=_OVERLAPPING_SPELLS + "SELECT TOP 50 o.*, l.name FROM overlaps o "
        "JOIN core.listing l ON l.listing_id = o.listing_id",
    ),
    Check(
        "members_of_two_indices",
        _IN_TWO_INDICES + "SELECT CAST(COUNT(*) AS FLOAT) FROM twice",
        warn=0,
        fail=1e9,  # the next universe replay settles them; until then each is looked at
        higher_is_better=False,
        describe="spells of one listing in two of a market's indices at once (they never share a member)",
        examples=_IN_TWO_INDICES + "SELECT TOP 50 t.*, l.name FROM twice t "
        "JOIN core.listing l ON l.listing_id = t.listing_id ORDER BY t.start_date",
    ),
    Check(
        "members_before_first_trade",
        _BEFORE_FIRST_TRADE + "SELECT CAST(COUNT(*) AS FLOAT) FROM early",
        warn=0,
        fail=1e9,  # the next universe replay clips them; until then each is looked at
        higher_is_better=False,
        describe="membership spells starting over a week before the listing's first price (a join the vendor missed)",
        examples=_BEFORE_FIRST_TRADE + "SELECT TOP 50 e.*, l.name FROM early e "
        "JOIN core.listing l ON l.listing_id = e.listing_id ORDER BY e.first_trade DESC",
    ),
    Check(
        "bars_outside_their_range",
        "SELECT CAST(COUNT(*) AS FLOAT) FROM equity.price_bar WHERE high < low "
        "OR [close] > high * 1.0001 OR [close] < low * 0.9999",
        warn=0,
        fail=1e9,
        higher_is_better=False,
        describe="daily bars whose close lies outside their own high / low (a vendor mixing two days)",
    ),
    Check(
        "rows_without_provenance",
        "SELECT "
        + " + ".join(f"(SELECT CAST(COUNT(*) AS FLOAT) FROM {t} WHERE batch_id IS NULL)" for t in PROVENANCED),
        warn=0,
        fail=1e9,
        higher_is_better=False,
        describe="canonical rows that do not name the raw batch they came from",
    ),
    Check(
        "issuers_with_cik_without_facts",
        """SELECT COALESCE(CAST(SUM(CASE WHEN f.issuer_id IS NULL THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0), 0)
           FROM core.identifier i
           LEFT JOIN (SELECT DISTINCT issuer_id FROM equity.financial_fact) f ON f.issuer_id = i.entity_id
           WHERE i.entity_type = 'issuer' AND i.id_type = 'CIK'""",
        warn=0.10,
        fail=0.50,
        higher_is_better=False,
        describe="matched companies with no financial statements loaded (none matched, as in London: nothing missing)",
    ),
    Check(
        "annual_reports_without_text_measures",
        """SELECT COALESCE(CAST(SUM(CASE WHEN m.batch_id IS NULL THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0), 0)
           FROM ingest.batch b
           LEFT JOIN (SELECT DISTINCT batch_id FROM equity.text_measure) m ON m.batch_id = b.batch_id
           WHERE b.dataset = 'annual_report'""",
        warn=0.01,
        fail=0.10,
        higher_is_better=False,
        describe="landed 10-K documents with no text measures (a load stopped between landing and measuring)",
    ),
)


def by_name(name: str) -> Check:
    """A core check by its name."""
    return next(c for c in CORE_CHECKS if c.name == name)
