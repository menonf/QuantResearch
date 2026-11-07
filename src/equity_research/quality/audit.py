"""Does the data follow the architecture, and is it fit to backtest on? Read-only.

    findings = audit(engine)            # [area, rule, value, ok, note] - every rule of docs/ARCHITECTURE.md
    print(findings[~findings["ok"]])

The quality gates (:mod:`.checks`) run after every load and stop what follows; the audit is the slower, fuller look
after a build: the shape of the data (one listing per share line, no code held twice at once, membership spells that
make sense, accounts dated before they were used, every row traceable to its raw file) and its fitness for purpose -
whether, year by year, the members of each index have prices, accounts and share counts known at the time, and
whether every vendor the preferences name actually supplies data.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pycountry
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research.db.schema import VENDORS
from equity_research.quality.checks import _MEMBER_DAYS_CTE, PROVENANCED
from equity_research.raw.store import sha256_file
from equity_research.settings import get_settings, vendor_preferences

#: years whose member-days must be this fully priced (an index member without a price drops out of a backtest)
YEAR_COVERAGE = 0.995
#: share of each year-end's members with accounts / a share count known by then
YEAR_END_COVERAGE = 0.95
FACT_AGE_DAYS = 400  # accounts older than this on a year end count as missing (the research's MAX_AGE_DAYS)
SHARES_AGE_DAYS = 400
CHECKSUM_SAMPLE = 150


@dataclass(frozen=True)
class Finding:
    """One rule's outcome."""

    area: str
    rule: str
    value: object
    ok: bool
    note: str = ""


class _Audit:
    def __init__(self, engine: Engine, raw_dir: Path) -> None:
        self.engine = engine
        self.raw_dir = raw_dir
        self.findings: list[Finding] = []

    def q(self, sql: str, **params: Any) -> list[Any]:
        with self.engine.connect() as c:
            return list(c.execute(sa.text(sql), params).all())

    def one(self, sql: str, **params: Any) -> Any:
        return self.q(sql, **params)[0][0]

    def rule(self, area: str, text: str, measure: Callable[[], tuple[object, bool, str]]) -> None:
        try:
            value, ok, note = measure()
        except Exception as exc:  # a rule that cannot be measured is a finding too, never a crash
            value, ok, note = None, False, f"could not measure: {type(exc).__name__}: {exc}"[:300]
        self.findings.append(Finding(area, text, value, bool(ok), note))

    def none(self, area: str, text: str, sql: str, note: str = "") -> None:
        """A rule that holds when ``sql`` counts nothing."""
        self.rule(area, text, lambda: (n := int(self.one(sql)), n == 0, note if n else ""))


def audit(engine: Engine, *, raw_dir: Path | None = None, sample: int = CHECKSUM_SAMPLE) -> pd.DataFrame:
    """Every rule, ``[area, rule, value, ok, note]`` (see the module docstring)."""
    a = _Audit(engine, raw_dir or get_settings().raw_dir)
    _schema(a)
    _identity(a)
    _membership(a)
    _prices(a)
    _accounts(a)
    _fitness(a)
    _lineage(a, sample)
    _ops(a)
    return pd.DataFrame([f.__dict__ for f in a.findings])


# ---------------------------------------------------------------- the shape of the data


def _schema(a: _Audit) -> None:
    from equity_research.db.migrate import head_revision  # noqa: PLC0415 - alembic is slow to import

    a.rule(
        "schema",
        "migrated to the latest revision",
        lambda: (v := a.one("SELECT version_num FROM alembic_version"), v == head_revision(), ""),
    )
    a.rule(
        "schema",
        "ref.vendor holds the fixed vendor ids",
        lambda: (
            got := {c: v for v, c in a.q("SELECT vendor_id, vendor_code FROM ref.vendor")},
            got == {c: v[0] for c, v in VENDORS.items()},
            "",
        ),
    )


def _identity(a: _Audit) -> None:
    a.none(
        "identity",
        "every equity listing has a company",
        "SELECT COUNT(*) FROM ref.listing WHERE instrument_type = 'equity' AND issuer_id IS NULL",
    )
    a.none(
        "identity",
        "index levels have no company",
        "SELECT COUNT(*) FROM ref.listing WHERE instrument_type = 'index' AND issuer_id IS NOT NULL",
    )
    a.none(
        "identity",
        "every listing has an identifier",
        "SELECT COUNT(*) FROM ref.listing l WHERE NOT EXISTS (SELECT 1 FROM ref.identifier i "
        "WHERE i.entity_type = 'listing' AND i.entity_id = l.listing_id)",
    )
    a.none(
        "identity",
        "identifiers point at an existing listing or company",
        "SELECT COUNT(*) FROM ref.identifier i WHERE (i.entity_type = 'listing' AND NOT EXISTS "
        "(SELECT 1 FROM ref.listing l WHERE l.listing_id = i.entity_id)) OR (i.entity_type = 'issuer' AND NOT EXISTS "
        "(SELECT 1 FROM ref.issuer s WHERE s.issuer_id = i.entity_id))",
    )
    a.none(
        "identity",
        "an identifier's validity ends after it starts",
        "SELECT COUNT(*) FROM ref.identifier WHERE valid_to IS NOT NULL AND valid_to <= valid_from",
    )
    overlaps = dict(
        a.q(
            """SELECT x.id_type, COUNT(*) FROM ref.identifier x JOIN ref.identifier y
                 ON x.id_type = y.id_type AND x.value = y.value AND x.entity_type = y.entity_type
                AND x.entity_id < y.entity_id
                AND x.valid_from < COALESCE(y.valid_to, '9999-12-31') AND y.valid_from < COALESCE(x.valid_to,
                    '9999-12-31')
               GROUP BY x.id_type"""
        )
    )
    for id_type in ("RIC", "ISIN", "CUSIP", "CIK", "TICKER", "YAHOO", "FIGI", "NORGATE", "SIMFIN"):
        n = int(overlaps.get(id_type, 0))
        a.findings.append(Finding("identity", f"no {id_type} held by two entities at once", n, n == 0))
    a.none(
        "identity",
        "every company has a listing",
        "SELECT COUNT(*) FROM ref.issuer s WHERE NOT EXISTS (SELECT 1 FROM ref.listing l WHERE l.issuer_id = "
        "s.issuer_id)",
    )
    a.none(
        "identity",
        "one listing per share line (an ISIN open on one listing only)",
        "SELECT COUNT(*) FROM (SELECT value FROM ref.identifier WHERE id_type = 'ISIN' AND entity_type = 'listing' "
        "AND valid_to IS NULL GROUP BY value HAVING COUNT(DISTINCT entity_id) > 1) x",
    )
    codes = {c.alpha_2 for c in pycountry.countries}

    def homes() -> tuple[object, bool, str]:
        bad = [
            d
            for (d,) in a.q("SELECT domicile_country FROM ref.issuer WHERE domicile_country IS NOT NULL")
            if d not in codes
        ]
        return len(bad), not bad, str(sorted(set(bad))[:5]) if bad else ""

    a.rule("identity", "countries are ISO codes", homes)
    a.none(
        "identity",
        "one open classification per company, vendor and scheme",
        "SELECT COUNT(*) FROM (SELECT issuer_id FROM ref.classification WHERE valid_to IS NULL "
        "GROUP BY issuer_id, vendor_id, scheme HAVING COUNT(*) > 1) x",
    )


def _membership(a: _Audit) -> None:
    a.none(
        "members",
        "a spell ends after it starts",
        "SELECT COUNT(*) FROM ref.index_membership WHERE end_date IS NOT NULL AND end_date <= start_date",
    )
    a.none(
        "members",
        "no overlapping spells of one listing in one index",
        "SELECT COUNT(*) FROM ref.index_membership x JOIN ref.index_membership y ON x.index_id = y.index_id "
        "AND x.listing_id = y.listing_id AND x.vendor_id = y.vendor_id AND x.start_date < y.start_date "
        "AND COALESCE(x.end_date, '9999-12-31') > y.start_date",
    )
    a.none(
        "members",
        "members are equities (or closed-end funds: UK investment trusts)",
        "SELECT COUNT(*) FROM ref.index_membership m JOIN ref.listing l ON l.listing_id = m.listing_id "
        "WHERE l.instrument_type NOT IN ('equity', 'fund')",
    )
    a.none(
        "members",
        "no member before its first trade",
        "SELECT COUNT(*) FROM ref.index_membership m JOIN (SELECT listing_id, MIN(trade_date) AS d "
        "FROM market.price_bar GROUP BY listing_id) f ON f.listing_id = m.listing_id "
        "WHERE m.start_date < DATEADD(day, -7, f.d)",
    )
    a.none(
        "members",
        "every index with members names the listing carrying its level",
        "SELECT COUNT(*) FROM ref.index_definition WHERE level_listing_id IS NULL "
        "AND index_id IN (SELECT index_id FROM ref.index_membership)",
    )
    # the FTSE SmallCap has no fixed size (everything below the 350 big enough to qualify)
    expected = {"SP500": 500, "SP400": 400, "SP600": 600, "FTSE100": 100, "FTSE250": 250}

    def sizes() -> tuple[object, bool, str]:
        rows = a.q(
            """WITH d AS (SELECT DATEFROMPARTS(y, 6, 30) AS day FROM (VALUES (2005),(2007),(2009),(2011),(2013),(2015),
                      (2017),(2019),(2021),(2023),(2025)) v(y))
               SELECT x.code, d.day, COUNT(m.listing_id) FROM d CROSS JOIN ref.index_definition x
               LEFT JOIN ref.index_membership m ON m.index_id = x.index_id AND m.start_date <= d.day
                    AND (m.end_date IS NULL OR m.end_date > d.day)
               WHERE d.day >= (SELECT MIN(start_date) FROM ref.index_membership)
                 AND x.index_id IN (SELECT index_id FROM ref.index_membership)
               GROUP BY x.code, d.day"""
        )
        off = [f"{c} {d}: {n}" for c, d, n in rows if c in expected and abs(n - expected[c]) > 0.03 * expected[c]]
        return len(off), not off, "; ".join(off[:6])

    a.rule(
        "members", "index sizes within 3% of 500 / 400 / 600, 100 / 250 (each other 30 June the history covers)", sizes
    )


def _prices(a: _Audit) -> None:
    a.none(
        "prices",
        "a close lies within its day's high and low",
        "SELECT COUNT(*) FROM market.price_bar WHERE high < low OR [close] > high * 1.0001 OR [close] < low * 0.9999",
    )
    a.none(
        "prices",
        "no bars on a weekend",
        "SELECT COUNT(*) FROM market.price_bar WHERE DATEPART(weekday, trade_date) IN (1, 7)",
    )
    a.none(
        "prices",
        "no bars after a listing's last day",
        "SELECT COUNT(*) FROM market.price_bar p JOIN ref.listing l ON l.listing_id = p.listing_id "
        "WHERE l.delisted_on IS NOT NULL AND p.trade_date > l.delisted_on",
    )
    a.none(
        "prices",
        "prices in a currency research reads (dollars, pounds, pence as GBX, or one with an exchange rate)",
        "SELECT COUNT(*) FROM market.price_bar p "
        "WHERE p.currency IS NOT NULL AND p.currency NOT IN ('USD', 'GBP', 'GBX') "
        "AND NOT EXISTS (SELECT 1 FROM ref.listing x WHERE x.instrument_type = 'fx' AND x.currency = p.currency)",
    )
    a.none(
        "prices",
        # a line can change the currency it is quoted in (IHG: pence to dollars, 2026): its latest bar is in its own;
        # an exchange rate is one unit of its currency, priced in dollars
        "a listing's latest bar in the listing's currency",
        "SELECT COUNT(*) FROM ref.listing l CROSS APPLY (SELECT TOP 1 p.currency FROM market.price_bar p "
        "WHERE p.listing_id = l.listing_id ORDER BY p.trade_date DESC) b "
        "WHERE b.currency IS NOT NULL AND l.currency IS NOT NULL AND b.currency <> l.currency "
        "AND l.instrument_type <> 'fx'",
    )
    a.none(
        "prices",
        "a dividend declared in another currency has its exchange rate",
        "SELECT COUNT(DISTINCT a.currency) FROM market.corporate_action a WHERE a.action_type = 'cash_dividend' "
        "AND a.currency IS NOT NULL AND a.currency <> 'USD' AND NOT EXISTS (SELECT 1 FROM ref.listing l "
        "WHERE l.instrument_type = 'fx' AND l.currency = a.currency)",
    )
    a.none("prices", "volume is not negative", "SELECT COUNT(*) FROM market.price_bar WHERE volume < 0")
    a.none(
        "prices",
        "a split has a positive ratio, a dividend a positive amount",
        "SELECT COUNT(*) FROM market.corporate_action WHERE (action_type IN ('split', 'spinoff') AND NOT ratio > 0) "
        "OR (action_type = 'cash_dividend' AND NOT amount > 0)",
    )


def _accounts(a: _Audit) -> None:
    a.none(
        "accounts",
        "period types are Q / YTD / FY / I",
        "SELECT COUNT(*) FROM fund.financial_fact WHERE period_type NOT IN ('Q', 'YTD', 'FY', 'I')",
    )
    a.none(
        "accounts",
        "units are dollars, pounds or shares",
        "SELECT COUNT(*) FROM fund.financial_fact WHERE unit NOT IN ('USD', 'GBP', 'shares')",
    )
    a.none(
        "accounts",
        "an instant has no start, a flow has one",
        "SELECT COUNT(*) FROM fund.financial_fact WHERE (period_type = 'I' AND period_start IS NOT NULL) "
        "OR (period_type <> 'I' AND period_start IS NULL)",
    )
    a.none(
        "accounts",
        "a quarter is 80-100 days, a year so far 160-290, a year 350-380",
        "SELECT COUNT(*) FROM fund.financial_fact WHERE (period_type = 'Q' AND DATEDIFF(day, period_start, period_end) "
        "NOT BETWEEN 79 AND 100) OR (period_type = 'FY' AND DATEDIFF(day, period_start, period_end) NOT "
        "BETWEEN 349 AND 380) OR (period_type = 'YTD' AND DATEDIFF(day, period_start, period_end) NOT "
        "BETWEEN 160 AND 290)",
    )
    a.none(
        "accounts",
        "no figure about a period that had not ended when published",
        "SELECT COUNT(*) FROM fund.financial_fact WHERE period_end > CAST(published_at AS DATE)",
    )
    a.none(
        "accounts",
        "figures belong to a company with a listing",
        "SELECT COUNT(*) FROM fund.financial_fact f WHERE NOT EXISTS (SELECT 1 FROM ref.listing l WHERE "
        "l.issuer_id = f.issuer_id)",
    )
    a.none(
        "accounts",
        "SEC figures name a filing that exists",
        "SELECT COUNT(*) FROM fund.financial_fact f WHERE f.vendor_id = 2 AND f.filing_ref IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM fund.filing g WHERE g.issuer_id = f.issuer_id AND g.vendor_id = 2 AND g.filing_ref = "
        "f.filing_ref)",
    )


# ---------------------------------------------------------------- fit for backtesting and index reconstruction


def _fitness(a: _Audit) -> None:
    prefs = vendor_preferences()

    def yearly() -> tuple[object, bool, str]:
        rows = a.q(
            _MEMBER_DAYS_CTE
            + """SELECT YEAR(d.trade_date) AS y, SUM(CASE WHEN t.trade_date IS NULL THEN 0 ELSE t.priced END),
                        SUM(CASE WHEN t.trade_date IS NULL THEN 0 ELSE t.members END), COUNT(t.trade_date)
                 FROM by_day d LEFT JOIN trading_days t ON t.country = d.country AND t.trade_date = d.trade_date
                 GROUP BY YEAR(d.trade_date) ORDER BY y"""
        )
        start = int(a.one("SELECT YEAR(MIN(start_date)) FROM ref.index_membership"))
        cover = {int(y): (p / m if m else 0.0) for y, p, m, _ in rows if int(y) >= start}
        low = {y: round(c, 4) for y, c in cover.items() if c < YEAR_COVERAGE}
        return (low or "every year"), not low, "per year: " + ", ".join(f"{y} {c:.2%}" for y, c in cover.items())

    a.rule("fitness", f"members priced on >= {YEAR_COVERAGE:.1%} of their days, every year since the first", yearly)

    account_vendors = ", ".join(f"'{v}'" for v in prefs.fundamentals)  # our own vendor codes, not input

    def year_ends(kind: str) -> tuple[object, bool, str]:
        exists = {
            "accounts": f"""EXISTS (SELECT 1 FROM fund.financial_fact f JOIN ref.vendor v ON v.vendor_id = f.vendor_id
                              WHERE f.issuer_id = l.issuer_id AND v.vendor_code IN ({account_vendors})
                                AND f.item = 'total_assets'
                                AND f.published_at < d.day AND f.published_at >= DATEADD(day, -{FACT_AGE_DAYS},
                                    d.day))""",
            "shares": f"""(EXISTS (SELECT 1 FROM market.share_count s WHERE s.listing_id = m.listing_id
                              AND s.as_of_date < d.day AND s.as_of_date >= DATEADD(day, -{SHARES_AGE_DAYS}, d.day))
                           OR EXISTS (SELECT 1 FROM fund.financial_fact f WHERE f.issuer_id = l.issuer_id
                              AND f.vendor_id = 2 AND f.item IN ('shares_cover', 'shares_outstanding')
                              AND f.published_at < d.day AND f.published_at >= DATEADD(day, -{SHARES_AGE_DAYS},
                                  d.day)))""",
        }[kind]
        rows = a.q(
            f"""WITH first_year AS (SELECT YEAR(MIN(start_date)) AS y FROM ref.index_membership),
                years AS (SELECT TOP (80) f.y + ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1 AS y
                          FROM sys.all_objects CROSS JOIN first_year f),
                d AS (SELECT DATEFROMPARTS(y, 12, 31) AS day FROM years WHERE y < YEAR(GETDATE()))
                SELECT y, COUNT(*), SUM(has) FROM (
                    SELECT YEAR(d.day) AS y, CASE WHEN {exists} THEN 1 ELSE 0 END AS has
                    FROM d JOIN ref.index_membership m ON m.start_date <= d.day AND (m.end_date IS NULL OR
                        m.end_date > d.day)
                    JOIN ref.listing l ON l.listing_id = m.listing_id
                ) x GROUP BY y ORDER BY y""",
        )
        cover = {int(y): k / n for y, n, k in rows if n}
        low = {y: round(c, 3) for y, c in cover.items() if c < YEAR_END_COVERAGE}
        return (low or "every year"), not low, "per year end: " + ", ".join(f"{y} {c:.0%}" for y, c in cover.items())

    a.rule(
        "fitness",
        f"members at each year end with accounts ({', '.join(prefs.fundamentals)}) published in the year before "
        f"(value, quality)",
        lambda: year_ends("accounts"),
    )
    a.rule(
        "fitness",
        "members at each year end with a share count known then (market value, index weights)",
        lambda: year_ends("shares"),
    )

    def benchmarks() -> tuple[object, bool, str]:
        rows = a.q(
            """SELECT u.code, i.value, MIN(p.trade_date), MAX(p.trade_date) FROM ref.universe u
               LEFT JOIN ref.identifier i ON i.entity_type = 'listing' AND i.entity_id = u.benchmark_listing_id
                    AND i.id_type IN ('YAHOO', 'RIC')
               LEFT JOIN market.price_bar p ON p.listing_id = u.benchmark_listing_id
               WHERE EXISTS (SELECT 1 FROM ref.universe_component c JOIN ref.index_membership m
                             ON m.index_id = c.index_id WHERE c.universe_id = u.universe_id)
               GROUP BY u.code, i.value"""
        )
        newest = a.one("SELECT MAX(trade_date) FROM market.price_bar")
        start = a.one("SELECT MIN(start_date) FROM ref.index_membership")
        bad = [
            f"{code} ({value}): {lo}..{hi}"
            for code, value, lo, hi in rows
            if lo is None or hi is None or lo > start or (newest - hi).days > 7
        ]
        return len(bad), not bad, "; ".join(bad[:6]) or f"all from {start} to {newest}"

    a.rule("fitness", "every universe's benchmark priced from the first membership to the newest day", benchmarks)
    a.rule(
        "fitness",
        "today's members have a sector",
        lambda: (
            missing := int(
                a.one(
                    """SELECT COUNT(DISTINCT l.issuer_id) FROM ref.index_membership m
                       JOIN ref.listing l ON l.listing_id = m.listing_id
                       WHERE m.end_date IS NULL AND NOT EXISTS (SELECT 1 FROM ref.classification c
                             WHERE c.issuer_id = l.issuer_id AND c.valid_to IS NULL)"""
                )
            ),
            missing <= 5,
            f"{missing} companies without one" if missing else "",
        ),
    )

    def preferred() -> tuple[object, bool, str]:
        tables = {
            "prices": "market.price_bar",
            "members": "ref.index_membership",
            "fundamentals": "fund.financial_fact",
            "classification": "ref.classification",
        }
        have = {
            what: {
                c
                for (c,) in a.q(
                    f"SELECT DISTINCT v.vendor_code FROM {t} x JOIN ref.vendor v ON v.vendor_id = x.vendor_id"
                )
            }
            for what, t in tables.items()
        }
        wanted = {
            "prices": prefs.prices[:1],
            "members": prefs.members[:1],
            "fundamentals": prefs.fundamentals[:1],
            "classification": prefs.classification[:1],
        }
        missing = [f"{what}: {v}" for what, vs in wanted.items() for v in vs if v not in have[what]]
        return len(missing), not missing, "; ".join(missing) or "the first preferred vendor of each kind has data"

    a.rule("fitness", "every first-preferred vendor supplies its data", preferred)


# ---------------------------------------------------------------- lineage and operations


def _lineage(a: _Audit, sample: int) -> None:
    for table in PROVENANCED:
        a.none(
            "lineage", f"{table}: every row names its raw batch", f"SELECT COUNT(*) FROM {table} WHERE batch_id IS NULL"
        )
    batches = a.q("SELECT batch_id, raw_path, checksum, status FROM ingest.batch")

    def files() -> tuple[object, bool, str]:
        missing = [b for b in batches if b[3] != "superseded" and not (a.raw_dir / b[1]).exists()]
        return f"{len(missing)} missing of {len(batches)}", not missing, ""

    def checksums() -> tuple[object, bool, str]:
        present = [b for b in batches if (a.raw_dir / b[1]).exists()]
        picked = random.Random(1).sample(present, k=min(sample, len(present)))
        bad = [b[0] for b in picked if sha256_file(a.raw_dir / b[1]) != b[2]]
        return len(bad), not bad, f"batches {bad[:5]}" if bad else ""

    def orphans() -> tuple[object, bool, str]:
        registered = {str(Path(b[1])).lower() for b in batches}
        loose = [
            p
            for p in a.raw_dir.rglob("*")
            if p.is_file() and "_staging" not in p.parts and str(p.relative_to(a.raw_dir)).lower() not in registered
        ]
        return len(loose), not loose, ", ".join(str(p.relative_to(a.raw_dir)) for p in loose[:3])

    a.rule("lineage", "every batch's raw file is in the raw zone", files)
    a.rule("lineage", f"raw files unchanged (sha256 of {sample} sampled)", checksums)
    a.rule("lineage", "no raw file without its batch", orphans)

    def unread() -> tuple[object, bool, str]:
        rows = a.q(
            "SELECT v.vendor_code, b.dataset, COUNT(*) FROM ingest.batch b JOIN ref.vendor v ON v.vendor_id = "
            "b.vendor_id "
            "WHERE b.status = 'landed' GROUP BY v.vendor_code, b.dataset"
        )
        return sum(n for *_, n in rows), not rows, ", ".join(f"{v}/{d} {n}" for v, d, n in rows[:6])

    a.rule("lineage", "every landed batch was read", unread)


def _ops(a: _Audit) -> None:
    a.none(
        "ops",
        "no job left running (a crash records its failure)",
        "SELECT COUNT(*) FROM ops.job_run WHERE status = 'running'",
    )
