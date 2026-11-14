"""The LSEG load end to end on a scratch database, LSEG replaced by a fake answering in LSEG's shapes.

Three runs: one stopped by LSEG's daily limit, one that resumes and finishes, one that finds nothing to do; then a
refresh of the last month, and its repeat.
"""

import datetime as dt
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research.db import engine as db_engine
from equity_research.db import migrate
from equity_research.ingest.vendors.refinitiv import fields as f
from equity_research.ingest.vendors.refinitiv.client import LsegClient, ThrottledError
from equity_research.pipelines.load_estimates import STEPS as STEPS_E
from equity_research.pipelines.load_estimates import load_estimates
from equity_research.pipelines.load_lseg import load_lseg
from equity_research.pipelines.refresh import written
from equity_research.settings import get_settings

pytestmark = pytest.mark.integration

SCRATCH = "equity_research_test"
START, END = "2024-01-01", "2024-03-29"
MEMBERS = {".SPX": ["AAA.N", "BBB.OQ"], ".MID": ["CCC.N"], ".SPCY": ["DDD.N"]}
RETIRED = "OLD.N^B24"  # left the S&P 500 on 2024-02-15
RENAMED = {"CCO.N^A24": "CCC.N"}  # the same shares under an older RIC (a ticker change on 2024-01-10)


def _security(ric: str) -> str:
    """A 9-character code per security: a renamed RIC shares its successor's."""
    return RENAMED.get(ric, ric).split(".")[0].ljust(6, "X")[:6] + "100"


def _within(answer: pd.DataFrame, date_column: str, params: dict[str, Any] | None) -> pd.DataFrame:
    """The rows LSEG would send for ``params``' window (a refresh's few days often hold no event at all)."""
    p = params or {}
    days = pd.to_datetime(answer[date_column])
    return answer[(days >= pd.Timestamp(p.get("SDate", START))) & (days <= pd.Timestamp(p.get("EDate", END)))]


class FakeLseg:
    """Answers like LSEG for a handful of RICs; ``throttle_after`` requests, it says "Too many requests"."""

    def __init__(self, throttle_after: int | None = None) -> None:
        self.requests = 0
        self.lseg_requests = 0
        self.skipped: list[str] = []
        self.throttle_after = throttle_after

    def allowance(self) -> dict[str, int] | None:
        return None

    def _count(self, cost: int = 1) -> None:
        if self.throttle_after is not None and self.requests >= self.throttle_after:
            raise ThrottledError("LSEG limit reached (fake)")
        self.requests += 1
        self.lseg_requests += cost  # a history: one per RIC, as LSEG counts

    def get_data(  # noqa: PLR0911, PLR0912 - one answer per kind of request
        self, universe: Sequence[str], fields_: Sequence[str], params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        self._count()
        rics = list(universe)
        if fields_ == f.SNAPSHOT_FIELDS:
            chain = rics[0][2:]
            return pd.DataFrame(
                {"Instrument": MEMBERS[chain], "Exchange Ticker": [r.split(".")[0] for r in MEMBERS[chain]]}
            )
        if fields_ == f.EVENT_FIELDS:
            if rics == [".MID"]:  # the old RIC left and the new one joined: one company, renamed
                return pd.DataFrame(
                    {
                        "Instrument": [".MID", ".MID"],
                        "Date": ["2024-01-10", "2024-01-10"],
                        "Constituent RIC": ["CCO.N^A24", "CCC.N"],
                        "Constituent Name": ["C", "C"],
                        "Change": ["Leaver", "Joiner"],
                    }
                )
            if rics != [".SPX"]:
                return pd.DataFrame(columns=["Instrument", "Date", "Constituent RIC", "Constituent Name", "Change"])
            return pd.DataFrame(
                {
                    "Instrument": [".SPX"],
                    "Date": ["2024-02-15"],
                    "Constituent RIC": [RETIRED],
                    "Constituent Name": ["Old"],
                    "Change": ["Leaver"],
                }
            )
        if fields_ == f.INSTRUMENT_FIELDS:
            return pd.DataFrame(
                {
                    "Instrument": rics,
                    "Company Common Name": [f"{r} Inc" if not r.startswith(".") else "" for r in rics],
                    "ISIN": [f"US{_security(r)}0" if not r.startswith(".") else "" for r in rics],
                    "SEDOL": "",
                    "CUSIP": [_security(r) if not r.startswith(".") else "" for r in rics],
                    "Exchange Country Code": "US",
                    "Exchange Ticker": [r.split(".")[0] for r in rics],
                }
            )
        if fields_ == f.DOMICILE_FIELDS:  # LSEG answers under its own column name, whatever it is
            return pd.DataFrame(
                {
                    "Instrument": rics,
                    "Country ISO Code of Headquarters": ["IE" if r == "BBB.OQ" else "US" for r in rics],
                }
            )
        if fields_ == f.CLASSIFICATION_FIELDS:
            return pd.DataFrame(
                {
                    "Instrument": rics,
                    "GICS Sector Name": "Energy",
                    "GICS Industry Group Name": "Energy",
                    "GICS Industry Name": "Oil",
                    "TRBC Economic Sector Name": "Energy",
                }
            )
        if fields_ == f.DIVIDEND_FIELDS:
            dividends = pd.DataFrame(
                {
                    "Instrument": ["BBB.OQ", "BBB.OQ"],
                    "Dividend Ex Date": ["2023-11-09", "2024-02-09"],  # the first in the prices' year before START
                    "Gross Dividend Amount": [0.5, 0.5],
                    "Dividend Type": ["Regular", "Regular"],
                }
            )
            return _within(dividends, "Dividend Ex Date", params)
        if fields_ == f.CAPITAL_CHANGE_FIELDS:
            changes = pd.DataFrame(
                {"Instrument": ["AAA.N"], "Capital Change Effective Date": ["2024-02-01"], "Adjustment Factor": [0.5]}
            )
            return _within(changes, "Capital Change Effective Date", params)
        if fields_ == f.DAILY_FIELDS:  # a refresh's prices: one request for the whole call
            return self._daily(rics, str((params or {})["SDate"]), str((params or {})["EDate"]))
        if fields_ == f.SHARES_FIELDS:
            return pd.DataFrame({"Instrument": rics, "Issue Default Shares Outstanding": 1e9, "Date": "2024-01-31"})
        if fields_ == f.FREE_FLOAT_FIELDS:
            return pd.DataFrame({"Instrument": rics, "Free Float (Percent)": 90.0, "Date": "2024-01-15"})
        if fields_ == f.ANNOUNCEMENT_FIELDS:
            return pd.DataFrame(
                {"Instrument": rics, "Period End Date": "2023-12-31", "Original Announcement Date Time": "2024-02-10"}
            )
        if fields_ == f.STATEMENT_FIELDS:
            return pd.DataFrame(
                {
                    "Instrument": rics,
                    "Period End Date": "2023-12-31",
                    **{c: 100.0 for c, _, _ in f.STATEMENT_ITEMS.values()},
                }
            )
        if any(list(fields_) == f.annual_fields(x) for x in f.ANNUAL_ITEMS):  # one yearly item per request
            return pd.DataFrame({"Instrument": rics, "Period End Date": "2023-12-31", "Value": 7.0})
        raise AssertionError(f"unexpected request {fields_}")

    def _daily(self, rics: list[str], start: str, end: str) -> pd.DataFrame:
        rows = []
        for r in rics:
            if r == RETIRED:  # LSEG repeats a retired share's last bar, dated on its last day
                rows += [{"Instrument": r, "Date": "2024-02-14", "Price Close": 1.0, "Volume": 0}] * 3
                continue
            for d in pd.bdate_range(max(start, START), min(end, END)):
                rows.append({"Instrument": r, "Date": d, "Price Close": self._close(r, d), "Volume": 1000})
        return pd.DataFrame(rows, columns=["Instrument", "Date", "Price Close", "Volume"])

    @staticmethod
    def _close(ric: str, day: pd.Timestamp) -> float:
        base = 100.0 + sum(map(ord, ric)) % 50
        return base / 2 if ric == "AAA.N" and day >= pd.Timestamp("2024-02-01") else base  # a 2-for-1 split

    def get_history(self, universe: Sequence[str], fields_: Sequence[str], start: str, end: str) -> pd.DataFrame:
        self._count(len(universe))
        rows = []
        for r in universe:
            last = "2024-02-14" if r == RETIRED else END  # the retired share stopped trading
            for d in pd.bdate_range(max(start, START), min(end, last)):
                px = self._close(r, d)
                rows.append(
                    {
                        "Instrument": r,
                        "Date": d,
                        "TRDPRC_1": px,
                        "OPEN_PRC": px,
                        "HIGH_1": px,
                        "LOW_1": px,
                        "ACVOL_UNS": 1000,
                    }
                )
        return pd.DataFrame(rows, columns=["Instrument", "Date", *f.HISTORY_FIELDS])


@pytest.fixture(scope="module")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    """A new, migrated database next to the configured one (never on Azure), dropped afterwards."""
    settings = get_settings()
    if "database.windows.net" in settings.database.server:
        pytest.skip("the scratch database is created on a local SQL Server only")
    mp = pytest.MonkeyPatch()
    mp.setenv("EQR_DB_SERVER", settings.database.server)
    mp.setenv("EQR_DB_NAME", SCRATCH)
    mp.setenv("EQR_DATA_DIR", str(tmp_path_factory.mktemp("lseg_data")))
    get_settings.cache_clear()
    db_engine.get_engine.cache_clear()
    try:
        db_engine.create_database(get_settings().database)
        migrate.upgrade_to_latest()
        yield db_engine.get_engine()
    finally:
        db_engine.get_engine().dispose()
        master = sa.create_engine(
            db_engine.connection_url(
                get_settings().database.__class__(**{**get_settings().database.__dict__, "name": "master"})
            ),
            isolation_level="AUTOCOMMIT",
        )
        with master.connect() as c:
            c.execute(sa.text(f"ALTER DATABASE [{SCRATCH}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE"))
            c.execute(sa.text(f"DROP DATABASE [{SCRATCH}]"))
        master.dispose()
        mp.undo()
        get_settings.cache_clear()
        db_engine.get_engine.cache_clear()


def _count(engine: Engine, sql: str) -> int:
    with engine.connect() as c:
        return int(c.execute(sa.text(sql)).scalar_one())


def test_a_throttled_load_resumes_and_a_repeat_does_nothing(scratch: Engine, tmp_path: Path) -> None:
    stopped = load_lseg(scratch, start=START, end=END, client=cast(LsegClient, FakeLseg(throttle_after=5)))
    assert not stopped["complete"] and "run the load again" in stopped["stopped"]

    resumed_client = FakeLseg()
    done = load_lseg(scratch, start=START, end=END, client=cast(LsegClient, resumed_client))
    assert done["complete"], done
    assert done["universe"]["reused"] >= 5  # the five answered before the stop were read back, not asked again

    assert _count(scratch, "SELECT COUNT(*) FROM ref.listing") == 9  # 5 share lines (one retired) + 4 index levels
    renamed = (
        "SELECT COUNT(DISTINCT entity_id) FROM ref.identifier WHERE id_type = 'RIC' AND value IN ('CCC.N', 'CCO.N^A24')"
    )
    assert _count(scratch, renamed) == 1  # one listing carries both RICs
    assert _count(scratch, "SELECT COUNT(*) FROM ref.universe WHERE benchmark_listing_id IS NOT NULL") == 4
    # the renamed line left under its old RIC and joined under its new one the same day: one unbroken spell
    assert _count(scratch, "SELECT COUNT(*) FROM ref.index_membership") == 5
    assert _count(scratch, "SELECT COUNT(*) FROM ref.index_membership WHERE batch_id IS NULL") == 0
    of_retired = f"JOIN ref.identifier i ON i.entity_type = 'listing' AND i.value = '{RETIRED}'"
    with scratch.connect() as c:

        def one(sql: str) -> Any:
            return c.execute(sa.text(sql)).scalar_one()

        assert (
            one(
                f"SELECT CONVERT(VARCHAR(10), m.end_date, 23) FROM ref.index_membership m {of_retired} "
                "AND i.entity_id = m.listing_id"
            )
            == "2024-02-15"
        )
        split = c.execute(
            sa.text(
                "SELECT ratio, CONVERT(VARCHAR(10), ex_date, 23) FROM market.corporate_action "
                "WHERE action_type = 'split'"
            )
        ).one()
        assert (split[0], split[1]) == (2.0, "2024-02-01")
        # actions are asked from the first price, not from START: momentum and returns there need them too
        assert one("SELECT MIN(ex_date) FROM market.corporate_action").isoformat() == "2023-11-09"
        assert (
            one(f"SELECT COUNT(*) FROM market.price_bar p {of_retired} AND i.entity_id = p.listing_id") > 0
        )  # its history
        delisted = one(
            f"SELECT CONVERT(VARCHAR(10), l.delisted_on, 23) FROM ref.listing l {of_retired} "
            "AND i.entity_id = l.listing_id"
        )
        assert delisted == "2024-02-14"  # its last trading day
        assert one("SELECT COUNT(*) FROM ref.classification WHERE sector = 'Energy' AND valid_to IS NULL") == 5
        assert one("SELECT COUNT(*) FROM ref.issuer WHERE domicile_country = 'IE'") == 1  # registered abroad
    assert _count(scratch, "SELECT COUNT(*) FROM market.share_count WHERE kind = 'float'") == 5
    assert _count(scratch, "SELECT COUNT(*) FROM fund.financial_fact WHERE item = 'revenue'") == 5
    # the yearly items, one field per request, dated by the year's announcement from the quarterly statements
    assert (
        _count(scratch, "SELECT COUNT(*) FROM fund.financial_fact WHERE item = 'depreciation' AND period_type = 'FY'")
        == 5
    )
    assert (
        _count(scratch, "SELECT COUNT(*) FROM fund.financial_fact WHERE item = 'loans_net' AND period_type = 'I'") == 5
    )
    assert _count(scratch, "SELECT COUNT(*) FROM market.price_bar WHERE vendor_id = 1") > 300

    repeat_client = FakeLseg()
    again = load_lseg(scratch, start=START, end=END, client=cast(LsegClient, repeat_client))
    assert again["complete"] and repeat_client.requests == 0  # everything was landed already
    changes = written(again)  # as notebook 02 counts them: a repeat writes nothing, memberships included
    assert changes[["inserted", "updated"]].to_numpy().sum() == 0, changes.to_string()


def test_the_loaded_database_copies_exactly_and_only_once(scratch: Engine, tmp_path: Path) -> None:
    from dataclasses import replace  # noqa: PLC0415

    from equity_research.pipelines.copy_database import CopyRefusedError, copy_database  # noqa: PLC0415

    target_db = replace(get_settings().database, name=f"{SCRATCH}_copy")
    db_engine.create_database(target_db)
    target = db_engine.make_engine(target_db)
    try:
        migrate.upgrade_to_latest(target)
        report = copy_database(scratch, target)
        assert all(t.ok for t in report), [t for t in report if not t.ok]
        assert sum(t.target_rows for t in report) > 300
        with pytest.raises(CopyRefusedError, match="already holds data"):
            copy_database(scratch, target)
    finally:
        target.dispose()
        master = sa.create_engine(
            db_engine.connection_url(replace(target_db, name="master")), isolation_level="AUTOCOMMIT"
        )
        with master.connect() as c:
            c.execute(sa.text(f"ALTER DATABASE [{target_db.name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE"))
            c.execute(sa.text(f"DROP DATABASE [{target_db.name}]"))
        master.dispose()


def test_a_refresh_asks_prices_by_the_call_and_a_window_without_events_is_fine(scratch: Engine) -> None:
    since = dt.date(2024, 3, 1)  # after the split and the dividend: LSEG has no event to send
    client = FakeLseg()
    refreshed = load_lseg(scratch, start=START, end=END, since=since, client=cast(LsegClient, client))
    assert refreshed["complete"], refreshed
    assert refreshed["actions"]["corporate_actions"]["inserted"] == 0
    # four share lines alive in March, one get_data call; only the index levels are histories (one per RIC)
    assert refreshed["prices"]["requests"] == 2
    assert client.lseg_requests == client.requests + 3  # 4 levels in one history: 3 more than one request

    repeat = FakeLseg()
    again = load_lseg(scratch, start=START, end=END, since=since, client=cast(LsegClient, repeat))
    assert again["complete"] and repeat.requests == 0  # a rerun the same day reads every answer back


class FakeEstimates(FakeLseg):
    """LSEG's estimate answers: every per-share figure quoted in today's shares, as LSEG restates them."""

    def get_data(
        self, universe: Sequence[str], fields_: Sequence[str], params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        p = params or {}
        stems = {m: f.consensus_fields(m) for m in f.ESTIMATE_STEMS}
        name = {
            "eps": "Earnings Per Share",
            "revenue": "Revenue",
            "ebit": "EBIT",
            "dps": "Dividend Per Share",
            "cfps": "Cash Flow Per Share",
            "roe": "Return On Equity",
        }
        for measure, fields_of in stems.items():
            if list(fields_) == fields_of:
                self._count()
                ends = pd.date_range(p["SDate"], p["EDate"], freq="ME")
                n = name[measure]
                return pd.DataFrame(
                    [
                        {
                            "Instrument": r,
                            f"{n} - Mean": 2.0,
                            "Calc Date": d,
                            "Period End Date": "2024-12-31",
                            f"{n} - Median": 2.0,
                            f"{n} - Standard Deviation": 0.1,
                            f"{n} - Number of Included Estimates": 5,
                        }
                        for r in universe
                        for d in ends
                    ]
                )
            if list(fields_) == f.actual_fields(measure):
                self._count()
                return pd.DataFrame(
                    {
                        "Instrument": list(universe),
                        f"{name[measure]} - Actual": 4.0,
                        "Period End Date": "2023-12-31",
                        "Report Date": "2024-02-10 07:00:00",
                    }
                )
        if list(fields_) == f.REVISION_FIELDS:
            # asked as a history, LSEG never answers these: the load must ask one date at a time
            assert "EDate" not in p and "Frq" not in p, p
            self._count()
            return pd.DataFrame(
                {
                    "Instrument": list(universe),
                    "Number of Analysts Revising Up": 1,
                    "Number of Analysts Revising Down": 2,
                }
            )
        return super().get_data(universe, fields_, params)


def test_estimates_load_in_the_shares_of_their_date_and_resume(scratch: Engine) -> None:
    stopped = load_estimates(scratch, start=START, end=END, client=cast(LsegClient, FakeEstimates(throttle_after=7)))
    assert not stopped["complete"] and "run the load again" in stopped["stopped"]
    client = FakeEstimates()
    done = load_estimates(scratch, start=START, end=END, client=cast(LsegClient, client))
    assert done["complete"], done
    assert done["consensus"]["reused"] >= 7  # what was answered before the stop is read back
    periods = sum(len(p) for p in f.ESTIMATE_PERIODS.values())
    # five share lines (the renamed one asked under both RICs counts once) x two month ends (January and February;
    # the window ends on 29 March) x every measure and period
    assert _count(scratch, "SELECT COUNT(*) FROM fund.estimate") == 5 * 2 * periods
    aaa = "(SELECT entity_id FROM ref.identifier WHERE id_type = 'RIC' AND value = 'AAA.N')"
    eps = (
        "SELECT mean FROM fund.estimate WHERE measure = '{m}' AND period = 'FY1' "
        f"AND listing_id = {aaa} AND as_of_date = '{{d}}'"
    )
    with scratch.connect() as c:

        def one(sql: str) -> Any:
            return c.execute(sa.text(sql)).scalar_one()

        assert one(eps.format(m="eps", d="2024-01-31")) == 4.0  # before the 2-for-1 split: twice today's figure
        assert one(eps.format(m="eps", d="2024-02-29")) == 2.0
        assert one(eps.format(m="revenue", d="2024-01-31")) == 2.0  # totals are not per share
        actual = f"SELECT value FROM fund.estimate_actual WHERE measure = 'eps' AND listing_id = {aaa}"
        assert one(actual) == 8.0  # FY2023 in the shares of its year end
        old = "(SELECT entity_id FROM ref.identifier WHERE id_type = 'RIC' AND value = 'OLD.N^B24')"
        revised = f"SELECT COUNT(*) FROM fund.estimate_revision WHERE listing_id = {old}"
        assert one(revised) == 2  # a member at the end of January only (FY1 and FY2), not after it left
        for table in ("estimate", "estimate_revision", "estimate_actual"):
            assert one(f"SELECT COUNT(*) FROM fund.{table} WHERE batch_id IS NULL") == 0
    repeat = FakeEstimates()
    again = load_estimates(scratch, start=START, end=END, client=cast(LsegClient, repeat))
    assert again["complete"] and repeat.requests == 0
    written = [again[step][rows] for step, rows in zip(STEPS_E, ("estimates", "actuals", "revisions"), strict=True)]
    assert all(w["inserted"] == 0 and w["updated"] == 0 for w in written)  # read back, and nothing changed


def test_a_stopped_step_still_reports_what_it_did(scratch: Engine) -> None:
    from equity_research.db.schema import metadata  # noqa: PLC0415

    with scratch.begin() as conn:  # an empty database again: this test loads from nothing
        for t in reversed(metadata.sorted_tables):
            if t.name != "vendor":
                conn.execute(t.delete())
    stopped = load_lseg(scratch, start=START, end=END, client=cast(LsegClient, FakeLseg(throttle_after=30)))
    assert not stopped["complete"]
    steps = [s for s in ("universe", "prices", "actions", "shares", "statements") if s in stopped]
    # the step the limit stopped is in the report (it used to vanish), after every step that finished
    assert steps[0] == "universe" and len(steps) > 1
    assert stopped["universe"]["memberships"]["inserted"] > 0


# ---------------------------------------------------------------- London
UK_MEMBERS = {".FTSE": ["AAA.L", "TRST.L"], ".FTMC": ["MMM.L"], ".FTSC": []}  # AAA: also a New York ticker here
UK_FLOATED = {"MMM.L": "2024-02-01"}  # a member today with no joiner event that first traded after the start
UK_CONSOLIDATED = {"AAA.L": "2024-02-15"}  # a 1-for-10 LSEG does not report: the price x10, the shares /10
UK_LEVELS = {".FTSE", ".FTMC", ".FTSC", ".FTASXGBPT", ".FTASXXMGBPT"}


class FakeLondon(FakeLseg):
    """LSEG's London answers: pence, official closes, PermIDs, half-years. Asking for a US RIC fails the test."""

    def _london(self, universe: Sequence[str]) -> list[str]:
        rics = list(universe)
        foreign = [
            r
            for r in rics
            if not (r.split("^")[0].endswith(".L") or r in UK_LEVELS or r[2:] in UK_MEMBERS or r.endswith("="))
        ]
        assert not foreign, f"a London load asked for {foreign}"
        return rics

    def get_data(  # noqa: PLR0911 - one answer per kind of request
        self, universe: Sequence[str], fields_: Sequence[str], params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        rics = self._london(universe)
        if fields_ == f.SNAPSHOT_FIELDS:
            self._count()
            members = UK_MEMBERS[rics[0][2:]]
            return pd.DataFrame({"Instrument": members, "Exchange Ticker": [r.split(".")[0] for r in members]})
        if fields_ == f.EVENT_FIELDS:
            self._count()
            return pd.DataFrame(columns=["Instrument", "Date", "Constituent RIC", "Constituent Name", "Change"])
        if fields_ == f.INSTRUMENT_FIELDS:
            self._count()
            share = [not r.startswith(".") for r in rics]
            return pd.DataFrame(
                {
                    "Instrument": rics,
                    "Company Common Name": [f"{r} plc" if s else "" for r, s in zip(rics, share, strict=True)],
                    "ISIN": [f"GB{_security(r)}0" if s else "" for r, s in zip(rics, share, strict=True)],
                    "SEDOL": "",
                    "CUSIP": "",
                    "Exchange Country Code": ["GB" if s else "" for s in share],
                    "Exchange Ticker": [r.split(".")[0] if s else "" for r, s in zip(rics, share, strict=True)],
                }
            )
        if fields_ == f.UK_COMPANY_FIELDS:
            self._count()
            return pd.DataFrame(
                {
                    "Instrument": rics,
                    "Organization PermID": [4_000_000 + sum(map(ord, r)) for r in rics],  # one per company
                    "Country ISO Code of Incorporation": ["JE" if r == "MMM.L" else "GB" for r in rics],
                    "TRBC Industry Name": [f.INVESTMENT_TRUST_INDUSTRY if r == "TRST.L" else "Banks" for r in rics],
                }
            )
        if fields_ == f.DIVIDEND_CURRENCY_FIELDS:
            self._count()
            dividends = pd.DataFrame(
                {
                    "Instrument": ["AAA.L"],
                    "Dividend Ex Date": ["2024-02-08"],
                    "Gross Dividend Amount": [0.25],
                    "Dividend Type": ["Interim"],
                    "Currency": ["GBP"],
                }
            )
            return _within(dividends[dividends["Instrument"].isin(rics)], "Dividend Ex Date", params)
        if fields_ == f.UK_STATEMENT_FIELDS:
            self._count()
            yearly = (params or {}).get("Frq") == "FY"
            ends = ["2023-12-31"] if yearly else ["2023-06-30", "2023-12-31"]
            rows = [
                {
                    "Instrument": r,
                    "Period End Date": e,
                    **{col: None for col, _, _ in f.UK_STATEMENT_ITEMS.values()},
                    "Revenue from Business Activities - Total": 200.0 if yearly else 100.0,
                    "Total Assets": 1000.0,
                    "Original Announcement Date Time": "2024-02-20" if e.endswith("12-31") else "2023-08-01",
                }
                for r in rics
                for e in ends
            ]
            return pd.DataFrame(rows)
        if fields_ == f.UK_CAPITAL_CHANGE_FIELDS:
            self._count()
            return pd.DataFrame(columns=["Instrument", "Capital Change Ex Date", "Adjustment Factor"])
        if fields_ == f.QUOTE_CURRENCY_FIELDS:
            self._count()
            return pd.DataFrame({"Instrument": rics, "CF_CURR": "GBp"})
        if fields_ == f.SHARES_FIELDS:  # month ends; the consolidated line's count falls tenfold
            self._count()
            ends = ["2024-01-31", "2024-02-29"]
            return pd.DataFrame(
                [
                    {
                        "Instrument": r,
                        "Issue Default Shares Outstanding": 1e8 if r in UK_CONSOLIDATED and e > "2024-02" else 1e9,
                        "Date": e,
                    }
                    for r in rics
                    for e in ends
                ]
            )
        if fields_ == f.PRICE_CURRENCY_FIELDS:  # a refresh's prices, each with today's quote currency
            daily = self._daily(rics, str((params or {})["SDate"]), str((params or {})["EDate"]))
            daily["Price Close"] = 250.0
            return daily.assign(Currency="GBp")
        return super().get_data(rics, fields_, params)

    def get_history(self, universe: Sequence[str], fields_: Sequence[str], start: str, end: str) -> pd.DataFrame:
        rics = self._london(universe)
        self._count(len(rics))
        rows = [
            {
                "Instrument": r,
                "Date": d,
                **dict.fromkeys(fields_, 25.0 if d < pd.Timestamp(UK_CONSOLIDATED.get(r, START)) else 250.0),
            }
            for r in rics
            for d in pd.bdate_range(max(start, UK_FLOATED.get(r, START)), min(end, END))
        ]
        return pd.DataFrame(rows, columns=["Instrument", "Date", *fields_])


class FakeLondonEstimates(FakeEstimates, FakeLondon):
    """London's estimate requests: asked in pounds, for London's RICs only."""

    def get_data(
        self, universe: Sequence[str], fields_: Sequence[str], params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        if list(fields_) in [f.consensus_fields(m) for m in f.ESTIMATE_STEMS]:
            assert (params or {}).get("Curn") == "GBP", params  # else Vodafone's come in euros, Shell's in dollars
            self._london(universe)
        return super().get_data(universe, fields_, params)


def test_london_loads_beside_new_york_without_touching_it(scratch: Engine) -> None:
    before = load_lseg(scratch, start=START, end=END, client=cast(LsegClient, FakeLseg()))  # New York, all landed
    assert before["complete"], before
    done = load_lseg(scratch, start=START, end=END, client=cast(LsegClient, FakeLondon()), market="UK")
    assert done["complete"], done
    # all in pence: the consolidation's jump is read (no split explains it yet) and is not a change of currency
    assert done["quotes"]["quotes"] == {"listings_read": 1, "listings_with_other_currencies": 0}
    with scratch.connect() as c:

        def rows(sql: str) -> list[Any]:
            return [tuple(r) for r in c.execute(sa.text(sql)).all()]

        london = rows(
            "SELECT i.value, l.instrument_type, l.currency, s.incorporation_country FROM ref.listing l "
            "JOIN ref.identifier i ON i.entity_type = 'listing' AND i.entity_id = l.listing_id AND i.id_type = 'RIC' "
            "JOIN ref.issuer s ON s.issuer_id = l.issuer_id WHERE l.exchange_mic = 'XLON' ORDER BY i.value"
        )
        assert london == [
            ("AAA.L", "equity", "GBX", "GB"),
            ("MMM.L", "equity", "GBX", "JE"),
            ("TRST.L", "fund", "GBX", "GB"),  # an investment trust
        ]
        # one ticker, two markets: both recorded
        tickers = rows("SELECT market FROM ref.identifier WHERE id_type = 'TICKER' AND value = 'AAA' ORDER BY market")
        assert tickers == [("GB",), ("US",)]
        assert rows("SELECT COUNT(*) FROM ref.identifier WHERE entity_type = 'issuer' AND id_type = 'PERMID'") == [(3,)]
        currencies = rows(
            "SELECT DISTINCT l.instrument_type, p.currency FROM market.price_bar p "
            "JOIN ref.listing l ON l.listing_id = p.listing_id "
            "WHERE (l.exchange_mic = 'XLON' OR l.currency = 'GBP') AND l.instrument_type <> 'fx' "
            "ORDER BY 1, 2"
        )
        assert currencies == [("equity", "GBX"), ("fund", "GBX"), ("index", "GBP")]  # shares in pence, levels in pounds
        assert rows("SELECT currency, amount FROM market.corporate_action WHERE currency = 'GBP'") == [("GBP", 0.25)]
        revenue = rows(
            "SELECT period_type, value, unit FROM fund.financial_fact f "
            "JOIN ref.listing l ON l.issuer_id = f.issuer_id "
            "WHERE l.exchange_mic = 'XLON' AND f.item = 'revenue' AND l.instrument_type = 'equity' "
            "GROUP BY period_type, value, unit ORDER BY period_type"
        )
        assert revenue == [("FY", 200.0, "GBP"), ("YTD", 100.0, "GBP")]  # the second half is in the year
        # the pound in dollars, day by day: what research turns dividends declared abroad into pounds with
        fx = rows(
            "SELECT l.currency, MIN(p.currency), COUNT(*) FROM ref.listing l JOIN market.price_bar p "
            "ON p.listing_id = l.listing_id WHERE l.instrument_type = 'fx' GROUP BY l.currency ORDER BY l.currency"
        )
        days = len(pd.bdate_range(START, END))
        assert fx == [("EUR", "USD", days), ("GBP", "USD", days)]  # euros: a currency London lines are quoted in
        # a float with no joiner event is a member from its first trade, not from the start of the window
        spells = rows(
            "SELECT m.start_date FROM ref.index_membership m JOIN ref.identifier i ON i.entity_type = 'listing' "
            "AND i.entity_id = m.listing_id AND i.id_type = 'RIC' AND i.value = 'MMM.L'"
        )
        assert spells == [(dt.date(2024, 2, 1),)]
        # a consolidation LSEG does not report, confirmed by the share count once the shares step has run
        splits = rows(
            "SELECT a.ex_date, a.ratio FROM market.corporate_action a "
            "JOIN ref.identifier i ON i.entity_type = 'listing' "
            "AND i.entity_id = a.listing_id AND i.id_type = 'RIC' AND i.value = 'AAA.L' WHERE a.action_type = 'split'"
        )
        assert splits == [(dt.date(2024, 2, 15), 0.1)]
    assert done["universe_after_prices"]["requests"] == 0  # replayed from the raw zone
    assert done["actions_after_shares"]["requests"] == 0
    assert done["actions"]["splits_confirmed_by_share_count"] == 0  # no counts yet on the first pass
    assert done["actions_after_shares"]["splits_confirmed_by_share_count"] == 1

    uk_estimates = load_estimates(
        scratch, start=START, end=END, client=cast(LsegClient, FakeLondonEstimates()), market="UK", steps=["consensus"]
    )
    assert uk_estimates["complete"], uk_estimates

    new_york = FakeLseg()
    again = load_lseg(scratch, start=START, end=END, client=cast(LsegClient, new_york))
    assert again["complete"] and new_york.requests == 0  # every US request is described as before, and read back

    # London's refresh, then its repeat the same day: nothing asked again, nothing written (notebook 02's check)
    since = dt.date(2024, 3, 1)
    refreshed = load_lseg(
        scratch, start=START, end=END, since=since, client=cast(LsegClient, FakeLondon()), market="UK"
    )
    assert refreshed["complete"], refreshed
    repeat = FakeLondon()
    twice = load_lseg(scratch, start=START, end=END, since=since, client=cast(LsegClient, repeat), market="UK")
    assert twice["complete"] and repeat.requests == 0
    assert written(twice)[["inserted", "updated"]].to_numpy().sum() == 0, written(twice).to_string()
