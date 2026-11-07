"""Load analysts' estimates (I/B/E/S, through LSEG): consensus history, reported actuals and revision counts.

    report = load_estimates(engine)                              # 2005 to the last month end: ~3,200 requests
    report = refresh_estimates(engine)                           # the month ends since the newest we hold, if any
    report = load_estimates(engine, market="UK")                 # London's listings, in pounds

Steps (each needs the listings ``load_lseg`` created, and uses its splits):

1. ``consensus`` - for every measure and forecast period of :data:`~.fields.ESTIMATE_PERIODS`, the consensus as it
   stood at each month end (mean, median, standard deviation, number of estimates, the fiscal year's end), for every
   equity listing, in :data:`BUCKETS` groups of RICs: one call per group answers the whole history.
2. ``actuals`` - each fiscal year's reported figure and when it was reported (:data:`~.fields.ACTUAL_MEASURES`).
3. ``revisions`` - how many analysts raised and lowered their EPS estimate for FY1 and FY2, at each month end, for
   that day's index members only. LSEG answers these one date at a time: asked as a history the request never
   answers and lseg-data re-sends it, at a request each, until stopped.

**Per-share figures in the shares of their own date.** LSEG restates per-share estimates for every later split (Apple's
2015 consensus comes back divided by its 2020 split), so a figure fetched again after a split would change. Each is
put back into the shares of its own date with LSEG's splits from ``market.corporate_action``
(:func:`~.normalize.in_shares_of_their_date`), like prices, which are stored unadjusted.

The load ends at the last month end before today (:func:`last_month_end`): consensus is monthly, and a fixed end keeps
every request's description the same all month, so a load resumed on another day reads back what it landed instead of
paying for it again. :func:`refresh_estimates` asks only once a new month end has passed - ~400 requests a month.

Landing and resuming are ``load_lseg``'s: every request is landed under a description of what was asked, and a
request landed before is read back, not asked again - so a load stopped by LSEG's daily limit (or by a request that
ran away, :class:`~.client.RunawayRequestError`) continues where it stopped when run again.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import upsert
from equity_research.ingest.vendors.refinitiv import VENDOR, rics
from equity_research.ingest.vendors.refinitiv import fields as f
from equity_research.ingest.vendors.refinitiv import normalize as norm
from equity_research.ingest.vendors.refinitiv.client import LsegClient, RunawayRequestError, ThrottledError
from equity_research.pipelines.load_lseg import MARKETS, START, LsegLoad, StepReport, buckets, market_rics
from equity_research.raw.store import mark

log = logging.getLogger(__name__)

STEPS = ("consensus", "actuals", "revisions")
#: RIC groups per measure and period: ~165 RICs a call, which LSEG answers with 20 years of months in about a minute
#: for ~4 requests (checked 2026-10-30). Fixed, so a RIC added changes only its own group's request.
BUCKETS = 20
#: RICs per revisions call (one date): 1,000 cost 2 requests (checked 2026-10-30).
REVISION_CHUNK = 1000
CONSENSUS_KEY = ["listing_id", "vendor_id", "measure", "period", "as_of_date"]


def last_month_end(today: dt.date) -> dt.date:
    """The last calendar month end before ``today``: the newest month LSEG's monthly consensus has closed."""
    return (pd.Timestamp(today) - pd.offsets.MonthEnd(1)).date()


def month_ends(start: str, end: str) -> list[str]:
    """Calendar month ends from ``start`` to ``end`` (the days LSEG dates monthly consensus by)."""
    return [d.date().isoformat() for d in pd.date_range(start, end, freq="ME")]


def live_first(rows: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """One row per ``key``: a listing asked under two RICs (renamed) keeps its live RIC's answer."""
    retired = rows["ric"].map(rics.is_retired)
    return (
        rows.assign(_retired=retired)
        .sort_values("_retired", kind="stable")
        .drop_duplicates(list(key))
        .drop(columns="_retired")
    )


class EstimatesLoad(LsegLoad):
    """The steps of the module docstring over one engine and client."""

    def splits(self) -> pd.DataFrame:
        """``[listing_id, ex_date, ratio]``: LSEG's splits - the ones its per-share estimates are restated for."""
        a = schema.corporate_action
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(a.c.listing_id, a.c.ex_date, a.c.ratio).where(
                    a.c.action_type == "split", a.c.vendor_id == vendor_id(VENDOR)
                )
            ).all()
        out = pd.DataFrame(rows, columns=["listing_id", "ex_date", "ratio"])
        return out.assign(ex_date=pd.to_datetime(out["ex_date"]), ratio=out["ratio"].astype(float))

    def fetched_on(self, batch_id: int) -> pd.Timestamp:
        """The day a batch was fetched: the shares LSEG's per-share figures in it are quoted in."""
        b = schema.batch
        with self.engine.connect() as conn:
            when = conn.execute(sa.select(b.c.fetched_at).where(b.c.batch_id == batch_id)).scalar_one()
        return pd.Timestamp(when).normalize()

    @staticmethod
    def _listed(rows: pd.DataFrame, key: Sequence[str], ids: dict[str, int]) -> pd.DataFrame:
        """``rows`` keyed by listing (``ids``: RIC -> listing, read once per step - not per call) and vendor."""
        rows = rows.assign(listing_id=rows["ric"].map(ids), vendor_id=vendor_id(VENDOR)).dropna(subset=["listing_id"])
        return live_first(rows.astype({"listing_id": int}), key).drop(columns="ric")

    def currency(self) -> dict[str, str]:
        """The ``Curn`` parameter of this market's estimate requests (none for the US: its requests stay as landed)."""
        return {} if self.market.estimate_currency is None else {"Curn": self.market.estimate_currency}

    def consensus(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Monthly consensus snapshots of every measure and forecast period."""
        report = report or StepReport()  # the caller's, so work done before a stop is still reported
        window = {"SDate": (since or pd.Timestamp(self.start).date()).isoformat(), "EDate": self.end, **self.currency()}
        splits, ids, equities = self.splits(), self.listing_ids(), self.equities()
        for measure, periods in f.ESTIMATE_PERIODS.items():
            for period in periods:
                params = {**window, **f.MONTHLY_HISTORY, "Period": period}
                for chunk in buckets(equities, BUCKETS):
                    batch_id, raw = self.fetch("estimates", chunk, f.consensus_fields(measure), params, report)
                    rows = self._listed(norm.consensus(raw).assign(measure=measure, period=period), CONSENSUS_KEY, ids)
                    if measure in f.PER_SHARE_MEASURES:
                        rows = norm.in_shares_of_their_date(
                            rows, ["mean", "median", "std_dev"], "as_of_date", splits, self.fetched_on(batch_id)
                        )
                    with self.engine.begin() as conn:
                        report.add(
                            "estimates", upsert(conn, schema.estimate, rows, key=CONSENSUS_KEY, batch_id=batch_id)
                        )
                        mark(conn, batch_id, "normalized")
        return report

    def actuals(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Each fiscal year's reported figure, per measure, and when it was reported."""
        report = report or StepReport()
        # a refresh asks the year before ``since`` too: a fiscal year is reported months after it ends
        begin = (pd.Timestamp(since) - pd.DateOffset(years=1)).date() if since else pd.Timestamp(self.start).date()
        params = {"SDate": begin.isoformat(), "EDate": self.end, **f.YEARLY_ACTUALS, **self.currency()}
        key = ["listing_id", "vendor_id", "measure", "period_end"]
        splits, ids, equities = self.splits(), self.listing_ids(), self.equities()
        for measure in f.ACTUAL_MEASURES:
            for chunk in buckets(equities, BUCKETS):
                batch_id, raw = self.fetch("estimate_actuals", chunk, f.actual_fields(measure), params, report)
                rows = self._listed(norm.actuals(raw).assign(measure=measure), key, ids)
                if measure in f.PER_SHARE_MEASURES:
                    rows = norm.in_shares_of_their_date(
                        rows, ["value"], "period_end", splits, self.fetched_on(batch_id)
                    )
                with self.engine.begin() as conn:
                    report.add("actuals", upsert(conn, schema.estimate_actual, rows, key=key, batch_id=batch_id))
                    mark(conn, batch_id, "normalized")
        return report

    def members_on(self) -> pd.DataFrame:
        """``[listing_id, start_date, end_date]``: every index membership spell (any vendor's)."""
        m = schema.index_membership
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(m.c.listing_id, m.c.start_date, m.c.end_date).distinct()).all()
        out = pd.DataFrame(rows, columns=["listing_id", "start_date", "end_date"])
        return out.assign(start_date=pd.to_datetime(out["start_date"]), end_date=pd.to_datetime(out["end_date"]))

    def revisions(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """EPS revision counts at each month end, asked one date at a time for that day's members."""
        report = report or StepReport()
        spells = self.members_on()
        ids = self.listing_ids()
        ric_of: dict[int, list[str]] = {}
        for ric, listing in ids.items():
            if not rics.is_index(ric):
                ric_of.setdefault(listing, []).append(ric)
        for day in month_ends((since or pd.Timestamp(self.start).date()).isoformat(), self.end):
            d = pd.Timestamp(day)
            inside = spells[(spells["start_date"] <= d) & (spells["end_date"].isna() | (spells["end_date"] > d))]
            members = sorted({r for listing in inside["listing_id"] for r in ric_of.get(int(listing), [])})
            for period in f.REVISION_PERIODS:
                for i in range(0, len(members), REVISION_CHUNK):
                    chunk = members[i : i + REVISION_CHUNK]
                    # one date only (SDate, no EDate / Frq): asked as a history these counts never answer
                    params = {"SDate": day, "Period": period}
                    batch_id, raw = self.fetch("estimate_revisions", chunk, f.REVISION_FIELDS, params, report)
                    key = ["listing_id", "vendor_id", "measure", "period", "as_of_date"]
                    rows = self._listed(norm.revisions(raw, day).assign(measure="eps", period=period), key, ids)
                    with self.engine.begin() as conn:
                        report.add(
                            "revisions", upsert(conn, schema.estimate_revision, rows, key=key, batch_id=batch_id)
                        )
                        mark(conn, batch_id, "normalized")
        return report


def load_estimates(
    engine: Engine,
    *,
    start: str = START,
    end: str | None = None,
    since: dt.date | None = None,
    steps: Sequence[str] = STEPS,
    client: LsegClient | None = None,
    market: str = "US",
) -> dict[str, Any]:
    """Run ``steps`` (see the module docstring); ``complete`` False when LSEG's limit or a runaway request stopped it.

    ``since`` turns the full load into a top-up: consensus and revisions from that day, actuals from a year before.
    ``end`` defaults to :func:`last_month_end`. ``market``: whose listings (a ``load_lseg`` market code).
    """
    end = end or last_month_end(dt.date.today()).isoformat()
    load = EstimatesLoad(engine, client or LsegClient(), start=start, end=end, market=MARKETS[market])
    report: dict[str, Any] = {"complete": False, "stopped": None, "market": market}
    with ops.job("load_estimates" if market == "US" else f"load_estimates_{market.lower()}", engine) as run:
        try:
            for step in steps:
                progress = StepReport()
                try:
                    getattr(load, step)(since, progress)
                finally:
                    report[step] = progress.as_dict()  # also what a stopped step did before the stop
                log.info("estimates %s: %s", step, report[step])
            report["complete"] = True
        except (ThrottledError, RunawayRequestError) as exc:
            report["stopped"] = f"{exc} - run the load again later: it continues where it stopped"
            log.warning(report["stopped"])
        report["requests"] = load.client.requests
        report["lseg_requests"] = load.client.lseg_requests
        allowance = load.client.allowance()
        report["allowance_left"] = None if allowance is None else allowance.get("daily_request_remaining")
        report["skipped_rics"] = load.client.skipped[:50]
        run.finish(rows=load.client.requests, detail=report)
    return report


def newest_estimate(engine: Engine, market: str = "US") -> dt.date | None:
    """The newest consensus date held from LSEG for ``market``'s listings, or None for none."""
    e = schema.estimate
    mine = set(market_rics(engine, MARKETS[market]).values())
    if not mine:
        return None
    with engine.connect() as conn:
        held = conn.execute(
            sa.select(e.c.listing_id, sa.func.max(e.c.as_of_date))
            .where(e.c.vendor_id == vendor_id(VENDOR))
            .group_by(e.c.listing_id)
        ).all()
    days = [d for listing, d in held if listing in mine and d is not None]
    return pd.Timestamp(max(days)).date() if days else None


def refresh_estimates(
    engine: Engine, *, today: dt.date | None = None, client: LsegClient | None = None, market: str = "US"
) -> dict[str, Any]:
    """The month ends since the newest consensus held (everything when there is none); nothing until a new one passed.

    A daily refresh therefore costs nothing on most days and ~400 requests on the first run after a month end.
    """
    end = last_month_end(today or dt.date.today())
    newest = newest_estimate(engine, market)
    if newest is not None and newest >= end:
        return {"complete": True, "stopped": None, "up_to_date": f"consensus held to {newest}; next after {end:%B}"}
    since = None if newest is None else newest + dt.timedelta(days=1)
    return load_estimates(engine, since=since, end=end.isoformat(), client=client, market=market)


PROBE_RICS = ["AAPL.OQ", "BRKb.N", "JPM.N", "TWTR.K^J22"]  # a share, a share class, a bank, a retired one


def probe(client: LsegClient | None = None) -> pd.DataFrame:
    """One small question per dataset and measure, read by the load's own code: ``[dataset, ok, rows, detail]``.

    Run before a long load: a field LSEG renamed shows here in a minute. Nothing is stored; ~20 requests.
    """
    c = client or LsegClient()
    history = {"SDate": "2015-01-01", "EDate": "2015-03-31", **f.MONTHLY_HISTORY}
    tries: list[tuple[str, Any]] = [
        (
            f"consensus {m}",
            lambda m=m: norm.consensus(c.get_data(PROBE_RICS, f.consensus_fields(m), {**history, "Period": "FY1"})),
        )
        for m in f.ESTIMATE_STEMS
    ]
    tries += [
        (
            f"actuals {m}",
            lambda m=m: norm.actuals(
                c.get_data(
                    PROBE_RICS, f.actual_fields(m), {"SDate": "2013-01-01", "EDate": "2015-12-31", **f.YEARLY_ACTUALS}
                )
            ),
        )
        for m in f.ACTUAL_MEASURES
    ]
    tries.append(
        (
            "revisions",
            lambda: norm.revisions(
                c.get_data(PROBE_RICS, f.REVISION_FIELDS, {"SDate": "2015-06-30", "Period": "FY1"}), "2015-06-30"
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    stopped = False
    for dataset, ask in tries:
        if stopped:
            rows.append({"dataset": dataset, "ok": False, "rows": 0, "detail": "not tried: LSEG stopped answering"})
            continue
        try:
            answer = ask()
            rows.append(
                {
                    "dataset": dataset,
                    "ok": len(answer) > 0,
                    "rows": len(answer),
                    "detail": "" if len(answer) else "no rows - check the fields",
                }
            )
        except (ThrottledError, RunawayRequestError) as exc:
            stopped = True
            rows.append({"dataset": dataset, "ok": False, "rows": 0, "detail": str(exc)[:200]})
        except Exception as exc:
            rows.append({"dataset": dataset, "ok": False, "rows": 0, "detail": f"{type(exc).__name__}: {exc}"[:300]})
    return pd.DataFrame(rows)
