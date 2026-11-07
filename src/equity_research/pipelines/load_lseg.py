"""Load everything LSEG supplies, from nothing or as a daily top-up - and resume where LSEG's daily limit stopped it.

    report = load_lseg(engine)                       # the whole history from START (several days within LSEG limits)
    report = load_lseg(engine, since=date(2026, 10, 1))   # a refresh: only what changed since then
    report = load_lseg(engine, market="UK")          # London: the FTSE 100, 250 and SmallCap (:data:`MARKETS`)

Steps, in dependency order (each later step needs the listings of the first):

1. ``universe`` - today's members and every joiner / leaver of the S&P 500, 400 and 600 since ``start``; the
   listings - one per share line: RICs with the same ISIN are one security whose code changed
   (:func:`~equity_research.ingest.vendors.refinitiv.normalize.share_lines`), so they share a listing and its
   history - their companies and codes (ISIN, CUSIP, SEDOL, ticker, every RIC), today's GICS sector; our indices
   and universes (:mod:`equity_research.ingest.reference`); membership.
2. ``prices`` - unadjusted daily closes and volumes, in 11-year windows (LSEG counts one request per RIC per
   window), for every listing from 400 days before it joined an index (momentum needs a year of history) until it
   left; index levels for the whole span. A refresh asks its few days by ``get_data`` instead: one request per
   call of 50 RICs, where histories would cost one per RIC - ~30 requests a day rather than ~1,600. A build then
   replays step 1 from the raw zone (no requests): only now are there first trades to start its spells on.
3. ``actions`` - dividends; splits (confirmed by the price, else by the share count) and spin-offs. A build replays
   it after step 4 (no requests), when there are share counts to confirm a split by.
4. ``shares`` - monthly share counts per share class, and the free float.
5. ``statements`` - quarterly statements and the day each quarter was first announced.
``fx`` (London) - after ``actions``: the daily exchange rates of every currency the market's dividends are declared
   in, as ``fx`` listings (one unit of the currency, in US dollars) - research converts such dividends on their ex-date.
``quotes`` (London) - each bar's currency: some lines change the currency they are quoted in, which LSEG's histories do
   not say (:func:`~.normalize.quote_currencies`).
6. ``annuals`` - yearly statement items LSEG answers only one field at a time (cash flows, depreciation, inventories,
   unusual items, bank items: ``fields.ANNUAL_ITEMS``), dated by the fiscal year's announcement from step 5.

A load covers one market (:class:`Market`): the US (the S&P indices) or London (the FTSE UK indices: official closes
in pence, half-yearly statements as first reported in pounds, no ``annuals`` step - its statements carry those items).
Each market's requests name only its own listings, so loading one never changes the other's request descriptions.

Every request is landed in the raw zone under a description of exactly what was asked (dataset, fields, parameters,
RICs). A request already landed is read back, not asked again - so after :class:`ThrottledError` (LSEG's daily
limit), running the load again continues where it stopped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.db.upsert import UpsertResult, upsert
from equity_research.ingest.classification import record as record_classification
from equity_research.ingest.identity import OPEN_START, add_identifiers, anchor_map, ensure_listings, merge_listings
from equity_research.ingest.membership import clip_to_first_trade, coalesce, disjoint
from equity_research.ingest.reference import INDEXES, level_codes, write_reference
from equity_research.ingest.vendors.refinitiv import VENDOR, rics
from equity_research.ingest.vendors.refinitiv import fields as f
from equity_research.ingest.vendors.refinitiv import normalize as norm
from equity_research.ingest.vendors.refinitiv.client import LsegClient, RequestShapeError, ThrottledError
from equity_research.raw.store import find_batch, land_frame, mark, read_frame, status

log = logging.getLogger(__name__)

START = "2005-01-01"
#: On a build, a step replayed (the same day: from the raw zone, no requests) once a later step has given it what it
#: reads: members are clipped to their listing's first trade, and splits confirmed by the share count.
REPLAYED_AFTER = {"prices": "universe", "shares": "actions"}
HISTORY_DAYS = 400
STEPS = ("universe", "prices", "actions", "shares", "statements", "annuals")
#: Requests per dataset: RICs are spread over a fixed number of buckets by a hash of their code, so a RIC added or
#: removed changes only its own bucket - every other request keeps its description and is read back, not asked
#: again. Sized for LSEG: ~165 RICs per instrument request, ~40 per price year, ~40 per action, ~20 per share or
#: statement request.
BUCKETS = {"instruments": 20, "actions": 80, "shares": 170, "statements": 170, "annuals": 10}
#: Price windows: LSEG returns about 3,000 days per history request and counts one request per RIC, so a window of
#: 11 years (~2,770 trading days) costs a RIC one request where calendar years cost eleven. Windows start from the
#: first year of the build (2003-2013, 2014-2024, 2025-...), so a build resumed on another day asks the same windows.
PRICE_WINDOW_YEARS = 11
#: RICs per price call: a history costs one request per RIC (the chunk only groups what is landed); a refresh's
#: get_data costs one per call, and 50 RICs over a month or so stays a modest answer
PRICE_CHUNK = 50
DONE_TOLERANCE_DAYS = 14
REQUEST_COLUMN = "_request"


@dataclass(frozen=True)
class Market:
    """What one market's load asks LSEG for and how it stores the answers."""

    code: str
    country: str  # our indices' country (``reference``)
    chains: Mapping[str, str]  # our index code -> the chain listing its members
    #: its listings' exchanges; ``None``: every exchange no other market claims (and listings without one)
    exchanges: frozenset[str] | None
    history_fields: tuple[str, ...]
    price_currency: str  # its shares' prices as stored (``GBX``: pence)
    level_currency: str  # its index levels' and accounts' currency
    dividend_fields: tuple[str, ...]
    steps: tuple[str, ...]
    buckets: Mapping[str, int]
    company_fields: bool = False  # PermID, incorporation and TRBC industry asked with the listings
    half_years: bool = False  # statements half-yearly as first reported (else quarterly, LSEG's latest)
    #: fields asked before HISTORY_FIELDS was narrowed, whose landed answers are reused
    history_also_from: tuple[tuple[str, ...], ...] = ()
    #: the currency estimates are asked in (``Curn``); None: as LSEG quotes them (the US's, in dollars)
    estimate_currency: str | None = None
    #: capital changes asked by this date (London: the ex-date), and the trading days after it a price may confirm on
    capital_change_fields: tuple[str, ...] = tuple(f.CAPITAL_CHANGE_FIELDS)
    capital_change_columns: Mapping[str, str] = field(default_factory=lambda: dict(f.CAPITAL_CHANGE_COLUMNS))
    split_days_after: int = 1
    #: currencies its shares have been quoted in besides its own, whose rates the fx step always loads
    quote_currencies: tuple[str, ...] = ()
    #: a refresh's prices (``get_data``); London's with the currency each is quoted in today
    daily_fields: tuple[str, ...] = tuple(f.DAILY_FIELDS)


US = Market(
    "US",
    "US",
    f.MEMBERSHIP_CHAINS,
    None,
    tuple(f.HISTORY_FIELDS),
    "USD",
    "USD",
    tuple(f.DIVIDEND_FIELDS),
    STEPS,
    BUCKETS,
    history_also_from=(tuple(f.HISTORY_FIELDS_FULL),),
)
#: London: ~1,400 RICs since 2005 against the US's ~3,300, so fewer buckets keep requests about as full.
UK = Market(
    "UK",
    "GB",
    f.UK_MEMBERSHIP_CHAINS,
    frozenset({"XLON"}),
    tuple(f.UK_HISTORY_FIELDS),
    norm.PENCE,
    "GBP",
    tuple(f.DIVIDEND_CURRENCY_FIELDS),
    ("universe", "prices", "actions", "fx", "quotes", "shares", "statements"),
    {"instruments": 10, "actions": 35, "shares": 70, "statements": 40},
    company_fields=True,
    half_years=True,
    estimate_currency="GBP",  # else per-share estimates come in the reporting currency: Vodafone euros, Shell dollars
    capital_change_fields=tuple(f.UK_CAPITAL_CHANGE_FIELDS),
    capital_change_columns=f.UK_CAPITAL_CHANGE_COLUMNS,
    split_days_after=3,  # a rights issue's ex-date can lead its price's fall by days (Hammerson, 2020)
    quote_currencies=("GBP", "USD", "EUR"),
    daily_fields=tuple(f.PRICE_CURRENCY_FIELDS),
)
MARKETS = {m.code: m for m in (US, UK)}
#: Names of levels that are benchmarks, not one of our indices.
LEVEL_NAMES = {
    ".SPCOMP": "S&P Composite 1500",
    ".FTASXGBPT": "FTSE All-Share Total Return",
    ".FTASXXMGBPT": "FTSE All-Share ex Investment Trusts Total Return",
}


@dataclass
class StepReport:
    """What one step did."""

    requests: int = 0
    reused: int = 0
    rows: dict[str, UpsertResult] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def add(self, what: str, result: UpsertResult) -> None:
        """Count rows written under ``what``."""
        self.rows[what] = self.rows.get(what, UpsertResult()) + result

    def as_dict(self) -> dict[str, Any]:
        """For the job log and notebooks."""
        return {
            "requests": self.requests,
            "reused": self.reused,
            **{k: v.__dict__ for k, v in self.rows.items()},
            **self.notes,
        }


def price_windows(first: pd.Timestamp, last: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """``[(lo, hi)]``: :data:`PRICE_WINDOW_YEARS`-year windows from ``first``'s year to ``last``, clipped to both."""
    out = []
    for year in range(first.year, last.year + 1, PRICE_WINDOW_YEARS):
        lo = max(first, pd.Timestamp(year, 1, 1))
        hi = min(last, pd.Timestamp(year + PRICE_WINDOW_YEARS - 1, 12, 31))
        out.append((lo, hi))
    return out


def buckets(items: Sequence[str], n: int) -> list[list[str]]:
    """``items`` in ``n`` stable groups: an item's group depends only on the item itself."""
    groups: dict[int, list[str]] = {}
    for item in sorted(set(items)):
        groups.setdefault(int(hashlib.sha1(item.encode()).hexdigest()[:8], 16) % n, []).append(item)
    return [groups[k] for k in sorted(groups)]


def _with_request(answer: pd.DataFrame, key: str) -> pd.DataFrame:
    """The answer with its request in every row, so two requests answered alike stay two batches.

    An empty answer (common: no dividends in a window) becomes one row holding just the request; it reads back as an
    empty frame once the request column is dropped.
    """
    return answer.assign(**{REQUEST_COLUMN: key}) if not answer.empty else pd.DataFrame({REQUEST_COLUMN: [key]})


def request_key(dataset: str, universe: Sequence[str], fields_: Sequence[str], params: dict[str, Any]) -> str:
    """A stable description of one request: what was asked of whom (the RIC list as a hash and a count)."""
    digest = hashlib.sha1("|".join(universe).encode()).hexdigest()[:12]
    spec = json.dumps({"fields": list(fields_), "params": params}, sort_keys=True)
    return f"lseg:{dataset}:{spec}:{len(universe)} RICs {digest}"[:1024]


def market_levels(market: Market) -> list[str]:
    """The RICs of ``market``'s index levels and benchmarks (``reference``)."""
    return [value for id_type, value in level_codes(market.country) if id_type == "RIC"]


def market_rics(engine: Engine, market: Market) -> dict[str, int]:
    """``{RIC: listing_id}`` of ``market``'s listings: on its exchanges, and index levels not another market's.

    Each market's requests name only its own RICs, so loading London never changes a US request's description (a
    resumed US load still reads back what it landed) and a US request never asks for London's shares.
    """
    with engine.connect() as conn:
        every = anchor_map(conn, "listing", "RIC")
        mics = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.exchange_mic)).all())
    foreign = {v for t, v in level_codes() if t == "RIC"} - set(market_levels(market))
    claimed = frozenset().union(*(m.exchanges or frozenset() for m in MARKETS.values()))

    def here(ric: str, listing_id: int) -> bool:
        if ric in foreign or rics.is_fx(ric):  # exchange rates are every market's: the fx step keeps them
            return False
        if rics.is_index(ric):
            return True
        mic = mics.get(listing_id)
        return mic in market.exchanges if market.exchanges is not None else mic not in claimed

    return {r: i for r, i in every.items() if here(r, i)}


class LsegLoad:
    """The steps of the module docstring over one engine and client."""

    def __init__(self, engine: Engine, client: LsegClient, *, start: str, end: str, market: Market = US) -> None:
        self.engine = engine
        self.client = client
        self.start = start
        self.end = end
        self.market = market

    # ------------------------------------------------------------ requests, landed

    def fetch(
        self,
        dataset: str,
        universe: Sequence[str],
        fields_: Sequence[str],
        params: dict[str, Any],
        report: StepReport,
        *,
        as_of: str | None = None,
        history: bool = False,
        also_from: Sequence[Sequence[str]] = (),
    ) -> tuple[int, pd.DataFrame]:
        """``(batch_id, LSEG's answer)``: from the raw zone when this exact request was landed before.

        ``as_of`` dates a request whose answer changes from day to day although its parameters do not (today's
        member list): it is part of the description, never sent to LSEG. ``history`` asks ``get_history`` (params
        ``start`` and ``end``) instead of ``get_data``. ``also_from``: wider field lists whose landed answers hold
        everything asked for (a request made before the fields were narrowed), reused rather than asked again.
        """
        p = params if as_of is None else {**params, "as_of": as_of}
        key = request_key(dataset, universe, fields_, p)
        with self.engine.connect() as conn:
            found = find_batch(conn, vendor=VENDOR, dataset=dataset, source=key)
            for wider in also_from:
                if found is None:
                    found = find_batch(
                        conn, vendor=VENDOR, dataset=dataset, source=request_key(dataset, universe, wider, p)
                    )
        if found is not None:
            report.reused += 1
            return found.batch_id, read_frame(found).drop(columns=REQUEST_COLUMN, errors="ignore")
        skipped_before = len(self.client.skipped)
        if history:
            answer = self.client.get_history(universe, fields_, params["start"], params["end"])
        else:
            answer = self.client.get_data(universe, fields_, params)
        report.requests += 1
        if len(self.client.skipped) > skipped_before and len(universe) == 1:
            # LSEG refused the whole request: nothing to keep, and it is asked again next time
            raise RuntimeError(f"LSEG refused {dataset} for {universe[0]} - see the log for its message")
        with self.engine.begin() as conn:
            batch = land_frame(conn, vendor=VENDOR, dataset=dataset, source=key, frame=_with_request(answer, key))
        return batch.batch_id, answer

    def listing_ids(self) -> dict[str, int]:
        """``{RIC: listing_id}`` of this market's listings (:func:`market_rics`)."""
        return market_rics(self.engine, self.market)

    def levels(self) -> list[str]:
        """The RICs of this market's index levels and benchmarks."""
        return market_levels(self.market)

    def bucket_count(self, dataset: str) -> int:
        """How many requests a dataset's RICs are spread over, for this market."""
        return self.market.buckets.get(dataset, BUCKETS[dataset])

    # ------------------------------------------------------------ 1. universe

    def universe(self, report: StepReport | None = None) -> StepReport:
        """Members, listings, codes, sectors, our indices and universes, memberships."""
        report = report or StepReport()  # the caller's, so work done before a stop is still reported
        today = dt.date.today().isoformat()
        intervals: dict[str, tuple[int, pd.DataFrame, list[str]]] = {}  # index -> (events batch, spells, members)
        used: list[int] = []  # every batch this step reads: marked normalized once its rows are written
        for code, chain in self.market.chains.items():
            snap_id, snapshot = self.fetch("index_members", [f"0#{chain}"], f.SNAPSHOT_FIELDS, {}, report, as_of=today)
            events_id, events = self.fetch(
                "index_events", [chain], f.EVENT_FIELDS, {"SDATE": self.start, "EDATE": today, "IC": "B"}, report
            )
            intervals[code] = (events_id, norm.memberships(snapshot, events, self.start), norm.members_today(snapshot))
            used += [snap_id, events_id]
        members = sorted({r for _, frame, _ in intervals.values() for r in frame["ric"]})
        described = self._describe(sorted(set(members) | set(self.levels())), report)
        attr, classes, batch_of, class_batch_of, batch_id, also_read = described
        used += sorted(set(batch_of.values()) | set(class_batch_of.values()) | set(also_read))
        lines = norm.share_lines(attr)  # every code -> the RIC anchoring its share line
        report.notes["codes_left_out"] = sorted(set(attr["ric"]) - set(lines))
        anchors = attr[attr["ric"].isin(set(lines.values()))]
        frame = norm.listings(anchors, classes)
        frame = frame.assign(batch_id=frame["anchor"].map(batch_of))
        with self.engine.begin() as conn:
            listing_ids, line_of = self._write_listings(conn, frame, lines, batch_id, report)
            self._classify(conn, frame.assign(batch_id=frame["anchor"].map(class_batch_of)), listing_ids, report)
            aliases = pd.DataFrame(
                [(code, "RIC", code) for code, anchor in lines.items() if code != anchor and rics.is_ric(code)],
                columns=["ric", "id_type", "value"],
            )
            codes = pd.concat([norm.listing_identifiers(anchors), aliases], ignore_index=True)
            codes = codes.assign(entity_id=codes["ric"].map(line_of), batch_id=codes["ric"].map(batch_of))
            columns = [
                "entity_type",
                "entity_id",
                "id_type",
                "value",
                "valid_from",
                "valid_to",
                "vendor_id",
                "batch_id",
            ]
            conflicts = add_identifiers(
                conn,
                codes.dropna(subset=["entity_id"]).assign(
                    entity_type="listing", valid_from=OPEN_START, valid_to=None, vendor_id=vendor_id(VENDOR)
                )[columns],
                batch_id=None,
            )
            report.notes["identifier_conflicts"] = len(conflicts)
            index_ids = write_reference(conn)
            p = schema.price_bar
            first_trade = pd.Series(
                dict(
                    conn.execute(sa.select(p.c.listing_id, sa.func.min(p.c.trade_date)).group_by(p.c.listing_id)).all()
                )
            )
            # a market's indices never share a member: one the replay leaves in two is put in one (chains largest first)
            spells = pd.concat(
                [self._spells(index_ids[code], frame_, line_of) for code, (_, frame_, _) in intervals.items()],
                ignore_index=True,
            )
            today_ = {
                (index_ids[code], line_of[r]) for code, (*_, rics_) in intervals.items() for r in rics_ if r in line_of
            }
            priority = {index_ids[code]: rank for rank, code in enumerate(intervals)}
            spells, counts = disjoint(spells, self.start, today_, priority)
            report.notes["memberships_moved_by_events"] = counts["moved"]
            report.notes["membership_days_by_priority"] = counts["by_priority"]
            for code, (events_id, _, _) in intervals.items():
                index_id = index_ids[code]
                own = spells[spells["index_id"] == index_id]
                self._replace_memberships(conn, index_id, own, report, events_id, first_trade)
            for used_id in sorted(set(used)):
                mark(conn, used_id, "normalized")
        return report

    def _describe(
        self, everything: list[str], report: StepReport
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int], int | None, list[int]]:
        """``(attributes, classification, {RIC: instruments batch}, {RIC: classification batch}, a batch, others)``.

        Codes, names, domicile (and, where the market asks, PermID, incorporation and TRBC industry) of every RIC, in
        buckets; ``others`` are the other batches read. A RIC LSEG describes not at all stays as a bare listing, so its
        prices can still load.
        """
        attrs, classes, homes, firms, batch_id = [], [], [], [], None
        batch_of: dict[str, int] = {}
        class_batch_of: dict[str, int] = {}
        also_read: list[int] = []
        for chunk in buckets(everything, self.bucket_count("instruments")):
            batch_id, raw = self.fetch("instruments", chunk, f.INSTRUMENT_FIELDS, {}, report)
            attrs.append(norm.instruments(raw, level_currency=self.market.level_currency))
            batch_of |= dict.fromkeys(chunk, batch_id)
            class_id, raw = self.fetch("classification", chunk, f.CLASSIFICATION_FIELDS, {}, report)
            classes.append(norm.classification(raw))
            class_batch_of |= dict.fromkeys(chunk, class_id)
            home_id, home = self._domicile(chunk, report)
            homes.append(home)
            also_read += [home_id] if home_id is not None else []
            if self.market.company_fields:
                firm_id, firm = self.fetch("companies", chunk, f.UK_COMPANY_FIELDS, {}, report)
                firms.append(norm.companies(firm))
                also_read.append(firm_id)
        attr = pd.concat(attrs, ignore_index=True).drop_duplicates("ric")
        attr = attr.merge(pd.concat(homes, ignore_index=True).drop_duplicates("ric"), on="ric", how="left")
        if firms:
            attr = attr.merge(pd.concat(firms, ignore_index=True).drop_duplicates("ric"), on="ric", how="left")
        missing = sorted(set(everything) - set(attr["ric"]))
        if missing:
            attr = pd.concat([attr, pd.DataFrame({"ric": missing})], ignore_index=True)
        names = {v: i.name for i in INDEXES for t, v in i.levels if t == "RIC"} | LEVEL_NAMES
        attr["name"] = attr["name"].where(attr["name"].notna(), attr["ric"].map(names))
        return attr, pd.concat(classes, ignore_index=True), batch_of, class_batch_of, batch_id, also_read

    def _write_listings(
        self, conn: Connection, frame: pd.DataFrame, lines: dict[str, str], batch_id: int | None, report: StepReport
    ) -> tuple[dict[str, int], dict[str, int]]:
        """``({anchor RIC: listing_id}, {every code: listing_id})``: the listings and their companies, created or found.

        A company is named by its CUSIP; one without (London's, US-listed ones registered abroad) by LSEG's PermID
        where asked (:attr:`Market.company_fields`), else by its share's ISIN; a share LSEG gives none of these for
        (Arcosa) by the share's own RIC, until the SEC's CIK (matched by ticker) joins it to the right one - an equity
        listing never stands alone. Share lines recorded as separate listings before become one.
        """
        before = anchor_map(conn, "listing", "RIC")
        listing_cols = frame.drop(
            columns=["sector", "industry_group", "industry", "issuer_isin", "issuer_permid", "incorporated"]
        )
        ids = ensure_listings(
            conn, listing_cols, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor=VENDOR, batch_id=batch_id
        )
        abroad = frame["issuer_isin"].notna()
        by_permid = frame["issuer_permid"].notna()
        bare = frame["issuer_anchor"].isna() & ~abroad & ~by_permid & frame["instrument_type"].isin(["equity", "fund"])
        for rows, anchor_of, issuer_code in (
            (by_permid, frame["issuer_permid"], "PERMID"),
            (abroad, frame["issuer_isin"], "ISIN"),
            (bare, frame["anchor"], "RIC"),
        ):
            if rows.any():
                extra = ensure_listings(
                    conn,
                    listing_cols[rows].assign(issuer_anchor=anchor_of[rows]),
                    anchor_type="RIC",
                    issuer_anchor_type=issuer_code,
                    vendor=VENDOR,
                    batch_id=batch_id,
                )
                ids.issuers_created += extra.issuers_created
        line_of = {code: ids.listing_ids[anchor] for code, anchor in lines.items()}
        merged = 0
        for code, listing_id in line_of.items():
            if code in before and before[code] != listing_id:
                merge_listings(conn, listing_id, [before[code]])
                merged += 1
        report.notes |= {"listings": len(set(line_of.values())), "listings_created": ids.listings_created}
        report.notes |= {"issuers_created": ids.issuers_created, "listings_merged": merged}
        return ids.listing_ids, line_of

    def _domicile(self, chunk: list[str], report: StepReport) -> tuple[int | None, pd.DataFrame]:
        """``(batch, [ric, domicile])``; empty (and noted) when LSEG does not know the field - it must not stop."""
        try:
            batch_id, raw = self.fetch("domicile", chunk, f.DOMICILE_FIELDS, {}, report)
        except RequestShapeError as exc:
            report.notes["domicile"] = f"not loaded: {exc}"[:300]
            return None, pd.DataFrame(columns=["ric", "domicile"])
        return batch_id, norm.domicile(raw)

    def _classify(
        self,
        conn: Connection,
        frame: pd.DataFrame,
        listing_ids: dict[str, int],
        report: StepReport,
    ) -> None:
        """Today's GICS classification and domicile of each company (from any of its listings).

        ``frame[batch_id]`` is the classification batch each row came from: each is recorded under its own.
        """
        lid = frame["anchor"].map(listing_ids)
        issuers = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
        rows = frame.assign(issuer_id=lid.map(issuers)).dropna(subset=["issuer_id"]).drop_duplicates("issuer_id")
        opened = closed = 0
        for batch_id, group in rows.dropna(subset=["sector"]).groupby("batch_id", dropna=False):
            recorded = record_classification(
                conn,
                group,
                vendor=VENDOR,
                on=pd.Timestamp(self.end).date(),
                batch_id=None if pd.isna(batch_id) else int(str(batch_id)),
            )
            opened, closed = opened + recorded.opened, closed + recorded.closed
        report.notes["classifications"] = {"opened": opened, "closed": closed}
        incorporated = rows.dropna(subset=["incorporated"])
        if not incorporated.empty:
            report.add(
                "incorporations",
                upsert(
                    conn,
                    schema.issuer,
                    incorporated[["issuer_id", "incorporated"]].rename(
                        columns={"incorporated": "incorporation_country"}
                    ),
                    key=["issuer_id"],
                ),
            )
        homes = rows.dropna(subset=["issuer_domicile"])
        if not homes.empty:
            report.add(
                "domiciles",
                upsert(
                    conn,
                    schema.issuer,
                    homes[["issuer_id", "issuer_domicile"]].rename(columns={"issuer_domicile": "domicile_country"}),
                    key=["issuer_id"],
                ),
            )

    @staticmethod
    def _spells(index_id: int, intervals: pd.DataFrame, listing_ids: dict[str, int]) -> pd.DataFrame:
        """The index's replayed spells by listing: spells of codes that turned out to be one share line merged."""
        rows = pd.DataFrame(
            {
                "index_id": index_id,
                "listing_id": intervals["ric"].map(listing_ids),
                "start_date": intervals["start_date"],
                "end_date": intervals["end_date"],
                "vendor_id": vendor_id(VENDOR),
            }
        ).dropna(subset=["listing_id"])
        return coalesce(rows.astype({"listing_id": int}))

    def _replace_memberships(  # noqa: PLR0917 - one index's replay and what it is checked against
        self,
        conn: Connection,
        index_id: int,
        rows: pd.DataFrame,
        report: StepReport,
        batch_id: int,
        first_trade: pd.Series,
    ) -> None:
        """The index's membership as replayed today (a replay is the whole truth, so it replaces the stored one).

        No spell starts before the listing first traded (a member with no join event in the window is dated to its
        start - see ``ingest.membership``). Only what changed is written: a spell the replay no longer has is
        deleted, the others upserted - so a repeated replay writes nothing.
        """
        t = schema.index_membership
        rows, clipped = clip_to_first_trade(rows, first_trade)
        report.notes["memberships_clipped"] = int(report.notes.get("memberships_clipped", 0)) + clipped
        mine = (t.c.index_id == index_id, t.c.vendor_id == vendor_id(VENDOR))
        held = {
            (int(i), pd.Timestamp(d)) for i, d in conn.execute(sa.select(t.c.listing_id, t.c.start_date).where(*mine))
        }
        replayed = {(int(i), pd.Timestamp(d)) for i, d in zip(rows["listing_id"], rows["start_date"], strict=True)}
        gone = [{"listing_id": i, "start_date": d.date()} for i, d in sorted(held - replayed)]
        if gone:
            conn.execute(
                sa.delete(t).where(
                    *mine, t.c.listing_id == sa.bindparam("listing_id"), t.c.start_date == sa.bindparam("start_date")
                ),
                gone,
            )
        report.notes["memberships_deleted"] = int(report.notes.get("memberships_deleted", 0)) + len(gone)
        written = upsert(conn, t, rows, key=["index_id", "listing_id", "start_date"], batch_id=batch_id)
        report.add("memberships", written)

    # ------------------------------------------------------------ 2. prices

    def spans(self) -> pd.DataFrame:
        """``[ric, listing_id, first, last]``: the days each listing's prices are needed for."""
        t, d = schema.index_membership, schema.index_definition
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(t.c.listing_id, t.c.start_date, t.c.end_date)
                .join(d, d.c.index_id == t.c.index_id)
                .where(d.c.code.in_(list(self.market.chains)))
            ).all()
        spells = pd.DataFrame(rows, columns=["listing_id", "start_date", "end_date"])
        spells["end_date"] = pd.to_datetime(spells["end_date"]).fillna(pd.Timestamp(self.end))  # still a member
        m = spells.groupby("listing_id").agg(first=("start_date", "min"), last=("end_date", "max")).reset_index()
        ids = self.listing_ids()  # every RIC a share line was ever called: each is asked for over the whole span
        codes = pd.DataFrame({"ric": list(ids), "listing_id": list(ids.values())})
        out = m.merge(codes, on="listing_id")
        out["first"] = pd.to_datetime(out["first"]) - pd.Timedelta(days=HISTORY_DAYS)
        out["last"] = pd.to_datetime(out["last"])
        levels = [value for value in self.levels() if value in ids]
        whole = pd.DataFrame(
            {
                "ric": levels,
                "listing_id": [ids[r] for r in levels],
                "first": pd.Timestamp(self.start),
                "last": pd.Timestamp(self.end),
            }
        )
        return pd.concat([out, whole], ignore_index=True).dropna(subset=["ric"])

    def prices(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Unadjusted daily bars in windows of :data:`PRICE_WINDOW_YEARS` (from ``since`` only, for a refresh).

        LSEG answers a history once per RIC, up to about 3,000 days a request - so each RIC is asked once per window,
        not once a year. In a build, a listing priced to the end of its part of a window is done and not asked again
        (a load stopped by the allowance resumes with the rest); a refresh asks every listing from ``since`` again
        (vendors correct recent days), by ``get_data`` - one request per call of :data:`PRICE_CHUNK` RICs, not one
        per RIC - except index levels, which stay histories (a handful of RICs).
        """
        report = report or StepReport()  # the caller's, so work done before a stop is still reported
        spans = self.spans()
        ids = self.listing_ids()
        first = pd.Timestamp(since) if since else pd.Timestamp(self.start) - pd.Timedelta(days=HISTORY_DAYS)
        last = pd.Timestamp(self.end)
        for lo, hi in price_windows(first, last):
            alive = spans[(spans["first"] <= hi) & (spans["last"] >= lo)]
            window = {"start": lo.date().isoformat(), "end": hi.date().isoformat()}
            wanted = sorted(alive["ric"].unique())
            if since is None:
                done = self._priced_through(lo, hi, alive)
                wanted = [r for r in wanted if ids.get(r) not in done]
            levels = [r for r in wanted if r.startswith(".")]  # index levels: no volume, asked apart
            shares = [r for r in wanted if not r.startswith(".")]
            chunks = [shares[i : i + PRICE_CHUNK] for i in range(0, len(shares), PRICE_CHUNK)]
            if since is None:
                requests = [(chunk, list(self.market.history_fields)) for chunk in chunks]
            else:  # a refresh's few days: get_data costs one request a call, a history one per RIC
                daily = {"SDate": window["start"], "EDate": window["end"], **f.DAILY_PARAMS}
                for chunk in chunks:
                    batch_id, raw = self.fetch("prices", chunk, list(self.market.daily_fields), daily, report)
                    self._write_new(norm.daily_prices(raw, window["start"], window["end"]), ids, batch_id, report)
                requests = []
            requests += [(levels, f.LEVEL_HISTORY_FIELDS)] if levels else []
            for chunk, fields_ in requests:
                batch_id, raw = self.fetch(
                    "prices", chunk, fields_, window, report, history=True, also_from=self.market.history_also_from
                )
                self._write_new(norm.price_bars(raw), ids, batch_id, report)
            log.info("LSEG prices %s..%s: %d listings, %d RICs asked", lo.date(), hi.date(), len(alive), len(wanted))
        with self.engine.begin() as conn:
            self._close_retired(conn)
        return report

    def _close_retired(self, conn: Connection) -> None:
        """Retired listings get ``delisted_on`` and a ticker dated to their lifetime; live ones a ticker dated sooner.

        A listing is retired when every RIC it ever had is retired (a renamed share line keeps its old RIC as an
        alias and is still live). Its ticker is valid from its first price to its last, so another vendor naming the
        company by ticker and date (Norgate's ``LEH-200809``) finds it - and not today's holder of that ticker. A live
        listing's ticker that a retired one held before it is dated from the live one's first trade.
        """
        retired = """
            SELECT l.listing_id FROM ref.listing l
            WHERE EXISTS (SELECT 1 FROM ref.identifier i WHERE i.entity_type = 'listing' AND i.entity_id = l.listing_id
                          AND i.id_type = 'RIC' AND CHARINDEX('^', i.value) > 0)
              AND NOT EXISTS (SELECT 1 FROM ref.identifier i WHERE i.entity_type = 'listing'
                              AND i.entity_id = l.listing_id AND i.id_type = 'RIC' AND CHARINDEX('^', i.value) = 0)
        """
        conn.execute(
            sa.text(
                "UPDATE l SET delisted_on = p.last_day FROM ref.listing l JOIN ("
                "SELECT listing_id, MAX(trade_date) AS last_day FROM market.price_bar WHERE vendor_id = :v "
                f"GROUP BY listing_id) p ON p.listing_id = l.listing_id WHERE l.listing_id IN ({retired}) "
                "AND (l.delisted_on IS NULL OR l.delisted_on <> p.last_day)"
            ),
            {"v": vendor_id(VENDOR)},
        )
        # a leaver LSEG's event list missed (a company bought: S&P 600 members most often) is a member until its last
        # trade, not for ever
        conn.execute(
            sa.text(
                "UPDATE m SET end_date = DATEADD(day, 1, l.delisted_on) FROM ref.index_membership m "
                "JOIN ref.listing l ON l.listing_id = m.listing_id "
                f"WHERE m.vendor_id = :v AND m.end_date IS NULL AND l.listing_id IN ({retired}) "
                "AND l.delisted_on IS NOT NULL AND m.start_date <= l.delisted_on"
            ),
            {"v": vendor_id(VENDOR)},
        )
        lives = pd.DataFrame(
            conn.execute(
                sa.text(
                    "SELECT l.listing_id, i.value AS ric, MIN(p.trade_date) AS first_day, l.delisted_on, "
                    "MAX(CASE WHEN p.trade_date = l.delisted_on THEN p.batch_id END) AS batch_id "
                    "FROM ref.listing l JOIN ref.identifier i ON i.entity_type = 'listing' "
                    "AND i.entity_id = l.listing_id AND i.id_type = 'RIC' "
                    "JOIN market.price_bar p ON p.listing_id = l.listing_id AND p.vendor_id = :v "
                    f"WHERE l.listing_id IN ({retired}) AND l.instrument_type = 'equity' AND l.delisted_on IS NOT NULL "
                    "AND (l.exchange_mic IS NULL OR l.exchange_mic <> 'XLON') "  # London RICs do not spell tickers
                    "GROUP BY l.listing_id, i.value, l.delisted_on"
                ),
                {"v": vendor_id(VENDOR)},
            ).all(),
            columns=["listing_id", "ric", "first_day", "delisted_on", "batch_id"],
        )
        if lives.empty:
            return
        lives = lives.sort_values("ric").drop_duplicates("listing_id", keep="last")  # one ticker per retired line
        add_identifiers(
            conn,
            pd.DataFrame(
                {
                    "entity_type": "listing",
                    "entity_id": lives["listing_id"],
                    "id_type": "TICKER",
                    "value": lives["ric"].map(rics.ticker),
                    "valid_from": pd.to_datetime(lives["first_day"]),
                    "valid_to": pd.to_datetime(lives["delisted_on"]) + pd.Timedelta(days=1),
                    "vendor_id": vendor_id(VENDOR),
                    "batch_id": lives["batch_id"],  # the answer that showed its last day
                }
            ),
            batch_id=None,
        )
        # a live share's ticker held by a retired one before it: valid from the live share's first trade, not for
        # ever (Valaris has been VAL since 2019; Valspar was VAL until 2017)
        conn.execute(
            sa.text(
                """
                UPDATE t SET valid_from = f.first_day
                FROM ref.identifier t
                JOIN (SELECT listing_id, MIN(trade_date) AS first_day FROM market.price_bar WHERE vendor_id = :v
                      GROUP BY listing_id) f ON f.listing_id = t.entity_id
                WHERE t.entity_type = 'listing' AND t.id_type = 'TICKER' AND t.valid_to IS NULL
                  AND t.valid_from < f.first_day
                  AND EXISTS (SELECT 1 FROM ref.identifier o WHERE o.entity_type = 'listing' AND o.id_type = 'TICKER'
                              AND o.value = t.value AND o.market = t.market AND o.entity_id <> t.entity_id
                              AND o.valid_to IS NOT NULL AND o.valid_to <= f.first_day)
                  AND NOT EXISTS (SELECT 1 FROM ref.identifier d WHERE d.id_type = 'TICKER' AND d.value = t.value
                                  AND d.market = t.market AND d.valid_from = f.first_day)
                """
            ),
            {"v": vendor_id(VENDOR)},
        )

    def _priced_through(self, lo: pd.Timestamp, hi: pd.Timestamp, alive: pd.DataFrame) -> set[int]:
        """Listings whose LSEG prices in ``[lo, hi]`` reach the end of their part of it (within a fortnight).

        A listing priced only for part of the window (a run of one-year windows stopped in 2014) is not done: its
        newest bar is too early. One that stopped trading before it left the index is asked again each build - a
        request per such listing.
        """
        p = schema.price_bar
        with self.engine.connect() as conn:
            newest = dict(
                conn.execute(
                    sa.select(p.c.listing_id, sa.func.max(p.c.trade_date))
                    .where(p.c.vendor_id == vendor_id(VENDOR), p.c.trade_date.between(lo.date(), hi.date()))
                    .group_by(p.c.listing_id)
                ).all()
            )
        needed = alive.groupby("listing_id")["last"].max().where(lambda d: d <= hi, hi)
        slack = pd.Timedelta(days=DONE_TOLERANCE_DAYS)
        return {
            lid
            for lid, day in zip(needed.index.astype(int), needed, strict=True)
            if lid in newest and pd.Timestamp(newest[lid]) >= day - slack
        }

    def _write_new(self, bars: pd.DataFrame, ids: dict[str, int], batch_id: int, report: StepReport) -> None:
        """Write a price answer unless it was written before (a resumed load does not rewrite what it wrote)."""
        with self.engine.connect() as conn:
            if status(conn, batch_id) == "normalized":
                return
        self._write_bars(bars, ids, batch_id, report)

    def _write_bars(self, bars: pd.DataFrame, ids: dict[str, int], batch_id: int, report: StepReport) -> None:
        level = bars["ric"].map(rics.is_index).astype(bool)
        currency = pd.Series(self.market.price_currency, index=bars.index).mask(level, self.market.level_currency)
        if "currency" in bars.columns:  # the answer says what each price is quoted in (London's refresh)
            currency = bars["currency"].where(bars["currency"].notna() & ~level, currency)
        rows = bars.assign(listing_id=bars["ric"].map(ids), vendor_id=vendor_id(VENDOR), currency=currency)
        rows = rows.dropna(subset=["listing_id"]).drop(columns="ric")
        with self.engine.begin() as conn:
            report.add(
                "price_bars",
                upsert(conn, schema.price_bar, rows, key=["listing_id", "vendor_id", "trade_date"], batch_id=batch_id),
            )
            mark(conn, batch_id, "normalized")

    # ------------------------------------------------------------ 3. dividends and splits

    def equities(self) -> list[str]:
        """Every share RIC of this market we hold a listing for (investment trusts too: they are index members)."""
        return sorted(r for r in self.listing_ids() if not rics.is_index(r))

    def actions(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Dividends; splits confirmed by the price (or the share count) and spin-offs."""
        report = report or StepReport()  # the caller's, so work done before a stop is still reported
        ids = self.listing_ids()
        window = {"SDate": (since or pd.Timestamp(self.start).date()).isoformat(), "EDate": self.end}
        unverified = by_count = 0
        for chunk in buckets(self.equities(), self.bucket_count("actions")):
            batch_id, raw = self.fetch("dividends", chunk, list(self.market.dividend_fields), window, report)
            div = norm.dividends(raw).assign(action_type="cash_dividend", ratio=None)
            fields_ = list(self.market.capital_change_fields)
            changes_id, raw = self.fetch("capital_changes", chunk, fields_, window, report)
            closes = self._closes(chunk, ids)
            changes = norm.capital_changes(
                raw, closes, columns=self.market.capital_change_columns, days_after=self.market.split_days_after
            )
            by_price = int(changes["verified"].sum())
            changes = norm.confirm_by_share_count(
                changes,
                closes,
                self._counts(chunk, ids),
                days_after=self.market.split_days_after,
                held=self._held_splits(chunk, ids),
            )
            by_count += int(changes["verified"].sum()) - by_price
            unverified += int((~changes["verified"]).sum())
            # a split the price does not confirm is not one: a real split always shows in unadjusted prices, and a
            # phantom one (Covidien's "100-for-1" of 2011-03-21) would multiply that day's return a hundredfold
            confirmed = changes[changes["verified"]].drop(columns="verified")
            parts = [div, confirmed.assign(amount=None)]
            rows = pd.concat([p for p in parts if not p.empty] or parts[:1], ignore_index=True)
            rows = rows.assign(listing_id=rows["ric"].map(ids), vendor_id=vendor_id(VENDOR)).dropna(
                subset=["listing_id"]
            )
            with self.engine.begin() as conn:
                report.add(
                    "corporate_actions",
                    upsert(
                        conn,
                        schema.corporate_action,
                        rows.drop(columns="ric"),
                        key=["listing_id", "vendor_id", "action_type", "ex_date"],
                        batch_id=batch_id,
                    ),
                )
                for read in (batch_id, changes_id):
                    mark(conn, read, "normalized")
        report.notes["splits_left_out_unconfirmed_by_price"] = unverified
        report.notes["splits_confirmed_by_share_count"] = by_count
        return report

    def _of_chunk(
        self, chunk: list[str], ids: dict[str, int], table: sa.Table, *columns: str, **where: object
    ) -> pd.DataFrame:
        """``[ric, *columns]``: LSEG's rows of ``table`` for the chunk's listings, under the chunk's codes."""
        listing = {ids[r]: r for r in chunk if r in ids}
        if not listing:
            return pd.DataFrame(columns=["ric", *columns])
        t = table
        query = sa.select(t.c.listing_id, *(t.c[c] for c in columns)).where(
            t.c.listing_id.in_(list(listing)),
            t.c.vendor_id == vendor_id(VENDOR),
            *(t.c[k] == v for k, v in where.items()),
        )
        with self.engine.connect() as conn:
            out = pd.DataFrame(conn.execute(query).all(), columns=["listing_id", *columns])
        return out.assign(ric=out["listing_id"].map(listing)).drop(columns="listing_id")

    def _closes(self, chunk: list[str], ids: dict[str, int]) -> pd.DataFrame:
        out = self._of_chunk(chunk, ids, schema.price_bar, "trade_date", "close")
        return out.assign(trade_date=pd.to_datetime(out["trade_date"]))

    def _counts(self, chunk: list[str], ids: dict[str, int]) -> pd.DataFrame:
        """``[ric, as_of_date, shares]`` outstanding - none yet on a build's first pass (``shares`` comes later)."""
        out = self._of_chunk(chunk, ids, schema.share_count, "as_of_date", "shares", kind="outstanding")
        return out.assign(as_of_date=pd.to_datetime(out["as_of_date"]), shares=out["shares"].astype(float))

    def _held_splits(self, chunk: list[str], ids: dict[str, int]) -> pd.DataFrame:
        """``[ric, ex_date, ratio]``: the splits and spin-offs already stored for the chunk's listings."""
        a = schema.corporate_action
        out = pd.concat(
            [self._of_chunk(chunk, ids, a, "ex_date", "ratio", action_type=kind) for kind in ("split", "spinoff")],
            ignore_index=True,
        )
        return out.dropna(subset=["ratio"]).astype({"ratio": float})

    def fx(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Daily exchange rates of the currencies this market's dividends are declared in, and of its own.

        Each is an ``fx`` listing - one unit of the currency, priced in US dollars - so research can turn a dividend
        into its universe's currency on the ex-date. A currency without a known RIC (:data:`~.fields.FX_RICS`) is
        noted, not guessed.
        """
        report = report or StepReport()
        a = schema.corporate_action
        mine = sorted(set(self.listing_ids().values()))
        with self.engine.connect() as conn:
            declared = {
                str(c)
                for c, listing in conn.execute(
                    sa.select(a.c.currency, a.c.listing_id)
                    .where(a.c.vendor_id == vendor_id(VENDOR), a.c.currency.is_not(None))
                    .distinct()
                ).all()
                if listing in mine
            }
        wanted = (declared | {self.market.level_currency} | set(self.market.quote_currencies)) - {"USD"}
        report.notes["fx_unknown_currencies"] = sorted(wanted - set(f.FX_RICS))
        quotes = {f.FX_RICS[c][0]: c for c in sorted(wanted & set(f.FX_RICS))}
        if not quotes:
            return report
        frame = pd.DataFrame(
            {
                "anchor": list(quotes),
                "issuer_anchor": None,
                "issuer_name": None,
                "name": [f"{c} in US dollars" for c in quotes.values()],
                "instrument_type": "fx",
                "currency": list(quotes.values()),
            }
        )
        in_dollars = {f.FX_RICS[c][0]: f.FX_RICS[c][1] for c in quotes.values()}
        first = pd.Timestamp(since) if since else pd.Timestamp(self.start) - pd.Timedelta(days=HISTORY_DAYS)
        ids: dict[str, int] = {}
        for lo, hi in price_windows(first, pd.Timestamp(self.end)):
            window = {"start": lo.date().isoformat(), "end": hi.date().isoformat()}
            batch_id, raw = self.fetch("fx", sorted(quotes), f.FX_HISTORY_FIELDS, window, report, history=True)
            if not ids:  # the listings name the first answer that priced them
                with self.engine.begin() as conn:
                    ids = ensure_listings(
                        conn, frame, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor=VENDOR, batch_id=batch_id
                    ).listing_ids
            with self.engine.connect() as conn:
                if status(conn, batch_id) == "normalized":
                    continue
            bars = norm.fx_bars(raw, in_dollars)
            rows = bars.assign(listing_id=bars["ric"].map(ids), vendor_id=vendor_id(VENDOR), currency="USD")
            with self.engine.begin() as conn:
                report.add(
                    "fx_bars",
                    upsert(
                        conn,
                        schema.price_bar,
                        rows.dropna(subset=["listing_id"]).drop(columns="ric"),
                        key=["listing_id", "vendor_id", "trade_date"],
                        batch_id=batch_id,
                    ),
                )
                mark(conn, batch_id, "normalized")
        return report

    def pounds_per_unit(self) -> pd.DataFrame:
        """``date x currency``: the value of one unit in the market's currency, from the ``fx`` listings' rates.

        Its sub-unit too (pence: a hundredth); dollars from the market's own rate (each rate is in dollars per unit).
        """
        own = self.market.level_currency
        p, t = schema.price_bar, schema.listing
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(t.c.currency, p.c.trade_date, p.c.close)
                .join(t, t.c.listing_id == p.c.listing_id)
                .where(t.c.instrument_type == "fx")
            ).all()
        dollars = pd.DataFrame(rows, columns=["currency", "trade_date", "close"]).pivot_table(
            index="trade_date", columns="currency", values="close", aggfunc="last"
        )
        dollars.index = pd.DatetimeIndex(pd.to_datetime(dollars.index))
        if own not in dollars.columns:
            return pd.DataFrame()
        out = dollars.div(dollars[own], axis=0).assign(USD=1.0 / dollars[own])
        return out.assign(**{norm.PENCE: 0.01} if own == "GBP" else {}).sort_index().ffill()

    def quotes(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Each London bar's currency: today's quote (``CF_CURR``), and before a change of quote currency the earlier.

        Only listings that need it are read back: those quoted today in something else than the market's sub-unit, and
        those with a day's move over 5x (:func:`~.normalize.quote_currencies`). A listing's own currency becomes its
        latest bar's.
        """
        report = report or StepReport()
        ids = self.listing_ids()
        live = [r for r in self.equities() if not rics.is_retired(r)]
        today = dt.date.today().isoformat()
        now: dict[int, str] = {}
        used: list[int] = []
        for chunk in buckets(live, self.bucket_count("instruments")):
            batch_id, raw = self.fetch("quote_currency", chunk, f.QUOTE_CURRENCY_FIELDS, {}, report, as_of=today)
            used.append(batch_id)
            for ric, currency in norm.current_currencies(raw).itertuples(index=False):
                now[ids[ric]] = str(currency)
        rates = self.pounds_per_unit()
        if rates.empty:
            report.notes["quotes"] = "no exchange rates: run the fx step first"
            return report
        mine = set(ids.values())
        default = self.market.price_currency
        with self.engine.connect() as conn:
            jumpers = {
                int(i)
                for (i,) in conn.execute(
                    sa.text(
                        "SELECT DISTINCT listing_id FROM (SELECT listing_id, [close], LAG([close]) OVER "
                        "(PARTITION BY listing_id ORDER BY trade_date) AS prev FROM market.price_bar "
                        "WHERE vendor_id = :v) x "
                        "WHERE prev > 0 AND [close] > 0 AND ABS(LOG([close] / prev)) > LOG(5.0)"
                    ),
                    {"v": vendor_id(VENDOR)},
                ).all()
            }
        todo = sorted((jumpers | {i for i, c in now.items() if c != default}) & mine)
        changed = 0
        p, a = schema.price_bar, schema.corporate_action
        for listing_id in todo:
            with self.engine.connect() as conn:
                bars = pd.DataFrame(
                    conn.execute(
                        sa.select(p.c.trade_date, p.c.close, p.c.currency)
                        .where(p.c.listing_id == listing_id, p.c.vendor_id == vendor_id(VENDOR))
                        .order_by(p.c.trade_date)
                    ).all(),
                    columns=["trade_date", "close", "currency"],
                )
                quiet = {
                    pd.Timestamp(d)
                    for (d,) in conn.execute(
                        sa.select(a.c.ex_date).where(
                            a.c.listing_id == listing_id, a.c.action_type.in_(["split", "spinoff"])
                        )
                    ).all()
                }
            if bars.empty:
                continue
            found = norm.quote_currencies(bars, now.get(listing_id, default), rates, quiet)
            moved = bars.assign(currency=found)[found != bars["currency"]]
            if moved.empty:
                continue
            with self.engine.begin() as conn:
                upsert(
                    conn,
                    schema.price_bar,
                    moved[["trade_date", "currency"]].assign(listing_id=listing_id, vendor_id=vendor_id(VENDOR)),
                    key=["listing_id", "vendor_id", "trade_date"],
                    batch_id=used[0] if used else None,
                )
                upsert(
                    conn,
                    schema.listing,
                    pd.DataFrame({"listing_id": [listing_id], "currency": [str(found.iloc[-1])]}),
                    key=["listing_id"],
                )
            changed += 1
        with self.engine.begin() as conn:
            for read in used:
                mark(conn, read, "normalized")
        report.notes["quotes"] = {"listings_read": len(todo), "listings_with_other_currencies": changed}
        return report

    # ------------------------------------------------------------ 4. share counts

    def shares(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Monthly share counts per share class and the free float."""
        report = report or StepReport()  # the caller's, so work done before a stop is still reported
        ids = self.listing_ids()
        window = {"SDate": (since or pd.Timestamp(self.start).date()).isoformat(), "EDate": self.end, **f.MONTHLY}
        for chunk in buckets(self.equities(), self.bucket_count("shares")):
            batch_id, shares_raw = self.fetch("shares_outstanding", chunk, f.SHARES_FIELDS, window, report)
            float_id, float_raw = self.fetch("free_float", chunk, f.FREE_FLOAT_FIELDS, window, report)
            counts = norm.share_counts(shares_raw, float_raw)
            rows = counts.assign(listing_id=counts["ric"].map(ids), vendor_id=vendor_id(VENDOR)).dropna(
                subset=["listing_id"]
            )
            with self.engine.begin() as conn:
                report.add(
                    "share_counts",
                    upsert(
                        conn,
                        schema.share_count,
                        rows.drop(columns="ric"),
                        key=["listing_id", "vendor_id", "kind", "as_of_date"],
                        batch_id=batch_id,
                    ),
                )
                for read in (batch_id, float_id):
                    mark(conn, read, "normalized")
        return report

    # ------------------------------------------------------------ 5. statements

    def statements(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Quarterly statements as company facts, dated by their first announcement (London: half-years)."""
        report = report or StepReport()  # the caller's, so work done before a stop is still reported
        if self.market.half_years:
            return self._half_years(since, report)
        ids = self.listing_ids()
        with self.engine.connect() as conn:
            issuer_of = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
        begin = (pd.Timestamp(since) - pd.DateOffset(years=1)).date() if since else pd.Timestamp(self.start).date()
        window = {"SDate": begin.isoformat(), "EDate": self.end, **f.QUARTERLY}
        in_dollars = {**window, **f.STATEMENT_PARAMS}
        for chunk in buckets(self.equities(), BUCKETS["statements"]):
            announced_id, announced_raw = self.fetch("announcements", chunk, f.ANNOUNCEMENT_FIELDS, window, report)
            batch_id, raw = self.fetch("statements", chunk, f.STATEMENT_FIELDS, in_dollars, report)
            facts = norm.statements(raw, norm.announcements(announced_raw))
            facts = facts.assign(issuer_id=facts["ric"].map(ids).map(issuer_of), vendor_id=vendor_id(VENDOR))
            facts = facts.dropna(subset=["issuer_id"]).drop_duplicates(
                ["issuer_id", "item", "period_type", "period_end", "published_at"]  # two share classes, one company
            )
            with self.engine.begin() as conn:
                report.add(
                    "financial_facts",
                    upsert(
                        conn,
                        schema.financial_fact,
                        facts.drop(columns="ric"),
                        key=["issuer_id", "vendor_id", "item", "period_type", "period_end", "published_at"],
                        batch_id=batch_id,
                        insert_only=True,  # the first figure we saw stays: later restatements do not overwrite it
                    ),
                )
                for read in (batch_id, announced_id):
                    mark(conn, read, "normalized")
        return report

    def _half_years(self, since: dt.date | None, report: StepReport) -> StepReport:
        """Half-years and fiscal years as first reported, in the market's currency at each period's rate.

        A period is inside a request's window by its announcement, so a refresh asks from a year before ``since``:
        a fiscal year announced months after it ended is still caught.
        """
        ids = self.listing_ids()
        with self.engine.connect() as conn:
            issuer_of = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
        begin = (pd.Timestamp(since) - pd.DateOffset(years=1)).date() if since else pd.Timestamp(self.start).date()
        window = {"SDate": begin.isoformat(), "EDate": self.end}
        currency = {"Curn": self.market.level_currency}
        for chunk in buckets(self.equities(), self.bucket_count("statements")):
            halves_id, halves = self.fetch(
                "statements_halves",
                chunk,
                f.UK_STATEMENT_FIELDS,
                {**window, **f.UK_STATEMENT_PARAMS, **currency},
                report,
            )
            years_id, years = self.fetch(
                "statements_years", chunk, f.UK_STATEMENT_FIELDS, {**window, **f.UK_YEARLY_PARAMS, **currency}, report
            )
            facts = norm.half_year_statements(halves, years, unit=self.market.level_currency)
            facts = facts.assign(issuer_id=facts["ric"].map(ids).map(issuer_of), vendor_id=vendor_id(VENDOR))
            facts = facts.dropna(subset=["issuer_id"]).drop_duplicates(
                ["issuer_id", "item", "period_type", "period_end", "published_at"]  # two share lines, one company
            )
            with self.engine.begin() as conn:
                report.add(
                    "financial_facts",
                    upsert(
                        conn,
                        schema.financial_fact,
                        facts.drop(columns="ric"),
                        key=["issuer_id", "vendor_id", "item", "period_type", "period_end", "published_at"],
                        batch_id=halves_id,
                        insert_only=True,
                    ),
                )
                for read in (halves_id, years_id):
                    mark(conn, read, "normalized")
        return report

    # ------------------------------------------------------------ 6. yearly statement items

    def announcements_held(self) -> pd.DataFrame:
        """``[ric, period_end, announced_at]``: when each period LSEG reported was first announced, from step 5's facts.

        Read from the facts already stored - asking LSEG again would cost a request per bucket.
        """
        fact = schema.financial_fact
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(fact.c.issuer_id, fact.c.period_end, sa.func.min(fact.c.published_at))
                .where(fact.c.vendor_id == vendor_id(VENDOR), fact.c.item == "revenue")
                .group_by(fact.c.issuer_id, fact.c.period_end)
            ).all()
            issuer_of = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
        held = pd.DataFrame(rows, columns=["issuer_id", "period_end", "announced_at"])
        rics_of = pd.DataFrame(
            [(ric, issuer_of.get(listing)) for ric, listing in self.listing_ids().items()], columns=["ric", "issuer_id"]
        )
        out = rics_of.merge(held, on="issuer_id")[["ric", "period_end", "announced_at"]]
        return out.assign(
            period_end=pd.to_datetime(out["period_end"]), announced_at=pd.to_datetime(out["announced_at"])
        )

    def annuals(self, since: dt.date | None = None, report: StepReport | None = None) -> StepReport:
        """Yearly statement items, one field per request, as company facts dated by their year's announcement."""
        report = report or StepReport()
        ids = self.listing_ids()
        with self.engine.connect() as conn:
            issuer_of = dict(conn.execute(sa.select(schema.listing.c.listing_id, schema.listing.c.issuer_id)).all())
        begin = (pd.Timestamp(since) - pd.DateOffset(years=1)).date() if since else pd.Timestamp(self.start).date()
        window = {"SDate": begin.isoformat(), "EDate": self.end, **f.YEARLY_STATEMENT_PARAMS}
        announced = self.announcements_held()
        for field_ in f.ANNUAL_ITEMS:
            for chunk in buckets(self.equities(), BUCKETS["annuals"]):
                batch_id, raw = self.fetch("annual_statements", chunk, f.annual_fields(field_), window, report)
                facts = norm.annual_statement(raw, field_, announced)
                facts = facts.assign(issuer_id=facts["ric"].map(ids).map(issuer_of), vendor_id=vendor_id(VENDOR))
                facts = facts.dropna(subset=["issuer_id"]).drop_duplicates(
                    ["issuer_id", "item", "period_type", "period_end", "published_at"]
                )
                with self.engine.begin() as conn:
                    report.add(
                        "financial_facts",
                        upsert(
                            conn,
                            schema.financial_fact,
                            facts.drop(columns="ric"),
                            key=["issuer_id", "vendor_id", "item", "period_type", "period_end", "published_at"],
                            batch_id=batch_id,
                            insert_only=True,
                        ),
                    )
                    mark(conn, batch_id, "normalized")
        return report


def load_lseg(
    engine: Engine,
    *,
    start: str = START,
    end: str | None = None,
    since: dt.date | None = None,
    steps: Sequence[str] | None = None,
    client: LsegClient | None = None,
    market: str = "US",
    indices: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run ``steps`` (see the module docstring); returns a report with ``complete`` False when LSEG's limit stopped it.

    ``since`` turns a full load into a refresh: prices, actions and share counts from that day, statements from a
    year before it (late restatements and announcements); members and listings are always replayed in full.
    ``market`` is a :data:`MARKETS` code; ``steps`` default to that market's. ``indices`` limits a trial build to some
    of the market's indices (``["FTSE100"]``).
    """
    end = end or dt.date.today().isoformat()
    chosen = MARKETS[market]
    if indices is not None:
        chosen = replace(chosen, chains={code: chosen.chains[code] for code in indices})
    steps = chosen.steps if steps is None else steps
    unknown = [s for s in steps if s not in chosen.steps]
    if unknown:
        raise ValueError(f"no step {unknown} for market {market}; its steps: {chosen.steps}")
    load = LsegLoad(engine, client or LsegClient(), start=start, end=end, market=chosen)
    report: dict[str, Any] = {"complete": False, "stopped": None, "market": market}
    with ops.job("load_lseg" if market == "US" else f"load_lseg_{market.lower()}", engine) as run:
        try:
            for step in steps:
                progress = StepReport()
                try:
                    if step == "universe":
                        load.universe(progress)
                    else:
                        getattr(load, step)(since, progress)
                finally:
                    report[step] = progress.as_dict()  # also what a stopped step did before the stop
                log.info("LSEG %s: %s", step, report[step])
                again = REPLAYED_AFTER.get(step)
                if since is None and again in steps[: list(steps).index(step)]:
                    progress = StepReport()
                    if again == "universe":
                        load.universe(progress)
                    else:
                        load.actions(None, progress)
                    report[f"{again}_after_{step}"] = progress.as_dict()
            report["complete"] = True
        except ThrottledError as exc:
            report["stopped"] = f"{exc} - run the load again later: it continues where it stopped"
            log.warning(report["stopped"])
        report["requests"] = load.client.requests
        report["lseg_requests"] = load.client.lseg_requests
        allowance = load.client.allowance()
        report["allowance_left"] = None if allowance is None else allowance.get("daily_request_remaining")
        report["skipped_rics"] = load.client.skipped[:50]
        run.finish(rows=load.client.requests, detail=report)
    return report


PROBE_RICS = ["AAPL.OQ", "BRKb.N", "LIN.OQ", "LEH.N^I08"]  # a share, a share class, a company abroad, a retired one


def probe(client: LsegClient | None = None) -> pd.DataFrame:
    """One small request per dataset, read by the same code as the load: ``[dataset, ok, rows, detail]``.

    Run before a long load: a field LSEG renamed or a parameter it no longer takes shows here in a minute, not hours
    into the load. Nothing is stored. Stops at LSEG's daily limit, marking the rest not tried.
    """
    c = client or LsegClient()
    window = {"SDate": "2024-01-01", "EDate": "2024-12-31"}
    quarterly = {**window, **f.QUARTERLY}
    tries: list[tuple[str, Any]] = [
        (
            "index members",
            lambda: norm.memberships(
                c.get_data(["0#.SPX"], f.SNAPSHOT_FIELDS),
                c.get_data([".SPX"], f.EVENT_FIELDS, {"SDATE": "2024-01-01", "EDATE": "2024-06-30", "IC": "B"}),
                "2024-01-01",
            ),
        ),
        ("instruments", lambda: norm.instruments(c.get_data(PROBE_RICS, f.INSTRUMENT_FIELDS))),
        ("classification", lambda: norm.classification(c.get_data(PROBE_RICS, f.CLASSIFICATION_FIELDS))),
        ("domicile", lambda: norm.domicile(c.get_data(PROBE_RICS, f.DOMICILE_FIELDS)).dropna()),
        ("prices", lambda: norm.price_bars(c.get_history(PROBE_RICS, f.HISTORY_FIELDS, "2008-09-08", "2008-09-12"))),
        ("prices, one whole window", lambda: _whole_window(c)),
        ("dividends", lambda: norm.dividends(c.get_data(PROBE_RICS, f.DIVIDEND_FIELDS, window))),
        (
            "capital changes",
            lambda: norm.capital_changes(
                c.get_data(
                    ["AAPL.OQ", "NVDA.OQ"], f.CAPITAL_CHANGE_FIELDS, {"SDate": "2020-01-01", "EDate": "2024-12-31"}
                ),
                pd.DataFrame(columns=["ric", "trade_date", "close"]),
            ),
        ),
        (
            "share counts",
            lambda: norm.share_counts(
                c.get_data(PROBE_RICS, f.SHARES_FIELDS, {**window, **f.MONTHLY}),
                c.get_data(PROBE_RICS, f.FREE_FLOAT_FIELDS, {**window, **f.MONTHLY}),
            ),
        ),
        ("announcements", lambda: norm.announcements(c.get_data(PROBE_RICS, f.ANNOUNCEMENT_FIELDS, quarterly))),
        (
            "statements in USD",
            lambda: norm.statements(
                c.get_data(PROBE_RICS, f.STATEMENT_FIELDS, {**window, **f.STATEMENT_PARAMS}),
                pd.DataFrame(columns=["ric", "period_end", "announced_at"]),
            ),
        ),
        (
            "yearly items (one field per request)",
            lambda: norm.annual_statement(
                c.get_data(
                    PROBE_RICS,
                    f.annual_fields("TR.DepreciationAmort"),
                    {**window, **f.YEARLY_STATEMENT_PARAMS},
                ),
                "TR.DepreciationAmort",
                pd.DataFrame(columns=["ric", "period_end", "announced_at"]),
            ),
        ),
    ]
    return _run_probe(c, tries)


def _run_probe(c: LsegClient, tries: list[tuple[str, Any]]) -> pd.DataFrame:
    """``[dataset, ok, rows, detail]`` for each try, then the allowance left; a try's exception is its detail."""
    rows: list[dict[str, Any]] = []
    stopped = False
    for dataset, run in tries:
        if stopped:
            rows.append({"dataset": dataset, "ok": False, "rows": 0, "detail": "not tried: LSEG limit reached"})
            continue
        try:
            answer = run()
            rows.append(
                {
                    "dataset": dataset,
                    "ok": len(answer) > 0,
                    "rows": len(answer),
                    "detail": "" if len(answer) else "no rows - check the fields",
                }
            )
        except ThrottledError as exc:
            stopped = True
            rows.append({"dataset": dataset, "ok": False, "rows": 0, "detail": str(exc)[:200]})
        except Exception as exc:
            rows.append({"dataset": dataset, "ok": False, "rows": 0, "detail": f"{type(exc).__name__}: {exc}"[:300]})
    limits = c.allowance()
    left = None if limits is None else limits.get("daily_request_remaining")
    rows.append(
        {
            "dataset": "allowance left today",
            "ok": left is not None,
            "rows": left or 0,
            "detail": "Workspace did not say"
            if limits is None
            else f"of {limits.get('daily_request_max')} requests; "
            f"{limits.get('daily_volume_remaining', 0) // 1000:,} of {limits.get('daily_volume_max', 0) // 1000:,} MB",
        }
    )
    return pd.DataFrame(rows)


def _whole_window(c: LsegClient) -> pd.DataFrame:
    """Apple's bars for one price window from 2014, in one request: LSEG must answer all of it.

    LSEG cuts a history at about 3,000 days; the error names the days answered, from which a shorter
    :data:`PRICE_WINDOW_YEARS` follows.
    """
    lo, hi = pd.Timestamp("2014-01-01"), pd.Timestamp(2014 + PRICE_WINDOW_YEARS - 1, 12, 31)
    bars = norm.price_bars(c.get_history(["AAPL.OQ"], f.HISTORY_FIELDS, lo.date().isoformat(), hi.date().isoformat()))
    expected = len(pd.bdate_range(lo, hi))
    if len(bars) < 0.95 * expected or bars["trade_date"].min() > lo + pd.Timedelta(days=10):
        first = bars["trade_date"].min().date() if len(bars) else None
        raise ValueError(
            f"LSEG answered {len(bars)} of ~{expected} days of a {PRICE_WINDOW_YEARS}-year window (from {first}) - "
            "shorten PRICE_WINDOW_YEARS"
        )
    return bars


class ProbeExpectationError(ValueError):
    """LSEG answered, but not what a UK load relies on (a field renamed, a currency or rule changed)."""


def _expect(ok: object, what: str) -> None:
    if not ok:
        raise ProbeExpectationError(what)


def _value(cell: object) -> float:
    """An answer's cell as a number (LSEG's columns come typed as objects)."""
    return float(str(cell))


def _uk_members(c: LsegClient) -> pd.DataFrame:
    frames = []
    for code, chain in f.UK_MEMBERSHIP_CHAINS.items():
        snapshot = c.get_data([f"0#{chain}"], f.SNAPSHOT_FIELDS)
        _expect(len(snapshot) >= 90, f"{chain}: {len(snapshot)} members today")
        events = c.get_data([chain], f.EVENT_FIELDS, {"SDATE": START, "EDATE": dt.date.today().isoformat(), "IC": "B"})
        spells = norm.memberships(snapshot, events, START).assign(index=code)
        _expect(len(events) > 0 and pd.to_datetime(events["Date"]).min().year <= 2006, f"{chain}: events from 2006")
        frames.append(spells)
    out = pd.concat(frames, ignore_index=True)
    abroad = sorted(out.loc[out["ric"].map(rics.exchange_mic) != "XLON", "ric"].unique())
    _expect(not abroad, f"members not on the London Stock Exchange: {abroad[:10]}")
    return out


def _uk_closes(c: LsegClient) -> pd.DataFrame:
    bars = c.get_history(["VOD.L", "CS.L^D07"], f.UK_HISTORY_FIELDS, "2006-03-01", "2006-03-07")
    _expect(set(bars["Instrument"]) == {"VOD.L", "CS.L^D07"}, "official closes of a live and a retired RIC")
    _expect(bars["OFF_CLOSE"].notna().all(), "OFF_CLOSE blank")
    return bars


def _uk_currency(c: LsegClient) -> pd.DataFrame:
    days = {**f.DAILY_PARAMS, "SDate": "2026-10-19", "EDate": "2026-10-23"}
    px = c.get_data(["VOD.L"], f.PRICE_CURRENCY_FIELDS, days)
    _expect(set(px["Currency"].dropna()) == {"GBp"}, f"prices in {set(px['Currency'].dropna())}, not pence")
    return px


def _uk_codes(c: LsegClient) -> pd.DataFrame:
    attrs = norm.instruments(c.get_data(["VOD.L", "SJP.L"], f.INSTRUMENT_FIELDS))
    _expect(attrs["isin"].notna().all() and attrs["sedol"].notna().all(), "ISIN and SEDOL")
    _expect(attrs["cusip"].isna().all(), "a London share with a CUSIP")
    raw = c.get_data(["SJP.L"], ["TR.ExchangeTicker"])
    _expect(raw.iloc[0, -1] == "STJ", "St James's Place's exchange ticker is STJ, not its RIC's root")
    return attrs


def _uk_companies(c: LsegClient) -> pd.DataFrame:
    raw = c.get_data(["GLEN.L", "SMT.L"], f.UK_COMPANY_FIELDS)
    out = raw.rename(columns={"Instrument": "ric", **f.UK_COMPANY_COLUMNS}).set_index("ric")
    _expect(out.loc["GLEN.L", "incorporated"] == "JE", "Glencore incorporated in Jersey")
    _expect(out.loc["SMT.L", "trbc_industry"] == f.INVESTMENT_TRUST_INDUSTRY, "Scottish Mortgage an investment trust")
    _expect(out["permid"].notna().all(), "PermIDs")
    return out


def _uk_dividends(c: LsegClient) -> pd.DataFrame:
    raw = c.get_data(["SHEL.L"], f.DIVIDEND_CURRENCY_FIELDS, {"SDate": "2024-01-01", "EDate": "2024-12-31"})
    _expect(set(raw["Currency"].dropna()) == {"GBP"}, f"Shell's London dividends in {set(raw['Currency'])}")
    return raw


def _uk_first_reported(c: LsegClient) -> pd.DataFrame:
    item = ["TR.F.TotRevenue.periodenddate", "TR.F.TotRevenue", "TR.F.NetIncAfterMinIntr"]
    window = {"SDate": "2013-01-01", "EDate": "2015-12-31"}  # a period is in the window by its announcement
    halves = c.get_data(["TSCO.L"], item, {**window, **f.UK_STATEMENT_PARAMS})
    years = c.get_data(["TSCO.L"], item, {**window, **f.UK_YEARLY_PARAMS})
    restated = c.get_data(["TSCO.L"], item, {**window, **f.UK_YEARLY_PARAMS, "ReportingState": "Rsdt"})
    revenue = halves.columns[2]
    h = halves.set_index(pd.to_datetime(halves["Period End Date"]))[revenue].astype(float)
    y = years.set_index(pd.to_datetime(years["Period End Date"]))
    fy = pd.Timestamp("2015-02-28")
    _expect(abs(h.loc["2014-08-23"] + h.loc[fy] - _value(y.loc[fy, revenue])) < 1e6, "two halves make the year")
    net = years.columns[3]
    orig = _value(y.loc["2013-02-23", net])
    rsdt = _value(restated.set_index(pd.to_datetime(restated["Period End Date"])).loc["2013-02-23", net])
    _expect(orig != rsdt, f"Tesco's FY2013 net income as first reported ({orig:,.0f}) equals its restated one")
    return halves


def _uk_estimates(c: LsegClient) -> pd.DataFrame:
    fields_ = ["TR.EPSMean", "TR.EPSMean.calcdate"]
    months = {"SDate": "2015-01-01", "EDate": "2015-03-31", "Frq": "M", "Period": "FY1"}
    own = c.get_data(["SHEL.L"], fields_, months)
    gbp = c.get_data(["SHEL.L"], fields_, {**months, "Curn": "GBP"})
    rate = _value(own.iloc[0, 1]) / _value(gbp.iloc[0, 1])
    _expect(1.4 < rate < 1.6, f"dollars per pound {rate:.3f} in January 2015 (that month's rate is ~1.51)")
    return gbp


def _uk_levels(c: LsegClient) -> pd.DataFrame:
    bars = c.get_history(list(f.UK_LEVELS.values()), ["TRDPRC_1"], "2005-01-04", "2005-01-07")
    _expect(set(bars["Instrument"]) == set(f.UK_LEVELS.values()), "total-return levels from 2005")
    return bars


def probe_uk(client: LsegClient | None = None) -> pd.DataFrame:
    """What a London (FTSE UK) load relies on, one small request each: ``[dataset, ok, rows, detail]``.

    Each try checks a fact recorded in ``fields`` (London section) - not only that rows came back - so a change at LSEG
    shows here before a load is built on it. Nothing is stored; ~20 requests.
    """
    c = client or LsegClient()
    tries: list[tuple[str, Any]] = [
        ("members: FTSE 100 / 250 / SmallCap since 2005", lambda: _uk_members(c)),
        ("official closes (live and retired)", lambda: _uk_closes(c)),
        ("prices in pence", lambda: _uk_currency(c)),
        ("codes: ISIN, SEDOL, exchange ticker", lambda: _uk_codes(c)),
        ("company: PermID, incorporation, investment trusts", lambda: _uk_companies(c)),
        ("dividends in pounds", lambda: _uk_dividends(c)),
        ("half-years as first reported", lambda: _uk_first_reported(c)),
        ("estimates in pounds at each month's rate", lambda: _uk_estimates(c)),
        ("total-return levels", lambda: _uk_levels(c)),
    ]
    return _run_probe(c, tries)
