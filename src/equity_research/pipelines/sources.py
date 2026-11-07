"""The data sources a build or a refresh runs - one entry per vendor adapter.

    checks = probe_sources()                    # each source's questions with known answers, before a long load
    reports = run_sources(engine, "build")      # every source in vendor_preferences().sources, in order
    reports = run_sources(engine, "refresh")    # the same, asking each only for what changed lately

Adding a vendor is one adapter plus one entry in :data:`SOURCES`; choosing it is a setting (``EQR_SOURCES``). The order
matters: the first source creates the listings the later ones attach their codes and prices to.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research.db import schema
from equity_research.db.schema import vendor_id
from equity_research.pipelines.load_benchmarks import load_benchmarks
from equity_research.pipelines.load_estimates import load_estimates, refresh_estimates
from equity_research.pipelines.load_estimates import probe as probe_estimates
from equity_research.pipelines.load_financedatabase import load_financedatabase
from equity_research.pipelines.load_lseg import MARKETS, load_lseg, market_rics
from equity_research.pipelines.load_lseg import probe as probe_lseg
from equity_research.pipelines.load_lseg import probe_uk as probe_lseg_uk
from equity_research.pipelines.load_norgate import load_norgate
from equity_research.pipelines.load_norgate import probe as probe_norgate
from equity_research.pipelines.load_sec import load_sec
from equity_research.pipelines.load_simfin import load_simfin
from equity_research.pipelines.load_simfin import probe as probe_simfin
from equity_research.pipelines.load_tiingo import load_tiingo
from equity_research.pipelines.load_tiingo import probe as probe_tiingo
from equity_research.settings import vendor_preferences

Mode = Literal["build", "refresh"]
PRICE_OVERLAP_DAYS = 10  # a refresh asks again for the last days: vendors correct recent prices
SEC_BUILD_MAX_AGE_DAYS = 7.0
SEC_REFRESH_MAX_AGE_DAYS = 30.0  # the SEC rebuilds its file nightly; companies report quarterly
SIMFIN_MAX_AGE_DAYS = 7.0


@dataclass(frozen=True)
class Source:
    """One vendor's loads: from nothing (or continuing an unfinished build), and the regular top-up."""

    vendor: str
    supplies: str
    build: Callable[[Engine], dict[str, Any]]
    refresh: Callable[[Engine], dict[str, Any]]
    #: what it needs on this computer before it can run
    needs: str = "nothing (public files)"
    #: questions with known answers, asked before a long load: ``[check, ok, detail...]``; None for public files
    probe: Callable[[], pd.DataFrame] | None = None


def price_since(engine: Engine, vendor: str, market: str | None = None) -> dt.date | None:
    """Where a vendor's price refresh starts (:func:`refresh_start`); None (everything) without a price yet.

    With ``market`` (an LSEG market code), from that market's newest price: London's refresh must not start from New
    York's newest day, nor New York's from London's.
    """
    p = schema.price_bar
    query = sa.select(p.c.listing_id, sa.func.max(p.c.trade_date)).where(p.c.vendor_id == vendor_id(vendor))
    with engine.connect() as conn:
        newest_of = dict(conn.execute(query.group_by(p.c.listing_id)).all())
    if market is not None:
        mine = set(market_rics(engine, MARKETS[market]).values())
        newest_of = {k: v for k, v in newest_of.items() if k in mine}
    return refresh_start(max(newest_of.values())) if newest_of else None


def refresh_start(newest: dt.date) -> dt.date:
    """The first of the month a few days before ``newest`` - asked again because vendors correct recent days.

    The month, not the day: a refresh that failed after writing prices moves ``newest`` on, and a start that moved
    with it would change every request's description - the rerun would pay LSEG for all of it again instead of
    reading back what it landed. The longer window costs nothing: LSEG counts requests, not days.
    """
    return (newest - dt.timedelta(days=PRICE_OVERLAP_DAYS)).replace(day=1)


SOURCES: dict[str, Source] = {
    "refinitiv": Source(
        "refinitiv",
        "index members, listings and codes, sectors, daily prices, dividends and splits, share counts, statements",
        build=load_lseg,
        refresh=lambda engine: load_lseg(engine, since=price_since(engine, "refinitiv", "US")),
        needs="LSEG Workspace running and signed in; up to 10,000 requests a day (a full build takes ~7,400)",
        probe=probe_lseg,
    ),
    "refinitiv_uk": Source(
        "refinitiv",
        "London (FTSE 100 / 250 / SmallCap): members, listings, official closes in pence, dividends, splits, share "
        "counts, half-yearly statements as first reported in pounds",
        build=lambda engine: load_lseg(engine, market="UK"),
        refresh=lambda engine: load_lseg(engine, market="UK", since=price_since(engine, "refinitiv", "UK")),
        needs="LSEG Workspace running; ~4,000 requests from nothing (resumable), a few hundred a day after",
        probe=probe_lseg_uk,
    ),
    "estimates": Source(
        "refinitiv",
        "analysts' estimates (I/B/E/S): monthly consensus history, reported actuals, revision counts",
        build=load_estimates,
        refresh=refresh_estimates,  # asks only once a new month end has passed
        needs="LSEG Workspace running; after refinitiv (its listings and splits); ~3,200 requests from nothing, "
        "~400 on the first refresh after each month end",
        probe=probe_estimates,
    ),
    "estimates_uk": Source(
        "refinitiv",
        "analysts' estimates (I/B/E/S) of London's listings, in pounds",
        build=lambda engine: load_estimates(engine, market="UK"),
        refresh=lambda engine: refresh_estimates(engine, market="UK"),
        needs="LSEG Workspace running; after refinitiv_uk (its listings and splits)",
    ),
    "yahoo": Source(
        "yahoo",
        "benchmark index levels and funds (^GSPC, SPY, RSP, ^SPXEW, SPTM, ^SP400, ^SP600)",
        build=load_benchmarks,
        refresh=load_benchmarks,
    ),
    "norgate": Source(
        "norgate",
        "index members with history, listings incl. delisted, unadjusted prices, splits and dividends, GICS",
        build=load_norgate,
        refresh=lambda engine: load_norgate(engine, since=price_since(engine, "norgate")),
        needs="Norgate Data Updater running with a subscription (a trial holds about two years)",
        probe=probe_norgate,
    ),
    "financedatabase": Source(
        "financedatabase",
        "codes (ISIN, FIGI, Yahoo symbol), the country a company is run from, its own sector labels - enriches only",
        build=load_financedatabase,
        refresh=load_financedatabase,
    ),
    "tiingo": Source(
        "tiingo",
        "unadjusted daily prices, dividends and splits of the listings other sources found (by ticker, cross-checked)",
        build=load_tiingo,
        refresh=load_tiingo,  # it continues from each listing's newest bar either way
        needs="an API key (`eqr set-key tiingo`); a free account allows few requests an hour - runs resume",
        probe=probe_tiingo,
    ),
    "simfin": Source(
        "simfin",
        "company accounts, standardised (dated by first publication where the SEC confirms them), its sector labels",
        build=lambda engine: load_simfin(engine, max_age_days=SIMFIN_MAX_AGE_DAYS),
        refresh=lambda engine: load_simfin(engine, max_age_days=SIMFIN_MAX_AGE_DAYS),
        needs="an API key (`eqr set-key simfin`); run after sec (its figures are checked against the SEC's)",
        probe=probe_simfin,
    ),
    "sec": Source(
        "sec",
        "company accounts as first reported, with every correction",
        build=lambda engine: load_sec(engine, max_age_days=SEC_BUILD_MAX_AGE_DAYS),
        refresh=lambda engine: load_sec(engine, max_age_days=SEC_REFRESH_MAX_AGE_DAYS),
        needs="a contact for the SEC (EQR_SEC_USER_AGENT or keyring sec_user_agent: name and e-mail)",
    ),
}


def describe_sources(names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """``[source, supplies, needs, in this build]`` for every registered source - the ones this build runs first."""
    chosen = list(names or vendor_preferences().sources)
    order = chosen + sorted(n for n in SOURCES if n not in chosen)
    rows = [
        {"source": n, "supplies": SOURCES[n].supplies, "needs": SOURCES[n].needs, "in this build": n in chosen}
        for n in order
    ]
    return pd.DataFrame(rows)


def probe_sources(names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Each source's probe, one table: ``[source, check, ok, detail]`` (a probe that cannot run says why)."""
    frames = []
    for name in names or vendor_preferences().sources:
        source = SOURCES[name]
        if source.probe is None:
            out = pd.DataFrame([{"check": "nothing to ask first", "ok": True, "detail": source.needs}])
        else:
            try:
                out = source.probe().rename(columns={"dataset": "check"})
            except Exception as exc:  # one source failing to answer must not hide the others
                out = pd.DataFrame([{"check": "probe", "ok": False, "detail": f"{type(exc).__name__}: {exc}"[:300]}])
        frames.append(out.assign(source=name))
    table = pd.concat(frames, ignore_index=True)
    return table[["source", "check", "ok", "detail"]]


def run_sources(engine: Engine, mode: Mode, names: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    """Run each source's ``mode`` load in order; ``{vendor: its report}``."""
    out = {}
    for name in names or vendor_preferences().sources:
        if name not in SOURCES:
            raise KeyError(f"no source {name!r}; known: {sorted(SOURCES)}")
        source = SOURCES[name]
        out[name] = source.build(engine) if mode == "build" else source.refresh(engine)
    return out
