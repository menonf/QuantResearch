"""Everything a backtest needs for one universe, read from the local research cache.

    data = load_universe_data(cache, "SP1500", "2010-01-01", "2024-12-31")

A universe is in its indices' currency (dollars for the S&P universes, pounds for the FTSE ones): prices quoted in a
sub-unit (London's pence) are turned into it, and anything in another currency stops research or is left out. Index
members that are closed-end funds (UK investment trusts) are left out of a universe - their accounts are a portfolio's.
Buying a London share pays stamp duty where its company is incorporated in the UK or Ireland (``buy_tax_bps``).

Inputs are chosen by vendor, never hard-wired: prices take ``price_vendors`` in priority order per listing and day,
a company's accounts on each date from the first of ``fundamentals_vendor`` that can answer for it then (a later one
answers while the preferred one cannot yet - :func:`accounts_as_of`), memberships from the first of the member
vendors that has the index - by default the project's :func:`~equity_research.settings.vendor_preferences`. The same
functions build the panels whichever vendor supplies them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from equity_research.research.accounting import accounting_items, as_of
from equity_research.research.cache import ResearchCache
from equity_research.research.panels import (
    by_listing,
    close_panel,
    latest_counts,
    membership_mask,
    split_adjusted,
    split_factor,
    splits_between,
    total_returns,
    trading_days,
)
from equity_research.research.portfolio import rebalance_dates
from equity_research.research.ratios import lookup, market_values, ratio_panel
from equity_research.research.reads import (
    as_vendors,
    corporate_actions,
    financial_facts,
    first_preferred,
    id_list,
    listing_issuers,
    outstanding_counts,
)
from equity_research.settings import vendor_preferences

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseData:
    """Daily ``date x listing_id`` panels plus the ratios, market values and share changes on rebalance dates."""

    universe: str
    close: pd.DataFrame
    adjusted: pd.DataFrame
    returns: pd.DataFrame
    members: pd.DataFrame
    segment: pd.DataFrame
    cost_bps: pd.DataFrame
    borrow_bps: pd.DataFrame
    #: ``date x listing_id``: a tax on purchases only, beside ``cost_bps`` (UK stamp duty; 0 where none)
    buy_tax_bps: pd.DataFrame
    rebalances: pd.DatetimeIndex
    ratios: pd.DataFrame
    benchmark: pd.Series
    #: ``rebalance date x listing_id``: the listing's share of its company's market value (:func:`listing_values`)
    market_cap: pd.DataFrame
    #: ``rebalance date x listing_id``: log change of the company's share count over the year before the date
    share_change: pd.DataFrame
    #: ``rebalance date x listing_id``: the latest quarter's standardised earnings surprise known before the date
    earnings_surprise: pd.DataFrame
    #: ``rebalance date x listing_id``: yearly growth of trailing earnings per share over :data:`GROWTH_YEARS`
    eps_growth: pd.DataFrame
    #: ``rebalance date x listing_id``: yearly growth of trailing sales per share over :data:`GROWTH_YEARS`
    sales_growth: pd.DataFrame
    #: what every amount is in (prices, dividends, accounts): the universe's indices' currency
    currency: str = "USD"


#: Research adds and divides amounts, so they must all be in one currency: a universe's indices' (the default when
#: an index does not say).
RESEARCH_CURRENCY = "USD"
#: Fact units research can use as they are besides money in the research currency: share counts, and pure numbers
#: (regulatory capital ratios).
COUNT_UNITS = ("shares", "pure")
#: A sub-unit prices are quoted in -> (its currency, sub-units in one): London quotes pence.
SUBUNITS = {"GBX": ("GBP", 100.0)}
#: Stamp duty on buying a London share, by the country its company is incorporated in (UK 0.5%, Ireland 1%); none on
#: sales, on shorts (taken as contracts for difference) or on a company incorporated elsewhere (Jersey, Guernsey).
STAMP_DUTY_BPS = {"GB": 50.0, "IE": 100.0}
LONDON = "XLON"
#: One price per listing and day: the first preferred vendor's.
PRICE_KEY = ("listing_id", "trade_date")


class CurrencyError(ValueError):
    """Amounts in another currency reached research, which cannot convert them."""


def check_currency(currencies: pd.Series, what: str, currency: str = RESEARCH_CURRENCY) -> None:
    """Stop when any amount is in another currency than ``currency`` (a missing one is allowed)."""
    other = currencies.dropna()
    other = other[other != currency]
    if not other.empty:
        raise CurrencyError(
            f"{what}: {len(other):,} rows in {sorted(other.unique())[:5]} - this universe is in "
            f"{currency} only; convert them in the adapter or leave those listings out"
        )


def in_major_units(bars: pd.DataFrame) -> pd.DataFrame:
    """``bars`` with prices quoted in a sub-unit (:data:`SUBUNITS`: pence) turned into their currency (pounds).

    Stored as quoted (a fact); turned when read, so a London price, its pound dividends and its accounts in pounds
    are on one scale.
    """
    out = bars.copy()
    for subunit, (currency, per) in SUBUNITS.items():
        quoted = out["currency"] == subunit
        out.loc[quoted, "close"] = out.loc[quoted, "close"] / per
        out.loc[quoted, "currency"] = currency
    return out


def in_currency(bars: pd.DataFrame, rates: pd.DataFrame, currency: str) -> pd.DataFrame:
    """``bars`` in ``currency``: a sub-unit in its currency, any other at its day's rate.

    Pence become pounds (:func:`in_major_units`); a London line once quoted in dollars or euros (Coca-Cola
    Europacific, InterContinental Hotels) is converted at the day's rate (:func:`fx_rates`: the last known). A bar
    without a rate keeps its currency, so :func:`check_currency` names it.
    """
    out = in_major_units(bars)
    other = out["currency"].notna() & (out["currency"] != currency)
    if not other.any() or currency not in rates.columns:
        return out
    known = rates.sort_index().ffill()
    days = pd.DatetimeIndex(pd.to_datetime(out.loc[other, "trade_date"]))
    at = known.index.searchsorted(days, side="right") - 1
    rows = np.where(at >= 0, at, 0)
    theirs = known[currency].to_numpy()[rows]
    own = [
        known[c].to_numpy()[r] if c in known.columns else np.nan
        for c, r in zip(out.loc[other, "currency"].astype(str), rows, strict=True)
    ]
    factor = np.where(at >= 0, np.asarray(own, dtype=float) / theirs, np.nan)
    converted = pd.Series(factor, index=out.index[other]).notna()
    idx = converted[converted].index
    out.loc[idx, "close"] = out.loc[idx, "close"] * pd.Series(factor, index=out.index[other]).loc[idx]
    out.loc[idx, "currency"] = currency
    return out


def in_research_currency(facts: pd.DataFrame, currency: str = RESEARCH_CURRENCY) -> pd.DataFrame:
    """The facts research can use: in ``currency`` or counts (:data:`COUNT_UNITS`); the rest left out, and logged."""
    usable = facts["unit"].isin((currency, *COUNT_UNITS))
    if not usable.all():
        dropped = facts.loc[~usable]
        log.warning(
            "left out %d facts of %d companies in %s",
            len(dropped),
            dropped["issuer_id"].nunique(),
            sorted(dropped["unit"].unique())[:5],
        )
    return facts[usable].drop(columns="unit")


def load_universe_data(
    cache: ResearchCache,
    universe: str,
    start: str,
    end: str,
    *,
    price_vendors: tuple[str, ...] | None = None,
    fundamentals_vendor: str | tuple[str, ...] | None = None,
    rebalance: str = "ME",
    history_days: int = 400,
    include_last_day: bool = False,
) -> UniverseData:
    """Load ``universe`` over ``start..end`` (prices from ``history_days`` earlier, for momentum and betas).

    ``include_last_day`` adds the last trading day to the decision dates (a screen as of today, not month end).
    The benchmark level takes ``price_vendors`` first, then any other vendor for the days they miss: it is reported,
    never traded, so filling a vendor's gap with another's level of the same index is safe.
    """
    components = cache.query(
        """SELECT uc.index_id, uc.segment, uc.cost_bps, uc.borrow_bps, uc.priority, u.benchmark_listing_id
           FROM ref.universe u JOIN ref.universe_component uc ON uc.universe_id = u.universe_id WHERE u.code = ?""",
        [universe],
    )
    if components.empty:
        # name the folder: an empty cache is usually a kernel started without EQR_DB_NAME, reading another database's
        raise KeyError(
            f"universe {universe!r} is not in the cache at {cache.path} - sync it first, or set EQR_DB_SERVER and "
            "EQR_DB_NAME for the database whose cache you meant"
        )
    prefs = vendor_preferences()
    price_vendors = price_vendors or prefs.prices
    account_vendors = as_vendors(fundamentals_vendor or prefs.fundamentals)
    currency = universe_currency(cache, components["index_id"])
    members = memberships(cache, components["index_id"], prefs.members).merge(
        components[["index_id", "segment", "priority"]], on="index_id"
    )
    members = members[members["listing_id"].isin(operating_companies(cache, members["listing_id"]))]
    listings = members["listing_id"].drop_duplicates()
    benchmark_id = components["benchmark_listing_id"].dropna()
    price_ids = pd.concat([listings, benchmark_id]).drop_duplicates()
    first_day = (pd.Timestamp(start) - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")

    bars = cache.query(
        f"SELECT listing_id, vendor, trade_date, close, currency FROM market.price_bar "
        f"WHERE listing_id IN ({id_list(price_ids)}) AND trade_date BETWEEN ? AND ?",
        [first_day, end],
    )
    bars = in_currency(bars, rates := fx_rates(cache), currency)
    check_currency(bars["currency"], "prices", currency)
    is_benchmark = bars["listing_id"].isin(benchmark_id.astype(int).tolist())
    # a benchmark takes any vendor's level for the days the preferred ones miss (it is reported, never traded)
    bars = first_preferred(bars[bars["vendor"].isin(price_vendors) | is_benchmark], price_vendors, PRICE_KEY)
    actions = price_actions(corporate_actions(cache, price_ids), bars, price_vendors, benchmark_id)
    actions = dividends_in(actions, rates, currency)
    close = close_panel(bars)
    close = close.loc[trading_days(close)]  # a vendor's holiday prints are not trading days
    days = pd.DatetimeIndex(close.index)
    stocks = close.columns[close.columns.isin(listings.astype(int).tolist())]

    member = membership_mask(members, days, stocks)
    # a listing's segment on each day: the highest-priority component index it belongs to
    segment = pd.DataFrame(None, index=days, columns=stocks, dtype=object)
    cost = pd.DataFrame(float("nan"), index=days, columns=stocks)
    borrow = pd.DataFrame(float("nan"), index=days, columns=stocks)
    ordered = components.sort_values("priority")
    for index_id, seg, cost_bps, borrow_bps in zip(
        ordered["index_id"].tolist(),
        ordered["segment"].tolist(),
        ordered["cost_bps"].astype(float).tolist(),
        ordered["borrow_bps"].astype(float).fillna(0.0).tolist(),
        strict=True,
    ):
        inside = membership_mask(members[members["index_id"] == index_id], days, stocks) & segment.isna()
        segment = segment.mask(inside, seg)
        cost = cost.mask(inside, cost_bps)
        borrow = borrow.mask(inside, borrow_bps)
    tax = stamp_duty(cache, stocks).reindex(stocks).fillna(0.0)
    buy_tax = pd.DataFrame(np.tile(tax.to_numpy(), (len(days), 1)), index=days, columns=stocks).where(cost.notna())

    in_range = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    rebalances = rebalance_dates(in_range, rebalance)
    if include_last_day and len(in_range):
        rebalances = rebalances.union(pd.DatetimeIndex([in_range.max()]))
    listing_issuer = listing_issuers(cache, listings)
    facts = financial_facts(cache, listing_issuer, account_vendors)
    accounts = _accounts(
        vendor_items(facts, account_vendors, currency),
        listing_issuer,
        close[stocks],
        actions,
        member.reindex(rebalances),
        counts=own_counts(outstanding_counts(cache, listings), price_vendors),
    )

    returns = total_returns(close, actions)
    bench = (
        returns[int(benchmark_id.iloc[0])]
        if len(benchmark_id) and int(benchmark_id.iloc[0]) in returns
        else pd.Series(dtype=float)
    )
    return UniverseData(
        universe=universe,
        close=close[stocks],
        adjusted=split_adjusted(close[stocks], actions),
        returns=returns[stocks],
        members=member,
        segment=segment,
        cost_bps=cost,
        borrow_bps=borrow,
        buy_tax_bps=buy_tax,
        rebalances=rebalances,
        ratios=accounts["ratios"],
        benchmark=bench,
        market_cap=accounts["market_cap"],
        share_change=accounts["share_change"],
        earnings_surprise=accounts["earnings_surprise"],
        eps_growth=accounts["eps_growth"],
        sales_growth=accounts["sales_growth"],
        currency=currency,
    )


def fx_rates(cache: ResearchCache) -> pd.DataFrame:
    """``date x currency``: US dollars per unit of each currency held as an ``fx`` listing (dollars themselves: 1)."""
    rows = cache.query(
        "SELECT l.currency, p.trade_date, p.close FROM market.price_bar p "
        "JOIN ref.listing l ON l.listing_id = p.listing_id WHERE l.instrument_type = 'fx'"
    )
    rates = rows.pivot_table(index="trade_date", columns="currency", values="close", aggfunc="last")
    rates.index = pd.DatetimeIndex(pd.to_datetime(rates.index))
    return rates.sort_index().assign(USD=1.0)


def dividends_in(actions: pd.DataFrame, rates: pd.DataFrame, currency: str) -> pd.DataFrame:
    """``actions`` with every cash dividend in ``currency``, without the ``currency`` column.

    One declared in another currency (Anglo American's dollars on its London line, Old Mutual's rand) is converted at
    the rate of its ex-date, else the last before it (:func:`fx_rates`); one without a rate is left out, and logged -
    a dividend read in the wrong currency would be a wrong return. One without a currency is in its listing's.
    """
    if "currency" not in actions.columns:
        return actions
    foreign = (
        (actions["action_type"] == "cash_dividend") & actions["currency"].notna() & (actions["currency"] != currency)
    )
    if not foreign.any():
        return actions.drop(columns="currency")
    out = actions.copy()
    known = rates.sort_index().ffill()
    days = pd.DatetimeIndex(pd.to_datetime(out.loc[foreign, "ex_date"]))
    position = known.index.searchsorted(days, side="right") - 1
    has_day = position >= 0
    rows = np.where(has_day, position, 0)
    theirs = known[currency].to_numpy()[rows] if currency in known.columns else np.full(len(rows), np.nan)
    own = [
        known[c].to_numpy()[r] if c in known.columns else np.nan
        for c, r in zip(out.loc[foreign, "currency"].astype(str), rows, strict=True)
    ]
    factor = np.where(has_day, np.asarray(own, dtype=float) / theirs, np.nan)
    out.loc[foreign, "amount"] = out.loc[foreign, "amount"] * factor
    missing = foreign & out["amount"].isna()
    if missing.any():
        log.warning(
            "left out %d dividends in %s: no exchange rate to %s on their ex-date",
            int(missing.sum()),
            sorted(out.loc[missing, "currency"].unique()),
            currency,
        )
    return out[~missing].drop(columns="currency").reset_index(drop=True)


def price_actions(
    actions: pd.DataFrame, bars: pd.DataFrame, price_vendors: tuple[str, ...], benchmarks: pd.Series
) -> pd.DataFrame:
    """The corporate actions that go with the prices chosen, without the ``vendor`` column.

    A stock takes the actions of ``price_vendors``; a benchmark those of the vendor that supplied most of its prices -
    SPTM is priced by Yahoo alone, and without Yahoo's dividends and split its return would miss its dividends and
    show the 2017 split as a crash.
    """
    is_benchmark = actions["listing_id"].isin(benchmarks.astype(int).tolist())
    counts = bars.groupby(["listing_id", "vendor"]).size().rename("bars").reset_index()
    priced_by = counts.sort_values("bars").drop_duplicates("listing_id", keep="last").set_index("listing_id")["vendor"]
    own = actions["vendor"] == actions["listing_id"].map(priced_by)
    keep = (~is_benchmark & actions["vendor"].isin(price_vendors)) | (is_benchmark & own)
    return actions.loc[keep].drop(columns="vendor").reset_index(drop=True)


def own_counts(counts: pd.DataFrame, price_vendors: tuple[str, ...]) -> pd.DataFrame:
    """``[listing_id, as_of_date, shares]``: each listing's own outstanding count, from the first price vendor with it.

    A share line's own count is dated and never restated for later splits - unlike a company-level count, which a
    vendor may restate (LSEG's statements carry today's share basis back in time: Alphabet's 2015 count in its
    post-2022 shares, twenty times too many) - so a market value built from it is right.
    """
    c = first_preferred(counts[counts["vendor"].isin(price_vendors)], price_vendors, ["listing_id", "as_of_date"])
    return c[["listing_id", "as_of_date", "shares"]].reset_index(drop=True)


def _accounts(
    items: Mapping[str, pd.DataFrame],
    listing_issuer: pd.Series,
    close: pd.DataFrame,
    actions: pd.DataFrame,
    members: pd.DataFrame,
    *,
    counts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """The accounts-based panels on the decision dates (``members``' index), by :class:`UniverseData` field name.

    Market values come from each listing's own share count (:func:`own_counts`) where it has one: the listing's value
    weights it, its company's value (the sum over its listings) prices the ratios, and the change in its count is
    the issuance. The company's count from its accounts fills only where a listing has no count of its own. Per-share
    growth divides by the company's own counts too, never restated, so a split years ago does not read as growth.
    """
    dates, stocks = pd.DatetimeIndex(members.index), close.columns
    empty = pd.DataFrame(float("nan"), index=dates, columns=stocks)
    names = ("market_cap", "share_change", "earnings_surprise", "eps_growth", "sales_growth")
    if not items:
        return {"ratios": pd.DataFrame()} | dict.fromkeys(names, empty)
    factor = split_factor(actions, pd.DatetimeIndex(close.index), stocks)
    own = latest_counts(counts, actions, dates, stocks)
    listed = close.reindex(dates) * own
    company = company_values(listed, listing_issuer)
    fund = by_listing(accounts_as_of(items, dates), listing_issuer)
    fund = fund[fund.index.get_level_values("listing_id").isin(stocks)]
    where = fund.index
    fund = fund.assign(
        listed_value=lookup(
            company, where.get_level_values("date").to_series(), where.get_level_values("listing_id").to_series()
        )
    )
    values = market_values(fund, close, factor)
    from_accounts = listing_values(
        _panel(values["market_cap"].where(fund["listed_value"].isna().to_numpy()), dates, stocks),
        members,
        listing_issuer,
    )
    market_cap = listed.where(members).combine_first(from_accounts)
    own_before = in_shares_of(latest_counts(counts, actions, _a_year_earlier(dates), stocks), dates, actions)
    by_statements = log_change(
        _panel(values["shares"], dates, stocks), shares_a_year_before(items, dates, listing_issuer, stocks, actions)
    )
    share_change = log_change(own, own_before).combine_first(by_statements)
    company_shares = company_values(own, listing_issuer)
    earlier = pd.DatetimeIndex(dates - pd.DateOffset(years=GROWTH_YEARS))
    shares_then = company_values(
        in_shares_of(latest_counts(counts, actions, earlier, stocks), dates, actions, years=GROWTH_YEARS),
        listing_issuer,
    )
    then = items_years_before(items, dates, listing_issuer, stocks, ("net_income", "revenue"), years=GROWTH_YEARS)
    growth = {
        out: per_share_growth(_panel(fund[item], dates, stocks), company_shares, then[item], shares_then, GROWTH_YEARS)
        for out, item in (("eps_growth", "net_income"), ("sales_growth", "revenue"))
    }
    return {
        "ratios": ratio_panel(fund, close, factor),
        "market_cap": market_cap,
        "share_change": share_change,
        "earnings_surprise": _panel(fund["earnings_surprise"], dates, stocks),
        **growth,
    }


#: The span per-share growth is measured over, in years (the GARP index's three-year growth rates).
GROWTH_YEARS = 3


def per_share_growth(
    now: pd.DataFrame, shares_now: pd.DataFrame, then: pd.DataFrame, shares_then: pd.DataFrame, years: int
) -> pd.DataFrame:
    """Yearly compound growth of ``now / shares_now`` over ``then / shares_then``, ``years`` apart.

    Missing unless both per-share amounts are positive: growth from a loss (or into one) has no compound rate.
    """
    a = now / shares_now.where(shares_now > 0)
    b = then / shares_then.where(shares_then > 0)
    ratio = (a / b).where((a > 0) & (b > 0))
    return pd.DataFrame(np.power(ratio.to_numpy(dtype=float), 1.0 / years) - 1.0, index=now.index, columns=now.columns)


def items_years_before(
    items: Mapping[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    listing_issuer: pd.Series,
    listings: pd.Index,
    names: Sequence[str],
    *,
    years: int,
) -> dict[str, pd.DataFrame]:
    """``{item: date x listing_id}``: each item as known ``years`` before each date, labelled with the date itself.

    The vendor is chosen as on that earlier day (:func:`accounts_as_of`).
    """
    earlier = pd.Series(dates - pd.DateOffset(years=years), index=dates)
    known = by_listing(accounts_as_of(items, pd.DatetimeIndex(earlier.unique()), only=list(names)), listing_issuer)
    known = known[known.index.get_level_values("listing_id").isin(listings)].reset_index()
    label = pd.DataFrame({"earlier": earlier.to_numpy(), "label": earlier.index})
    known = known.merge(label, left_on="date", right_on="earlier")
    out = {}
    for name in names:
        panel = known.pivot_table(index="label", columns="listing_id", values=name, aggfunc="last")
        out[name] = panel.reindex(index=dates, columns=listings)
    return out


def _a_year_earlier(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates - pd.DateOffset(years=1))


def in_shares_of(earlier: pd.DataFrame, dates: pd.DatetimeIndex, actions: pd.DataFrame, years: int = 1) -> pd.DataFrame:
    """Counts dated ``years`` before each of ``dates`` (``earlier``, row by row), in the shares of the later date.

    Splits between the two dates adjust the count, so a split is not mistaken for an issue of new shares.
    """
    long = earlier.set_axis(dates).rename_axis(index="date", columns="listing_id").reset_index()
    long = long.melt(id_vars="date", var_name="listing_id", value_name="shares").dropna(subset=["shares"])
    if long.empty:
        return pd.DataFrame(np.nan, index=dates, columns=earlier.columns)
    after = long["date"] - pd.DateOffset(years=years)
    long["shares"] = long["shares"].to_numpy() * splits_between(actions, long["listing_id"], after, long["date"])
    wide = long.pivot_table(index="date", columns="listing_id", values="shares", aggfunc="last")
    return wide.reindex(index=dates, columns=earlier.columns)


def company_values(listed: pd.DataFrame, listing_issuer: pd.Series) -> pd.DataFrame:
    """``date x listing_id``: each listing's company total - the sum over its listings that have one.

    For values (each share line's market value) or share counts (each line's own count).
    """
    issuer = listing_issuer.reindex(listed.columns).to_numpy()
    total = listed.T.groupby(issuer).transform("sum").T
    known = listed.notna().astype(float).T.groupby(issuer).transform("sum").T
    return total.where(known > 0)


def _panel(values: pd.Series, dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """A ``(date, listing_id)`` series as a ``date x listing_id`` panel on ``dates`` and ``columns``."""
    return values.unstack("listing_id").reindex(index=dates, columns=columns)  # noqa: PD010 - long to wide


def shares_a_year_before(
    items: Mapping[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    listing_issuer: pd.Series,
    listings: pd.Index,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """``date x listing_id``: the company's share count known a year before each date, in the shares of the date.

    The count is the one research used on that earlier day (:func:`accounts_as_of`: the vendor chosen then). Splits
    between the count's publication and the date adjust it, so a split does not look like an issue of new shares
    (:func:`~equity_research.research.panels.splits_between` reads the split dates, not a price panel, which may begin
    after a count published two years earlier).
    """
    earlier = pd.Series(dates - pd.DateOffset(years=1), index=dates)
    known = by_listing(accounts_as_of(items, pd.DatetimeIndex(earlier.unique()), only=["shares"]), listing_issuer)
    known = known[known.index.get_level_values("listing_id").isin(listings)].reset_index()
    label = pd.DataFrame({"earlier": earlier.to_numpy(), "label": earlier.index})
    known = known.merge(label, left_on="date", right_on="earlier").dropna(subset=["shares"])
    known["shares"] = known["shares"].to_numpy() * splits_between(
        actions, known["listing_id"], known["known_at:shares"], known["label"]
    )
    panel = known.pivot_table(index="label", columns="listing_id", values="shares", aggfunc="last")
    return panel.reindex(index=dates, columns=listings)


def log_change(now: pd.DataFrame, before: pd.DataFrame) -> pd.DataFrame:
    """``log(now / before)`` where both are positive (a count of zero or below is a gap, not a change)."""
    ratio = now.where(now > 0) / before.where(before > 0)
    return pd.DataFrame(np.log(ratio.to_numpy(dtype=float)), index=now.index, columns=now.columns)


def listing_values(market_cap: pd.DataFrame, members: pd.DataFrame, listing_issuer: pd.Series) -> pd.DataFrame:
    """Each member listing's share of its company's market value; non-members missing.

    The value is divided among the company's member listings that day: GOOG and GOOGL each carry half of Alphabet,
    not all of it twice.
    """
    issuer = listing_issuer.reindex(market_cap.columns)
    classes = members.astype(float).T.groupby(issuer.to_numpy()).transform("sum").T
    return market_cap.where(members) / classes.where(classes > 0)


#: The item that says a vendor can describe a company on a date: trailing net income needs four reported quarters,
#: so a vendor whose filings have only just begun (the SEC's first XBRL filings, 2009-2011) does not answer yet.
ANSWERS = "net_income"


def vendor_items(
    facts: pd.DataFrame, vendors: tuple[str, ...], currency: str = RESEARCH_CURRENCY
) -> dict[str, pd.DataFrame]:
    """Each of ``vendors``' accounting items, in preference order, each built from that vendor's facts alone.

    :func:`~equity_research.research.accounting.accounting_items` per vendor, of its facts in ``currency``; a vendor
    with no usable facts is left out.
    """
    out: dict[str, pd.DataFrame] = {}
    for vendor in vendors:
        own = in_research_currency(facts[facts["vendor"] == vendor].drop(columns="vendor"), currency)
        if not own.empty:
            out[vendor] = accounting_items(own)
    return out


def accounts_as_of(
    items: Mapping[str, pd.DataFrame], dates: pd.DatetimeIndex, only: Sequence[str] | None = None
) -> pd.DataFrame:
    """What was known on each date, all of a company's figures on a date from one vendor.

    ``(date, issuer_id)`` x items (with ``period_end:`` and ``known_at:`` columns) and ``vendor``. The vendor is the
    first in ``items``' order that knows the company's trailing net income (:data:`ANSWERS`) on the date, else the
    first that knows anything about it. So a company's figures never mix two vendors on one date, and a later vendor
    answers while the preferred one cannot yet - LSEG during the year the SEC's first XBRL filings need to build four
    quarters, where cutting LSEG at the SEC's first period end (whose comparative figures were published years later)
    left most companies with no accounts in 2008-2011. ``only`` limits the items looked up.
    """
    known = []
    for rank, (vendor, its) in enumerate(items.items()):
        wanted = None if only is None else {*only, ANSWERS}
        cols = [c for c in its.columns if wanted is None or c.split(":")[-1] in wanted]
        k = as_of(its[cols], dates, with_period_end=True)
        silent = k[ANSWERS].isna() if ANSWERS in k.columns else pd.Series(True, index=k.index)
        known.append(k.assign(vendor=vendor, _rank=rank, _silent=silent))
    if not known:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "issuer_id"]))
    both = pd.concat(known).reset_index().sort_values(["date", "issuer_id", "_silent", "_rank"], kind="stable")
    best = both.drop_duplicates(["date", "issuer_id"], keep="first").drop(columns=["_rank", "_silent"])
    return best.set_index(["date", "issuer_id"]).sort_index()


def universe_currency(cache: ResearchCache, index_ids: pd.Series) -> str:
    """The one currency of a universe's indices (:data:`RESEARCH_CURRENCY` when they do not say)."""
    found = cache.query(
        f"SELECT DISTINCT currency FROM ref.index_definition WHERE index_id IN ({id_list(index_ids)}) "
        "AND currency IS NOT NULL"
    )["currency"].tolist()
    if len(found) > 1:
        raise CurrencyError(f"a universe's indices are in {sorted(found)}: research needs one currency")
    return str(found[0]) if found else RESEARCH_CURRENCY


def operating_companies(cache: ResearchCache, listing_ids: pd.Series) -> set[int]:
    """The listings that are a company's shares - not a closed-end fund (``fund``: a UK investment trust)."""
    rows = cache.query(
        f"SELECT listing_id FROM ref.listing WHERE listing_id IN ({id_list(listing_ids)}) AND instrument_type <> 'fund'"
    )
    return {int(i) for i in rows["listing_id"]}


def stamp_duty(cache: ResearchCache, listing_ids: pd.Index) -> pd.Series:
    """``listing_id -> bps`` charged on buying it: a London share's, by where its company is incorporated.

    :data:`STAMP_DUTY_BPS`; a listing elsewhere, or of a company incorporated elsewhere, is not in the answer.
    """
    rows = cache.query(
        "SELECT l.listing_id, i.incorporation_country FROM ref.listing l "
        "JOIN ref.issuer i ON i.issuer_id = l.issuer_id "
        f"WHERE l.listing_id IN ({id_list(listing_ids.to_series())}) AND l.exchange_mic = '{LONDON}'"
    )
    return rows.set_index("listing_id")["incorporation_country"].map(STAMP_DUTY_BPS).dropna().astype(float)


def memberships(cache: ResearchCache, index_ids: pd.Series, vendors: tuple[str, ...]) -> pd.DataFrame:
    """``[index_id, listing_id, start_date, end_date]``: each index's members from the first of ``vendors`` that has it.

    Two vendors' histories of one index are never mixed.
    """
    rows = cache.query(
        f"SELECT index_id, listing_id, start_date, end_date, vendor FROM ref.index_membership "
        f"WHERE index_id IN ({id_list(index_ids)})"
    )
    rank = {v: i for i, v in enumerate(vendors)}
    rows = rows[rows["vendor"].isin(rank)].assign(_rank=lambda d: d["vendor"].map(rank))
    best = rows.groupby("index_id")["_rank"].transform("min")
    return rows[rows["_rank"] == best].drop(columns=["vendor", "_rank"]).reset_index(drop=True)


def members_over_time(cache: ResearchCache, frequency: str = "ME") -> pd.DataFrame:
    """``date x index code``: how many members each index had at each period end (a quick check of the history)."""
    codes = cache.query("SELECT index_id, code FROM ref.index_definition")
    spells = memberships(cache, codes["index_id"], vendor_preferences().members).merge(codes, on="index_id")
    if spells.empty:
        return pd.DataFrame()
    start = pd.to_datetime(spells["start_date"]).min()
    days = pd.date_range(start, pd.Timestamp.today().normalize(), freq=frequency)
    out = {}
    for code, g in spells.groupby("code"):
        begin = pd.to_datetime(g["start_date"]).to_numpy()
        end = pd.to_datetime(g["end_date"]).fillna(pd.Timestamp.max).to_numpy()
        out[str(code)] = [int(((begin <= d) & (end > d)).sum()) for d in days.to_numpy()]
    return pd.DataFrame(out, index=days)
