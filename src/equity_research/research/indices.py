"""Rebuild an index from its members - weighted by market value, or equally - and compare it with the real thing.

    data = load_index_data(cache, "SP500", "2010-01-01", "2026-10-22")
    cap = cap_weighted(data.price_returns, data.market_cap, data.members)         # vs ^GSPC (prices only)
    cap_tr = cap_weighted(data.total_returns, data.market_cap, data.members)      # vs SPY (with dividends)
    ew = equal_weighted(data.total_returns, data.members)                         # vs RSP
    tracking(cap["return"], vendor_returns(cache, "^GSPC", total=False))

Nothing is known before it was: a member on day ``t`` earns its return from ``t-1`` to ``t`` with the weight its
market value had at the close of ``t-1``, and the share count behind that value was published before ``t-1``.

An index is one index's code (``SP500``) or a universe's (``FTSEAS``: the FTSE 100, 250 and SmallCap together, as the
FTSE All-Share is built), in the indices' own currency (pounds for London's: pence and lines quoted in dollars turned
into them). A listing in several of a universe's indices on one day belongs to the first by the universe's priority
(``segment``): the FTSE 100, 250 and SmallCap never share a member, but LSEG's event lists leave some listings in two.

Share counts: the listing's free float where a vendor has it (what S&P and FTSE weight by), else the company's own count
from its SEC filings. Expect the market-value version to differ from the official index a little more than the
equal-weighted one does: a company count includes shares that never trade, and some filings are mis-tagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from equity_research.research.accounting import accounting_items, as_of
from equity_research.research.cache import ResearchCache
from equity_research.research.data import (
    dividends_in,
    fx_rates,
    in_currency,
    in_major_units,
    operating_companies,
    universe_currency,
)
from equity_research.research.data import memberships as preferred_memberships
from equity_research.research.panels import (
    by_listing,
    close_panel,
    latest_counts,
    membership_mask,
    price_returns,
    splits_between,
    total_returns,
    trading_days,
)
from equity_research.research.portfolio import rebalance_dates
from equity_research.research.ratios import lookup, plausible_market_cap
from equity_research.research.reads import id_list
from equity_research.settings import vendor_preferences

TRADING_DAYS_PER_YEAR = 252
#: the facts behind a market value and its plausibility check (revenue and book equity)
INDEX_ITEMS = (
    "shares_cover",
    "shares_outstanding",
    "shares_weighted_basic",
    "revenue",
    "equity_parent",
    "equity_total",
)


@dataclass(frozen=True)
class IndexData:
    """Daily ``date x listing_id`` panels of one index's members."""

    code: str
    close: pd.DataFrame
    price_returns: pd.DataFrame
    total_returns: pd.DataFrame
    members: pd.DataFrame
    segment: pd.DataFrame  # the code of the index each member belongs to that day (None: not a member)
    shares: pd.DataFrame
    share_source: pd.DataFrame  # "float" (the listing's free float), "sec" (the company's count) or ""
    market_cap: pd.DataFrame
    names: pd.Series


def sec_shares(
    items: pd.DataFrame,
    listing_issuer: pd.Series,
    actions: pd.DataFrame,
    members: pd.DataFrame,
    close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The company's reported share count per member listing and day, in that day's shares.

    A count is in the shares of the day it was published (filings restate for a split before they are filed), so
    only splits after that day adjust it. An issuer with several member listings (two share classes) has its
    count divided among them. With ``close``, a count whose market value is implausible
    (:func:`~equity_research.research.ratios.plausible_market_cap`) is dropped for the day.
    """
    days = pd.DatetimeIndex(members.index)
    cols = [c for name in ("shares", "revenue", "equity") for c in (name, f"known_at:{name}")]
    known = as_of(items[cols], days, with_period_end=True)
    panel = by_listing(known, listing_issuer).dropna(subset=["shares"]).reset_index()
    panel = panel[panel["listing_id"].isin(members.columns)]
    since = splits_between(actions, panel["listing_id"], panel["known_at:shares"], panel["date"])
    panel["shares"] = panel["shares"].to_numpy() * since
    if close is not None:  # a count implying an absurd market value is a mis-scaled filing: dropped
        price = lookup(close, panel["date"], panel["listing_id"])
        cap = plausible_market_cap(
            pd.Series(price * panel["shares"].to_numpy(), index=panel.index), panel["revenue"], panel["equity"]
        )
        panel = panel[cap.notna() | np.isnan(price)]
    shares = panel.pivot_table(index="date", columns="listing_id", values="shares", aggfunc="last")
    shares = shares.reindex(index=days, columns=members.columns)
    issuer = listing_issuer.reindex(members.columns)
    classes = members.astype(float).T.groupby(issuer.to_numpy()).transform("sum").T  # member listings per issuer
    return shares / classes.where(classes > 0, 1.0)


def float_shares(
    counts: pd.DataFrame, actions: pd.DataFrame, members: pd.DataFrame, max_age_days: int = 400
) -> pd.DataFrame:
    """A vendor's free-float count per listing and day (the latest dated before the day), in that day's shares.

    ``counts``: ``[listing_id, as_of_date, shares]``. A float count belongs to one listing (one share class) and
    leaves out shares that do not trade (founders', governments'), as index providers weight.
    """
    return latest_counts(counts, actions, pd.DatetimeIndex(members.index), members.columns, max_age_days)


def index_components(cache: ResearchCache, code: str) -> pd.DataFrame:
    """``[index_id, code]`` of the indices ``code`` names, by priority: one index (``SP500``), else a universe's."""
    found = cache.query("SELECT index_id, code FROM ref.index_definition WHERE code = ?", [code])
    if found.empty:
        found = cache.query(
            "SELECT uc.index_id, d.code FROM ref.universe u "
            "JOIN ref.universe_component uc ON uc.universe_id = u.universe_id "
            "JOIN ref.index_definition d ON d.index_id = uc.index_id WHERE u.code = ? ORDER BY uc.priority",
            [code],
        )
    if found.empty:
        raise KeyError(f"no index or universe {code!r} in the cache at {cache.path}")
    return found.reset_index(drop=True)


def segments(
    memberships: pd.DataFrame, components: pd.DataFrame, days: pd.DatetimeIndex, columns: pd.Index
) -> pd.DataFrame:
    """``date x listing_id``: the code of the first of ``components`` (by priority) the listing is a member of that day.

    ``memberships``: ``[index_id, listing_id, start_date, end_date]``; ``components``: ``[index_id, code]`` in priority
    order. A listing in two indices on one day is counted once, in the first.
    """
    segment = pd.DataFrame(None, index=days, columns=columns, dtype=object)
    for index_id, name in zip(components["index_id"].tolist(), components["code"].tolist(), strict=True):
        inside = membership_mask(memberships[memberships["index_id"] == index_id], days, columns) & segment.isna()
        segment = segment.mask(inside, name)
    return segment


def load_index_data(
    cache: ResearchCache,
    code: str,
    start: str,
    end: str,
    *,
    vendors: tuple[str, ...] | None = None,
    funds: bool = True,
) -> IndexData:
    """Members' prices, returns, share counts and market values for index ``code`` (``SP500`` is the S&P 500).

    Prices from ``vendors`` (default: the preferred price vendors), members from the preferred member vendor. With
    ``funds=False`` closed-end funds (UK investment trusts) are left out, as research universes leave them out.
    """
    vendors = vendors or vendor_preferences().prices
    components = index_components(cache, code)
    indices = components["index_id"]
    memberships = preferred_memberships(cache, indices, vendor_preferences().members)
    if not funds:
        memberships = memberships[memberships["listing_id"].isin(operating_companies(cache, memberships["listing_id"]))]
    ids = id_list(memberships["listing_id"])
    in_vendors = ", ".join(repr(v) for v in vendors)
    bars = cache.query(
        f"SELECT listing_id, trade_date, close, currency FROM market.price_bar WHERE listing_id IN ({ids}) "
        f"AND vendor IN ({in_vendors}) AND trade_date BETWEEN ? AND ?",
        [start, end],
    )
    currency = universe_currency(cache, indices)
    rates = fx_rates(cache)
    # London's pence beside dividends in pounds; a line once quoted in dollars or euros at the day's rate
    bars = in_currency(bars, rates, currency).drop(columns="currency")
    actions = cache.query(
        f"SELECT listing_id, action_type, ex_date, amount, ratio, currency FROM market.corporate_action "
        f"WHERE listing_id IN ({ids}) AND vendor IN ({in_vendors})"
    )
    # dividends declared abroad (Anglo American's dollars on its London line), in the index's currency
    actions = dividends_in(actions, rates, currency)
    close = close_panel(bars)
    close = close.loc[trading_days(close)]
    days = pd.DatetimeIndex(close.index)
    segment = segments(memberships, components, days, close.columns)
    members = segment.notna()
    listings = cache.query(f"SELECT listing_id, issuer_id, name FROM ref.listing WHERE listing_id IN ({ids})")
    listing_issuer = listings.set_index("listing_id")["issuer_id"]
    facts = cache.query(
        "SELECT issuer_id, item, period_type, period_start, period_end, value, published_at FROM fund.financial_fact "
        f"WHERE vendor = 'sec' AND item IN ({', '.join(repr(i) for i in INDEX_ITEMS)}) "
        f"AND issuer_id IN ({id_list(listing_issuer)})"
    )
    floats = cache.query(
        f"SELECT listing_id, as_of_date, shares FROM market.share_count WHERE kind = 'float' AND listing_id IN ({ids})"
    )
    by_float = float_shares(floats, actions, members)
    by_sec = sec_shares(accounting_items(facts), listing_issuer, actions, members, close)
    shares = by_float.combine_first(by_sec)  # a listing's own float count first, the company's count otherwise
    source = pd.DataFrame(
        np.where(by_float.notna(), "float", np.where(by_sec.notna(), "sec", "")), index=days, columns=close.columns
    )
    return IndexData(
        code=code,
        close=close,
        price_returns=price_returns(close, actions),
        total_returns=total_returns(close, actions),
        members=members,
        segment=segment,
        shares=shares,
        share_source=source,
        market_cap=close * shares,
        names=listings.set_index("listing_id")["name"],
    )


def cap_weighted(returns: pd.DataFrame, market_cap: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """``[return, names, covered]`` per day: the members' returns weighted by yesterday's market value.

    ``covered`` is the share of the day's members that could be weighted (a price, a return and a share count);
    the others are left out and the rest re-weighted.
    """
    cap = market_cap.shift(1).where(members)
    usable = cap.notna() & returns.notna() & (cap > 0)
    weights = cap.where(usable)
    weights = weights.div(weights.sum(axis=1), axis=0)
    out = pd.DataFrame(
        {
            "return": (weights * returns).sum(axis=1, min_count=1),
            "names": usable.sum(axis=1),
            "covered": usable.sum(axis=1) / members.sum(axis=1).replace(0, np.nan),
        }
    )
    return out.iloc[1:]


def equal_weighted(returns: pd.DataFrame, members: pd.DataFrame, rebalance: str = "QE") -> pd.DataFrame:
    """``[return, names]`` per day: every member at the same weight, reset at each period end (quarterly, like RSP).

    Between resets the weights drift with prices; a company leaving the index hands its weight to the ones joining
    (shared equally), or to the others in proportion when nobody joins. A missing return counts as no change.
    """
    r = returns.to_numpy(dtype=float)
    m = members.to_numpy(dtype=bool)
    days = pd.DatetimeIndex(returns.index)
    resets = set(np.flatnonzero(days.isin(rebalance_dates(days, rebalance))))
    n_days = len(days)
    w = np.zeros(r.shape[1])
    ret = np.full(n_days, np.nan)
    names = np.zeros(n_days, dtype=int)
    for t in range(n_days):
        if t > 0 and w.sum() > 0:
            rt = np.nan_to_num(r[t], nan=0.0)
            ret[t] = float(w @ rt)
            w = w * (1.0 + rt)
            w = w / w.sum()
        tomorrow = m[t + 1] if t + 1 < n_days else m[t]
        held = w > 0
        if t in resets or not held.any():
            w = tomorrow / tomorrow.sum() if tomorrow.any() else np.zeros_like(w)
        else:
            leaving, joining = held & ~tomorrow, tomorrow & ~held
            freed = float(w[leaving].sum())
            w[leaving] = 0.0
            if joining.any():
                w[joining] = freed / joining.sum()
            elif w.sum() > 0:
                w = w / w.sum()
        names[t] = int((w > 0).sum())
    out = pd.DataFrame({"return": ret, "names": names}, index=days)
    return out.iloc[1:]


def vendor_returns(
    cache: ResearchCache,
    symbol: str,
    *,
    total: bool,
    vendors: tuple[str, ...] | None = None,
    id_type: str = "YAHOO",
) -> pd.Series:
    """Daily returns of a benchmark found by its code (a Yahoo symbol; ``id_type="RIC"``: LSEG's, as London's levels).

    With dividends (``total``) or prices only; prices come from the first of ``vendors`` (default: the preferred
    benchmark vendors) that has any for it.
    """
    listing = cache.query(
        "SELECT entity_id FROM ref.identifier WHERE entity_type = 'listing' AND id_type = ? AND value = ?",
        [id_type, symbol],
    )
    if listing.empty:
        raise KeyError(f"{symbol} is not loaded - run the benchmark load first")
    lid = int(listing["entity_id"].iloc[0])
    order = vendors or vendor_preferences().benchmarks
    for vendor in order:
        bars = cache.query(
            "SELECT listing_id, trade_date, close, currency FROM market.price_bar WHERE listing_id = ? AND vendor = ?",
            [lid, vendor],
        )
        bars = in_major_units(bars).drop(columns="currency")
        if bars.empty:
            continue
        actions = cache.query(
            "SELECT listing_id, action_type, ex_date, amount, ratio FROM market.corporate_action "
            "WHERE listing_id = ? AND vendor = ?",
            [lid, vendor],
        )
        close = close_panel(bars)
        returns = total_returns(close, actions) if total else price_returns(close, actions)
        return returns[lid].dropna().rename(symbol)
    raise KeyError(f"{symbol}: no prices from {order}")


def tracking(ours: pd.Series, theirs: pd.Series) -> dict[str, float]:
    """How closely ``ours`` follows ``theirs`` on the days both have a return."""
    both = pd.concat({"ours": ours, "theirs": theirs}, axis=1, sort=True).dropna()
    years = len(both) / TRADING_DAYS_PER_YEAR
    growth = (1.0 + both).prod()
    diff = both["ours"] - both["theirs"]
    return {
        "days": float(len(both)),
        "ours_per_year": float(growth["ours"] ** (1 / years) - 1) if years else float("nan"),
        "theirs_per_year": float(growth["theirs"] ** (1 / years) - 1) if years else float("nan"),
        "tracking_error": float(diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "correlation": float(both["ours"].corr(both["theirs"])),
        "beta": float(both["ours"].cov(both["theirs"]) / both["theirs"].var()),
    }


def calendar_years(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Each series' return per calendar year (only over the days all of them have)."""
    both = pd.concat(returns, axis=1, sort=True).dropna()
    years = pd.DatetimeIndex(both.index).year
    return (1.0 + both).groupby(years).prod() - 1.0
