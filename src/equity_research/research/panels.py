"""Point-in-time panels (``date x listing_id``) from canonical facts. Pure functions.

    close = close_panel(bars)                              # unadjusted
    close = close.loc[trading_days(close)]                 # drop a vendor's holiday prints
    adj = split_adjusted(close, actions)                   # comparable across splits
    ret = total_returns(close, actions)                    # price change + dividends
    member = membership_mask(memberships, close.index, close.columns)

Adjusted series are derived here from unadjusted closes and corporate actions - never taken from a vendor - so every
vendor's prices are treated alike and a new split does not rewrite stored history. Prices and returns are adjusted for
splits and for the capital changes that lower the price (spin-offs, rights issues: :func:`return_factor`); share
counts for splits only - a spin-off does not change how many shares a holder has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def close_panel(bars: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """``date x listing_id`` of one bar column from ``[listing_id, trade_date, <column>]`` rows."""
    panel = bars.pivot_table(index="trade_date", columns="listing_id", values=column, aggfunc="last")
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).astype("datetime64[ns]")
    return panel.sort_index()


def trading_days(close: pd.DataFrame, min_share: float = 0.5) -> pd.DatetimeIndex:
    """The days on which at least ``min_share`` of the listings alive (between their first and last price) traded.

    A vendor may print index levels on an exchange holiday; kept, that day would leave every stock without a price and
    cost each one its next day's return.
    """
    priced = close.notna()
    alive = priced.cummax() & priced[::-1].cummax()[::-1]
    share = priced.sum(axis=1) / alive.sum(axis=1).replace(0, float("nan"))
    return pd.DatetimeIndex(close.index[share >= min_share])


def split_factor(actions: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Per day and listing: the product of split ratios with ex-dates **after** the day (1 when none).

    Dividing a day's unadjusted close by it expresses the price in today's shares.
    """
    factor = pd.DataFrame(1.0, index=index, columns=columns)
    splits = actions[(actions["action_type"] == "split") & actions["ratio"].notna() & (actions["ratio"] > 0)]
    days = pd.DatetimeIndex(factor.index)
    for lid, ex_date, ratio in zip(
        splits["listing_id"].astype(int).tolist(),
        pd.to_datetime(splits["ex_date"]).tolist(),
        splits["ratio"].astype(float).tolist(),
        strict=True,
    ):
        if lid in factor.columns:
            factor.loc[days < ex_date, lid] *= ratio
    return factor


def splits_between(actions: pd.DataFrame, listing_ids: pd.Series, after: pd.Series, until: pd.Series) -> np.ndarray:
    """Per row: the product of the listing's split ratios with ex-dates in ``(after, until]`` (1.0 when none).

    Multiplying a share count dated ``after`` by it gives the count in the shares of ``until``.
    """
    rows = pd.DataFrame(
        {
            "row": np.arange(len(listing_ids)),
            "listing_id": pd.Series(listing_ids).astype(int).to_numpy(),
            "after": pd.to_datetime(pd.Series(after)).to_numpy(),
            "until": pd.to_datetime(pd.Series(until)).to_numpy(),
        }
    )
    splits = actions[(actions["action_type"] == "split") & actions["ratio"].notna() & (actions["ratio"] > 0)]
    splits = splits.assign(listing_id=splits["listing_id"].astype(int), ex_date=pd.to_datetime(splits["ex_date"]))[
        ["listing_id", "ex_date", "ratio"]
    ]
    hit = rows.merge(splits, on="listing_id")
    hit = hit[(hit["ex_date"] > hit["after"]) & (hit["ex_date"] <= hit["until"])]
    out = np.ones(len(rows))
    product = hit.groupby("row")["ratio"].prod()
    out[product.index.to_numpy()] = product.to_numpy(dtype=float)
    return out


def latest_counts(
    counts: pd.DataFrame,
    actions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    listings: pd.Index,
    max_age_days: int = 400,
) -> pd.DataFrame:
    """``date x listing_id``: each listing's latest share count dated strictly before each date, in that date's shares.

    ``counts``: ``[listing_id, as_of_date, shares]`` - a count of one share line (outstanding, or the free float),
    in the shares of its own date, so only splits after it adjust it. A count older than ``max_age_days`` is too
    stale to use.
    """
    days = pd.DataFrame({"date": pd.DatetimeIndex(dates).astype("datetime64[ns]")})
    c = counts.assign(
        listing_id=counts["listing_id"].astype(int),
        as_of_date=pd.to_datetime(counts["as_of_date"]).astype("datetime64[ns]"),
    )
    c = c[c["listing_id"].isin(listings)].sort_values("as_of_date")
    if c.empty:
        return pd.DataFrame(np.nan, index=dates, columns=listings)
    grid = days.merge(pd.DataFrame({"listing_id": c["listing_id"].unique()}), how="cross").sort_values("date")
    hit = pd.merge_asof(
        grid, c, left_on="date", right_on="as_of_date", by="listing_id", allow_exact_matches=False
    ).dropna(subset=["shares"])
    hit = hit[hit["as_of_date"] >= hit["date"] - pd.Timedelta(days=max_age_days)]
    hit["shares"] = hit["shares"].to_numpy() * splits_between(
        actions, hit["listing_id"], hit["as_of_date"], hit["date"]
    )
    panel = hit.pivot_table(index="date", columns="listing_id", values="shares", aggfunc="last")
    return panel.reindex(index=dates, columns=listings)


#: The version of the way returns and adjusted prices are derived here. Cached research artifacts are keyed by it
#: (:mod:`~equity_research.research.store`), so a change rebuilds them instead of reading results of the old method.
#: 1: splits only; 2 (2026-11-14): spin-offs and rights issues too (:func:`return_factor`).
RETURNS_VERSION = 2

#: How many trading days after the vendor's date a spin-off's price drop may fall on. LSEG dates many US spin-offs a
#: day early (the distribution date, not the ex-date) and some London ones two (Prudential's demerger of M&G, 2019):
#: of the drops of 5 % or more the price confirms, 99 % of London's and 93 % of the US's fall on the vendor's date or
#: the two trading days after; further out a match is as likely chance.
SPINOFF_DAYS_AFTER = 2
#: How closely the day's move must match a spin-off's drop to confirm it: within half the drop (log). Unconfirmed, it
#: is left out, like a split the price does not confirm - applied on the wrong day it would book a gain no holder had
#: (HSBC's rights issue of 2009, dated by LSEG six trading days before its price fell).
SPINOFF_TOLERANCE = 0.5


def spinoff_days(actions: pd.DataFrame, close: pd.DataFrame, days_after: int = SPINOFF_DAYS_AFTER) -> pd.DataFrame:
    """``[listing_id, ex_date, ratio]``: the spin-offs that lower the price, each on the day its price drops.

    ``spinoff`` rows are every capital change near 1 (a spin-off, a rights issue, a bonus issue): ``ratio`` > 1 lowers
    the price by ``1 / ratio - 1``. Each is placed on the vendor's date or one of the ``days_after`` trading days after
    it, whichever the split-adjusted price moves nearest that drop on, and kept only when that move confirms it (within
    :data:`SPINOFF_TOLERANCE` of the drop); one dated before the panel's second day is kept on its own date (it scales
    every day alike). Rows with ``ratio`` < 1 - London's share consolidations that come with a special payout - are
    left out: the price does not rise by them (checked 2026-11-14: applying them made 80 % of their days' returns
    worse).
    """
    rows = actions[(actions["action_type"] == "spinoff") & actions["ratio"].notna() & (actions["ratio"] > 1)]
    rows = rows[rows["listing_id"].astype(int).isin(close.columns)]
    out = pd.DataFrame(
        {
            "listing_id": rows["listing_id"].astype(int).to_numpy(),
            "ex_date": pd.to_datetime(rows["ex_date"]).to_numpy(),
            "ratio": rows["ratio"].astype(float).to_numpy(),
        }
    )
    if out.empty:
        return out
    adjusted = split_adjusted(close[out["listing_id"].unique()], actions, spinoffs=False)
    moves = np.log(adjusted / adjusted.shift(1))
    days = pd.DatetimeIndex(close.index)
    placed: list[pd.Timestamp | None] = []
    for lid, ex_date, ratio in zip(out["listing_id"], out["ex_date"], out["ratio"], strict=True):
        first = int(days.searchsorted(ex_date))
        if first == 0:  # before the panel: it scales every day alike, and the first day has no move to compare
            placed.append(pd.Timestamp(ex_date))
            continue
        move = moves[lid].reindex(days[first : first + 1 + days_after]).dropna()
        drop = -np.log(ratio)
        miss = np.abs(move.to_numpy() - drop)
        best = int(np.argmin(miss)) if len(move) else -1
        placed.append(move.index[best] if best >= 0 and miss[best] < SPINOFF_TOLERANCE * abs(drop) else None)
    out = out.assign(ex_date=pd.DatetimeIndex(placed))
    return out[out["ex_date"].notna()].reset_index(drop=True)


def return_factor(close: pd.DataFrame, actions: pd.DataFrame, spinoffs: bool = True) -> pd.DataFrame:
    """Per day and listing: the split factor (:func:`split_factor`) times the spin-off ratios after the day.

    Dividing a day's close by it gives a price whose day-to-day changes are the holder's: a spin-off's value is
    treated as reinvested in the parent on its ex-date, the way index providers and CRSP treat it.
    """
    index = pd.DatetimeIndex(close.index)
    factor = split_factor(actions, index, close.columns)
    if not spinoffs:
        return factor
    days = pd.DatetimeIndex(factor.index)
    for lid, ex_date, ratio in spinoff_days(actions, close).itertuples(index=False, name=None):
        factor.loc[days < ex_date, lid] *= ratio
    return factor


def split_adjusted(close: pd.DataFrame, actions: pd.DataFrame, spinoffs: bool = True) -> pd.DataFrame:
    """Closes expressed in today's shares (a 4-for-1 split divides earlier closes by 4).

    Closes before a spin-off are scaled down by its value too, unless ``spinoffs=False`` (splits only).
    """
    if not spinoffs:
        return close / split_factor(actions, pd.DatetimeIndex(close.index), close.columns)
    return close / return_factor(close, actions)


def dividends_panel(actions: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Cash dividend per share (in that day's shares) on each ex-date; 0 elsewhere."""
    div = actions[(actions["action_type"] == "cash_dividend") & actions["amount"].notna()]
    panel = div.pivot_table(index="ex_date", columns="listing_id", values="amount", aggfunc="sum")
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).astype("datetime64[ns]")
    return panel.reindex(index=index, columns=columns).fillna(0.0)


def total_returns(close: pd.DataFrame, actions: pd.DataFrame, spinoffs: bool = True) -> pd.DataFrame:
    """Daily total returns: ``(adj_t + dividend_t / factor_t) / adj_{t-1} - 1``, ``factor`` from :func:`return_factor`.

    NaN where either day has no price - a gap is not a zero return. ``spinoffs=False``: adjusted for splits only, so
    a spin-off's day books the parent's drop as a loss (the returns before 2026-11-14).
    """
    index = pd.DatetimeIndex(close.index)
    factor = return_factor(close, actions, spinoffs)
    adjusted = close / factor
    dividends = dividends_panel(actions, index, close.columns) / factor
    return (adjusted + dividends) / adjusted.shift(1) - 1.0


def price_returns(close: pd.DataFrame, actions: pd.DataFrame, spinoffs: bool = True) -> pd.DataFrame:
    """Daily adjusted price returns (no dividends)."""
    return split_adjusted(close, actions, spinoffs).pct_change(fill_method=None)


def membership_mask(memberships: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """True where a listing was a member on that day; ``[listing_id, start_date, end_date]`` with exclusive ends."""
    mask = pd.DataFrame(False, index=index, columns=columns)
    days = pd.DatetimeIndex(mask.index)
    for lid, start, end in zip(
        memberships["listing_id"].astype(int).tolist(),
        pd.to_datetime(memberships["start_date"]).tolist(),
        pd.to_datetime(memberships["end_date"]).tolist(),
        strict=True,
    ):
        if lid not in mask.columns:
            continue
        inside = (days >= start) & (days < end) if pd.notna(end) else days >= start
        mask.loc[inside, lid] = True
    return mask


def by_listing(issuer_panel: pd.DataFrame, listing_issuer: pd.Series) -> pd.DataFrame:
    """Expand a ``(date, issuer_id)`` panel to ``(date, listing_id)`` - each listing gets its issuer's values."""
    mapping = listing_issuer.dropna().astype(int).rename("issuer_id").rename_axis("listing_id").reset_index()
    out = issuer_panel.reset_index().merge(mapping, on="issuer_id", how="inner")
    return out.drop(columns="issuer_id").set_index(["date", "listing_id"]).sort_index()
