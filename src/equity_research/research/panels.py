"""Point-in-time panels (``date x listing_id``) from canonical facts. Pure functions.

    close = close_panel(bars)                              # unadjusted
    close = close.loc[trading_days(close)]                 # drop a vendor's holiday prints
    adj = split_adjusted(close, actions)                   # comparable across splits
    ret = total_returns(close, actions)                    # price change + dividends
    member = membership_mask(memberships, close.index, close.columns)

Adjusted series are derived here from unadjusted closes and corporate actions - never taken from a vendor - so every
vendor's prices are treated alike and a new split does not rewrite stored history.
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


def split_adjusted(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Closes expressed in today's shares (a 4-for-1 split divides earlier closes by 4)."""
    return close / split_factor(actions, pd.DatetimeIndex(close.index), close.columns)


def dividends_panel(actions: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Cash dividend per share (in that day's shares) on each ex-date; 0 elsewhere."""
    div = actions[(actions["action_type"] == "cash_dividend") & actions["amount"].notna()]
    panel = div.pivot_table(index="ex_date", columns="listing_id", values="amount", aggfunc="sum")
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).astype("datetime64[ns]")
    return panel.reindex(index=index, columns=columns).fillna(0.0)


def total_returns(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Daily total returns: ``(adj_t + dividend_t / factor_t) / adj_{t-1} - 1``.

    NaN where either day has no price - a gap is not a zero return.
    """
    index = pd.DatetimeIndex(close.index)
    factor = split_factor(actions, index, close.columns)
    adjusted = close / factor
    dividends = dividends_panel(actions, index, close.columns) / factor
    return (adjusted + dividends) / adjusted.shift(1) - 1.0


def price_returns(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Daily split-adjusted price returns (no dividends)."""
    return split_adjusted(close, actions).pct_change(fill_method=None)


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
