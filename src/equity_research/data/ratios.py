"""Valuation and quality ratios at each decision date, from point-in-time accounting items and prices.

    fund = by_listing(as_of(accounting_items(facts), dates, with_period_end=True), listing_issuer)
    ratios = ratio_panel(fund, close, factor)          # (date, listing_id) x RATIOS
    values = market_values(fund, close, factor)        # (date, listing_id) x [shares, market_cap]

Prices are the decision date's, so price multiples are always current (no stored period-end market caps):

* **market cap** = the company's value from its listings' own share counts where the caller supplies it
  (``listed_value``: :func:`~equity_research.data.universe.own_counts`); else the listing's unadjusted close on (or up
  to :data:`PRICE_LOOKBACK_DAYS` before) the date x the share count of the latest filing, adjusted for splits between
  that filing's publication and the date;
* **enterprise value** = market cap + total debt - cash.

The 18 ratios keep the names of the previous project's set so they can be checked against the vendor's; the magic
formula's two inputs (Greenblatt) are added: ``EBIT/EV`` and ``ROC`` = EBIT / (net working capital + fixed assets),
gross profitability (Novy-Marx): ``Gross Profit/TA``, and two from the cash-flow statement: ``FCF/EV`` (operating cash
flow less capital spending, per dollar of enterprise value) and ``Accruals/TA`` (profit not backed by cash, Sloan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_LOOKBACK_DAYS = 5

RATIOS = [
    "EV/EBIT",
    "Op. In./(NWC+FA)",
    "P/E",
    "P/B",
    "P/S",
    "Op. In./Interest Expense",
    "Working Capital Ratio",
    "RoE",
    "ROCE",
    "Debt/Equity",
    "Debt Ratio",
    "Cash Ratio",
    "Asset Turnover",
    "Gross Profit Margin",
    "(CA-CL)/TA",
    "RE/TA",
    "EBIT/TA",
    "Book Equity/TL",
    "EBIT/EV",
    "ROC",
    "Gross Profit/TA",
    "FCF/EV",
    "Accruals/TA",
]


def lookup(panel: pd.DataFrame, dates: pd.Series, listings: pd.Series, *, ffill_days: int = 0) -> np.ndarray:
    """``panel[date, listing]`` for each pair; with ``ffill_days`` the last value up to that many rows earlier."""
    stacked_index = pd.DatetimeIndex(panel.index)
    wanted = pd.DatetimeIndex(pd.to_datetime(dates))
    grid = panel.reindex(stacked_index.union(wanted).unique().sort_values())
    if ffill_days:
        grid = grid.ffill(limit=ffill_days)
    rows = grid.index.get_indexer(wanted)
    cols = grid.columns.get_indexer(pd.Index(listings))
    values = grid.to_numpy(dtype=float)
    out = np.full(len(rows), np.nan)
    ok = (rows >= 0) & (cols >= 0)
    out[ok] = values[rows[ok], cols[ok]]
    return out


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    """``a / b`` with zero denominators as missing (never +/- infinity)."""
    return a / b.replace(0, np.nan)


#: A market value this many times both the company's yearly revenue and its book equity comes from a mis-scaled
#: share count, not from the market (the largest real multiples are tens: NVIDIA at its peak was ~40x and ~60x).
IMPLAUSIBLE_MULTIPLE = 100.0


def plausible_market_cap(mkt_cap: pd.Series, revenue: pd.Series, equity: pd.Series) -> pd.Series:
    """``mkt_cap`` where it is believable, missing where it is not.

    Not believable: above :data:`IMPLAUSIBLE_MULTIPLE` times both revenue and book equity, both positive (a company
    with no revenue or negative equity is never judged by it).
    """
    too_big = (
        (revenue > 0)
        & (equity > 0)
        & (mkt_cap > IMPLAUSIBLE_MULTIPLE * revenue)
        & (mkt_cap > IMPLAUSIBLE_MULTIPLE * equity)
    )
    return mkt_cap.where(~too_big)


def market_values(fund: pd.DataFrame, close: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """``(date, listing_id)`` x ``[shares, market_cap]``: the company's share count and market value on each date.

    The count is in the shares of the date; the value is at the date's price, missing where implausible
    (:func:`plausible_market_cap`) - or, where ``fund`` has a ``listed_value`` column (the company's value from its
    listings' own share counts), that value. ``fund`` needs ``shares``, ``known_at:shares``, ``revenue`` and ``equity``;
    ``close`` and ``factor`` are as for :func:`ratio_panel`.
    """
    f = fund.reset_index()
    price = lookup(close, f["date"], f["listing_id"], ffill_days=PRICE_LOOKBACK_DAYS)
    split_since = lookup(factor, f["known_at:shares"], f["listing_id"], ffill_days=PRICE_LOOKBACK_DAYS) / lookup(
        factor, f["date"], f["listing_id"], ffill_days=PRICE_LOOKBACK_DAYS
    )
    shares = f["shares"] * pd.Series(split_since, index=f.index).fillna(1.0)
    mkt_cap = plausible_market_cap(pd.Series(price, index=f.index) * shares, f["revenue"], f["equity"])
    # the company's value from its listings' own share counts, where the caller has it, beats one from its accounts
    listed = _optional(f, "listed_value")
    out = pd.DataFrame({"shares": shares, "market_cap": listed.where(listed.notna(), mkt_cap)})
    out.index = pd.MultiIndex.from_frame(f[["date", "listing_id"]])
    return out


def _optional(f: pd.DataFrame, name: str) -> pd.Series:
    """Column ``name`` of ``f``, or all missing - an item only some vendors report."""
    return f[name] if name in f.columns else pd.Series(np.nan, index=f.index)


def ratio_panel(fund: pd.DataFrame, close: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """``(date, listing_id)`` x :data:`RATIOS`.

    Args:
        fund: ``(date, listing_id)`` x accounting items (:data:`~equity_research.data.accounting.ITEMS`) with a
            ``known_at:shares`` column (``as_of(..., with_period_end=True)``): the count is in that day's shares.
        close: ``date x listing_id`` unadjusted closes.
        factor: ``date x listing_id`` split factors (:func:`~equity_research.data.panels.split_factor`).
    """
    f = fund.reset_index()
    mkt_cap = pd.Series(market_values(fund, close, factor)["market_cap"].to_numpy(), index=f.index)
    ev = mkt_cap + f["total_debt"].fillna(0.0) - f["cash"].fillna(0.0)
    nwc = f["current_assets"] - f["current_liabilities"]
    out = pd.DataFrame(index=f.index)
    out["EV/EBIT"] = _div(ev, f["ebit"])
    out["Op. In./(NWC+FA)"] = _div(f["operating_income"], nwc + f["fixed_assets"])
    out["P/E"] = _div(mkt_cap, f["net_income"])
    out["P/B"] = _div(mkt_cap, f["equity"])
    out["P/S"] = _div(mkt_cap, f["revenue"])
    out["Op. In./Interest Expense"] = _div(f["operating_income"], f["interest_expense"].abs())
    out["Working Capital Ratio"] = _div(f["current_assets"], f["current_liabilities"])
    out["RoE"] = _div(f["net_income"], f["equity"])
    out["ROCE"] = _div(f["ebit"], f["equity"] + f["total_debt"])
    out["Debt/Equity"] = _div(f["total_debt"], f["equity"])
    out["Debt Ratio"] = _div(f["total_debt"], f["total_assets"])
    out["Cash Ratio"] = _div(f["cash"], f["current_liabilities"])
    out["Asset Turnover"] = _div(f["revenue"], f["total_assets"])
    out["Gross Profit Margin"] = _div(f["gross_profit"], f["revenue"])
    out["(CA-CL)/TA"] = _div(nwc, f["total_assets"])
    out["RE/TA"] = _div(f["retained_earnings"], f["total_assets"])
    out["EBIT/TA"] = _div(f["ebit"], f["total_assets"])
    out["Book Equity/TL"] = _div(f["equity"], f["total_liabilities"])
    # a yield over a negative EV (cash above value and debt) flips its sign: a profitable cash box would rank dearest
    out["EBIT/EV"] = _div(f["ebit"], ev.where(ev > 0))
    out["ROC"] = _div(f["ebit"], nwc + f["fixed_assets"])
    out["Gross Profit/TA"] = _div(f["gross_profit"], f["total_assets"])
    cash_flow, capex = _optional(f, "operating_cash_flow"), _optional(f, "capex")
    # a company that reports no capital spending (most banks) has no free cash flow here, not all of it free
    out["FCF/EV"] = _div(cash_flow - capex, ev.where(ev > 0))  # a negative EV (cash above value) flips the sign
    out["Accruals/TA"] = _div(f["net_income"] - cash_flow, f["total_assets"])
    out.index = pd.MultiIndex.from_frame(f[["date", "listing_id"]])
    return out[RATIOS]
