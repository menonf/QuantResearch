"""What the risk model reads beyond a universe's backtest data, from the local research cache.

    data = load_universe_data(cache, "SP1500", "2005-01-01", "2024-12-31")
    inputs = load_risk_inputs(cache, data)              # + dollar volume, five years of accounts, industries

Choices by vendor follow the project's preferences, as in :mod:`~equity_research.research.data`: volume from the
first price vendor with a bar, accounts from the first fundamentals vendor that answers for the company on the date.
Industries are today's classification (the history starts the day it was first recorded, so earlier years use
today's sector - a company that moved, Alphabet into Communication Services in 2018, is in its new one throughout).
"""

from __future__ import annotations

import pandas as pd

from equity_research.research.cache import ResearchCache
from equity_research.research.classification import classification_on
from equity_research.research.data import (
    PRICE_KEY,
    UniverseData,
    company_values,
    in_major_units,
    items_years_before,
    vendor_items,
)
from equity_research.research.reads import as_vendors, financial_facts, first_preferred, id_list, listing_issuers
from equity_research.research.risk_styles import Accounts, RiskInputs
from equity_research.settings import vendor_preferences

#: Items read on each exposure date, and those also read one to four years before it (averages, trends, ROE).
ITEMS_NOW = (
    "total_assets",
    "revenue",
    "operating_cash_flow",
    "equity",
    "total_debt",
    "cash",
    "net_income",
    "operating_income",
    "ebit",
    "interest_expense",
    "current_liabilities",
)
ITEMS_BEFORE = ("total_assets", "revenue", "operating_cash_flow", "equity")
YEARS = 5


def load_risk_inputs(
    cache: ResearchCache,
    data: UniverseData,
    *,
    level: str = "sector",
    price_vendors: tuple[str, ...] | None = None,
    fundamentals_vendor: str | tuple[str, ...] | None = None,
) -> RiskInputs:
    """The risk model's inputs for ``data``'s listings, with exposure dates on its rebalance dates (month ends).

    ``level`` is the classification level the industry factors use: GICS ``sector`` by default, because the vendor
    gives only the sector for many companies that left the indices - a finer level would leave them unclassified.
    """
    prefs = vendor_preferences()
    price_vendors = price_vendors or prefs.prices
    vendors = as_vendors(fundamentals_vendor or prefs.fundamentals)
    listings = data.returns.columns
    dates = data.rebalances
    listing_issuer = listing_issuers(cache, listings)

    days = pd.DatetimeIndex(data.returns.index)
    bars = cache.query(
        f"SELECT listing_id, vendor, trade_date, close, volume, currency FROM market.price_bar "
        f"WHERE listing_id IN ({id_list(listings)}) AND trade_date BETWEEN ? AND ?",
        [days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d")],
    )
    bars = first_preferred(in_major_units(bars[bars["vendor"].isin(price_vendors)]), price_vendors, PRICE_KEY)
    bars = bars.assign(dollars=bars["close"] * bars["volume"], trade_date=pd.to_datetime(bars["trade_date"]))
    dollar_volume = bars.pivot_table(index="trade_date", columns="listing_id", values="dollars", aggfunc="last")

    items = vendor_items(financial_facts(cache, listing_issuer, vendors), vendors, data.currency)
    years = [
        items_years_before(items, dates, listing_issuer, listings, ITEMS_NOW if k == 0 else ITEMS_BEFORE, years=k)
        for k in range(YEARS)
    ]

    classes = classification_on(cache)[level]
    industry = listing_issuer.map(classes).reindex(listings).rename("industry")
    return RiskInputs(
        returns=data.returns,
        adjusted=data.adjusted,
        members=data.members,
        dollar_volume=dollar_volume.reindex(index=days, columns=listings),
        dates=dates,
        listing_value=data.market_cap.reindex(columns=listings),
        company_value=company_values(data.market_cap.reindex(columns=listings), listing_issuer),
        accounts=Accounts(years),
        industry=industry,
        listing_issuer=listing_issuer,
    )
