"""What a model learns from: features of each company on each decision date, from prices, volume and accounts.

    dollars = dollar_volume(cache, data)                       # daily traded value (the one read from the cache)
    price = price_features(data, dollars, dates)               # momentum, reversal, volatility, liquidity, tails
    accounts = fundamental_features(data, dates)               # the book's ratios, size, issuance, surprise, growth
    calendar = calendar_features(dates)                        # the same for every company on a date
    long = stack(price | accounts, data.members)               # (date, listing_id) x feature, members only

The families follow *Machine Learning for Trading* (chapter 8); every feature is known at the close of its date,
built from daily total returns (``returns``), split-adjusted prices and traded value, and the accounts as published.
Each is a raw value; how features are scaled (percentiles, z-scores) is the model's choice, not the feature's.

1. **Momentum and reversal** - returns over 12 and 6 months skipping the latest month (Jegadeesh and Titman, 1993),
   the latest month and week (short-term reversal, Jegadeesh 1990; Lehmann 1990), and residual momentum (the market
   taken out, Blitz, Huij and Martens 2011).
2. **Risk** - volatility over a month and a year, idiosyncratic volatility over three months (the part the market does
   not explain, Ang et al. 2006), beta over a year, the largest daily return in a month (Bali, Cakici and Whitelaw
   2011) and the skewness of daily returns over a year.
3. **Liquidity** - average daily traded value, turnover (traded value over market value), Amihud's (2002) illiquidity
   (the average absolute return per million traded) and the share of days without a price change (Lesmond, Ogden
   and Trzcinka 1999). Spreads come from closes alone: Roll's (1984) estimate, twice the square root of minus the
   covariance of consecutive returns. Our bars have no daily high and low, so Corwin and Schultz's (2012) high-low
   estimate is not possible.
4. **Accounts** - the 18 ratios of Damon Lee's book (``ml.BOOK_CLIPS``, clipped), size (market value), the change in
   share count over a year, the latest standardised earnings surprise, and growth of earnings and sales per share.
5. **Calendar** - month of the year (as a point on a circle, so December lies next to January), January, and the
   turn of the month. They are the same for every company on a date: alone they rank nobody, so they matter only to a
   model that combines them with other features.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from equity_research.research import factors as fx
from equity_research.research import ml
from equity_research.research.cache import ResearchCache
from equity_research.research.data import PRICE_KEY, UniverseData, in_major_units
from equity_research.research.portfolio import trailing_beta
from equity_research.research.reads import first_preferred, id_list
from equity_research.settings import vendor_preferences

#: Trading days in a week, a month, three months and a year.
WEEK, MONTH, QUARTER, YEAR = 5, 21, 63, 252
#: The price-and-volume features, in the order :func:`price_features` returns them.
PRICE_FEATURES = (
    "mom_12_1",
    "mom_6_1",
    "ret_1m",
    "ret_1w",
    "resid_mom",
    "vol_1m",
    "vol_1y",
    "idio_vol_3m",
    "beta_1y",
    "max_ret_1m",
    "skew_1y",
    "dollar_volume_1m",
    "turnover_1m",
    "amihud_1m",
    "zero_days_3m",
    "roll_spread_3m",
)
#: The accounts features beyond the book's ratios.
ACCOUNT_EXTRAS = ("size", "share_change", "earnings_surprise", "eps_growth", "sales_growth")


def dollar_volume(
    cache: ResearchCache, data: UniverseData, price_vendors: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """``day x listing_id``: each day's close times volume, in major units, from the preferred price vendor."""
    vendors = price_vendors or vendor_preferences().prices
    days = pd.DatetimeIndex(data.returns.index)
    bars = cache.query(
        f"SELECT listing_id, vendor, trade_date, close, volume, currency FROM market.price_bar "
        f"WHERE listing_id IN ({id_list(data.returns.columns)}) AND trade_date BETWEEN ? AND ?",
        [days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d")],
    )
    bars = first_preferred(in_major_units(bars[bars["vendor"].isin(vendors)]), vendors, PRICE_KEY)
    bars = bars.assign(dollars=bars["close"] * bars["volume"], trade_date=pd.to_datetime(bars["trade_date"]))
    wide = bars.pivot_table(index="trade_date", columns="listing_id", values="dollars", aggfunc="last")
    return wide.reindex(index=days, columns=data.returns.columns)


def market_return(data: UniverseData) -> pd.Series:
    """The members' equal-weighted daily return: the market the risk features measure against."""
    members = data.members.reindex(index=data.returns.index, columns=data.returns.columns, fill_value=False)
    return data.returns.where(members.astype(bool)).mean(axis=1)


def roll_spread(returns: pd.DataFrame, window: int = QUARTER) -> pd.DataFrame:
    """Roll's (1984) spread: ``2 sqrt(-cov(r_t, r_t-1))`` over ``window`` days; 0 where the covariance is positive."""
    cov = returns.rolling(window, min_periods=window // 2).cov(returns.shift(1))
    out = 2 * (-cov).clip(lower=0.0).pow(0.5)
    return out.where(cov.notna())


def amihud(returns: pd.DataFrame, dollars: pd.DataFrame, window: int = MONTH) -> pd.DataFrame:
    """Amihud's (2002) illiquidity: the average of ``|return| / traded value`` (per million) over ``window`` days."""
    ratio = returns.abs() / (dollars.where(dollars > 0) / 1e6)
    out: pd.DataFrame = ratio.rolling(window, min_periods=window // 2).mean()
    return out


def idiosyncratic_volatility(returns: pd.DataFrame, market: pd.Series, window: int = QUARTER) -> pd.DataFrame:
    """The annualised volatility of the part of each daily return the market does not explain, over ``window`` days."""
    var = returns.rolling(window, min_periods=window // 2).var()
    cov = returns.rolling(window, min_periods=window // 2).cov(market)
    mvar = market.rolling(window, min_periods=window // 2).var()
    residual = (var - cov.pow(2).div(mvar, axis=0)).clip(lower=0.0)
    out = (residual * YEAR).pow(0.5)
    return out


def price_features(data: UniverseData, dollars: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """:data:`PRICE_FEATURES` on ``dates`` (``date x listing_id``), each from data up to and including its date."""
    r, p = data.returns, data.adjusted
    market = market_return(data)
    active = r.notna()
    traded = dollars.where(active)

    def on(panel: pd.DataFrame) -> pd.DataFrame:
        """A daily panel sampled on the decision dates."""
        return panel.reindex(index=dates, columns=r.columns)

    dv = traded.rolling(MONTH, min_periods=MONTH // 2).mean()
    return {
        "mom_12_1": on(fx.momentum(p, YEAR, MONTH)),
        "mom_6_1": on(fx.momentum(p, 126, MONTH)),
        "ret_1m": on(p / p.shift(MONTH) - 1.0),
        "ret_1w": on(p / p.shift(WEEK) - 1.0),
        "resid_mom": fx.residual_momentum(r, market, dates).reindex(columns=r.columns),
        "vol_1m": on(r.rolling(MONTH, min_periods=MONTH // 2).std() * np.sqrt(YEAR)),
        "vol_1y": on(r.rolling(YEAR, min_periods=YEAR // 2).std() * np.sqrt(YEAR)),
        "idio_vol_3m": on(idiosyncratic_volatility(r, market)),
        "beta_1y": on(trailing_beta(r, market)),
        "max_ret_1m": on(r.rolling(MONTH, min_periods=MONTH // 2).max()),
        "skew_1y": on(r.rolling(YEAR, min_periods=YEAR // 2).skew()),
        "dollar_volume_1m": on(dv),
        "turnover_1m": on(dv) / data.market_cap.reindex(index=dates, columns=r.columns).where(lambda v: v > 0),
        "amihud_1m": on(amihud(r, traded)),
        "zero_days_3m": on((r == 0).where(active).rolling(QUARTER, min_periods=QUARTER // 2).mean()),
        "roll_spread_3m": on(roll_spread(r)),
    }


def fundamental_features(data: UniverseData, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """The book's 18 ratios (clipped) and :data:`ACCOUNT_EXTRAS` on ``dates``, as published by then."""
    columns = data.returns.columns
    book = ml.book_features(data.ratios)
    ratios = {
        name: book[name].unstack("listing_id").reindex(index=dates, columns=columns)  # noqa: PD010
        for name in book.columns
    }
    extras = {
        "size": data.market_cap,
        "share_change": data.share_change,
        "earnings_surprise": data.earnings_surprise,
        "eps_growth": data.eps_growth,
        "sales_growth": data.sales_growth,
    }
    return ratios | {name: extras[name].reindex(index=dates, columns=columns) for name in ACCOUNT_EXTRAS}


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """``date x feature``: the month as a point on a circle, January, and whether the date is a month's last day."""
    d = pd.DatetimeIndex(dates)
    angle = 2 * np.pi * (d.month - 1) / 12
    nxt = pd.Series(d).shift(-1)
    month_end = (pd.Series(d.month) != nxt.dt.month).to_numpy()
    return pd.DataFrame(
        {
            "month_sin": np.sin(angle),
            "month_cos": np.cos(angle),
            "january": (d.month == 1).astype(float),
            "month_end": month_end.astype(float),
        },
        index=d,
    )


def stack(panels: Mapping[str, pd.DataFrame], members: pd.DataFrame) -> pd.DataFrame:
    """``(date, listing_id) x feature`` for the members on each panel date (the dates of the first panel)."""
    first = next(iter(panels.values()))
    on = members.reindex(index=first.index, columns=first.columns, fill_value=False).astype(bool)
    rows, cols = np.nonzero(on.to_numpy())
    index = pd.MultiIndex.from_arrays([on.index[rows], on.columns[cols]], names=["date", "listing_id"])
    values = {
        name: panel.reindex(index=first.index, columns=first.columns).to_numpy(dtype=float)[rows, cols]
        for name, panel in panels.items()
    }
    return pd.DataFrame(values, index=index)


def unstack(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``feature -> date x listing_id``: the panels of a :func:`stack`-ed frame (members only; others missing)."""
    return {name: long[name].unstack("listing_id") for name in long.columns}  # noqa: PD010 - long to wide
