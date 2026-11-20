"""The stock traits behind a factor dashboard, after S&P DJI's *Dashboard: S&P 500 Factor Indices*. Pure functions.

    traits = stock_characteristics(data, day, market)     # per stock: volatility, momentum, yields, beta, ...
    tilt = exposures(book, benchmark_weights, factor_z(traits, members))

The dashboard's period returns, risk and holdings comparisons know nothing of equities: they are in
:mod:`research_core.analytics.performance.periods` and :mod:`research_core.analytics.performance.holdings`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_research.data.universe import UniverseData
from equity_research.strategies.factor_strategy import StrategyConfig
from equity_research.strategies.factors import dividend_yield, momentum
from equity_research.strategies.garp import standard_z
from research_core.engine.portfolio import trailing_beta


def _factor_book(score: str, enter: float, exit_: float, weighting: str) -> StrategyConfig:
    """One long-only book of the S&P 500 chosen by ``score``, rebalanced quarterly, before the hold-out."""
    return StrategyConfig(
        family="sp500_factor_dashboard",
        universe="SP500",
        start="2009-12-01",
        end="2024-12-31",
        rebalance="QE",
        factors=((score, 1.0),),
        enter=enter,
        exit_=exit_,
        long_only=True,
        beta_neutral=False,
        weighting=weighting,
    )


#: The default dashboard's factor books, in the spirit of S&P's factor indices (which are not licensed to us): the
#: top fifth by value, quality and momentum weighted by market value; by low volatility, high beta and buybacks
#: (net share issuance) weighted equally; the top sixth (about 80 names) by dividend yield, weighted equally. A
#: notebook finds each in the registry by these settings and registers it the first time.
FACTOR_BOOKS: dict[str, StrategyConfig] = {
    "Value": _factor_book("value", 0.2, 0.25, "market_cap"),
    "Quality": _factor_book("quality", 0.2, 0.25, "market_cap"),
    "Momentum": _factor_book("momentum", 0.2, 0.25, "market_cap"),
    "Low Volatility": _factor_book("low_vol", 0.2, 0.25, "equal"),
    "High Beta": _factor_book("high_beta", 0.2, 0.25, "equal"),
    "High Dividend": _factor_book("dividend_yield", 0.16, 0.2, "equal"),
    "Buyback": _factor_book("issuance", 0.2, 0.25, "equal"),
}


def stock_characteristics(data: UniverseData, day: pd.Timestamp, market: pd.Series) -> pd.DataFrame:
    """``listing_id x trait`` on ``day``: what the dashboard averages over a book.

    Volatility and beta over the trailing year (beta to ``market``), the 12-1 month return, earnings, book and sales
    yields, the trailing year's dividend yield, return on equity, accruals, leverage, and market value in dollars.
    """
    ratios = data.ratios.xs(day, level="date") if day in data.ratios.index.get_level_values("date") else None
    returns = data.returns.loc[:day]

    def ratio(name: str) -> pd.Series:
        return ratios[name] if ratios is not None and name in ratios else pd.Series(dtype=float)

    def inverse(name: str) -> pd.Series:
        r = ratio(name)
        return (1.0 / r).where(r != 0)

    traits = {
        "volatility": returns.tail(252).std() * np.sqrt(252),
        "return_12_1": momentum(data.adjusted.loc[:day]).iloc[-1],
        "beta": trailing_beta(returns.tail(400), market.reindex(returns.tail(400).index)).iloc[-1],
        "earnings_to_price": inverse("P/E"),
        "book_to_price": inverse("P/B"),
        "sales_to_price": inverse("P/S"),
        "dividend_yield": dividend_yield(returns.tail(300), data.adjusted.loc[:day].tail(300)).iloc[-1],
        "roe": ratio("RoE"),
        "accruals": ratio("Accruals/TA"),
        "leverage": ratio("Debt/Equity"),
        "market_cap": data.market_cap.loc[day] if day in data.market_cap.index else pd.Series(dtype=float),
    }
    return pd.DataFrame(traits).reindex(data.members.columns)


def factor_z(traits: pd.DataFrame, members: pd.Series) -> pd.DataFrame:
    """``listing_id x factor``: the dashboard's seven factor scores, standardised among the members.

    Low volatility (minus volatility), momentum (the 12-1 return per unit of volatility), value (earnings, book and
    sales yields), high beta, dividend (the yield), quality (return on equity, minus accruals, minus leverage) and
    small size (minus the log of market value) - each winsorized and standardised like the GARP scores.
    """

    def z(values: pd.Series, *, reverse: bool = False) -> pd.Series:
        frame = values.to_frame().T
        # one row of membership under the same label as the values', or nothing would line up
        on = members.reindex(frame.columns).eq(True).to_numpy()
        member_frame = pd.DataFrame([on], index=frame.index, columns=frame.columns)
        return standard_z(frame, member_frame, reverse=reverse).iloc[0]

    def mean(*parts: pd.Series) -> pd.Series:
        return pd.concat(parts, axis=1).mean(axis=1)

    return pd.DataFrame(
        {
            "Low Volatility": z(traits["volatility"], reverse=True),
            "Momentum": z(traits["return_12_1"] / traits["volatility"]),
            "Value": mean(z(traits["earnings_to_price"]), z(traits["book_to_price"]), z(traits["sales_to_price"])),
            "High Beta": z(traits["beta"]),
            "Dividend": z(traits["dividend_yield"]),
            "Quality": mean(
                z(traits["roe"]),
                z(traits["accruals"], reverse=True),
                z(traits["leverage"].where(traits["leverage"] >= 0), reverse=True),
            ),
            "Small Size": z(np.log(traits["market_cap"].where(traits["market_cap"] > 0)), reverse=True),
        }
    )
