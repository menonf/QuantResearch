"""Books built on the risk model: a universe's own index weights plus tilts by a score (or tilts alone, long-short).

    model = risk_model_for(cache, data)                                  # notebook 4.2's model on data's month ends
    book = tilted_book(data, score, dates, model, risk=0.03, max_active=0.01)

What :func:`~research_core.engine.tilt.tilt` does on each rebalance date, fed from an equity universe: the benchmark
is the universe's members by market value (the same weights as
:func:`~equity_research.analytics.indices.cap_weighted`), and the risk model is notebook 4.2's (BFRE 2.0: market,
GICS sectors, styles, specific risk), estimated on the universe's own returns and forecast on its month ends. A
book that holds its index and spends a budget of active risk on its score, with no sector, size or beta bet, is what
notebook 3's long-only books were not: they bought the top slice of a ranking and weighted it equally or by size,
so most of their active return was the sector, size and beta that slice happened to have.

The risk model needs about a year of factor returns before its first forecast, so a strategy built this way loads its
data :data:`RISK_HISTORY` before its first book.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from equity_research.analytics.risk import risk_model as rm
from equity_research.analytics.risk.risk_data import load_risk_inputs
from equity_research.analytics.risk.risk_styles import substyles
from equity_research.data.cache import ResearchCache
from equity_research.data.universe import UniverseData
from research_core.engine.backtest import run_backtest
from research_core.engine.tilt import tilt

#: How long before a strategy's first book its data must start: the exposures of a year, then a year of factor returns
#: before the first covariance forecast (``RiskModelConfig.min_factor_days``).
RISK_HISTORY = pd.DateOffset(years=2)
#: Words a strategy's ``neutral`` setting may use for groups of the model's factors; any other word is a factor name.
GROUPS = ("sector", "size")


def risk_model_for(
    cache: ResearchCache,
    data: UniverseData,
    *,
    price_vendors: tuple[str, ...] | None = None,
    fundamentals_vendor: str | tuple[str, ...] | None = None,
) -> rm.RiskModel:
    """Notebook 4.2's risk model of ``data``'s universe, with exposures and forecasts on its rebalance dates."""
    inputs = load_risk_inputs(cache, data, price_vendors=price_vendors, fundamentals_vendor=fundamentals_vendor)
    exposures = rm.style_exposures(substyles(inputs), inputs)
    days = set(data.returns.index)
    return rm.build_risk_model(data.returns, exposures, [d for d in data.rebalances if d in days])


def neutral_factors(model: rm.RiskModel, neutral: Sequence[str]) -> list[str]:
    """The model's factors ``neutral`` names: ``sector`` is every industry factor, ``size`` the Size style."""
    out: list[str] = []
    for word in neutral:
        if word == "sector":
            out += list(model.exposures.industries)
        elif word == "size":
            out.append("Size")
        elif word in model.exposures.factors:
            out.append(word)
        else:
            raise ValueError(f"no factor {word!r} to hold neutral; use {GROUPS} or one of {model.exposures.factors}")
    return out


def index_weights(data: UniverseData, day: pd.Timestamp) -> pd.Series:
    """The universe by market value on ``day``: each member's value over the members' total (listings with a value)."""
    members = pd.Series(data.members.loc[day]).reindex(data.market_cap.columns).fillna(False).astype(bool)
    value = pd.Series(data.market_cap.loc[day]).where(members)
    value = value[value > 0].dropna()
    return value / value.sum()


def index_returns(data: UniverseData) -> pd.Series:
    """Daily returns of the universe by market value: :func:`index_weights` on each rebalance date, held until the next.

    ``data.market_cap`` is dated on rebalance dates only, so the index is held as an index fund holds it: weighted by
    market value at each month end, the weights drifting with prices in between (by the same backtest as a strategy,
    without costs). A strategy measured against it is measured against the portfolio its tilts start from.
    """
    book = pd.DataFrame({d: index_weights(data, d) for d in data.rebalances}).T
    book = book.reindex(columns=data.returns.columns).fillna(0.0)
    book.index = pd.DatetimeIndex(book.index)
    return run_backtest(book, data.returns, cost_bps=0.0).gross


def tilted_book(
    data: UniverseData,
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    model: rm.RiskModel,
    *,
    long_only: bool = True,
    risk: float = 0.03,
    max_active: float = 0.02,
    neutral: Sequence[str] = GROUPS,
    gross: float | None = None,
) -> pd.DataFrame:
    """``date x listing_id`` weights on ``dates``: the index plus tilts by ``score`` (or the tilts alone).

    A date the model has no forecast for holds the index (long-only) or nothing (long-short) - which the strategy's
    empty-book check then reports.
    """
    factors = neutral_factors(model, neutral)
    rows = {}
    for d in dates:
        bench = index_weights(data, d)
        if d not in model.covariance or bench.empty:
            rows[d] = bench if long_only else pd.Series(dtype=float)
            continue
        loadings, cov, specific = model.on(d)
        day_score = pd.Series(score.loc[d]) if d in score.index else pd.Series(dtype=float)
        rows[d] = tilt(
            bench,
            day_score,
            loadings,
            cov,
            specific,
            neutral=factors,
            risk=risk,
            max_active=max_active,
            long_only=long_only,
            gross=gross,
        )
    book = pd.DataFrame(rows).T.reindex(columns=data.close.columns).fillna(0.0)
    book.index = pd.DatetimeIndex(book.index)
    return book
