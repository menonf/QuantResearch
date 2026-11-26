"""A stock screen: rank a universe's members on one day by a mix of factors, and show the numbers behind the rank.

    data = load_universe_data(cache, "SP1500", "2025-06-01", "2026-10-22", include_last_day=True)
    table = screen(cache, data, {"value": 1.0, "quality": 1.0, "momentum": 1.0})
    table.head(25)

The scores are the same as a backtest's (:func:`~equity_research.strategies.factor_strategy.factor_scores`), so a screen
shows today's version of a book that was tested over history. Any registered strategy - a factor mix, GARP, QGARP -
is screened from its run (:func:`strategy_screen`), with the weight it would hold beside each score:

    today = snapshot(cache, load_run(engine, run_id=109).config)
    table = strategy_screen(cache, today)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from equity_research.data.cache import ResearchCache
from equity_research.data.reads import id_list
from equity_research.data.universe import UniverseData
from equity_research.strategies import factors as fx
from equity_research.strategies.factor_strategy import StrategyConfig, cache_factors, factor_scores
from equity_research.strategies.runs import Snapshot
from research_core.data.calendars import TRADING_DAYS

#: Ratios shown next to the scores (a reader checks a rank against them).
SHOWN_RATIOS = ("P/E", "P/B", "EV/EBIT", "RoE", "Gross Profit/TA", "Debt/Equity")


def tickers(cache: ResearchCache, listing_ids: pd.Index) -> pd.Series:
    """A short display code per listing: the current RIC's root (``AAPL.OQ`` -> ``AAPL``)."""
    ids = id_list(listing_ids)
    rics = cache.query(
        "SELECT entity_id AS listing_id, value FROM core.identifier WHERE entity_type = 'listing' AND id_type = 'RIC' "
        f"AND entity_id IN ({ids}) ORDER BY valid_from"
    )
    current = rics.drop_duplicates("listing_id", keep="last").set_index("listing_id")["value"]
    return current.str.split(".").str[0].str.replace("^", "", regex=False).reindex(listing_ids)


def screen(
    cache: ResearchCache, data: UniverseData, factors: Mapping[str, float], *, on: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Members on ``on`` (the last decision date by default) ranked by the weighted mix of ``factors``.

    Columns: ``rank``, ``ticker``, ``name``, ``segment``, ``score`` (the mix), one z-score per factor, the
    trailing 12-1 month return and yearly volatility, and :data:`SHOWN_RATIOS`.
    """
    day = pd.Timestamp(on) if on is not None else data.rebalances.max()
    if day not in data.rebalances:
        raise ValueError(f"{day.date()} is not a decision date of this data; use include_last_day or a month end")
    # a factor read from the cache (analysts' consensus revisions) is built as a strategy would build it
    extra = cache_factors(cache, data, StrategyConfig(family="screen", factors=tuple(dict(factors).items())))
    parts = {name: factor_scores(data, name, extra=extra) for name in factors}
    mix = pd.Series(fx.combine(parts, dict(factors)).loc[day])
    return _ranked(cache, data, day, mix, pd.DataFrame({name: panel.loc[day] for name, panel in parts.items()}))


def strategy_screen(cache: ResearchCache, today: Snapshot) -> pd.DataFrame:
    """The members ``today``'s strategy can score, ranked by its own score, with the weight it would hold.

    Columns as :func:`screen`'s, with ``weight`` after ``score`` (negative: short; 0: not held) and the strategy's
    own parts (factor z-scores, GARP's two composites, QGARP's composites) in place of the factors.
    """
    table = _ranked(cache, today.data, today.day, today.score, today.parts)
    table.insert(list(table.columns).index("score") + 1, "weight", today.weights.reindex(table.index).fillna(0.0))
    return table


def _ranked(
    cache: ResearchCache, data: UniverseData, day: pd.Timestamp, score: pd.Series, parts: pd.DataFrame
) -> pd.DataFrame:
    """Members on ``day`` with a ``score``, best first, with ``parts`` and the numbers a reader checks a rank by."""
    members = pd.Series(data.members.loc[day]).astype(bool)
    table = pd.DataFrame({"score": score.reindex(members.index[members.to_numpy()])}).dropna()
    for name in parts.columns:
        table[name] = parts[name].reindex(table.index)
    table["return_12_1"] = fx.momentum(data.adjusted).loc[day].reindex(table.index)
    table["volatility"] = (data.returns.loc[:day].tail(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)).reindex(table.index)
    if not data.ratios.empty and day in data.ratios.index.get_level_values("date"):
        ratios = data.ratios.xs(day, level="date")
        for ratio in SHOWN_RATIOS:
            if ratio in ratios.columns:
                table[ratio] = ratios[ratio].reindex(table.index)
    ids = id_list(table.index)
    names = cache.query(f"SELECT listing_id, name FROM core.listing WHERE listing_id IN ({ids})")
    table.insert(0, "segment", pd.Series(data.segment.loc[day]).reindex(table.index))
    table.insert(0, "name", names.set_index("listing_id")["name"].reindex(table.index))
    table.insert(0, "ticker", tickers(cache, table.index))
    table = table.sort_values("score", ascending=False)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.rename_axis("listing_id")
