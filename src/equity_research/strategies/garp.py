"""Growth at a reasonable price: the S&P 500 GARP strategy, rebuilt from its published rules.

    result = run_garp(cache, GarpConfig(start="2009-01-01", end="2024-12-31"), engine=engine)

After S&P Dow Jones Indices, *Indexing GARP Strategies: A Practitioner's Guide* (July 2019), and the S&P GARP Indices
methodology for the schedule. On each reference date - the last business day of May and November - among the S&P
500's members of the day:

1. **Ratios**: three-year growth of earnings per share and of sales per share (compound, per year); return on equity;
   financial leverage (total debt over book equity); earnings to price.
2. **Z-scores**: each ratio winsorized at its 2.5 and 97.5 percentiles, then standardised with the members' mean and
   standard deviation (leverage reversed: less debt scores higher). A return on equity that is positive only because
   a loss is divided by negative equity, and leverage over negative equity, take the worst score of the day - the
   paper assigns "the z-score of the 2.5 percentile ranked security", which only makes sense as a penalty.
3. **Two composites**, each an average of what is available and capped at +/-4: growth (the two growth z-scores) and
   quality & value (leverage, return on equity and earnings to price; earnings to price and one quality z-score are
   required). The weights inside each composite are equal, as published, unless a run sets ``growth_weights`` or
   ``quality_value_weights`` - a variant.
4. **Selection**: the 150 with the best growth z-score, then 75 of those by the quality & value z-score - the top 60
   always, current constituents still within the top 90 next, the rest in order until 75 (a buffer against turnover).
5. **Weights** in proportion to the growth score (1 + z above zero, 1 / (1 - z) below), each stock at most 5 % and
   each GICS sector at most 40 %.
6. The book is traded after the close of the third Friday of June and December, and drifts between.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from equity_research import ASSET_CLASS
from equity_research.analytics.indices import vendor_returns
from equity_research.data.cache import ResearchCache
from equity_research.data.sectors import listing_sectors
from equity_research.data.universe import UniverseData, load_universe_data
from equity_research.preferences import vendor_preferences
from research_core.engine.backtest import BacktestResult, run_backtest
from research_core.engine.finish import finish_run
from research_core.engine.registry import fence

#: The percentiles each ratio is winsorized at, and the bound on each composite z-score.
WINSOR = (0.025, 0.975)
Z_CAP = 4.0
#: The z-scores in each composite and their weights - equal, the published rule.
GROWTH_WEIGHTS: tuple[tuple[str, float], ...] = (("eps_growth", 1.0), ("sales_growth", 1.0))
QUALITY_VALUE_WEIGHTS: tuple[tuple[str, float], ...] = (("leverage", 1.0), ("roe", 1.0), ("earnings_yield", 1.0))

Weights = Mapping[str, float] | Sequence[tuple[str, float]]


def composite_weights(weights: Weights, published: tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
    """``weights`` as ``((z-score, weight), ...)`` in ``published``'s order, a z-score not named weighing zero.

    Only the ratios matter (a weighted mean), so they need not add up to one.
    """
    pairs = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    given, names = dict(pairs), [name for name, _ in published]
    if len(given) < len(pairs):
        raise ValueError("a z-score is weighted twice")
    unknown = sorted(set(given) - set(names))
    if unknown:
        raise ValueError(f"unknown z-scores {unknown}; choose from {names}")
    if any(w < 0 for w in given.values()) or not any(w > 0 for w in given.values()):
        raise ValueError("weights must be zero or more, and at least one above zero")
    return tuple((name, float(given.get(name, 0.0))) for name in names)


@dataclass(frozen=True)
class GarpConfig:
    """Every setting of a GARP run (hashed into its registry entry); the defaults are the published rules.

    ``growth_weights`` and ``quality_value_weights`` (a mapping or pairs) weigh the z-scores inside each composite;
    left at the published equal weights they are not part of the registered config, so runs from before they existed
    stay the same trial. The availability rules do not change with them: earnings to price and one quality z-score
    are still required, whatever their weights.
    """

    family: str = "garp_sp500"
    universe: str = "SP500"
    start: str = "2009-01-01"
    end: str = "2024-12-31"
    benchmark: str = "SPY"  # a fund's Yahoo symbol: measured with its dividends
    growth_count: int = 150
    count: int = 75
    automatic: int = 60
    keep_within: int = 90
    stock_cap: float = 0.05
    sector_cap: float = 0.40
    reference_months: tuple[int, ...] = (5, 11)
    growth_weights: Weights = GROWTH_WEIGHTS
    quality_value_weights: Weights = QUALITY_VALUE_WEIGHTS
    price_vendors: tuple[str, ...] = field(default_factory=lambda: vendor_preferences().prices)
    fundamentals_vendor: str | tuple[str, ...] = field(default_factory=lambda: vendor_preferences().fundamentals)
    open_holdout: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "growth_weights", composite_weights(self.growth_weights, GROWTH_WEIGHTS))
        object.__setattr__(
            self, "quality_value_weights", composite_weights(self.quality_value_weights, QUALITY_VALUE_WEIGHTS)
        )

    def registered(self) -> dict[str, Any]:
        """The config as the registry hashes it: a composite's weights only when they differ from the published."""
        out = asdict(self)
        for name, published in (("growth_weights", GROWTH_WEIGHTS), ("quality_value_weights", QUALITY_VALUE_WEIGHTS)):
            if out[name] == published:
                del out[name]
        return out


@dataclass(frozen=True)
class GarpResult:
    """The scores, the book, the backtest and the summary metrics of a run."""

    config: GarpConfig
    growth_z: pd.DataFrame
    quality_value_z: pd.DataFrame
    book: pd.DataFrame
    backtest: BacktestResult
    benchmark: pd.Series
    metrics: dict[str, Any]
    run_id: int | None = None


# ---------------------------------------------------------------- scores


def winsorized(raw: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Members' values clipped, date by date, to their own 2.5 and 97.5 percentiles; non-members missing.

    A bound is the value of the security ranked at that percentile (the paper's rule), not one interpolated between
    two securities - with an outlier among few names, interpolation would put the bound half-way to the outlier.
    """
    values = raw.where(members.reindex_like(raw).eq(True))  # a member missing from ``members`` is not one
    low = values.quantile(WINSOR[0], axis=1, interpolation="higher")
    high = values.quantile(WINSOR[1], axis=1, interpolation="lower")
    too_few = ~(low < high)  # a handful of names: the two ranks meet, and clipping would flatten them all
    low, high = low.mask(too_few, values.min(axis=1)), high.mask(too_few, values.max(axis=1))
    return values.clip(lower=low, upper=high, axis=0)


def standard_z(raw: pd.DataFrame, members: pd.DataFrame, *, reverse: bool = False) -> pd.DataFrame:
    """Winsorized values standardised with the day's members' mean and standard deviation (``reverse``: minus that)."""
    w = winsorized(raw, members)
    z = w.sub(w.mean(axis=1), axis=0).div(w.std(axis=1).replace(0.0, np.nan), axis=0)
    return -z if reverse else z


def _worst_where(z: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """``z``, with the flagged names given the day's lowest score."""
    worst = z.min(axis=1)
    return z.mask(
        flagged,
        pd.DataFrame(np.repeat(worst.to_numpy()[:, None], z.shape[1], axis=1), index=z.index, columns=z.columns),
    )


def _mean_of(parts: list[pd.DataFrame], weights: Sequence[float] | None = None) -> pd.DataFrame:
    """The weighted mean of the parts each name has (equal weights by default)."""
    stacked = np.stack([p.to_numpy(dtype=float) for p in parts])
    w = np.ones(len(parts)) if weights is None else np.asarray(weights, dtype=float)
    with np.errstate(invalid="ignore"):
        total = np.sum(~np.isnan(stacked) * w[:, None, None], axis=0)
        mean = np.nansum(stacked * w[:, None, None], axis=0) / np.where(total > 0, total, np.nan)
    return pd.DataFrame(mean, index=parts[0].index, columns=parts[0].columns)


def growth_z(
    eps_growth: pd.DataFrame, sales_growth: pd.DataFrame, members: pd.DataFrame, weights: Weights = GROWTH_WEIGHTS
) -> pd.DataFrame:
    """The growth composite: the (weighted) mean of the EPS- and sales-growth z-scores available, capped at +/-4."""
    parts = [standard_z(eps_growth, members), standard_z(sales_growth, members)]
    w = [weight for _, weight in composite_weights(weights, GROWTH_WEIGHTS)]
    return _mean_of(parts, w).clip(-Z_CAP, Z_CAP)


def quality_value_z(
    ratios: pd.DataFrame, members: pd.DataFrame, dates: pd.DatetimeIndex, weights: Weights = QUALITY_VALUE_WEIGHTS
) -> pd.DataFrame:
    """The quality & value composite: leverage, return on equity and earnings to price (see the module docstring)."""

    def panel(name: str) -> pd.DataFrame:
        return ratios[name].unstack("listing_id").reindex(index=dates, columns=members.columns)  # noqa: PD010

    negative_equity = panel("Book Equity/TL") < 0
    roe = panel("RoE")
    loss_over_negative = (roe > 0) & negative_equity
    roe_z = _worst_where(standard_z(roe.mask(loss_over_negative), members), loss_over_negative & members)
    leverage = panel("Debt/Equity")
    leverage_z = _worst_where(
        standard_z(leverage.mask(negative_equity), members, reverse=True), negative_equity & members
    )
    pe = panel("P/E")
    ep_z = standard_z((1.0 / pe).where(pe != 0), members)
    quality_known = roe_z.notna() | leverage_z.notna()
    w = [weight for _, weight in composite_weights(weights, QUALITY_VALUE_WEIGHTS)]
    return _mean_of([leverage_z, roe_z, ep_z], w).where(ep_z.notna() & quality_known).clip(-Z_CAP, Z_CAP)


def growth_score(z: pd.Series) -> pd.Series:
    """The weight-giving growth score: 1 + z above zero, 1 / (1 - z) below (always positive, 1 at zero)."""
    return pd.Series(np.where(z > 0, 1.0 + z, 1.0 / (1.0 - z)), index=z.index)


# ---------------------------------------------------------------- selection and weights


def select(growth: pd.Series, quality_value: pd.Series, held: set[int], config: GarpConfig) -> list[int]:
    """The constituents of one rebalance: sequential filtering with the buffer (see the module docstring)."""
    both = pd.DataFrame({"g": growth, "qv": quality_value}).dropna()
    candidates = both.sort_values("g", ascending=False, kind="stable").head(config.growth_count)
    ranked = candidates.sort_values("qv", ascending=False, kind="stable").index.astype(int).tolist()
    chosen = ranked[: config.automatic]
    for name in ranked[config.automatic : config.keep_within]:
        if len(chosen) >= config.count:
            break
        if name in held:
            chosen.append(name)
    for name in ranked:
        if len(chosen) >= config.count:
            break
        if name not in chosen:
            chosen.append(name)
    return chosen


def capped_weights(score: pd.Series, sectors: pd.Series, stock_cap: float, sector_cap: float) -> pd.Series:
    """Weights in proportion to ``score``, no stock above ``stock_cap`` and no sector above ``sector_cap``.

    What a cap removes goes to the names (or sectors) below their cap in proportion to their weight, repeated until
    nothing is over.
    """
    w = score / score.sum()
    sector = sectors.reindex(w.index).fillna("unknown")
    for _ in range(100):
        over = w > stock_cap + 1e-12
        if over.any():
            excess = float((w[over] - stock_cap).sum())
            w[over] = stock_cap
            room = w < stock_cap - 1e-12
            w[room] += excess * w[room] / w[room].sum()
        by_sector = w.groupby(sector).sum()
        heavy = by_sector[by_sector > sector_cap + 1e-12]
        for name, total in heavy.items():
            inside = sector == name
            excess = total - sector_cap
            w[inside] *= sector_cap / total
            light = ~sector.isin(heavy.index)
            w[light] += excess * w[light] / w[light].sum()
        if not over.any() and heavy.empty:
            break
    return w


def schedule(rebalances: pd.DatetimeIndex, days: pd.DatetimeIndex, months: tuple[int, ...]) -> pd.Series:
    """``reference date -> trade date``: each reference month end, and the third Friday of the following month.

    The trade happens after that Friday's close; on an exchange holiday, the trading day before it.
    """
    out = {}
    for ref in rebalances[rebalances.month.isin(months)]:
        first = (ref + pd.offsets.MonthBegin(1)).normalize()
        friday = first + pd.DateOffset(days=(4 - first.weekday()) % 7 + 14)
        trading = days[days <= friday]
        if len(trading) and trading[-1] > ref:
            out[ref] = trading[-1]
    return pd.Series(out, dtype="datetime64[ns]")


def garp_book(
    growth: pd.DataFrame,
    quality_value: pd.DataFrame,
    sectors: dict[pd.Timestamp, pd.Series],
    trades: pd.Series,
    config: GarpConfig,
) -> pd.DataFrame:
    """Target weights on the trade dates, chosen and weighted with each reference date's scores and sectors."""
    rows = {}
    held: set[int] = set()
    for ref_key, trade in trades.items():
        ref = pd.Timestamp(str(ref_key))
        g, qv = pd.Series(growth.loc[ref]), pd.Series(quality_value.loc[ref])
        chosen = select(g, qv, held, config)
        held = set(chosen)
        weights = capped_weights(growth_score(g.reindex(chosen)), sectors[ref], config.stock_cap, config.sector_cap)
        rows[trade] = weights.reindex(growth.columns).fillna(0.0)
    return pd.DataFrame(rows).T.reindex(columns=growth.columns).fillna(0.0)


# ---------------------------------------------------------------- the run


def garp_scores(data: UniverseData, config: GarpConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(growth, quality_value)``: the two composite z-scores of ``data``'s members on its rebalance dates."""
    members = data.members.reindex(data.rebalances).astype(bool)
    growth = growth_z(data.eps_growth, data.sales_growth, members, config.growth_weights)
    return growth, quality_value_z(data.ratios, members, data.rebalances, config.quality_value_weights)


def run_garp(cache: ResearchCache, config: GarpConfig, *, engine: Engine | None = None) -> GarpResult:
    """Fence, load, score, select and weight, backtest, and (with ``engine``) register the run."""
    holdout = fence(config.end, open_holdout=config.open_holdout)
    data: UniverseData = load_universe_data(
        cache,
        config.universe,
        config.start,
        config.end,
        price_vendors=config.price_vendors,
        fundamentals_vendor=config.fundamentals_vendor,
    )
    growth, quality_value = garp_scores(data, config)
    days = pd.DatetimeIndex(data.returns.index)
    trades = schedule(data.rebalances, days[days <= pd.Timestamp(config.end)], config.reference_months)
    sectors = {ref: listing_sectors(cache, data.members.columns, ref) for ref in trades.index}
    book = garp_book(growth, quality_value, sectors, trades, config)
    returns = data.returns[data.returns.index <= pd.Timestamp(config.end)]
    result = run_backtest(
        book, returns, cost_bps=data.cost_bps, borrow_bps=data.borrow_bps, buy_tax_bps=data.buy_tax_bps
    )
    benchmark = vendor_returns(cache, config.benchmark, total=True)
    metrics, run_id = finish_run(
        result,
        benchmark,
        config.registered(),
        asset_class=ASSET_CLASS,
        family=config.family,
        engine=engine,
        holdout=holdout,
    )
    return GarpResult(config, growth, quality_value, book, result, benchmark, metrics, run_id)
