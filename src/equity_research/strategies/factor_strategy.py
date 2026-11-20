"""A factor strategy end to end: one frozen config in, a recorded backtest out.

    cfg = StrategyConfig(family="value_quality_ls", universe="SP1500", factors=(("value", 1.0), ("quality", 1.0)))
    result = run_strategy(cache, cfg, engine=engine)      # fenced, backtested, registered
    result.metrics

The hold-out fence is checked first; with an ``engine`` the run is registered in ``research.run`` and the metrics
include the deflated Sharpe ratio for the number of configurations of the family tried so far. The metrics also
measure the run against its universe's benchmark (:func:`active_performance`): for a long-only book the deflated
Sharpe ratio of the *active* returns (``deflated_active_sharpe``) is the one that says whether the choice of names
added anything, since its own Sharpe ratio mostly measures the market.

``weighting`` sets how the chosen names share each leg: ``equal``, or ``market_cap`` - in proportion to their market
value, so a long-only book keeps its universe's mix of large and small companies (:func:`weighted_by`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import pandas as pd
from sqlalchemy.engine import Engine

from equity_research import ASSET_CLASS
from equity_research.data.cache import ResearchCache
from equity_research.data.universe import UniverseData, load_universe_data
from equity_research.preferences import vendor_preferences
from equity_research.strategies import factors as fx
from equity_research.strategies import learned_value as ml
from research_core.engine.backtest import BacktestResult, run_backtest
from research_core.engine.finish import finish_run
from research_core.engine.portfolio import beta_neutral, quantile_book, trailing_beta, weighted_by
from research_core.engine.registry import fence

#: Scores a model learns from past outcomes (:mod:`~equity_research.strategies.learned_value`), and where their
#: training data begins: a run using one loads data from here, though its book still starts at
#: ``StrategyConfig.start``.
LEARNED = frozenset({"ml_value"})
TRAIN_FROM = "2006-01-01"
#: How the chosen names share a leg (``StrategyConfig.weighting``).
WEIGHTINGS = ("equal", "market_cap")


@dataclass(frozen=True)
class StrategyConfig:
    """Every setting of a run (hashed into its registry entry)."""

    family: str
    universe: str = "SP1500"
    start: str = "2011-01-01"
    end: str = "2024-12-31"
    rebalance: str = "ME"
    factors: tuple[tuple[str, float], ...] = (("momentum", 1.0), ("value", 1.0), ("quality", 1.0), ("low_vol", 1.0))
    enter: float = 0.1
    exit_: float = 0.2
    long_only: bool = False
    beta_neutral: bool = True
    weighting: str = "equal"
    price_vendors: tuple[str, ...] = field(default_factory=lambda: vendor_preferences().prices)
    fundamentals_vendor: str | tuple[str, ...] = field(default_factory=lambda: vendor_preferences().fundamentals)
    open_holdout: bool = False


@dataclass(frozen=True)
class StrategyResult:
    """The book, the backtest and the summary metrics of a run."""

    config: StrategyConfig
    scores: pd.DataFrame
    book: pd.DataFrame
    backtest: BacktestResult
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: int | None = None


def factor_scores(data: UniverseData, name: str) -> pd.DataFrame:
    """One factor's z-scores on the rebalance dates, members only."""
    members = data.members.reindex(data.rebalances)
    builders: dict[str, Callable[[], pd.DataFrame]] = {
        "momentum": lambda: fx.cross_sectional_z(fx.momentum(data.adjusted).reindex(data.rebalances), members),
        "low_vol": lambda: fx.cross_sectional_z(fx.low_volatility(data.returns).reindex(data.rebalances), members),
        "value": lambda: fx.value(data.ratios, members),
        "quality": lambda: fx.quality(data.ratios, members),
        "issuance": lambda: fx.cross_sectional_z(fx.net_issuance(data.share_change).reindex(data.rebalances), members),
        "cash_flow_yield": lambda: fx.cash_flow_yield(data.ratios, members),
        "dividend_yield": lambda: fx.cross_sectional_z(
            fx.dividend_yield(data.returns, data.adjusted).reindex(data.rebalances), members
        ),
        "high_beta": lambda: fx.cross_sectional_z(
            trailing_beta(data.returns, data.returns.where(data.members).mean(axis=1)).reindex(data.rebalances),
            members,
        ),
        "accruals": lambda: fx.accruals(data.ratios, members),
        "earnings_surprise": lambda: fx.cross_sectional_z(data.earnings_surprise.reindex(data.rebalances), members),
        "ml_value": lambda: fx.cross_sectional_z(
            ml.learned_value(data.ratios, data.returns, data.members, data.rebalances), members
        ),
        "residual_momentum": lambda: fx.cross_sectional_z(
            fx.residual_momentum(data.returns, data.returns.where(data.members).mean(axis=1), data.rebalances), members
        ),
        "magic_formula": lambda: fx.magic_formula(data.ratios, members),
    }
    if name not in builders:
        raise KeyError(f"unknown factor {name!r}; choose from {sorted(builders)}")
    return builders[name]().reindex(index=data.rebalances, columns=data.members.columns)


def learns(config: StrategyConfig) -> bool:
    """Whether a score of ``config`` is learned from past outcomes (its data must start at :data:`TRAIN_FROM`)."""
    return any(name in LEARNED for name, _ in config.factors)


def strategy_scores(data: UniverseData, config: StrategyConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``(score, parts)``: the weighted mix of ``config``'s factors on ``data``'s rebalance dates, and each factor's."""
    weights = dict(config.factors)
    parts = {name: factor_scores(data, name) for name in weights}
    return fx.combine(parts, weights), parts


def strategy_book(
    data: UniverseData, config: StrategyConfig, score: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """The target weights ``config`` holds on ``dates``: chosen by ``score``, weighted, beta-neutral if asked.

    The buffer (enter at ``enter``, leave at ``exit_``) depends on what was held before, so ``dates`` from earlier
    than the day of interest make its book the one the strategy would really hold.
    """
    book = quantile_book(score, dates, enter=config.enter, exit_=config.exit_, long_only=config.long_only)
    if config.weighting == "market_cap":
        book = weighted_by(book, data.market_cap)
    if config.beta_neutral and not config.long_only:
        market = data.returns.where(data.members).mean(axis=1)
        book = beta_neutral(book, trailing_beta(data.returns, market).reindex(book.index))
    return book


def run_strategy(cache: ResearchCache, config: StrategyConfig, *, engine: Engine | None = None) -> StrategyResult:
    """Fence, load, score, build the book, backtest, and (with ``engine``) register the run."""
    if config.weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {config.weighting!r}; choose from {WEIGHTINGS}")
    holdout = fence(config.end, open_holdout=config.open_holdout)
    # a learned score needs years of outcomes to learn from before the backtest begins
    data = load_universe_data(
        cache,
        config.universe,
        min(config.start, TRAIN_FROM) if learns(config) else config.start,
        config.end,
        price_vendors=config.price_vendors,
        fundamentals_vendor=config.fundamentals_vendor,
        rebalance=config.rebalance,
    )
    scores, _ = strategy_scores(data, config)
    dates = data.rebalances[data.rebalances >= pd.Timestamp(config.start)]  # no holdings carried in from training
    book = strategy_book(data, config, scores, dates)
    returns = data.returns[data.returns.index <= pd.Timestamp(config.end)]
    result = run_backtest(
        book, returns, cost_bps=data.cost_bps, borrow_bps=data.borrow_bps, buy_tax_bps=data.buy_tax_bps
    )
    metrics, run_id = finish_run(
        result,
        data.benchmark,
        asdict(config),
        asset_class=ASSET_CLASS,
        family=config.family,
        engine=engine,
        holdout=holdout,
    )
    return StrategyResult(config, scores, book, result, metrics, run_id)
