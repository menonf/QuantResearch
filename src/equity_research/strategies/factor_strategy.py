"""A factor strategy end to end: one frozen config in, a recorded backtest out.

    cfg = StrategyConfig(family="value_quality_ls", universe="SP1500", factors=(("value", 1.0), ("quality", 1.0)))
    result = run_strategy(cache, cfg, engine=engine)      # fenced, backtested, registered
    result.metrics

The hold-out fence is checked first; with an ``engine`` the run is registered in ``research.run`` and the metrics
include the deflated Sharpe ratio for the number of configurations of the family tried so far. The metrics also
measure the run against its benchmark (:func:`active_performance`): for a long-only book the deflated Sharpe ratio of
the *active* returns (``deflated_active_sharpe``) is the one that says whether the choice of names added anything,
since its own Sharpe ratio mostly measures the market.

``weighting`` sets how the chosen names share each leg: ``equal``, or ``market_cap`` - in proportion to their market
value, so a long-only book keeps its universe's mix of large and small companies (:func:`weighted_by`).

Settings added after the first runs (:data:`LATER`; each default is how runs were made before it existed, and a
setting left at its default is not registered, so those runs keep their hash and repeating one is no new trial):

* ``mix`` - ``mean`` averages the factors' scores as they come, skipping a missing one; ``standardised`` first
  z-scores each factor again (an average of four z-scores - value, quality - spreads less than one z-score, so it
  weighed less than its share) and counts a missing factor as neutral (a name scored on two factors of five is no
  longer twice as likely to reach the top tenth).
* ``within_sectors`` - each factor ranked among its sector's members rather than the whole universe. Sectors are the
  companies' GICS sectors as classified today (the risk model's convention): the sector history starts on the day
  it was first recorded, so earlier dates would use the same classification anyway.
* ``construction`` - ``quantile`` buys the best slice (``enter`` / ``exit_``) and weights it; ``tilted`` holds the
  universe by market value plus tilts by the score on the risk model (:mod:`.construction`): ``risk`` a year of
  predicted active risk (a long-short book's volatility), at most ``max_active`` in a name either way, no exposure to
  the ``neutral`` factors (sectors, size) and no active beta; ``gross`` caps a long-short book's gross exposure.
* ``max_weight`` - a quantile book's largest weight in one name.
* ``benchmark`` - what the active measures are against: the universe's own (``None``: SPTM for the S&P 1500),
  ``cap_weighted`` / ``equal_weighted`` - the universe rebuilt from its members (by market value at each rebalance
  date, drifting with prices between them; equally, reset quarterly) - or a fund's Yahoo symbol.
* ``drift_band`` - a holding may drift this far (relative) from its target before it is traded back.
* ``cost_multiplier`` - every trading cost times this (a stress test of the costs).
* ``train_from`` - where a learned score's training data begins (default :data:`TRAIN_FROM`).

A run whose book is empty on a rebalance date - a score with no values yet, a learned model not yet fitted - is
refused (:class:`EmptyBookError`): the backtest would hold cash without saying so and be measured as if it had
invested.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from equity_research import ASSET_CLASS
from equity_research.analytics.indices import equal_weighted, vendor_returns
from equity_research.analytics.risk.risk_model import RiskModel
from equity_research.data.cache import ResearchCache
from equity_research.data.reads import corporate_actions
from equity_research.data.sectors import listing_sectors
from equity_research.data.universe import UniverseData, load_universe_data, with_corrections
from equity_research.preferences import vendor_preferences
from equity_research.strategies import construction as cn
from equity_research.strategies import factors as fx
from equity_research.strategies import learned_value as ml
from equity_research.strategies.qgarp_estimates import consensus_revision_factors, load_estimate_inputs
from research_core.engine.backtest import BacktestResult, run_backtest
from research_core.engine.finish import finish_run
from research_core.engine.portfolio import beta_neutral, capped, quantile_book, trailing_beta, weighted_by
from research_core.engine.registry import fence

#: Scores a model learns from past outcomes (:mod:`~equity_research.strategies.learned_value`), and where their
#: training data begins: a run using one loads data from here, though its book still starts at
#: ``StrategyConfig.start``.
LEARNED = frozenset({"ml_value", "ml_value_rank"})
TRAIN_FROM = "2006-01-01"
#: How the chosen names share a leg (``StrategyConfig.weighting``).
WEIGHTINGS = ("equal", "market_cap")
#: How factor scores are mixed (``StrategyConfig.mix``) and how a book is built (``construction``).
MIXES = ("mean", "standardised")
CONSTRUCTIONS = ("quantile", "tilted")
#: Factors read from the research cache beyond what :class:`UniverseData` holds (:func:`cache_factors`): the
#: revision of analysts' consensus EPS over four months (0.4 / 0.3 / 0.2 / 0.1), for the same fiscal year, over the
#: price - a true history, unlike the analysts' revision counts (``qgarp_estimates``).
CACHE_FACTORS = frozenset({"consensus_revisions"})
#: Benchmarks built from the universe's own members (``StrategyConfig.benchmark``); any other value is a fund symbol.
OWN_BENCHMARKS = ("cap_weighted", "equal_weighted")


@dataclass(frozen=True)
class StrategyConfig:
    """Every setting of a run (hashed into its registry entry; see :data:`LATER` for the settings added later)."""

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
    mix: str = "mean"
    within_sectors: bool = False
    construction: str = "quantile"
    risk: float = 0.03
    max_active: float = 0.02
    neutral: tuple[str, ...] = cn.GROUPS
    gross: float | None = None
    max_weight: float | None = None
    benchmark: str | None = None
    drift_band: float = 0.0
    cost_multiplier: float = 1.0
    train_from: str | None = None

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("weighting", self.weighting, WEIGHTINGS),
            ("mix", self.mix, MIXES),
            ("construction", self.construction, CONSTRUCTIONS),
        ):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; choose from {allowed}")
        object.__setattr__(self, "neutral", tuple(self.neutral))


#: The settings added after runs were registered, with the default that reproduces those runs.
LATER: dict[str, Any] = {
    "mix": "mean",
    "within_sectors": False,
    "construction": "quantile",
    "risk": 0.03,
    "max_active": 0.02,
    "neutral": cn.GROUPS,
    "gross": None,
    "max_weight": None,
    "benchmark": None,
    "drift_band": 0.0,
    "cost_multiplier": 1.0,
    "train_from": None,
}


def registered_settings(config: StrategyConfig) -> dict[str, Any]:
    """``config`` as the registry stores and hashes it: a later setting (:data:`LATER`) only where it is not default.

    A run that does not use a setting is stored as it was before the setting existed, so it keeps its hash and
    repeating it is not counted as a new trial.
    """
    settings = asdict(config)
    for name, default in LATER.items():
        value = settings[name]
        if (tuple(value) if isinstance(value, list) else value) == default:
            del settings[name]
    return settings


class EmptyBookError(ValueError):
    """A strategy held nothing on a rebalance date: its backtest would sit in cash and be measured as invested."""


@dataclass(frozen=True)
class StrategyResult:
    """The book, the backtest and the summary metrics of a run."""

    config: StrategyConfig
    scores: pd.DataFrame
    book: pd.DataFrame
    backtest: BacktestResult
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: int | None = None
    benchmark: pd.Series | None = None


def cache_factors(cache: ResearchCache, data: UniverseData, config: StrategyConfig) -> dict[str, pd.DataFrame]:
    """The raw panels of ``config``'s :data:`CACHE_FACTORS` on ``data``'s rebalance dates (none when it uses none)."""
    if not {name for name, _ in config.factors} & CACHE_FACTORS:
        return {}
    listings = data.close.columns
    actions = with_corrections(corporate_actions(cache, listings), config.price_vendors)
    actions = actions.drop(columns="vendor").reset_index(drop=True)
    unused = pd.DataFrame(np.nan, index=data.rebalances, columns=listings)  # what only the other estimate factors read
    inputs = load_estimate_inputs(
        cache, data.rebalances, data.close, actions, value=unused, shares=unused, total_assets=unused
    )
    return {"consensus_revisions": consensus_revision_factors(inputs)["consensus_revisions_4_mth"]}


def factor_scores(data: UniverseData, name: str, *, extra: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One factor's z-scores on the rebalance dates, members only (``extra``: the :func:`cache_factors` panels)."""
    members = data.members.reindex(data.rebalances)
    if name in CACHE_FACTORS:
        if extra is None or name not in extra:
            raise ValueError(f"factor {name!r} is read from the research cache: pass cache_factors(...) as extra")
        return fx.cross_sectional_z(extra[name].reindex(data.rebalances), members).reindex(
            index=data.rebalances, columns=data.members.columns
        )
    builders: dict[str, Callable[[], pd.DataFrame]] = {
        "momentum": lambda: fx.cross_sectional_z(fx.momentum(data.adjusted).reindex(data.rebalances), members),
        "low_vol": lambda: fx.cross_sectional_z(fx.low_volatility(data.returns).reindex(data.rebalances), members),
        "value": lambda: fx.value(data.ratios, members),
        "earnings_value": lambda: fx.earnings_value(data.ratios, members),
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
        "ml_value_rank": lambda: fx.cross_sectional_z(
            ml.learned_value_rank(data.ratios, data.returns, data.members, data.rebalances), members
        ),
        "residual_momentum": lambda: fx.cross_sectional_z(
            fx.residual_momentum(data.returns, data.returns.where(data.members).mean(axis=1), data.rebalances), members
        ),
        "magic_formula": lambda: fx.magic_formula(data.ratios, members),
    }
    if name not in builders:
        raise KeyError(f"unknown factor {name!r}; choose from {sorted([*builders, *CACHE_FACTORS])}")
    return builders[name]().reindex(index=data.rebalances, columns=data.members.columns)


def learns(config: StrategyConfig) -> bool:
    """Whether a score of ``config`` is learned from past outcomes (its data must start before its first book)."""
    return any(name in LEARNED for name, _ in config.factors)


def data_start(config: StrategyConfig) -> str:
    """Where ``config``'s data must begin: a learned score's training start, the risk model's history, or ``start``.

    The book still starts at ``config.start``: the earlier data only trains a model or estimates the risk model.
    """
    starts = [pd.Timestamp(config.start)]
    if learns(config):
        starts.append(pd.Timestamp(config.train_from or TRAIN_FROM))
    if config.construction == "tilted":
        starts.append(pd.Timestamp(config.start) - cn.RISK_HISTORY)
    return min(starts).strftime("%Y-%m-%d")


def strategy_scores(
    data: UniverseData,
    config: StrategyConfig,
    *,
    sectors: pd.Series | None = None,
    extra: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``(score, parts)``: the weighted mix of ``config``'s factors on ``data``'s rebalance dates, and each factor's.

    ``sectors`` (listing -> sector) is needed when ``config.within_sectors``: each factor is then ranked within them.
    ``extra`` holds the panels of the factors read from the cache (:func:`cache_factors`).
    """
    weights = dict(config.factors)
    parts = {name: factor_scores(data, name, extra=extra) for name in weights}
    members = data.members.reindex(data.rebalances)
    if config.within_sectors:
        if sectors is None:
            raise ValueError("ranking within sectors needs the listings' sectors")
        parts = {name: fx.within_groups(p, sectors, members) for name, p in parts.items()}
    if config.mix == "standardised":
        mixed = {name: fx.cross_sectional_z(p, members) for name, p in parts.items()}
        return fx.combine(mixed, weights, missing="neutral"), parts
    return fx.combine(parts, weights), parts


def strategy_book(
    data: UniverseData,
    config: StrategyConfig,
    score: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    model: RiskModel | None = None,
) -> pd.DataFrame:
    """The target weights ``config`` holds on ``dates``: chosen by ``score``, weighted, beta-neutral if asked.

    The buffer (enter at ``enter``, leave at ``exit_``) depends on what was held before, so ``dates`` from earlier
    than the day of interest make its book the one the strategy would really hold. A ``tilted`` construction needs
    the risk ``model`` (:func:`~equity_research.strategies.construction.risk_model_for`).
    """
    if config.construction == "tilted":
        if model is None:
            raise ValueError("a tilted book needs the risk model")
        return cn.tilted_book(
            data,
            score,
            dates,
            model,
            long_only=config.long_only,
            risk=config.risk,
            max_active=config.max_active,
            neutral=config.neutral,
            gross=config.gross,
        )
    book = quantile_book(score, dates, enter=config.enter, exit_=config.exit_, long_only=config.long_only)
    if config.weighting == "market_cap":
        book = weighted_by(book, data.market_cap)
    if config.max_weight is not None:
        book = capped(book, config.max_weight)
    if config.beta_neutral and not config.long_only:
        market = data.returns.where(data.members).mean(axis=1)
        book = beta_neutral(book, trailing_beta(data.returns, market).reindex(book.index))
    return book


def check_book(book: pd.DataFrame) -> None:
    """Refuse a book that holds nothing on a rebalance date (:class:`EmptyBookError`), naming when it first holds."""
    empty = book.index[(book != 0).sum(axis=1) == 0]
    if len(empty):
        held = book.index[(book != 0).sum(axis=1) > 0]
        first = f"it first holds names on {held[0]:%Y-%m-%d}" if len(held) else "it never holds a name"
        raise EmptyBookError(
            f"the book is empty on {len(empty)} rebalance date(s) from {empty[0]:%Y-%m-%d}: {first}. Start the "
            "run when every score it needs exists (a learned score trains for years first: set train_from earlier)"
        )


def strategy_benchmark(cache: ResearchCache, data: UniverseData, config: StrategyConfig) -> pd.Series:
    """The daily returns ``config``'s active measures are taken against (see the module docstring)."""
    if config.benchmark is None:
        return data.benchmark
    if config.benchmark == "cap_weighted":  # market values are dated on rebalance dates: held between them
        return cn.index_returns(data)
    if config.benchmark == "equal_weighted":
        return equal_weighted(data.returns, data.members)["return"]
    return vendor_returns(cache, config.benchmark, total=True)


def run_strategy(cache: ResearchCache, config: StrategyConfig, *, engine: Engine | None = None) -> StrategyResult:
    """Fence, load, score, build the book, backtest, and (with ``engine``) register the run."""
    holdout = fence(config.end, open_holdout=config.open_holdout)
    # a learned score needs years of outcomes to learn from, the risk model a year of returns, before the first book
    data = load_universe_data(
        cache,
        config.universe,
        data_start(config),
        config.end,
        price_vendors=config.price_vendors,
        fundamentals_vendor=config.fundamentals_vendor,
        rebalance=config.rebalance,
    )
    sectors = listing_sectors(cache, data.close.columns) if config.within_sectors else None
    scores, _ = strategy_scores(data, config, sectors=sectors, extra=cache_factors(cache, data, config))
    dates = data.rebalances[data.rebalances >= pd.Timestamp(config.start)]  # no holdings carried in from training
    model = (
        cn.risk_model_for(
            cache, data, price_vendors=config.price_vendors, fundamentals_vendor=config.fundamentals_vendor
        )
        if config.construction == "tilted"
        else None
    )
    book = strategy_book(data, config, scores, dates, model=model)
    check_book(book)
    returns = data.returns[data.returns.index <= pd.Timestamp(config.end)]
    result = run_backtest(
        book,
        returns,
        cost_bps=data.cost_bps * config.cost_multiplier,
        borrow_bps=data.borrow_bps,
        buy_tax_bps=data.buy_tax_bps,
        drift_band=config.drift_band,
    )
    benchmark = strategy_benchmark(cache, data, config)
    metrics, run_id = finish_run(
        result,
        benchmark,
        registered_settings(config),
        asset_class=ASSET_CLASS,
        family=config.family,
        engine=engine,
        holdout=holdout,
    )
    return StrategyResult(config, scores, book, result, metrics, run_id, benchmark)
