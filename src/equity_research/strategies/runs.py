"""Any registered strategy run, rebuilt: its config from the registry, its book on a day, its backtest again.

    run = load_run(engine, run_id=109)            # or family="multi_factor_lo": that family's latest run
    today = snapshot(cache, run.config)           # today's scores, what they are made of, the book it would hold
    again = rerun(cache, run.config)              # the registered backtest again - not registered, no trial added

``research.run`` keeps every run's whole config with its family, so a run id is all a notebook needs to show any
strategy: a factor mix (:class:`~equity_research.strategies.factor_strategy.StrategyConfig`), the S&P 500 GARP rules
(:class:`~equity_research.strategies.garp.GarpConfig`) or the QGARP strategy
(:class:`~equity_research.strategies.qgarp.QgarpConfig`). Which of the three a stored config is follows
from its settings - only GARP has ``growth_count``, only QGARP ``fix_revisions_direction`` - and a setting added
after the run takes its default, which is how the run was made.

Steps of a snapshot: load the strategy's universe from a year before the day (today by default) up to it, with the
day itself as a decision date; build the strategy's own scores; build its book over that year, so the buffer between
entering and leaving a name holds what the strategy would really hold; read the scores, their parts and the book on
the day. A snapshot never backtests, so it does not reach into the hold-out with a backtest and adds no trial.
A rerun calls the strategy's own runner without an engine: the backtest again on today's data and code, nothing
recorded (the registry counts trials by config, and the config is the registered one).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research.data.cache import ResearchCache
from equity_research.data.sectors import listing_sectors
from equity_research.data.universe import UniverseData, load_universe_data
from equity_research.strategies.construction import risk_model_for
from equity_research.strategies.factor_strategy import (
    StrategyConfig,
    cache_factors,
    data_start,
    run_strategy,
    strategy_book,
    strategy_scores,
)
from equity_research.strategies.garp import GarpConfig, garp_book, garp_scores, run_garp, schedule
from equity_research.strategies.qgarp import (
    COMPOSITES,
    HISTORY_DAYS,
    QgarpConfig,
    qgarp_book,
    qgarp_scores,
    run_qgarp,
)
from research_core.data.db import schema
from research_core.engine.backtest import BacktestResult

#: The three kinds of strategy a run can be.
AnyConfig = StrategyConfig | GarpConfig | QgarpConfig
#: How far before the day a snapshot's book starts, so its buffer holds what the strategy would really hold (GARP
#: reselects twice a year, so a year always includes its last selection).
BOOK_HISTORY = pd.DateOffset(years=1)


class RunResult(Protocol):
    """What every strategy's runner returns, whatever builds its book."""

    @property
    def book(self) -> pd.DataFrame:
        """``date x listing_id``: the target weights on each rebalance date."""
        ...

    @property
    def backtest(self) -> BacktestResult:
        """Daily returns, turnover, costs and the weights held each day."""
        ...

    @property
    def metrics(self) -> Mapping[str, Any]:
        """The summary measures, as registered."""
        ...

    @property
    def run_id(self) -> int | None:
        """The registry's id of the run (``None`` when it was not registered)."""
        ...


@dataclass(frozen=True)
class RegisteredRun:
    """One row of ``research.run`` with its config rebuilt."""

    run_id: int
    family: str
    config: AnyConfig
    metrics: dict[str, Any]
    finished_at: pd.Timestamp | None

    @property
    def label(self) -> str:
        """A short name for tables and charts: the run, its family and (for a factor mix) its factors."""
        what = self.family
        if isinstance(self.config, StrategyConfig):
            what += ": " + ", ".join(name for name, _ in self.config.factors)
        return f"run {self.run_id} · {what}"


@dataclass(frozen=True)
class Snapshot:
    """A strategy on one day: the score it ranks by, what that score is made of, and the book it would hold."""

    config: AnyConfig
    day: pd.Timestamp
    data: UniverseData
    #: ``listing_id -> score``, higher is better: the names the strategy can rank on the day
    score: pd.Series
    #: ``listing_id x part``: the scores the strategy's score is built from (factor z-scores, composites)
    parts: pd.DataFrame
    #: ``listing_id -> weight`` the strategy would hold if it rebalanced on the day (negative: short; 0: not held)
    weights: pd.Series


# ---------------------------------------------------------------- the registry


def _frozen(value: Any) -> Any:
    """JSON's lists back into the tuples a config holds (factors as pairs, vendors in order)."""
    return tuple(_frozen(v) for v in value) if isinstance(value, list) else value


def config_from(payload: Mapping[str, Any]) -> AnyConfig:
    """The config a registered run was made with, from its stored settings."""
    kind: type[AnyConfig]
    if "growth_count" in payload:
        kind = GarpConfig
    elif "fix_revisions_direction" in payload:
        kind = QgarpConfig
    else:
        kind = StrategyConfig
    unknown = sorted(set(payload) - {f.name for f in fields(kind)})
    if unknown:
        raise ValueError(f"a {kind.__name__} has no settings {unknown}: this run cannot be rebuilt")
    return kind(**{name: _frozen(value) for name, value in payload.items()})


def choose_run(rows: pd.DataFrame, run_id: int | None, family: str | None) -> pd.Series:
    """The row of ``rows`` (``run_id``, ``family``, ...) a notebook asked for.

    A run id names one run; a family alone, its latest. Given both, they must agree - a run id typed for another
    strategy is caught, not shown under the wrong name.
    """
    if run_id is None and family is None:
        raise ValueError("name a run id, a strategy family, or both")
    if run_id is not None:
        found = rows[rows["run_id"] == run_id]
        if found.empty:
            raise KeyError(f"run {run_id} is not registered")
        row = found.iloc[0]
        if family is not None and row["family"] != family:
            raise ValueError(f"run {run_id} is a {row['family']!r} run, not {family!r}")
        return row
    of_family = rows[rows["family"] == family]
    if of_family.empty:
        raise KeyError(f"no run of family {family!r}; registered: {sorted(rows['family'].unique())}")
    return of_family.sort_values("run_id").iloc[-1]


def registered_runs(engine: Engine, family: str | None = None) -> pd.DataFrame:
    """Every registered run (of ``family``), newest first: id, family, when, universe, period and key metrics."""
    r = schema.research_run
    query = sa.select(r.c.run_id, r.c.family, r.c.finished_at, r.c.config_json, r.c.metrics_json)
    if family is not None:
        query = query.where(r.c.family == family)
    with engine.connect() as conn:
        rows = pd.DataFrame(conn.execute(query.order_by(r.c.run_id.desc())).mappings().all())
    if rows.empty:
        return rows
    config = rows.pop("config_json").map(json.loads)
    metrics = rows.pop("metrics_json").map(lambda m: json.loads(m) if m else {})
    for key in ("universe", "start", "end"):
        rows[key] = config.map(lambda c, k=key: c.get(k))
    for key in ("cagr", "sharpe", "information_ratio", "deflated_sharpe"):
        rows[key] = metrics.map(lambda m, k=key: m.get(k))
    return rows


def newest_by_score(rows: pd.DataFrame) -> pd.DataFrame:
    """``score x metric`` from registry ``rows`` (``run_id``, ``config_json``, ``metrics_json``), newest run each.

    A score is named by the config's first factor (``"ml:knn"`` -> ``knn``): one family's runs that differ only in the
    score - each learner of a study - side by side, a configuration run again shown once, as last run.
    """
    if rows.empty:
        return pd.DataFrame()
    named = rows.assign(
        score=rows["config_json"].map(lambda c: str(json.loads(c)["factors"][0][0]).removeprefix("ml:"))
    )
    last = named.sort_values("run_id").drop_duplicates("score", keep="last")
    metrics = [{"run_id": int(r["run_id"]), **json.loads(r["metrics_json"] or "{}")} for _, r in last.iterrows()]
    return pd.DataFrame(metrics, index=pd.Index(last["score"], name="score"))


def score_metrics(engine: Engine, family: str) -> pd.DataFrame:
    """``score x metric``: each score's newest registered run in ``family`` (:func:`newest_by_score`)."""
    r = schema.research_run
    query = sa.select(r.c.run_id, r.c.config_json, r.c.metrics_json).where(r.c.family == family)
    with engine.connect() as conn:
        rows = pd.DataFrame(conn.execute(query).mappings().all(), columns=list(query.selected_columns.keys()))
    return newest_by_score(rows)


def load_run(engine: Engine, run_id: int | None = None, *, family: str | None = None) -> RegisteredRun:
    """The registered run ``run_id``, or ``family``'s latest; both given, they must agree (:func:`choose_run`)."""
    r = schema.research_run
    query = sa.select(r.c.run_id, r.c.family, r.c.config_json, r.c.metrics_json, r.c.finished_at)
    query = query.where(r.c.run_id == run_id) if run_id is not None else query.where(r.c.family == family)
    with engine.connect() as conn:
        rows = pd.DataFrame(conn.execute(query).mappings().all(), columns=list(query.selected_columns.keys()))
    row = choose_run(rows, run_id, family)
    payload = json.loads(row["config_json"])
    payload.pop("family", None)  # the registry adds it to the hashed config; the config has its own
    config = config_from({"family": row["family"], **payload})
    finished = row["finished_at"]
    return RegisteredRun(
        run_id=int(row["run_id"]),
        family=str(row["family"]),
        config=config,
        metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
        finished_at=None if pd.isna(finished) else pd.Timestamp(finished),
    )


def one_universe(configs: Iterable[AnyConfig]) -> str:
    """The universe every config shares - books compared side by side must be chosen from the same names."""
    universes = sorted({c.universe for c in configs})
    if len(universes) != 1:
        raise ValueError(f"these runs are in different universes {universes}: compare runs of one universe")
    return universes[0]


# ---------------------------------------------------------------- the strategy again


def rerun(cache: ResearchCache, config: AnyConfig) -> RunResult:
    """The backtest of ``config`` again, by its own runner, without registering it (no engine: no trial added)."""
    if isinstance(config, GarpConfig):
        return run_garp(cache, config)
    if isinstance(config, QgarpConfig):
        return run_qgarp(cache, config)
    return run_strategy(cache, config)


def matching_run(rows: pd.DataFrame, config: AnyConfig) -> int | None:
    """The newest ``run_id`` among registry ``rows`` (``run_id``, ``config_json``) made with exactly ``config``."""
    for _, row in rows.sort_values("run_id", ascending=False).iterrows():
        payload = json.loads(row["config_json"])
        payload.pop("family", None)
        try:
            same = config_from({"family": config.family, **payload}) == config
        except (KeyError, TypeError, ValueError):  # another kind of strategy registered under the family
            continue
        if same:
            return int(row["run_id"])
    return None


def registered_or_run(cache: ResearchCache, engine: Engine, config: AnyConfig) -> RegisteredRun:
    """The newest registered run of exactly ``config``; where the registry has none, it is run and registered first.

    For the notebooks that compare or screen registered strategies: they name a strategy by its settings rather
    than by a run id (ids differ from one database to the next), and a book is registered only the first time.
    """
    r = schema.research_run
    query = sa.select(r.c.run_id, r.c.config_json).where(r.c.family == config.family)
    with engine.connect() as conn:
        rows = pd.DataFrame(conn.execute(query).mappings().all(), columns=["run_id", "config_json"])
    run_id = matching_run(rows, config)
    if run_id is None:
        result: RunResult
        if isinstance(config, GarpConfig):
            result = run_garp(cache, config, engine=engine)
        elif isinstance(config, QgarpConfig):
            result = run_qgarp(cache, config, engine=engine)
        else:
            result = run_strategy(cache, config, engine=engine)
        run_id = result.run_id
    if run_id is None:  # a runner given an engine always registers
        raise LookupError(f"the run of {config.family} was not registered")
    return load_run(engine, run_id)


def snapshot(cache: ResearchCache, config: AnyConfig, *, on: str | pd.Timestamp | None = None) -> Snapshot:
    """``config``'s scores and book on the last trading day up to ``on`` (default today); no backtest, no trial."""
    end = pd.Timestamp(on if on is not None else pd.Timestamp.today()).normalize()
    book_from = end - BOOK_HISTORY
    if isinstance(config, GarpConfig):
        return _garp_snapshot(cache, config, book_from, end)
    if isinstance(config, QgarpConfig):
        return _qgarp_snapshot(cache, config, book_from, end)
    return _factor_snapshot(cache, config, book_from, end)


def _load(
    cache: ResearchCache,
    config: AnyConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    **options: Any,
) -> UniverseData:
    return load_universe_data(
        cache,
        config.universe,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        price_vendors=config.price_vendors,
        fundamentals_vendor=config.fundamentals_vendor,
        include_last_day=True,
        **options,
    )


def _on(panel: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    return pd.Series(panel.loc[day]) if day in panel.index else pd.Series(dtype=float)


def _factor_snapshot(
    cache: ResearchCache, config: StrategyConfig, book_from: pd.Timestamp, end: pd.Timestamp
) -> Snapshot:
    # a learned score is fitted on years of outcomes and the risk model needs its history, as in the backtest
    start = pd.Timestamp(data_start(replace(config, start=book_from.strftime("%Y-%m-%d"))))
    data = _load(cache, config, start, end, rebalance=config.rebalance)
    day = data.rebalances.max()
    sectors = listing_sectors(cache, data.close.columns) if config.within_sectors else None
    score, parts = strategy_scores(data, config, sectors=sectors, extra=cache_factors(cache, data, config))
    model = (
        risk_model_for(cache, data, price_vendors=config.price_vendors, fundamentals_vendor=config.fundamentals_vendor)
        if config.construction == "tilted"
        else None
    )
    book = strategy_book(data, config, score, data.rebalances[data.rebalances >= book_from], model=model)
    return Snapshot(
        config=config,
        day=day,
        data=data,
        score=_on(score, day),
        parts=pd.DataFrame({name: _on(panel, day) for name, panel in parts.items()}),
        weights=_on(book, day),
    )


def _garp_snapshot(cache: ResearchCache, config: GarpConfig, book_from: pd.Timestamp, end: pd.Timestamp) -> Snapshot:
    data = _load(cache, config, book_from, end)
    day = data.rebalances.max()
    growth, quality_value = garp_scores(data, config)
    # the selections of the year before (they decide the buffer), then one on the day itself
    trades = schedule(data.rebalances, pd.DatetimeIndex(data.returns.index), config.reference_months)
    trades = trades[trades < day]
    trades = pd.concat([trades, pd.Series([day], index=[day], dtype="datetime64[ns]")])
    sectors = {ref: listing_sectors(cache, data.members.columns, ref) for ref in trades.index}
    book = garp_book(growth, quality_value, sectors, trades, config)
    g, qv = _on(growth, day), _on(quality_value, day)
    # GARP ranks by quality & value among the best growth scores; the rest are not candidates on the day
    both = pd.DataFrame({"g": g, "qv": qv}).dropna()
    candidates = both.sort_values("g", ascending=False, kind="stable").head(config.growth_count).index
    return Snapshot(
        config=config,
        day=day,
        data=data,
        score=qv.reindex(candidates),
        parts=pd.DataFrame({"growth": g, "quality_value": qv}),
        weights=_on(book, day),
    )


def _qgarp_snapshot(cache: ResearchCache, config: QgarpConfig, book_from: pd.Timestamp, end: pd.Timestamp) -> Snapshot:
    data = _load(cache, config, book_from, end, history_days=HISTORY_DAYS)
    day = data.rebalances.max()
    _, comp, score = qgarp_scores(cache, data, config)
    book = qgarp_book(data, config, score, data.rebalances[data.rebalances >= book_from])
    on_day = comp[comp["date"] == day].set_index("listing_id")
    return Snapshot(
        config=config,
        day=day,
        data=data,
        # the book ranks by minus QGARP's percentile rank (1 = best); shown as a percentile, 100 = best, same order
        score=101.0 + _on(score, day),
        parts=on_day[[c for c in COMPOSITES if c in on_day.columns]],
        weights=_on(book, day),
    )
