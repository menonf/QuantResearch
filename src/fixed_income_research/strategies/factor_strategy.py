"""Factor investing in a bond index: one frozen config in, every fifth of every signal backtested, the best registered.

    config = FactorConfig(family="hy_factors", index="H0A0", start="2019-07-31", end="2024-12-31")
    run = run_factors(cache, config, engine=engine)      # fenced, scored, backtested, registered
    run.backtests["composite"][5].net                     # the composite's best fifth, after costs
    quantile_table(run)                                   # every fifth of every signal against the index

Steps:

1. **Fence** - a backtest ending in the hold-out (``EQR_HOLDOUT_START``) is refused unless the config opens it.
2. **Read** - the index's month ends up to ``end`` (:func:`load_index_data`).
3. **Returns** - each bond's excess (or total) return over the month ending each month end; a bond that left the
   index has the index's return that month (:func:`with_leavers`), so every book and the index are measured alike.
4. **Scores** (:func:`factor_scores`) - the factors of :mod:`.factors`, turned so that higher is better (cheap, strong
   momentum, small issuer, safe), on each month end's members, and their weighted composite.
5. **Books** - on each month end from ``start`` to ``end``, every fifth of every signal (:func:`quantile_books`),
   equally weighted or in proportion to the bonds' index weights. A book formed on month end *t* from the file of *t*
   earns month *t+1*'s return, read from the file of *t+1*: the engine's timing, a book set on a date earns from the
   next one. Nothing of the file of *t+1* reaches the book of *t* (:func:`point_in_time_check` shows it).
6. **Backtest** - each book through :func:`run_backtest` at twelve periods a year, ``cost_bps`` charged on what is
   traded; the index at its own weights and its members equally weighted are run the same way at no cost: the
   benchmarks.
7. **Register** - each signal's best fifth, the long-only book one could hold, measured against the index and recorded
   in ``research.run`` with its family's trial count (:func:`finish_run`).

The functions after :func:`run_factors` measure a finished run for the notebooks: every fifth against the index, the
best-less-worst spreads, a fifth's weight by group against the index's, and today's ranking of the members.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from fixed_income_research import ASSET_CLASS
from fixed_income_research.analytics.replication import weight_by_group, with_leavers
from fixed_income_research.data.index_data import IndexData, load_index_data
from fixed_income_research.strategies import factors as fx
from research_core.analytics.performance.metrics import active_performance, capture, performance
from research_core.data.cache import ResearchCache
from research_core.data.calendars import MONTHS
from research_core.engine.backtest import BacktestResult, run_backtest
from research_core.engine.finish import finish_run
from research_core.engine.portfolio import quantile_books, weighted_by
from research_core.engine.registry import fence

#: The factors a config can mix, as :func:`factor_scores` names them.
FACTORS = ("value", "momentum", "size", "low_risk")
#: How a fifth's bonds share it: ``equal``, or ``market_value`` - in proportion to their index weights.
WEIGHTINGS = ("equal", "market_value")
#: Which of the provider's monthly returns the books earn: over duration-matched government bonds, or in full.
RETURN_KINDS = ("excess", "total")


@dataclass(frozen=True)
class FactorConfig:
    """Every setting of a run (hashed into its registry entry)."""

    family: str
    index: str = "H0A0"
    start: str = "2019-07-31"  # the first book: the first month end every factor has a score (momentum's 6+1 months)
    end: str = "2024-12-31"
    factors: tuple[tuple[str, float], ...] = (("value", 1.0), ("momentum", 1.0), ("size", 1.0), ("low_risk", 1.0))
    returns: str = "excess"
    quantiles: int = 5
    weighting: str = "equal"
    cost_bps: float = 25.0  # one way, on what is traded: about half a high-yield bond's bid-ask spread
    momentum_months: int = 6
    momentum_skip: int = 1
    oas_change_months: int = 3
    maturity_share: float = 0.5
    rating_scale: str = "high_yield"
    open_holdout: bool = False

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("weighting", self.weighting, WEIGHTINGS),
            ("returns", self.returns, RETURN_KINDS),
            ("rating_scale", self.rating_scale, tuple(fx.RATING_SCALES)),
        ):
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; choose from {allowed}")
        unknown = {name for name, _ in self.factors} - set(FACTORS)
        if unknown:
            raise ValueError(f"unknown factors {sorted(unknown)}; choose from {FACTORS}")
        object.__setattr__(self, "factors", tuple((name, float(w)) for name, w in self.factors))


@dataclass(frozen=True)
class FactorRun:
    """A finished run: its inputs, every signal's scores, books and backtests, and the registered measures."""

    config: FactorConfig
    data: IndexData
    returns: pd.DataFrame  # month end x bond: the return over the month ending that day, leavers given the index's
    outcome: pd.DataFrame  # month end x bond: the return over the month that follows (missing for a bond that left)
    scores: dict[str, pd.DataFrame]  # signal -> month end x bond, higher is better, members only
    books: dict[str, dict[int, pd.DataFrame]]  # signal -> fifth -> target weights on the rebalance dates
    backtests: dict[str, dict[int, BacktestResult]]
    index: BacktestResult  # the index at its own weights
    equal_weighted: BacktestResult  # its members equally weighted
    metrics: dict[str, dict[str, Any]]  # signal -> its best fifth's registered measures
    run_ids: dict[str, int | None]

    @property
    def members(self) -> pd.DataFrame:
        """Month end x bond: in the index after that month end's rebalance."""
        return self.data.weights.notna()

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The rebalance dates: the month ends from ``start`` to ``end``."""
        d = self.data.dates
        return d[(d >= pd.Timestamp(self.config.start)) & (d <= pd.Timestamp(self.config.end))]

    @property
    def benchmark(self) -> pd.Series:
        """The index's monthly return over the backtest's months."""
        return self.index.net


def factor_scores(data: IndexData, config: FactorConfig) -> dict[str, pd.DataFrame]:
    """The config's factors on each month end's members, higher is better, and their ``composite``.

    Momentum is always measured on excess returns - a bond's credit performance, not the government yields it shares
    with every other bond of its duration - whichever return the books earn.
    """
    members = data.weights.notna()
    years = fx.years_to_maturity(data.maturity)
    build = {
        "value": lambda: fx.oas_value(
            data.analytics["oas_bp"], years, data.rating, data.industry, change_months=config.oas_change_months
        ),
        "momentum": lambda: fx.momentum(data.excess_returns, months=config.momentum_months, skip=config.momentum_skip),
        "size": lambda: -fx.issuer_weight(data.weights, data.issuer),
        "low_risk": lambda: (
            -fx.riskiness(
                years, data.rating, scale=fx.RATING_SCALES[config.rating_scale], maturity_share=config.maturity_share
            )
        ),
    }
    scores = {name: build[name]().reindex_like(members).where(members) for name, _ in config.factors}
    scores["composite"] = fx.composite(scores, dict(config.factors), members)
    return scores


def registered_settings(config: FactorConfig, signal: str) -> dict[str, Any]:
    """What the registry stores of one signal's run: the config, the signal, and the mix only where it matters.

    A single factor's book does not depend on the composite's weights, so changing them is no new trial of it.
    """
    settings = asdict(config) | {"signal": signal}
    if signal != "composite":
        del settings["factors"]
    return settings


def run_on(data: IndexData, config: FactorConfig, *, engine: Engine | None = None, holdout: bool = False) -> FactorRun:
    """Steps 3-7 on an index already read (:func:`run_factors` fences and reads first)."""
    end = pd.Timestamp(config.end)
    raw = data.excess_returns if config.returns == "excess" else data.total_returns
    earned = with_leavers(data.weights, raw)
    earned = earned.loc[earned.index <= end]
    members = data.weights.notna()
    dates = data.dates[(data.dates >= pd.Timestamp(config.start)) & (data.dates <= end)]
    if dates.empty:
        raise ValueError(f"no month end of {config.index} between {config.start} and {config.end}")

    def backtest(book: pd.DataFrame, cost: float) -> BacktestResult:
        return run_backtest(book, earned, cost_bps=cost, periods_per_year=MONTHS)

    scores = factor_scores(data, config)
    books: dict[str, dict[int, pd.DataFrame]] = {}
    for name, score in scores.items():
        fifths = quantile_books(score, dates, quantiles=config.quantiles)
        if config.weighting == "market_value":
            fifths = {q: weighted_by(book, data.weights) for q, book in fifths.items()}
        books[name] = fifths
    backtests = {name: {q: backtest(b, config.cost_bps) for q, b in fifths.items()} for name, fifths in books.items()}

    index_book = data.weights.loc[dates].fillna(0.0)
    held = members.loc[dates].astype(float)
    index = backtest(index_book, 0.0)
    equal = backtest(held.div(held.sum(axis=1), axis=0), 0.0)

    metrics: dict[str, dict[str, Any]] = {}
    run_ids: dict[str, int | None] = {}
    for name in scores:
        metrics[name], run_ids[name] = finish_run(
            backtests[name][config.quantiles],
            index.net,
            registered_settings(config, name),
            asset_class=ASSET_CLASS,
            family=config.family,
            engine=engine,
            holdout=holdout,
            periods_per_year=MONTHS,
        )
    outcome = fx.calendar_shift(raw, -1).where(members)
    return FactorRun(config, data, earned, outcome, scores, books, backtests, index, equal, metrics, run_ids)


def run_factors(cache: ResearchCache, config: FactorConfig, *, engine: Engine | None = None) -> FactorRun:
    """Fence, read the index, score, build every fifth, backtest, and (with ``engine``) register the best fifths."""
    holdout = fence(config.end, open_holdout=config.open_holdout)
    data = load_index_data(cache, config.index, end=config.end)
    return run_on(data, config, engine=engine, holdout=holdout)


def quantile_table(run: FactorRun, *, net: bool = True) -> pd.DataFrame:
    """One row per signal and fifth: its own measures, those against the index, and its turnover a year.

    ``net`` after the trading costs, else before them (the signal's quality alone). Turnover is what was bought and
    sold in a year, as a multiple of the book.
    """
    rows = {}
    benchmark = run.benchmark
    for name, fifths in run.backtests.items():
        for q, result in fifths.items():
            r = result.net if net else result.gross
            rows[(name, q)] = (
                performance(r, periods_per_year=MONTHS)
                | active_performance(r, benchmark)
                | capture(r, benchmark)
                | {"turnover": result.metrics(periods_per_year=MONTHS)["turnover"]}
            )
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["signal", "fifth"])


def fan(table: pd.DataFrame, measure: str = "active_return") -> pd.DataFrame:
    """Signal x fifth: one measure of :func:`quantile_table`, and the best fifth's less the worst's.

    A signal that sorts the bonds shows a measure rising from the worst fifth to the best, not only a good best fifth.
    """
    signals = table.index.get_level_values("signal").unique()
    out = table.reset_index().pivot_table(index="signal", columns="fifth", values=measure).reindex(signals)
    top, bottom = max(out.columns), min(out.columns)
    out[f"{top} less {bottom}"] = out[top] - out[bottom]
    return out


def spreads(run: FactorRun, *, net: bool = False) -> pd.DataFrame:
    """Month x signal: the best fifth's return less the worst's - what the signal sorted, whatever the market did."""
    top = run.config.quantiles
    return pd.DataFrame(
        {name: (f[top].net - f[1].net) if net else (f[top].gross - f[1].gross) for name, f in run.backtests.items()}
    )


def spread_table(spread: pd.DataFrame) -> pd.DataFrame:
    """Per signal: the spread's growth a year, volatility, Sharpe ratio, share of months positive and t-statistic."""
    rows = {}
    for name in spread.columns:
        s = spread[name].dropna()
        own = performance(s, periods_per_year=MONTHS)
        rows[name] = {
            "cagr": own["cagr"],
            "vol": own["vol"],
            "sharpe": own["sharpe"],
            "positive": float((s > 0).mean()),
            "t": float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 1 and s.std() > 0 else float("nan"),
            "months": len(s),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def active_weights(run: FactorRun, signal: str, fifth: int, groups: pd.DataFrame) -> pd.DataFrame:
    """Rebalance date x group: the fifth's weight in each group (an industry, a rating band) less the index's."""
    book = run.books[signal][fifth]
    index = run.data.weights.reindex(book.index).fillna(0.0)
    mine, theirs = weight_by_group(book, groups), weight_by_group(index, groups)
    return mine.sub(theirs, fill_value=0.0)


def _row(panel: pd.DataFrame, d: pd.Timestamp, columns: pd.Index) -> pd.Series:
    """The row of month end ``d`` over ``columns`` (missing where the panel has no such column)."""
    return panel.loc[[d]].reindex(columns=columns).iloc[0]


def point_in_time_check(data: IndexData, config: FactorConfig, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per date: each signal's largest change on that date when every later month end is deleted (0: no look-ahead).

    The scores of a month end must be the same whether the files after it exist or not - otherwise the book of that
    month end reads its own future. A bond scored with the later files and not without them (or the reverse) is a
    change too, of infinite size. Also the number of bonds the composite scored.
    """
    everything = factor_scores(data, config)
    rows = {}
    for d in dates:
        known = factor_scores(data.until(d), config)
        row = {"bonds": float(_row(everything["composite"], d, data.weights.columns).notna().sum())}
        for name, full in everything.items():
            later = _row(full, d, full.columns).to_numpy(dtype=float)
            then = _row(known[name], d, full.columns).to_numpy(dtype=float)
            if (np.isnan(later) != np.isnan(then)).any():
                row[name] = float("inf")
                continue
            both = ~np.isnan(later)
            row[name] = float(np.abs(later[both] - then[both]).max()) if both.any() else 0.0
        rows[d] = row
    return pd.DataFrame.from_dict(rows, orient="index").astype({"bonds": int}).rename_axis("month end")


def screen(
    data: IndexData, scores: Mapping[str, pd.DataFrame], date: str | pd.Timestamp, issuers: pd.Series
) -> pd.DataFrame:
    """Every member of ``date``, best composite first: who it is, its spread, and each signal's percentile (1 best).

    ``issuers`` names the issuer ids (``core.issuer``). The percentiles are among that month end's members.
    """
    d = pd.Timestamp(date)
    on = _row(data.weights, d, data.weights.columns).dropna().index
    table = pd.DataFrame(
        {
            "bond": data.names.reindex(on),
            "issuer": [issuers.get(int(i), "") if pd.notna(i) else "" for i in _row(data.issuer, d, on)],
            "rating": _row(data.rating, d, on),
            "industry": _row(data.industry, d, on),
            "years": _row(fx.years_to_maturity(data.maturity.loc[[d]]), d, on),
            "oas_bp": _row(data.analytics["oas_bp"], d, on),
            "weight": _row(data.weights, d, on),
        },
        index=on,
    )
    for name, score in scores.items():
        table[name] = _row(fx.percentile(score.loc[[d]]), d, on)
    return table.sort_values("composite", ascending=False).rename_axis("listing_id")
