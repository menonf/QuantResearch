"""A finished backtest measured and registered the same way, whatever built its book.

    metrics, run_id = finish_run(result, benchmark, asdict(config), asset_class="equity", family="value_ls",
                                 engine=engine, holdout=False)

Its own measures, those against the benchmark and - with an engine - the trial count, both deflated Sharpe ratios
and the registered run. Every strategy of every asset class ends here, so their registry entries compare.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from research_core.analytics.performance.metrics import active_performance, monthly_pair
from research_core.data.calendars import MONTHS, TRADING_DAYS
from research_core.engine.backtest import BacktestResult
from research_core.engine.registry import deflated_sharpe, record_run, trials_with


def finish_run(
    result: BacktestResult,
    benchmark: pd.Series,
    config: dict[str, Any],
    *,
    asset_class: str,
    family: str,
    engine: Engine | None,
    holdout: bool,
) -> tuple[dict[str, Any], int | None]:
    """``(metrics, run_id)`` of a finished backtest, measured and - with an ``engine`` - registered.

    Its own measures, those against ``benchmark``, and with an ``engine`` the trial count, both deflated Sharpe ratios
    and the registered run (``run_id`` None without one). Every strategy, whatever builds its book, is measured and
    registered the same way.
    """
    metrics: dict[str, Any] = result.metrics()
    pair = monthly_pair(result.net, benchmark)
    active = pair["r"] - pair["b"]
    if not benchmark.empty:
        metrics |= active_performance(result.net, benchmark)
    run_id = None
    if engine is not None:
        trials = trials_with(engine, family, config)
        metrics["trials"] = trials
        metrics["deflated_sharpe"] = _deflated(result.net.dropna(), float(metrics["sharpe"]), trials, TRADING_DAYS)
        if len(active) > 1:
            metrics["deflated_active_sharpe"] = _deflated(active, float(metrics["information_ratio"]), trials, MONTHS)
        run_id = record_run(engine, config, metrics, asset_class=asset_class, family=family, holdout_opened=holdout)
    return metrics, run_id


def deflated_active(net: pd.Series, benchmark: pd.Series, information_ratio: float, trials: int) -> float:
    """The deflated active Sharpe ratio :func:`finish_run` records, for another number of ``trials``.

    For a study whose trials span families - a second version designed after the first one's results is another try at
    the same question, so its runs are deflated by both families' trials.
    """
    pair = monthly_pair(net, benchmark)
    return _deflated(pair["r"] - pair["b"], information_ratio, trials, MONTHS)


def _deflated(returns: pd.Series, sharpe: float, trials: int, periods: int) -> float:
    """The deflated Sharpe ratio of ``returns`` (``periods`` a year; annualised ``sharpe``) after ``trials``."""
    r = returns.to_numpy(dtype=float)
    z = (r - r.mean()) / r.std() if r.std() > 0 else np.zeros_like(r)
    return deflated_sharpe(
        sharpe=sharpe,
        trials=trials,
        observations=len(r),
        skew=float(np.mean(z**3)),
        kurtosis=float(np.mean(z**4)),
        periods=periods,
    )
