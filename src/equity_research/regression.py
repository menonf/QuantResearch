"""The regression check (docs/ML4T_PLAN.md, rule 5): the existing research's key numbers, measured again and compared.

    now = measure(cache, engine)                      # strategies rerun and the risk model rebuilt, nothing registered
    save_baseline(now, snapshot)                      # kept in the artifact store under the data snapshot
    differences = compare(load_baseline(snapshot), measure(cache, engine))    # empty when nothing moved

Each phase of the machine-learning plan adds modules beside the existing research, under the rule that nothing
existing changes. This checks it:

1. **Strategies** - each family behind notebooks 3.1-3.6 and 4.1 (:data:`FAMILIES`): its latest registered run's
   configuration is backtested again by its own runner (:func:`runs.rerun`: no engine, so no trial is added), and its
   key metrics (:data:`KEYS`) are kept.
2. **Risk model** - notebook 4.2's model of the S&P 1500 is rebuilt the same way and its factor summary and average
   daily R-squared kept.
3. **Compare** - a baseline is keyed by the data snapshot, so a later measurement is compared with the baseline of
   the same data: a difference is then code, not data. :func:`compare` lists every number that moved by more than a
   rounding tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from equity_research.analytics.risk import risk_model as rm
from equity_research.analytics.risk.risk_data import load_risk_inputs
from equity_research.analytics.risk.risk_styles import substyles
from equity_research.data import artifacts as store
from equity_research.data.cache import ResearchCache
from equity_research.data.universe import load_universe_data
from equity_research.strategies.runs import load_run, rerun

#: The registry families of the existing strategy notebooks (3.1-3.6) and the factor dashboard (4.1).
FAMILIES = (
    "single_factor_ls",
    "multi_factor_ls",
    "multi_factor_lo",
    "value_only_lo",
    "garp_sp500",
    "barings_qgarp",
    "sp500_factor_dashboard",
)
#: The metrics kept for each strategy.
KEYS = ("cagr", "vol", "sharpe", "max_drawdown", "turnover", "active_return", "tracking_error", "information_ratio")
#: Notebook 4.2's risk model: universe, data window and first forecast.
RISK_MODEL = {"universe": "SP1500", "start": "2005-01-01", "end": "2024-12-31", "forecasts_from": "2007-01-01"}
#: Two measurements agree when every number is within this of the other (floating-point noise, not a change).
TOLERANCE = 1e-9


def strategy_numbers(cache: ResearchCache, engine: Engine, families: tuple[str, ...] = FAMILIES) -> pd.DataFrame:
    """``family x metric``: each family's latest registered run, backtested again on today's code and data."""
    # every registered run is read before the first rerun: the reruns take long enough for a serverless database to
    # pause, and the engine wakes it only when it is created
    runs = {family: load_run(engine, family=family) for family in families}
    rows = {}
    for family, run in runs.items():
        metrics = rerun(cache, run.config).metrics
        rows[family] = {"run_id": run.run_id, **{k: float(metrics[k]) for k in KEYS if k in metrics}}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("family")


def risk_model_numbers(cache: ResearchCache) -> pd.DataFrame:
    """``factor x measure``: notebook 4.2's factor summary, plus the average daily R-squared as one more row."""
    data = load_universe_data(cache, RISK_MODEL["universe"], RISK_MODEL["start"], RISK_MODEL["end"], history_days=500)
    inputs = load_risk_inputs(cache, data)
    exposures = rm.style_exposures(substyles(inputs), inputs)
    days = pd.DatetimeIndex(data.returns.index)
    model = rm.build_risk_model(data.returns, exposures, list(days[days >= pd.Timestamp(RISK_MODEL["forecasts_from"])]))
    summary = rm.factor_summary(model.factor_returns.loc[RISK_MODEL["forecasts_from"] :])
    summary.loc["average R2"] = np.nan
    summary.loc["average R2", summary.columns[0]] = float(model.r2.loc[RISK_MODEL["forecasts_from"] :].mean())
    return summary


def measure(cache: ResearchCache, engine: Engine) -> pd.DataFrame:
    """Every key number, long: ``part``, ``row``, ``measure``, ``value`` (strategies, then the risk model)."""
    parts = {"strategy": strategy_numbers(cache, engine), "risk_model": risk_model_numbers(cache)}
    return pd.concat([_long(frame, part) for part, frame in parts.items()], ignore_index=True)


def _long(frame: pd.DataFrame, part: str) -> pd.DataFrame:
    long = frame.rename_axis(index="row").reset_index().melt(id_vars="row", var_name="measure", value_name="value")
    return long.assign(part=part, row=long["row"].astype(str), measure=long["measure"].astype(str))[
        ["part", "row", "measure", "value"]
    ]


def compare(before: pd.DataFrame, after: pd.DataFrame, tolerance: float = TOLERANCE) -> pd.DataFrame:
    """The numbers of ``before`` and ``after`` (:func:`measure`) that differ, or are in only one: empty if none."""
    keys = ["part", "row", "measure"]
    both = before.merge(after, on=keys, how="outer", suffixes=("_before", "_after"), indicator=True)
    a, b = both["value_before"].astype(float), both["value_after"].astype(float)
    close = (a - b).abs() <= tolerance * np.maximum(1.0, np.maximum(a.abs(), b.abs()))
    same = (both["_merge"] == "both") & (close | (a.isna() & b.isna()))
    return both[~same].drop(columns="_merge").reset_index(drop=True)


def save_baseline(numbers: pd.DataFrame, snapshot: str | None, root: Path | None = None) -> Path:
    """Keep ``numbers`` as the baseline of data ``snapshot``."""
    return store.save(numbers, "regression", "baseline", {"families": list(FAMILIES)}, snapshot=snapshot, root=root)


def load_baseline(snapshot: str | None, root: Path | None = None) -> pd.DataFrame | None:
    """The baseline of data ``snapshot``, or ``None`` (none taken, or the data has changed since)."""
    return store.load("regression", "baseline", {"families": list(FAMILIES)}, snapshot=snapshot, root=root)
