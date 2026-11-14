"""Research discipline in code: every backtest is recorded, trials are counted, the hold-out is fenced.

    fence(end="2026-09-30")                                    # raises: the hold-out is closed
    run_id = record_run(engine, config, metrics, family="value_quality_ls", data_snapshot=snap)
    deflated_sharpe(sharpe=0.65, trials=trial_count(engine, "value_quality_ls"), observations=4700)

* **Registry.** ``research.run`` keeps the config, its hash, the code version, the data snapshot, the metrics and who
  ran it - so the number of variants tried is a fact, not a memory.
* **Deflated Sharpe ratio** (Bailey & Lopez de Prado, 2014): the probability that the true Sharpe exceeds what the
  best of ``trials`` independent no-skill strategies would show by chance. Below ~0.95 the result is not convincing.
* **Hold-out fence.** Periods from the hold-out start (``EQR_HOLDOUT_START``) are refused unless the hold-out is
  opened explicitly - once a strategy is final - and a run that opened it is marked.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research.db import schema
from equity_research.raw.store import code_version
from equity_research.settings import get_settings

EULER_GAMMA = 0.5772156649015329


class HoldoutClosedError(RuntimeError):
    """A backtest reached into the hold-out without opening it."""


def config_hash(config: Mapping[str, Any]) -> str:
    """16-character hash of a config (key order does not matter)."""
    canonical = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def fence(end: str | pd.Timestamp, *, open_holdout: bool = False, holdout_start: str | None = None) -> bool:
    """Refuse a backtest ending in the hold-out unless ``open_holdout``; returns whether the hold-out is touched."""
    start = pd.Timestamp(holdout_start or get_settings().holdout_start)
    touches = pd.Timestamp(end) >= start
    if touches and not open_holdout:
        raise HoldoutClosedError(
            f"the backtest ends {pd.Timestamp(end).date()}, inside the hold-out from {start.date()}: end it before, "
            "or open the hold-out explicitly (only for a final strategy)"
        )
    return bool(touches)


def record_run(
    engine: Engine,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    family: str,
    data_snapshot: pd.Timestamp | None = None,
    holdout_opened: bool = False,
    notes: str | None = None,
) -> int:
    """Register a finished backtest; returns its ``run_id``. ``family`` groups the variants of one idea."""
    payload = {"family": family, **config}
    with engine.begin() as conn:
        run_id = conn.execute(
            sa.insert(schema.research_run)
            .values(
                family=family,
                config_hash=config_hash(payload),
                config_json=json.dumps(payload, sort_keys=True, default=str),
                code_version=code_version(),
                data_snapshot=None if data_snapshot is None else pd.Timestamp(data_snapshot).to_pydatetime(),
                user_name=getpass.getuser(),
                finished_at=sa.func.sysutcdatetime(),
                holdout_opened=holdout_opened,
                metrics_json=json.dumps(dict(metrics), default=float),
                notes=notes,
            )
            .returning(schema.research_run.c.run_id)
        ).scalar_one()
    return int(run_id)


#: What a registered configuration of a machine-learning study is (docs/ML4T_PLAN.md): a fit judged on validation
#: folds, a strategy backtested before the hold-out, or the one run that opens the hold-out.
STAGES = ("training", "backtest", "holdout")


def record_search(
    engine: Engine,
    family: str,
    points: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    stage: str = "training",
    search: str | None = None,
    data_snapshot: pd.Timestamp | None = None,
) -> list[int]:
    """Register every point of a search as a trial of ``family``; returns the run ids in order.

    ``points`` are ``(config, metrics)`` pairs - each hyperparameter setting tried and how it scored on its folds.
    A search registered whole means :func:`trial_count`, and so every deflated Sharpe ratio of the family, counts all
    it tried, not only the setting it kept. Each config records its ``stage`` (one of :data:`STAGES`) and the
    ``search`` it belongs to; a point registered again (a search rerun) is not another trial. Only the ``holdout``
    stage marks its runs as having opened the hold-out.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
    tagged = {"stage": stage} | ({"search": search} if search is not None else {})
    return [
        record_run(
            engine,
            tagged | dict(config),
            metrics,
            family=family,
            data_snapshot=data_snapshot,
            holdout_opened=stage == "holdout",
        )
        for config, metrics in points
    ]


def trial_count(engine: Engine, family: str) -> int:
    """How many distinct configurations of ``family`` have been run."""
    r = schema.research_run
    with engine.connect() as conn:
        n = conn.execute(sa.select(sa.func.count(sa.distinct(r.c.config_hash))).where(r.c.family == family)).scalar()
    return int(n or 0)


def trials_with(engine: Engine, family: str, config: Mapping[str, Any]) -> int:
    """How many distinct configurations of ``family`` there are once ``config`` has run.

    One more than :func:`trial_count` only when ``config`` is new: running a registered configuration again (a
    notebook re-executed) is not another variant tried.
    """
    r = schema.research_run
    digest = config_hash({"family": family, **config})
    with engine.connect() as conn:
        seen = conn.execute(
            sa.select(r.c.run_id).where(r.c.family == family, r.c.config_hash == digest).limit(1)
        ).first()
    return trial_count(engine, family) + (0 if seen else 1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF by bisection (enough precision for a threshold)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def expected_max_sharpe(trials: int, observations: int) -> float:
    """Per-period Sharpe the best of ``trials`` no-skill strategies reaches by chance (variance 1 / observations)."""
    if trials <= 1:
        return 0.0
    sd = math.sqrt(1.0 / observations)
    return sd * ((1 - EULER_GAMMA) * _norm_ppf(1 - 1 / trials) + EULER_GAMMA * _norm_ppf(1 - 1 / (trials * math.e)))


def deflated_sharpe(
    *, sharpe: float, trials: int, observations: int, skew: float = 0.0, kurtosis: float = 3.0, periods: int = 252
) -> float:
    """Probability that the true Sharpe beats the best of ``trials`` lucky ones.

    ``sharpe`` is annualised; ``observations`` the number of return periods; ``skew`` / ``kurtosis`` of the returns.
    """
    sr = sharpe / math.sqrt(periods)  # per period
    sr0 = expected_max_sharpe(trials, observations)
    denom = math.sqrt(max(1 - skew * sr + (kurtosis - 1) / 4 * sr * sr, 1e-12))
    return _norm_cdf((sr - sr0) * math.sqrt(observations - 1) / denom)
