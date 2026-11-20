"""Data-quality checks used as gates, recorded in ``ops.check_result``.

    results = run_checks(engine, checks, run_id=run.run_id)     # each asset class writes its own checks
    gate(results)                       # raises if any check failed: nothing downstream runs on bad data

A check measures one number with SQL and compares it with a warning and a failure threshold. What the checks are
belongs to the package that knows the tables (``equity_research.quality.checks``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from research_core.data.db import schema


class GateFailedError(RuntimeError):
    """A data-quality check failed."""


@dataclass(frozen=True)
class Check:
    """One measurement and its thresholds; ``higher_is_better`` sets the direction."""

    name: str
    sql: str
    warn: float
    fail: float
    higher_is_better: bool = True
    describe: str = ""
    examples: str | None = None  # a query listing the rows behind a warning, for whoever investigates it

    def status(self, value: float) -> str:
        """``pass``, ``warn`` or ``fail`` for a measured value."""
        if self.higher_is_better:
            return "fail" if value < self.fail else "warn" if value < self.warn else "pass"
        return "fail" if value > self.fail else "warn" if value > self.warn else "pass"


@dataclass(frozen=True)
class CheckResult:
    """A check's outcome."""

    name: str
    status: str
    value: float
    threshold: str


def run_checks(engine: Engine, checks: Sequence[Check], *, run_id: int | None = None) -> list[CheckResult]:
    """Measure every check and record the outcomes."""
    results = []
    with engine.begin() as conn:
        for c in checks:
            raw = conn.execute(sa.text(c.sql)).scalar()
            value = float(raw) if raw is not None else float("nan")
            status = "fail" if math.isnan(value) else c.status(value)  # NaN: nothing to measure
            op = ">=" if c.higher_is_better else "<="
            threshold = f"warn {op} {c.warn}, fail {op} {c.fail}"
            results.append(CheckResult(c.name, status, value, threshold))
            conn.execute(
                sa.insert(schema.check_result).values(
                    run_id=run_id, check_name=c.name, status=status, value=f"{value:.6g}", threshold=threshold
                )
            )
    return results


def gate(results: Sequence[CheckResult]) -> None:
    """Stop the pipeline when a check failed."""
    failed = [r for r in results if r.status == "fail"]
    if failed:
        raise GateFailedError("; ".join(f"{r.name} = {r.value:.6g} ({r.threshold})" for r in failed))


def examples(engine: Engine, check: Check) -> pd.DataFrame:
    """The rows behind a check's value (empty when the check has no examples query)."""
    if check.examples is None:
        return pd.DataFrame()
    with engine.connect() as conn:
        result = conn.execute(sa.text(check.examples))
        return pd.DataFrame(result.all(), columns=list(result.keys()))
