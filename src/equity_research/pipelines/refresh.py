"""Bring everything up to date in one go - the daily (or monthly) job.

    report = refresh(engine)

1. **every source** in :func:`~equity_research.settings.vendor_preferences` ``.sources``, in order, asking only for
   what changed (:mod:`.sources`): by default LSEG (members and listings in full; prices, dividends, splits and share
   counts from a few days before the newest price we hold, statements from a year before), Yahoo (the last month of
   benchmarks) and the SEC (its file again only when ours is a month old). When LSEG's daily limit stops it, the other
   sources still run and the next refresh continues;
2. **quality checks**, recorded in ``ops.check_result``; a failed check stops here with :class:`GateFailedError`,
   before the research cache can see the data;
3. **research cache**: new and changed rows copied to this computer;
4. **raw zone clean-up**: old copies of files the SEC publishes again anyway are removed (:func:`~.raw.store.prune`).

Every step writes only what changed, so a refresh with nothing new writes nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from equity_research.pipelines.sources import run_sources
from equity_research.quality.checks import CORE_CHECKS, gate, run_checks
from equity_research.raw.store import prune
from equity_research.research.cache import ResearchCache

log = logging.getLogger(__name__)


def refresh(engine: Engine, *, cache: ResearchCache | None = None) -> dict[str, Any]:
    """Run the steps of the module docstring; returns one report per source and step."""
    report: dict[str, Any] = dict(run_sources(engine, "refresh"))
    results = run_checks(engine, CORE_CHECKS)
    report["checks"] = {r.name: {"status": r.status, "value": r.value} for r in results}
    gate(results)
    report["cache"] = (cache or ResearchCache()).sync(engine)
    report["pruned_raw_files"] = prune_raw(engine)
    return report


def prune_raw(engine: Engine) -> dict[str, float]:
    """The raw-zone clean-up (:func:`~equity_research.raw.store.prune`), summed up."""
    with engine.begin() as conn:
        pruned = prune(conn)
    return {"files": len(pruned), "megabytes": round(sum(p.bytes for p in pruned) / 1e6, 1)}


def written(report: Mapping[str, Any]) -> pd.DataFrame:
    """``[step, what, inserted, updated]``: every write counted in a pipeline report, however deeply nested."""
    rows: list[tuple[str, str, int, int]] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            if {"inserted", "updated"} <= set(node):
                rows.append((path[0], " / ".join(path[1:]) or path[0], int(node["inserted"]), int(node["updated"])))
                return
            for key, value in node.items():
                walk(value, (*path, str(key)))

    for step, part in report.items():
        walk(part, (str(step),))
    return pd.DataFrame(rows, columns=["step", "what", "inserted", "updated"])
