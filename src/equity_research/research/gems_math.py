"""Arithmetic shared by the Barings GEMS factors: Spark's division and production's growth sign rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """``a / b``, missing where ``b`` is zero or missing (Spark's null, never an infinity)."""
    return a / b.where(b != 0)


def growth_piecewise(current: pd.DataFrame, past: pd.DataFrame, years: int | None = None) -> pd.DataFrame:
    """Production's sign rules for growth from ``past`` to ``current``.

    ``years=None`` is the 1-year version ((c - p) / |p|); 3 or 5 the compound rate a year, with a deterioration
    between two losses compounded on its size and negated. From or to zero, or across a sign change, the growth is
    +1 / -1 (better / worse); any missing input leaves it missing.
    """
    c = current.to_numpy(dtype=float)
    p = past.reindex_like(current).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (c - p) / np.abs(p)
        g = r if years is None else (r + 1) ** (1.0 / years) - 1
        g_neg = g if years is None else -((np.abs(r) + 1) ** (1.0 / years) - 1)
    conds = [
        (p == 0) & (c == 0),
        (p == 0) & (c < 0),
        (p == 0) & (c > 0),
        (p > 0) & (c == 0),
        (p > 0) & (c > 0),
        (p > 0) & (c < 0),
        (p < 0) & (c == 0),
        (p < 0) & (c > 0),
    ]
    choices: list[Any] = [0.0, -1.0, 1.0, -1.0, g, -1.0, 1.0, g]
    if years is None:
        conds += [(p < 0) & (c < 0), p == c]
        choices += [g, 0.0]
    else:
        conds += [(p < 0) & (c < 0) & (c < p), (p < 0) & (c < 0) & (c > p), p == c]
        choices += [g_neg, g, 0.0]
    out = np.select(conds, [np.broadcast_to(np.asarray(x, dtype=float), c.shape) for x in choices], default=np.nan)
    return pd.DataFrame(out, index=current.index, columns=current.columns)


def rows_on(panel: pd.DataFrame, days: pd.DatetimeIndex, *, limit: int = 5) -> pd.DataFrame:
    """``panel``'s rows on ``days`` (a calendar day takes the last trading day up to ``limit`` rows before it)."""
    grid = panel.reindex(panel.index.union(days)).ffill(limit=limit)
    return grid.reindex(days)
