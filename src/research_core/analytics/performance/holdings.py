"""What books hold, compared: overlap, active share, concentration, weights by group, exposures to scores.

    overlap = portfolio_overlap(weights)                  # how much weight two books hold in common
    active_share(book, benchmark_weights)                 # 0 for the benchmark itself, 1 for nothing in common
    tilt = exposures(book, benchmark_weights, scores)     # the book's weighted score less the benchmark's

Pure functions over weight series indexed by instrument.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def portfolio_overlap(weights: Mapping[str, pd.Series]) -> pd.DataFrame:
    """``book x book``: the weight two books hold in common (the sum of the smaller of each stock's two weights)."""
    names = list(weights)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            both = pd.concat({"a": weights[a], "b": weights[b]}, axis=1).fillna(0.0)
            out.loc[a, b] = float(both.min(axis=1).sum())
    return out


def active_share(weights: pd.Series, benchmark: pd.Series) -> float:
    """Half the sum of the absolute weight differences: 0 for the benchmark itself, 1 for nothing in common."""
    both = pd.concat({"w": weights, "b": benchmark}, axis=1).fillna(0.0)
    return float((both["w"] - both["b"]).abs().sum() / 2)


def sector_weights(weights: pd.Series, sectors: pd.Series) -> pd.Series:
    """Weight per sector (a stock without one counts as ``unknown``)."""
    held = weights[weights > 0]
    return held.groupby(sectors.reindex(held.index).fillna("unknown").to_numpy()).sum()


def concentration(weights: pd.Series) -> float:
    """The Herfindahl-Hirschman index of the weights in percent: 10,000 for one stock, 100 for a hundred equal ones."""
    w = weights[weights > 0]
    return float(((100 * w / w.sum()) ** 2).sum())


def weighted_average(weights: pd.Series, values: pd.Series) -> float:
    """The weighted average of ``values``, over the held stocks that have a value."""
    both = pd.concat({"w": weights[weights > 0], "v": values}, axis=1, join="inner").dropna()
    return float((both["w"] * both["v"]).sum() / both["w"].sum()) if both["w"].sum() > 0 else float("nan")


def exposures(weights: pd.Series, benchmark: pd.Series, scores: pd.DataFrame) -> pd.Series:
    """Per factor: the book's weighted average score less the benchmark's (0 = the benchmark's own exposure)."""
    return pd.Series(
        {f: weighted_average(weights, scores[f]) - weighted_average(benchmark, scores[f]) for f in scores.columns}
    )
