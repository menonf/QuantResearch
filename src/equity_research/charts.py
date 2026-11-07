"""The few chart types the notebooks use, drawn one consistent way (matplotlib, in the ``notebooks`` group).

    growth({"Our rebuild": ours, "S&P 500 (^GSPC)": gspc}, "Growth of $100")
    yearly_bars(table, "Return per calendar year")

Colours carry identity in a fixed order (a series keeps its colour wherever it appears); lines are thin; every
chart with several series has a legend and labels at the line ends, so colour is never the only clue. Charts show
one quantity on one axis.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from matplotlib.ticker import FuncFormatter, PercentFormatter

#: Categorical colours, assigned in this order (validated for colour-blind separation on the light surface).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEGATIVE = "#e34948"


def style() -> None:
    """Apply the chart style to every figure that follows."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "legend.labelcolor": TEXT,
            "lines.linewidth": 1.6,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def _figure(title: str, subtitle: str | None = None, *, height: float = 4.2) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, height))
    ax.set_title(title, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=TEXT_SECONDARY, fontsize=9.5, va="bottom")
    return fig, ax


def _end_labels(ax: Axes, lines: Mapping[str, pd.Series], fmt: str) -> None:
    """Each series' name and last value at its right end (in text colour; the line beside it carries the colour)."""
    for name, raw in lines.items():
        s = raw.dropna()
        if s.empty:
            continue
        ax.annotate(
            f"{name}  {format(float(s.iloc[-1]), fmt)}",
            (s.index[-1], float(s.iloc[-1])),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT,
        )


def growth(
    series: Mapping[str, pd.Series], title: str, *, subtitle: str | None = None, log: bool = True, symbol: str = "$"
) -> Figure:
    """What 100 grows to, following each series of daily returns (from the first day they all have).

    ``symbol`` is the currency's on the axis (``£`` for London's indices).
    """
    both = pd.concat(series, axis=1, sort=True).dropna()
    value = 100 * (1 + both).cumprod()
    fig, ax = _figure(title, subtitle)
    for i, name in enumerate(value.columns):
        ax.plot(value.index, value[name], color=SERIES[i % len(SERIES)], label=name)
    if log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{symbol}{v:,.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    _end_labels(ax, {c: value[c] for c in value.columns}, ",.0f")
    ax.margins(x=0.02)
    ax.set_xlim(value.index[0], value.index[-1] + (value.index[-1] - value.index[0]) * 0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def lines(series: Mapping[str, pd.Series], title: str, *, subtitle: str | None = None, percent: bool = False) -> Figure:
    """One or more series over time on one axis (a single series needs no legend)."""
    fig, ax = _figure(title, subtitle)
    for i, (name, s) in enumerate(series.items()):
        ax.plot(s.index, s.to_numpy(), color=SERIES[i % len(SERIES)], label=name)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if len(series) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def drawdown(returns: pd.Series, title: str, *, subtitle: str | None = None) -> Figure:
    """How far below its previous best the series was on each day (0 % = at a new high)."""
    value = (1 + returns.dropna()).cumprod()
    below = value / value.cummax() - 1
    fig, ax = _figure(title, subtitle, height=3.4)
    ax.fill_between(below.index, below.to_numpy(), 0, color=NEGATIVE, alpha=0.35, linewidth=0)
    ax.plot(below.index, below.to_numpy(), color=NEGATIVE, linewidth=1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    worst = below.idxmin()
    ax.annotate(
        f"worst {below.min():.0%} ({worst:%b %Y})",
        (worst, below.min()),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=TEXT,
    )
    fig.tight_layout()
    return fig


def yearly_bars(table: pd.DataFrame, title: str, *, subtitle: str | None = None) -> Figure:
    """Grouped bars: one group per year (rows), one bar per series (columns, at most four)."""
    fig, ax = _figure(title, subtitle)
    n = len(table.columns)
    width = 0.8 / n
    x = np.arange(len(table))
    for i, name in enumerate(table.columns):
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            table[name].to_numpy(),
            width=width * 0.92,  # a gap of surface between neighbouring bars
            color=SERIES[i % len(SERIES)],
            label=name,
        )
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(x, [str(i) for i in table.index])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if n > 1:
        ax.legend(loc="upper left", ncols=n)
    fig.tight_layout()
    return fig


def ranked_bars(
    values: pd.Series,
    title: str,
    *,
    subtitle: str | None = None,
    percent: bool = False,
    reference: tuple[str, float] | None = None,
) -> Figure:
    """Horizontal bars, largest at the top, each labelled with its value.

    ``reference``: a named value drawn as a dashed line across the bars (a benchmark's return, say).
    """
    s = values.sort_values()
    fig, ax = _figure(title, subtitle, height=max(2.5, 0.32 * len(s) + 1.2))
    colors = [SERIES[0] if v >= 0 else NEGATIVE for v in s.to_numpy()]
    ax.barh([str(i) for i in s.index], s.to_numpy(), color=colors, height=0.7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for y, v in enumerate(s.to_numpy()):
        label = f"{v:.1%}" if percent else f"{v:.2f}"
        ax.annotate(
            label,
            (v, y),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if reference is not None:
        name, value = reference
        ax.axvline(value, color=SERIES[1], linestyle="--", linewidth=1.2)
        shown = f"{value:.1%}" if percent else f"{value:.2f}"
        ax.annotate(
            f"{name} {shown}",
            (value, len(s) - 0.5),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=9,
            color=SERIES[1],
            va="bottom",
        )
    fig.tight_layout()
    return fig


def labeled_scatter(
    points: pd.DataFrame,
    title: str,
    *,
    xlabel: str,
    ylabel: str,
    subtitle: str | None = None,
    highlight: str | None = None,
    percent: bool = True,
) -> Figure:
    """One labelled point per row (columns ``x`` and ``y``); ``highlight`` names a row drawn apart (the benchmark)."""
    fig, ax = _figure(title, subtitle, height=6.0)
    ax.grid(axis="x", visible=True)
    for name, row in points.iterrows():
        special = name == highlight
        ax.scatter(
            row["x"],
            row["y"],
            color=NEGATIVE if special else SERIES[0],
            marker="s" if special else "o",
            s=46 if special else 30,
            zorder=3,
        )
        ax.annotate(
            str(name),
            (row["x"], row["y"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            color=NEGATIVE if special else TEXT,
            fontweight="bold" if special else "normal",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.margins(x=0.12, y=0.08)
    fig.tight_layout()
    return fig


def radar(series: Mapping[str, pd.Series], title: str, *, limit: float = 1.0) -> Figure:
    """Exposures on a ring of axes (one per index entry), each series a filled outline; the benchmark is the 0 ring.

    Values beyond +/- ``limit`` are drawn at the edge, so one extreme exposure does not shrink the others.
    """
    names = list(next(iter(series.values())).index)
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = cast(PolarAxes, fig.add_subplot(projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    closed = np.append(angles, angles[0])
    ax.plot(closed, np.zeros(len(closed)) + limit, color=SERIES[1], linestyle="--", linewidth=1.2, label="Benchmark")
    for i, (label, s) in enumerate(series.items()):
        values = np.clip(s.reindex(names).to_numpy(dtype=float), -limit, limit) + limit
        values = np.append(values, values[0])
        color = SERIES[(i * 2) % len(SERIES)]
        ax.plot(closed, values, color=color, label=label)
        ax.fill(closed, values, color=color, alpha=0.25)
    ax.set_xticks(angles, names)
    ax.set_ylim(0, 2 * limit)
    ax.set_yticklabels([])
    ax.set_title(title, pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.18, -0.08), fontsize=9)
    fig.tight_layout()
    return fig


def fit_panels(
    x: np.ndarray,
    y: np.ndarray,
    curves: pd.DataFrame,
    title: str,
    *,
    labels: Mapping[str, str] | None = None,
    columns: int = 4,
) -> Figure:
    """Small multiples: the same points ``(x, y)`` in every panel, one column of ``curves`` (index: x) in each.

    For comparing how differently several models fit the same data, on shared axes.
    """
    rows = -(-len(curves.columns) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(10, 2.5 * rows + 0.6), sharex=True, sharey=True, squeeze=False)
    low, high = np.quantile(y, [0.01, 0.99])
    for ax, name in zip(axes.flat, curves.columns, strict=False):
        ax.scatter(x, y, s=4, color=TEXT_SECONDARY, alpha=0.35, linewidths=0)
        ax.plot(curves.index, curves[name].to_numpy(), color=SERIES[0], linewidth=1.8)
        ax.set_title((labels or {}).get(str(name), str(name)), fontsize=10)
        ax.set_ylim(low - 0.2 * (high - low), high + 0.2 * (high - low))
    for ax in list(axes.flat)[len(curves.columns) :]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig
