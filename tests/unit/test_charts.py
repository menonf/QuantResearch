import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import matplotlib as mpl

mpl.use("Agg")

from research_core import charts

DAYS = pd.bdate_range("2020-01-01", periods=300)
R = pd.Series(np.random.default_rng(0).normal(0.0004, 0.01, 300), index=DAYS)


def test_every_chart_draws_with_labels_and_a_legend_for_several_series() -> None:
    charts.style()
    fig = charts.growth({"A": R, "B": R * 0.5}, "Growth of $100")
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert {t.get_text().split("  ")[0] for t in ax.texts} == {"A", "B"}  # the end labels
    charts.lines({"Only": R.cumsum()}, "One line", percent=True)
    charts.drawdown(R, "Below the best")
    charts.yearly_bars(pd.DataFrame({"A": [0.1, -0.05], "B": [0.08, 0.02]}, index=[2020, 2021]), "Years")
    charts.ranked_bars(pd.Series({"x": 0.3, "y": -0.1}), "Ranked", percent=True)
