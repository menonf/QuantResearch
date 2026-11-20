"""How closely one daily return series follows another: tracking error, correlation, beta - overall and by year.

    tracking(ours, theirs)                                        # one comparison
    tracking_summary({"our copy vs the index": (ours, theirs)})   # one row per comparison
    yearly_tracking_errors(pairs)                                 # when a reconstruction drifts
    calendar_years({"ours": ours, "theirs": theirs})              # each series' return per calendar year

Daily, over the days both series have a return.
(:func:`~research_core.analytics.performance.metrics.active_performance` measures a book against a benchmark fund on
monthly returns instead, where stale daily closes would count as tracking error.)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from research_core.data.calendars import TRADING_DAYS


def tracking(ours: pd.Series, theirs: pd.Series, *, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """How closely ``ours`` follows ``theirs`` on the days both have a return (``periods_per_year`` of them a year)."""
    both = pd.concat({"ours": ours, "theirs": theirs}, axis=1, sort=True).dropna()
    years = len(both) / periods_per_year
    growth = (1.0 + both).prod()
    diff = both["ours"] - both["theirs"]
    return {
        "days": float(len(both)),
        "ours_per_year": float(growth["ours"] ** (1 / years) - 1) if years else float("nan"),
        "theirs_per_year": float(growth["theirs"] ** (1 / years) - 1) if years else float("nan"),
        "tracking_error": float(diff.std() * np.sqrt(periods_per_year)),
        "correlation": float(both["ours"].corr(both["theirs"])),
        "beta": float(both["ours"].cov(both["theirs"]) / both["theirs"].var()),
    }


#: How the notebooks show a :func:`tracking_summary` (``Styler.format``).
TRACKING_FORMAT = {
    "days": "{:,}",
    "ours_per_year": "{:.2%}",
    "theirs_per_year": "{:.2%}",
    "tracking_error": "{:.2%}",
    "correlation": "{:.4f}",
    "beta": "{:.3f}",
}


def tracking_summary(pairs: Mapping[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    """One row of :func:`tracking` per comparison: ``{name: (ours, theirs)}``, days as whole numbers."""
    summary = pd.DataFrame({name: tracking(ours, theirs) for name, (ours, theirs) in pairs.items()}).T
    return summary.astype({"days": int})


def yearly_tracking_error(ours: pd.Series, theirs: pd.Series, *, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """The tracking error within each calendar year (annualised): *when* a reconstruction drifts.

    A data problem - a missed split, an incomplete list of joiners and leavers - is usually confined to a period,
    which the whole-period figure hides.
    """
    both = pd.concat({"ours": ours, "theirs": theirs}, axis=1, sort=True).dropna()
    diff = both["ours"] - both["theirs"]
    by_year: pd.Series = diff.groupby(pd.DatetimeIndex(diff.index).year).std()
    return by_year * float(np.sqrt(periods_per_year))


def yearly_tracking_errors(pairs: Mapping[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    """``year x comparison``: :func:`yearly_tracking_error` of each pair."""
    return pd.DataFrame({name: yearly_tracking_error(ours, theirs) for name, (ours, theirs) in pairs.items()})


def calendar_years(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Each series' return per calendar year (only over the days all of them have)."""
    both = pd.concat(returns, axis=1, sort=True).dropna()
    years = pd.DatetimeIndex(both.index).year
    return (1.0 + both).groupby(years).prod() - 1.0
