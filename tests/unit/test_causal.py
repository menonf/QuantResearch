import numpy as np
import pandas as pd
import pytest

from equity_research.ml import causal as cz
from research_core.analytics.signals import validation as va

MONTHS = pd.date_range("2010-01-31", "2014-12-31", freq="ME")
NAMES = list(range(1, 201))


def _confounded(seed: int = 0, theta: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A confounder c drives the treatment d and the outcome y through its square; d's own effect is ``theta``."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([MONTHS, NAMES], names=["date", "listing_id"])
    c = rng.uniform(-0.5, 0.5, len(index))
    noise = rng.uniform(-0.5, 0.5, len(index))
    curve = c**2 - 1 / 12  # the confounding a straight-line control in c cannot see
    d = 0.5 * c + 2.0 * curve + 0.2 * noise
    y = theta * d + 0.2 * curve + rng.normal(0, 0.02, len(index))
    x = pd.DataFrame({"d": d, "c": c, "unrelated": rng.uniform(-0.5, 0.5, len(index))}, index=index)
    return x, pd.Series(y, index=index).unstack("listing_id")  # noqa: PD010


def _blocks() -> list[va.Fold]:
    return va.purged_kfold(MONTHS, va.next_date_ends(MONTHS), k=5, embargo=1)


def test_dml_removes_a_curved_confounder_that_naive_ols_does_not() -> None:
    x, y = _confounded()
    naive = cz.ols_effect(x, y, "d")
    linear = cz.ols_effect(x, y, "d", controls=["c"])
    est = cz.dml(x, y, treatment="d", controls=["c", "unrelated"], blocks=_blocks())
    assert naive["theta"] > 0.05  # the confounder's curve loads onto d
    assert abs(est["theta"] - 0.01) < 3 * est["se"] and est["t"] > 2  # the true effect, found
    assert linear["theta"] > 0.05  # a straight-line control misses the curve
    assert abs(est["theta"] - 0.01) < abs(linear["theta"] - 0.01) / 10
    assert est["rows"] == len(MONTHS) * len(NAMES)


def test_no_effect_is_found_where_there_is_none() -> None:
    x, y = _confounded(1, theta=0.0)
    est = cz.dml(x, y, treatment="d", controls=["c"], blocks=_blocks())
    assert abs(est["t"]) < 2.5 and est["p_value"] > 0.01


def test_the_factor_zoo_keeps_what_has_its_own_effect() -> None:
    x, y = _confounded(2)
    zoo = cz.factor_zoo(x, y, ["d", "unrelated"], _blocks())
    assert bool(zoo.loc["d", "discovery"]) and not bool(zoo.loc["unrelated", "discovery"])
    with pytest.raises(ValueError, match="unknown learner"):
        cz.dml(x, y, treatment="d", controls=["c"], blocks=_blocks(), learner="forest")
