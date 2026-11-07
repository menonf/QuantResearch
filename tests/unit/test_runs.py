import json
from dataclasses import asdict
from typing import cast

import pandas as pd
import pytest

from equity_research.research.barings_garp import BaringsGarpConfig
from equity_research.research.garp import GarpConfig
from equity_research.research.runs import choose_run, config_from, one_universe
from equity_research.research.strategy import StrategyConfig

VENDORS = {"price_vendors": ("refinitiv",), "fundamentals_vendor": ("sec", "refinitiv")}


def _stored(config: object) -> dict[str, object]:
    """A config as the registry keeps it: through JSON, so its tuples come back as lists."""
    return cast(dict[str, object], json.loads(json.dumps(asdict(config))))  # type: ignore[call-overload]


@pytest.mark.parametrize(
    "config",
    [
        StrategyConfig(family="multi_factor_lo", factors=(("value", 1.0), ("ml_value", 0.5)), **VENDORS),  # type: ignore[arg-type]
        GarpConfig(reference_months=(6, 12), **VENDORS),  # type: ignore[arg-type]
        BaringsGarpConfig(qgarp_weights={"pct_rank_value_composite": 1.0}, **VENDORS),  # type: ignore[arg-type]
    ],
)
def test_every_kind_of_run_is_rebuilt_from_what_the_registry_stored(config: object) -> None:
    rebuilt = config_from(_stored(config))
    assert type(rebuilt) is type(config) and rebuilt == config


def test_a_run_from_before_a_setting_existed_takes_its_default() -> None:
    old = _stored(StrategyConfig(family="multi_factor_ls", **VENDORS))  # type: ignore[arg-type]
    del old["weighting"]  # runs 1-9 were registered before weighting existed: they were equal-weighted
    rebuilt = config_from(old)
    assert isinstance(rebuilt, StrategyConfig) and rebuilt.weighting == "equal"


def test_a_setting_no_strategy_knows_is_refused_not_dropped() -> None:
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        config_from(_stored(StrategyConfig(family="x", **VENDORS)) | {"leverage": 2.0})  # type: ignore[arg-type]


RUNS = pd.DataFrame({"run_id": [11, 101, 109, 110], "family": ["multi_factor_lo"] * 3 + ["garp_sp500"]})


def test_a_run_is_chosen_by_its_id_or_as_its_familys_latest() -> None:
    assert choose_run(RUNS, 11, None)["run_id"] == 11
    assert choose_run(RUNS, 11, "multi_factor_lo")["run_id"] == 11
    assert choose_run(RUNS, None, "multi_factor_lo")["run_id"] == 109


def test_a_run_id_and_a_family_that_disagree_are_caught() -> None:
    with pytest.raises(ValueError, match="'garp_sp500' run"):
        choose_run(RUNS, 110, "multi_factor_lo")
    with pytest.raises(KeyError, match="not registered"):
        choose_run(RUNS, 999, None)
    with pytest.raises(KeyError, match="no run of family"):
        choose_run(RUNS, None, "unknown")
    with pytest.raises(ValueError, match="name a run id"):
        choose_run(RUNS, None, None)


def test_books_compared_side_by_side_share_a_universe() -> None:
    sp500 = StrategyConfig(family="a", universe="SP500", **VENDORS)  # type: ignore[arg-type]
    assert one_universe([sp500, GarpConfig(**VENDORS)]) == "SP500"  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different universes"):
        one_universe([sp500, StrategyConfig(family="b", universe="SP1500", **VENDORS)])  # type: ignore[arg-type]
