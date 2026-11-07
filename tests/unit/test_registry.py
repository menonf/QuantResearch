import pytest

from equity_research.research.registry import (
    HoldoutClosedError,
    config_hash,
    deflated_sharpe,
    expected_max_sharpe,
    fence,
)


def test_config_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_the_holdout_is_refused_unless_opened() -> None:
    assert fence("2024-12-31", holdout_start="2025-01-01") is False
    with pytest.raises(HoldoutClosedError, match="hold-out"):
        fence("2025-06-30", holdout_start="2025-01-01")
    assert fence("2025-06-30", open_holdout=True, holdout_start="2025-01-01") is True


def test_more_trials_deflate_the_same_sharpe() -> None:
    assert expected_max_sharpe(1, 4700) == 0.0
    one = deflated_sharpe(sharpe=0.65, trials=1, observations=4700)
    thirty = deflated_sharpe(sharpe=0.65, trials=30, observations=4700)
    hundred = deflated_sharpe(sharpe=0.65, trials=100, observations=4700)
    assert one > 0.99  # 18.6 years at Sharpe 0.65, tried once: convincing
    assert one > thirty > hundred  # the same result is less convincing the more variants were tried
