import pytest

from equity_research.quality.checks import Check, CheckResult, GateFailedError, gate


def test_thresholds_in_both_directions() -> None:
    coverage = Check("coverage", "", warn=0.999, fail=0.99)
    assert [coverage.status(v) for v in (0.9995, 0.995, 0.98)] == ["pass", "warn", "fail"]
    missing = Check("missing", "", warn=0.1, fail=0.5, higher_is_better=False)
    assert [missing.status(v) for v in (0.05, 0.2, 0.6)] == ["pass", "warn", "fail"]


def test_a_failed_check_stops_the_pipeline() -> None:
    gate([CheckResult("a", "pass", 1.0, ""), CheckResult("b", "warn", 0.5, "")])  # warnings do not stop it
    with pytest.raises(GateFailedError, match=r"b = 0\.1"):
        gate([CheckResult("b", "fail", 0.1, "warn >= 0.9, fail >= 0.5")])
