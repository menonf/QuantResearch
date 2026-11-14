"""Strategy rule sets: read from their files, checked, and identified by a checksum of their rules."""

from __future__ import annotations

import pandas as pd
import pytest

from equity_research.db.schema import RULE_TYPES
from equity_research.research import barings_garp as bg
from equity_research.research import rules as ru

GEMS = ru.RULE_FILES["gems-2026-11"]


def _rules(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ru.COLUMNS)


def test_a_rule_file_is_read_as_subject_type_and_value() -> None:
    rules = ru.read_rules(GEMS)
    assert list(rules.columns) == ru.COLUMNS
    assert len(rules) == 234 and rules["subject"].nunique() == 117  # a lower and an upper limit each
    assert set(rules["rule_type"]) == {"lower_limit", "upper_limit"} <= set(RULE_TYPES)
    assert not rules.duplicated(["subject", "rule_type"]).any() and rules["value"].notna().all()
    assert rules.equals(rules.sort_values(["subject", "rule_type"], ignore_index=True))  # one order
    assert GEMS.strategy == "barings_qgarp" and "gems-2026-11" in bg.RULE_SETS


def test_the_checksum_names_the_rules_not_their_order() -> None:
    rules = ru.read_rules(GEMS)
    assert ru.checksum(rules) == ru.checksum(rules.sample(frac=1.0, random_state=1))
    changed = rules.assign(value=[rules["value"].iloc[0] + 0.5, *rules["value"].iloc[1:]])
    assert ru.checksum(changed) != ru.checksum(rules)
    assert len(ru.checksum(rules)) == 64


def test_a_set_may_not_hold_an_unknown_type_a_missing_value_or_a_rule_twice() -> None:
    good = [("earnings_yield", "lower_limit", -0.5), ("earnings_yield", "upper_limit", 0.5)]
    assert len(ru.checked(_rules(good), "t")) == 2
    with pytest.raises(ValueError, match="no rule type"):
        ru.checked(_rules([*good, ("earnings_yield", "clip_at", 3.0)]), "t")
    with pytest.raises(ValueError, match="without a number"):
        ru.checked(_rules([*good, ("book_yield", "upper_limit", float("nan"))]), "t")
    with pytest.raises(ValueError, match="given twice"):
        ru.checked(_rules([*good, ("earnings_yield", "upper_limit", 0.6)]), "t")


def test_limits_are_one_row_per_subject() -> None:
    rules = _rules(
        [("a", "lower_limit", -1.0), ("a", "upper_limit", 1.0), ("b", "upper_limit", 5.0), ("b", "weight", 0.3)]
    )
    limits = ru.limits_of(rules).set_index("subject")
    assert list(limits.columns) == ["lower_limit", "upper_limit", "exclude_financials"]
    assert [float(limits["lower_limit"].iloc[0]), float(limits["upper_limit"].iloc[0])] == [-1.0, 1.0]
    assert limits["lower_limit"].isna().tolist() == [False, True]  # b has no lower limit
    assert float(limits["upper_limit"].iloc[1]) == 5.0
    assert limits["exclude_financials"].isna().all()  # no such rule in the set
    assert ru.limits_of(rules, ("weight",)).set_index("subject").loc["b", "weight"] == 0.3
