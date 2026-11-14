"""Strategies' rules: named sets a run applies, kept in the repository and shared through the database.

    spec = RULE_FILES["gems-2026-11"]                  # what the set is, and the file it comes from
    rules = read_rules(spec)                             # [subject, rule_type, value], as the file has them
    rules = cached_rules(cache, "gems-2026-11")         # the same from the shared tables, through the cache
    limits = limits_of(rules)                            # one row per subject: lower_limit, upper_limit, ...

A **rule** says something about a **subject** - a factor ("earnings_yield"), a composite, or ``*`` for the strategy as
a whole: its lower or upper limit, whether financials are left out of it, its weight, a threshold. A **rule set** is
the named collection one strategy applies together (Barings' cleaning limits of November 2026 are ``gems-2026-11``).

Steps:

1. **Files are the source.** Each set is a file beside this module (:data:`RULE_FILES`), versioned and tested with
   the code that reads it.
2. **The database shares them.** ``eqr load-rules`` (:mod:`~equity_research.pipelines.load_rules`) writes each set to
   ``research.rule_set`` / ``research.strategy_rule``, where everyone - and any other tool - reads the same rules;
   the cache sync copies them to each researcher.
3. **A set never changes.** A registered run names the set it used, so the set must still say what it said then:
   the loader refuses a file whose rules differ from the loaded set of the same code (:func:`checksum`). A changed
   rule is a new file under a new code.
4. **Research reads the cache** (:func:`cached_rules`), like every other table.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from equity_research.db.schema import RULE_TYPES
from equity_research.research.cache import ResearchCache

COLUMNS = ["subject", "rule_type", "value"]


@dataclass(frozen=True)
class RuleFile:
    """A rule set kept in the repository: what it is, and the file (beside this module) that holds its rules."""

    code: str  # what a run's settings name
    strategy: str  # the registry family it belongs to
    file: str
    description: str
    source: str  # where the rules came from
    #: the file's columns for a rule's subject, type and value (exports name them their own way)
    columns: tuple[str, str, str] = ("subject", "rule_type", "value")


RULE_FILES: dict[str, RuleFile] = {
    spec.code: spec
    for spec in (
        RuleFile(
            code="gems-2026-11",
            strategy="barings_qgarp",
            file="barings_qgarp_limits.csv",
            description="Barings QGARP cleaning limits: a lower and an upper limit for each of 117 scores; a value "
            "outside its limits is set missing before the factor is ranked",
            source="Barings' limits of November 2026 "
            "(github.com/menonf/PythonProjects FactorResearch/rule.csv @40ff28c)",
            columns=("barings_metric_name", "rule_type", "rule_value"),
        ),
    )
}


def read_rules(spec: RuleFile) -> pd.DataFrame:
    """The rules of ``spec``'s file: ``[subject, rule_type, value]``, sorted, one row per subject and type."""
    raw = pd.read_csv(Path(__file__).with_name(spec.file))
    rules = raw[list(spec.columns)].set_axis(COLUMNS, axis=1)
    rules["subject"] = rules["subject"].astype(str).str.strip().str.lower()
    rules["rule_type"] = rules["rule_type"].astype(str).str.strip()
    rules["value"] = pd.to_numeric(rules["value"], errors="coerce")
    return checked(rules, spec.code)


def checked(rules: pd.DataFrame, code: str) -> pd.DataFrame:
    """``rules`` sorted, after refusing what a set may not hold: an unknown type, a missing value, a rule twice."""
    unknown = sorted(set(rules["rule_type"]) - set(RULE_TYPES))
    if unknown:
        raise ValueError(f"rule set {code!r}: no rule type {unknown}; known: {list(RULE_TYPES)}")
    if rules["value"].isna().any():
        bad = rules.loc[rules["value"].isna(), "subject"].tolist()[:5]
        raise ValueError(f"rule set {code!r}: rules without a number, for {bad}")
    twice = rules[rules.duplicated(["subject", "rule_type"], keep=False)]
    if not twice.empty:
        raise ValueError(f"rule set {code!r}: a rule given twice, for {sorted(set(twice['subject']))[:5]}")
    return rules.sort_values(["subject", "rule_type"], ignore_index=True)[COLUMNS]


def checksum(rules: pd.DataFrame) -> str:
    """SHA-256 of ``rules`` in their one order and spelling: two sets with the same rules have the same checksum."""
    ordered = rules.sort_values(["subject", "rule_type"])
    lines = [
        f"{s}|{t}|{float(v)!r}"
        for s, t, v in zip(ordered["subject"], ordered["rule_type"], ordered["value"], strict=True)
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def limits_of(
    rules: pd.DataFrame, types: tuple[str, ...] = ("lower_limit", "upper_limit", "exclude_financials")
) -> pd.DataFrame:
    """One row per subject with a column for each of ``types`` (missing where the set has no such rule)."""
    wide = rules.pivot_table(index="subject", columns="rule_type", values="value", aggfunc="max")
    return wide.reindex(columns=list(types)).rename_axis(columns=None).reset_index()


def cached_rules(cache: ResearchCache, code: str) -> pd.DataFrame:
    """The rules of the set ``code`` from the shared tables, through the cache: ``[subject, rule_type, value]``."""
    found = cache.query(
        "SELECT r.subject, r.rule_type, r.value FROM research.strategy_rule r "
        "JOIN research.rule_set s ON s.rule_set_id = r.rule_set_id WHERE s.code = ? ORDER BY r.subject, r.rule_type",
        [code],
    )
    if found.empty:
        known = cache.query("SELECT code FROM research.rule_set ORDER BY code")["code"].tolist()
        raise KeyError(
            f"no rule set {code!r} in the research cache (it holds {known}): "
            "run `eqr load-rules`, then `eqr sync-cache`"
        )
    return found[COLUMNS]


def rule_sets(cache: ResearchCache) -> pd.DataFrame:
    """Every rule set the cache holds: its code, strategy, description, source, and how many rules it has."""
    return cache.query(
        "SELECT s.code, s.strategy, s.description, s.source, s.loaded_at, COUNT(r.subject) AS rules "
        "FROM research.rule_set s LEFT JOIN research.strategy_rule r ON r.rule_set_id = s.rule_set_id "
        "GROUP BY s.code, s.strategy, s.description, s.source, s.loaded_at ORDER BY s.code"
    )
