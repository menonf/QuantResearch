"""The repository's rule sets in the shared tables: loaded once, never changed, read back through the cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import scratch_database

from equity_research.data import artifacts as store
from equity_research.data.cache import ResearchCache
from equity_research.pipelines import load_rules as pipe
from equity_research.strategies import barings_garp as bg
from equity_research.strategies import rules as ru

pytestmark = pytest.mark.integration

CODE = "gems-2026-11"


def _count(engine: sa.Engine, table: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(sa.text(f"SELECT COUNT(*) FROM research.{table}")).scalar_one())


def test_rule_sets_are_loaded_once_and_never_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with scratch_database("equity_research_test_rules", tmp_path) as engine:
        assert pipe.load_rules(engine) == {CODE: "loaded"}
        assert (_count(engine, "rule_set"), _count(engine, "strategy_rule")) == (1, 234)
        with engine.connect() as conn:
            held = conn.execute(sa.text("SELECT strategy, checksum FROM research.rule_set")).one()
        assert held[0] == "barings_qgarp" and held[1] == ru.checksum(ru.read_rules(ru.RULE_FILES[CODE]))

        assert pipe.load_rules(engine) == {CODE: "unchanged"}  # a second run writes nothing
        assert (_count(engine, "rule_set"), _count(engine, "strategy_rule")) == (1, 234)

        # new words about the same rules: the description follows the repository, the rules stay
        spec = ru.RULE_FILES[CODE]
        reworded = ru.RuleFile(**{**spec.__dict__, "description": "reworded", "source": "elsewhere"})
        assert pipe.load_rules(engine, [reworded]) == {CODE: "described"}
        with engine.connect() as conn:
            words = conn.execute(sa.text("SELECT description, source FROM research.rule_set")).one()
        assert tuple(words) == ("reworded", "elsewhere")
        assert (_count(engine, "rule_set"), _count(engine, "strategy_rule")) == (1, 234)

        # the file changes under the same code: refused - a registered run names this set
        rules = ru.read_rules(ru.RULE_FILES[CODE])
        changed = rules.assign(value=[rules["value"].iloc[0] + 1.0, *rules["value"].iloc[1:]])
        monkeypatch.setattr(pipe, "read_rules", lambda spec: changed)
        with pytest.raises(pipe.RuleSetChangedError, match="never changed"):
            pipe.load_rules(engine)
        assert _count(engine, "strategy_rule") == 234


def test_research_reads_the_rules_through_the_cache(tmp_path: Path) -> None:
    with scratch_database("equity_research_test_rules", tmp_path) as engine:
        cache = ResearchCache()
        cache.sync(engine)
        with pytest.raises(KeyError, match="eqr load-rules"):  # nothing loaded yet: the error says what to run
            ru.cached_rules(cache, CODE)

        pipe.load_rules(engine)
        before = store.data_snapshot(cache)
        copied = cache.sync(engine)
        assert copied["research.rule_set"] == 1 and copied["research.strategy_rule"] == 234
        assert store.data_snapshot(cache) == before  # loading rules does not move the data version

        from_file = ru.read_rules(ru.RULE_FILES[CODE])
        pd.testing.assert_frame_equal(ru.cached_rules(cache, CODE), from_file, check_dtype=False)
        shared, packaged = bg.shared_rules(cache, CODE), bg.packaged_rules(CODE)
        assert shared is not None and packaged is not None
        pd.testing.assert_frame_equal(shared, packaged, check_dtype=False)
        assert bg.shared_rules(cache, None) is None
        listed = ru.rule_sets(cache).set_index("code")
        assert listed.loc[CODE, "rules"] == 234 and listed.loc[CODE, "strategy"] == "barings_qgarp"
