"""Write the repository's rule sets to the shared tables ``research.rule_set`` and ``research.strategy_rule``.

    report = load_rules(engine)        # {code: "loaded" | "unchanged" | "described"}

Steps:

1. read each file of :data:`~equity_research.research.rules.RULE_FILES` and check it (known rule types, a number for
   every rule, no rule twice);
2. a set not yet in the database is inserted with its rules and their checksum;
3. a set already there keeps its rules when its checksum is the file's - only its description and source follow
   the repository's wording ("described" when they changed) - and is **refused** when it is not: a registered run
   names the set it used, so a loaded set's rules may not change. A changed rule goes in a new file under a new code.

The rules come from the repository, not from a vendor: they have no raw-zone batch (as the indices and universes of
``ingest/reference.py`` have none).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from equity_research import ops
from equity_research.db import schema
from equity_research.research.rules import RULE_FILES, RuleFile, checksum, read_rules

log = logging.getLogger(__name__)


class RuleSetChangedError(RuntimeError):
    """A file's rules differ from the set already loaded under its code."""


def load_rules(engine: Engine, files: Iterable[RuleFile] | None = None) -> dict[str, str]:
    """Load every rule set of ``files`` (the repository's by default) that the database does not hold yet."""
    report: dict[str, str] = {}
    s, r = schema.rule_set, schema.strategy_rule
    with ops.job("load_rules", engine) as run:
        for spec in files if files is not None else RULE_FILES.values():
            rules = read_rules(spec)
            digest = checksum(rules)
            with engine.begin() as conn:
                held = conn.execute(
                    sa.select(s.c.checksum, s.c.description, s.c.source).where(s.c.code == spec.code)
                ).first()
                if held is not None:
                    if str(held.checksum) != digest:
                        raise RuleSetChangedError(
                            f"rule set {spec.code!r} is loaded with other rules than {spec.file} now holds: a loaded "
                            "set is never changed (runs name it) - give the changed rules a new code"
                        )
                    if (held.description, held.source) == (spec.description, spec.source):
                        report[spec.code] = "unchanged"
                    else:  # words about the set, not its rules: they may follow the repository
                        conn.execute(
                            sa.update(s)
                            .where(s.c.code == spec.code)
                            .values(description=spec.description, source=spec.source)
                        )
                        report[spec.code] = "described"
                    continue
                set_id = conn.execute(
                    sa.insert(s)
                    .values(
                        code=spec.code,
                        strategy=spec.strategy,
                        description=spec.description,
                        source=spec.source,
                        checksum=digest,
                    )
                    .returning(s.c.rule_set_id)
                ).scalar_one()
                rows = [
                    {"rule_set_id": int(set_id), "subject": str(subject), "rule_type": str(kind), "value": float(value)}
                    for subject, kind, value in zip(rules["subject"], rules["rule_type"], rules["value"], strict=True)
                ]
                conn.execute(sa.insert(r), rows)
                report[spec.code] = "loaded"
                log.info("rule set %s: %d rules loaded", spec.code, len(rules))
        run.finish(rows=sum(1 for v in report.values() if v == "loaded"), detail=dict(report))
    return report
