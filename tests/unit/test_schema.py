import re

import sqlalchemy as sa

from equity_research.db import schema

IN_LIST = re.compile(r"^(\w+) IN \((.*)\)$")


def test_every_allowed_code_fits_its_column() -> None:
    checked = 0
    for table in schema.metadata.tables.values():
        for constraint in table.constraints:
            if not isinstance(constraint, sa.CheckConstraint):
                continue
            match = IN_LIST.match(str(constraint.sqltext))
            if match is None:
                continue
            column, values = match.group(1), re.findall(r"'([^']*)'", match.group(2))
            length = table.c[column].type.length  # type: ignore[attr-defined]
            too_long = [v for v in values if len(v) > length]
            assert not too_long, f"{table.fullname}.{column} is {length} wide: {too_long}"
            checked += 1
    assert checked >= 9
