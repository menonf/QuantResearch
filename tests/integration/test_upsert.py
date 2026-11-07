import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Connection

from equity_research.db import schema
from equity_research.db.upsert import upsert

pytestmark = pytest.mark.integration


def _vendors(conn: Connection) -> dict[str, str]:
    rows = conn.execute(sa.select(schema.vendor.c.vendor_code, schema.vendor.c.name)).all()
    return {r[0]: r[1] for r in rows}


def test_upsert_inserts_then_writes_only_changes(conn: Connection) -> None:
    frame = pd.DataFrame({"vendor_id": [901, 902], "vendor_code": ["test_a", "test_b"], "name": ["A", "B"]})
    first = upsert(conn, schema.vendor, frame, key=["vendor_id"])
    assert (first.inserted, first.updated) == (2, 0)
    again = upsert(conn, schema.vendor, frame, key=["vendor_id"])
    assert (again.inserted, again.updated) == (0, 0)  # identical data writes nothing
    changed = upsert(conn, schema.vendor, frame.assign(name=["A", "B2"]), key=["vendor_id"])
    assert (changed.inserted, changed.updated) == (0, 1)
    assert _vendors(conn)["test_b"] == "B2"


def test_insert_only_never_updates(conn: Connection) -> None:
    upsert(
        conn,
        schema.vendor,
        pd.DataFrame({"vendor_id": [903], "vendor_code": ["test_c"], "name": ["C"]}),
        key=["vendor_id"],
    )
    result = upsert(
        conn,
        schema.vendor,
        pd.DataFrame({"vendor_id": [903], "vendor_code": ["test_c"], "name": ["other"]}),
        key=["vendor_id"],
        insert_only=True,
    )
    assert (result.inserted, result.updated) == (0, 0)
    assert _vendors(conn)["test_c"] == "C"


def test_identifier_values_are_case_sensitive(conn: Connection) -> None:
    issuer_id = conn.execute(
        sa.insert(schema.issuer).values(name="Test Co").returning(schema.issuer.c.issuer_id)
    ).scalar_one()
    rows = pd.DataFrame(
        {
            "entity_type": ["issuer", "issuer"],
            "entity_id": [issuer_id, issuer_id],
            "id_type": ["RIC", "RIC"],
            "value": ["ATH_pe", "ATH_PE"],  # two different vendor codes
            "valid_from": [pd.Timestamp("1900-01-01")] * 2,
            "vendor_id": [1, 1],
        }
    )
    result = upsert(conn, schema.identifier, rows, key=["id_type", "value", "valid_from", "entity_type"])
    assert result.inserted == 2
