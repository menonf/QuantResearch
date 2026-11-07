"""The audit on a scratch database built by (a fake) Norgate: the shape rules hold, and a planted defect is found."""

from collections.abc import Iterator
from typing import cast

import pytest
import sqlalchemy as sa
import test_load_norgate as norgate_case
from conftest import scratch_database
from sqlalchemy.engine import Engine

from equity_research.ingest.vendors.norgate.client import NorgateClient
from equity_research.pipelines.load_norgate import load_norgate
from equity_research.quality.audit import audit

pytestmark = pytest.mark.integration

SCRATCH = "equity_research_test_audit"
SHAPE = ("schema", "identity", "members", "prices", "accounts", "lineage", "ops")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    with scratch_database(SCRATCH, tmp_path_factory.mktemp("audit_data")) as engine:
        load_norgate(
            engine,
            start=norgate_case.START,
            end=norgate_case.END,
            client=cast(NorgateClient, norgate_case.FakeNorgate()),
        )
        yield engine


def test_a_clean_build_keeps_every_rule_of_its_shape(built: Engine) -> None:
    findings = audit(built)
    shape = findings[findings["area"].isin(SHAPE)]
    # the fake's handful of members is no 500-share index: the size rule alone is about scale, not shape
    broken = shape[~shape["ok"] & ~shape["rule"].str.startswith("index sizes")]
    assert broken.empty, broken.to_string()
    assert set(findings["area"]) >= {*SHAPE, "fitness"}
    assert findings["note"].str.startswith("could not measure").sum() == 0  # every rule's SQL runs


def test_a_share_counted_twice_in_an_index_is_found(built: Engine) -> None:
    with built.begin() as c:
        c.execute(
            sa.text(
                "INSERT INTO ref.index_membership (index_id, listing_id, start_date, end_date, vendor_id, batch_id) "
                "SELECT TOP 1 index_id, listing_id, DATEADD(day, 5, start_date), NULL, vendor_id, batch_id "
                "FROM ref.index_membership"
            )
        )
    findings = audit(built).set_index("rule")
    assert not findings.loc["no overlapping spells of one listing in one index", "ok"]
