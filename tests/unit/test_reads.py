from pathlib import Path

import pandas as pd

from equity_research.data.cache import ResearchCache
from equity_research.data.reads import (
    as_vendors,
    corporate_actions,
    financial_facts,
    first_preferred,
    id_list,
    listing_issuers,
    outstanding_counts,
)
from equity_research.data.universe import vendor_items


def _table(root: Path, name: str, frame: pd.DataFrame) -> None:
    (root / name).mkdir()
    frame.to_parquet(root / name / "part-1.parquet")


def _cache(root: Path) -> ResearchCache:
    _table(
        root,
        "core.vendor",
        pd.DataFrame({"vendor_id": [1, 2], "vendor_code": ["refinitiv", "sec"], "name": ["LSEG", "SEC"]}),
    )
    _table(root, "core.listing", pd.DataFrame({"listing_id": [10, 11, 20], "issuer_id": [1, 1, 2]}))
    _table(
        root,
        "equity.financial_fact",
        pd.DataFrame(
            {
                "issuer_id": [1, 1, 2],
                "vendor_id": [2, 1, 2],
                "item": ["revenue"] * 3,
                "period_type": ["Q"] * 3,
                "period_start": pd.to_datetime(["2024-01-01"] * 3),
                "period_end": pd.to_datetime(["2024-03-31"] * 3),
                "value": [100.0, 99.0, 50.0],
                "published_at": pd.to_datetime(["2024-05-01"] * 3),
                "unit": ["GBP", "GBP", "USD"],
            }
        ),
    )
    _table(
        root,
        "equity.corporate_action",
        pd.DataFrame(
            {
                "listing_id": [10, 20],
                "vendor_id": [1, 1],
                "action_type": ["cash_dividend", "split"],
                "ex_date": pd.to_datetime(["2024-03-14", "2024-06-03"]),
                "amount": [0.41, None],
                "ratio": [None, 2.0],
                "currency": ["GBP", None],
            }
        ),
    )
    _table(
        root,
        "equity.share_count",
        pd.DataFrame(
            {
                "listing_id": [10, 10],
                "vendor_id": [1, 1],
                "kind": ["outstanding", "float"],
                "as_of_date": pd.to_datetime(["2024-03-31"] * 2),
                "shares": [1000.0, 800.0],
            }
        ),
    )
    return ResearchCache(root)


def test_ids_reach_the_sql_once_each_and_as_numbers_only() -> None:
    assert id_list(pd.Series([3, 1.0, 3, None])) == "3, 1"
    assert id_list(pd.Index([7])) == "7"
    assert id_list([]) == "NULL"  # IN (NULL) matches nothing, where IN () is a syntax error


def test_one_vendor_or_several_is_a_tuple() -> None:
    assert as_vendors("sec") == ("sec",)
    assert as_vendors(["sec", "refinitiv"]) == ("sec", "refinitiv")


def test_the_first_preferred_vendor_answers_and_an_unnamed_one_only_fills_gaps() -> None:
    rows = pd.DataFrame(
        {
            "listing_id": [1, 1, 1, 2],
            "trade_date": ["d1", "d1", "d2", "d1"],
            "vendor": ["yahoo", "norgate", "yahoo", "refinitiv"],
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = first_preferred(rows, ("refinitiv", "norgate"), ["listing_id", "trade_date"]).sort_values("close")
    assert out["close"].tolist() == [2.0, 3.0, 4.0]  # norgate over yahoo on d1; yahoo alone has d2
    assert "_rank" not in out.columns


def test_the_shared_reads_return_what_every_research_loader_needs(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    issuer = listing_issuers(cache, pd.Index([10, 11]))
    assert issuer.to_dict() == {10: 1, 11: 1}

    facts = financial_facts(cache, issuer, ("sec",))
    assert facts["vendor"].tolist() == ["sec"] and facts["unit"].tolist() == ["GBP"]  # one company, one vendor

    actions = corporate_actions(cache, [10, 20])
    assert set(actions.columns) >= {"vendor", "currency"}  # dividends keep their currency for conversion
    counts = outstanding_counts(cache, [10])
    assert counts["shares"].tolist() == [1000.0]  # outstanding only, never the float


def test_a_pound_universe_keeps_its_pound_accounts(tmp_path: Path) -> None:
    # the QGARP inputs once read accounts without the universe's currency: every pound fact was left out
    facts = financial_facts(_cache(tmp_path), pd.Series([1]), ("sec",))
    assert "sec" in vendor_items(facts, ("sec",), "GBP")
    assert "sec" not in vendor_items(facts, ("sec",), "USD")
