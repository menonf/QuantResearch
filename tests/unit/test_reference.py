"""What trading in each segment costs: an all-in assumption that covers the broker's commission."""

from __future__ import annotations

from equity_research.ingest import reference as ref

#: Interactive Brokers' fixed commission in basis points (read 2026-11-12): $0.005 a share at the segments' median
#: share prices of 2024 in New York; 0.05 % of the trade in London.
IBKR_BPS = {"SP500": 0.5, "SP400": 0.75, "SP600": 1.5, "FTSE100": 5.0, "FTSE250": 5.0, "FTSESC": 5.0}


def test_each_segments_cost_covers_the_brokers_commission_with_room_for_the_spread() -> None:
    components = {c.index: c for u in ref.UNIVERSES for c in u.components}
    assert set(components) == set(IBKR_BPS)
    for index, c in components.items():
        assert c.cost_bps >= 2 * IBKR_BPS[index], index  # the commission is at most half of the all-in cost


def test_smaller_companies_cost_more_to_trade_and_to_borrow() -> None:
    for large, mid, small in ((ref.LARGE, ref.MID, ref.SMALL), (ref.UK_LARGE, ref.UK_MID, ref.UK_SMALL)):
        assert large.cost_bps < mid.cost_bps < small.cost_bps
        assert large.borrow_bps < mid.borrow_bps < small.borrow_bps
