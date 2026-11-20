import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Connection

from research_core.data.identity import (
    US_MARKETS,
    add_identifiers,
    anchor_map,
    ensure_listings,
    match_listings,
    merge_issuers,
    merge_listings,
)

pytestmark = pytest.mark.integration


def missing(value: object) -> bool:
    """A missing value (NaN, NaT, None) - typed ``object`` so the checker does not narrow what a frame holds."""
    return bool(pd.isna(value))  # type: ignore[call-overload]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor": ["T1", "T2", "T3"],
            "issuer_anchor": ["TST001", "TST001", None],  # two share classes of one company; an index level
            "issuer_name": ["Test Holdings", "Test Holdings", None],
            "name": ["Test Holdings A", "Test Holdings B", "Test Index"],
            "instrument_type": ["equity", "equity", "index"],
            "currency": ["USD", "USD", "USD"],
        }
    )


def test_listings_are_created_once_and_share_an_issuer(conn: Connection) -> None:
    first = ensure_listings(
        conn, _frame(), anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None
    )
    assert first.listings_created == 3 and first.issuers_created == 1
    again = ensure_listings(
        conn,
        _frame().assign(name=["A2", "B", "Test Index"]),
        anchor_type="RIC",
        issuer_anchor_type="CUSIP",
        vendor="refinitiv",
        batch_id=None,
    )
    assert again.listings_created == 0 and again.issuers_created == 0
    assert again.listing_ids == first.listing_ids
    assert anchor_map(conn, "issuer", "CUSIP")["TST001"] > 0
    name = conn.exec_driver_sql(
        "SELECT name FROM core.listing WHERE listing_id = ?", (first.listing_ids["T1"],)
    ).scalar()
    assert name == "A2"  # attributes follow the source


def test_an_identifier_held_by_another_entity_is_reported_not_reassigned(conn: Connection) -> None:
    ids = ensure_listings(
        conn, _frame(), anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None
    )
    row = {
        "entity_type": "listing",
        "id_type": "ISIN",
        "value": "US0000TEST01",
        "valid_from": None,
        "valid_to": None,
        "vendor_id": 1,
    }
    assert add_identifiers(conn, pd.DataFrame([{**row, "entity_id": ids.listing_ids["T1"]}]), batch_id=None).empty
    conflicts = add_identifiers(conn, pd.DataFrame([{**row, "entity_id": ids.listing_ids["T2"]}]), batch_id=None)
    assert len(conflicts) == 1 and conflicts.loc[0, "held_by"] == ids.listing_ids["T1"]
    held = conn.exec_driver_sql("SELECT entity_id FROM core.identifier WHERE value = 'US0000TEST01'").scalars().all()
    assert held == [ids.listing_ids["T1"]]


def test_merging_issuers_moves_listings_identifiers_and_filings(conn: Connection) -> None:
    frame = _frame().assign(
        issuer_anchor=pd.Series(["TST001", "TST002", None], dtype="object")
    )  # one company recorded as two issuers
    ids = ensure_listings(conn, frame, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None)
    issuers = anchor_map(conn, "issuer", "CUSIP")
    a, b = issuers["TST001"], issuers["TST002"]
    filing = (
        "INSERT INTO equity.filing (issuer_id, vendor_id, filing_ref, form, period_end, published_at) "
        "VALUES (?, 1, ?, 'announcement', '2024-03-31', '2024-04-25')"
    )
    conn.exec_driver_sql(filing, (a, "announcement:2024-03-31"))
    conn.exec_driver_sql(filing, (b, "announcement:2024-03-31"))  # the same filing on both: a duplicate after the merge
    conn.exec_driver_sql(filing, (b, "announcement:2024-06-30"))
    conn.exec_driver_sql("UPDATE equity.filing SET loaded_at = '2000-01-01' WHERE issuer_id = ?", (b,))
    merge_issuers(conn, a, [b])
    listing_issuers = (
        conn.execute(
            sa.text("SELECT issuer_id FROM core.listing WHERE listing_id IN :ids").bindparams(
                sa.bindparam("ids", expanding=True)
            ),
            {"ids": [ids.listing_ids["T1"], ids.listing_ids["T2"]]},
        )
        .scalars()
        .all()
    )
    assert set(listing_issuers) == {a}
    refs = (
        conn.exec_driver_sql("SELECT filing_ref FROM equity.filing WHERE issuer_id = ? ORDER BY filing_ref", (a,))
        .scalars()
        .all()
    )
    assert refs == ["announcement:2024-03-31", "announcement:2024-06-30"]
    assert anchor_map(conn, "issuer", "CUSIP")["TST002"] == a  # the identifier moved too
    assert conn.exec_driver_sql("SELECT COUNT(*) FROM core.issuer WHERE issuer_id = ?", (b,)).scalar() == 0
    moved = conn.exec_driver_sql(
        "SELECT loaded_at FROM equity.filing WHERE filing_ref = 'announcement:2024-06-30' AND issuer_id = ?", (a,)
    ).scalar()
    assert moved is not None and moved.year > 2000  # stamped, so an incremental cache sync picks the move up


def test_merging_listings_keeps_one_price_per_day_and_moves_codes(conn: Connection) -> None:
    frame = _frame().assign(
        anchor=["NEW.N", "OLD.N^A20", "IDX"], issuer_anchor=pd.Series(["TST001", "TST001", None], dtype="object")
    )
    ids = ensure_listings(conn, frame, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None)
    new, old = ids.listing_ids["NEW.N"], ids.listing_ids["OLD.N^A20"]
    bar = "INSERT INTO equity.price_bar (listing_id, vendor_id, trade_date, [close]) VALUES (?, 1, ?, ?)"
    conn.exec_driver_sql(bar, (old, "2020-01-02", 9.0))
    conn.exec_driver_sql(bar, (old, "2020-01-03", 9.5))  # both RICs priced this day
    conn.exec_driver_sql(bar, (new, "2020-01-03", 10.0))
    conn.exec_driver_sql(bar, (new, "2020-01-06", 10.5))
    merge_listings(conn, new, [old])
    closes = conn.exec_driver_sql(
        "SELECT CONVERT(VARCHAR(10), trade_date, 23), [close] FROM equity.price_bar WHERE listing_id = ? ORDER BY 1",
        (new,),
    ).all()
    assert [tuple(r) for r in closes] == [("2020-01-02", 9.0), ("2020-01-03", 10.0), ("2020-01-06", 10.5)]
    assert anchor_map(conn, "listing", "RIC")["OLD.N^A20"] == new  # the old code now names the line
    assert conn.exec_driver_sql("SELECT COUNT(*) FROM core.listing WHERE listing_id = ?", (old,)).scalar() == 0


def test_a_vendors_securities_match_by_isin_then_ticker_on_the_day(conn: Connection) -> None:
    frame = _frame().assign(anchor=["AAA.N", "OLDCO.N^A15", "BBB.N"], issuer_anchor=["TST001", "TST002", "TST003"])
    ids = ensure_listings(conn, frame, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None)
    a, old, b = ids.listing_ids["AAA.N"], ids.listing_ids["OLDCO.N^A15"], ids.listing_ids["BBB.N"]
    codes = pd.DataFrame(
        {
            "entity_type": "listing",
            "entity_id": [a, old, a, b, b],
            "id_type": ["ISIN", "TICKER", "TICKER", "TICKER", "CUSIP"],
            "value": ["US000TEST001", "TSTX", "TSTX", "TSTY", "000TST999"],
            "valid_from": pd.to_datetime(["1900-01-01", "1900-01-01", "2015-02-01", "1900-01-01", "1900-01-01"]),
            "valid_to": [None, pd.Timestamp("2015-02-01"), None, None, None],  # TSTX passed from the old company to AAA
            "vendor_id": 1,
        }
    )
    add_identifiers(conn, codes, batch_id=None)
    conn.exec_driver_sql(  # one ticker, two listings on the same day: a person must decide
        "INSERT INTO core.identifier (entity_type, entity_id, id_type, value, valid_from, vendor_id) "
        "VALUES ('listing', ?, 'TICKER', 'TSTY', '1900-01-02', 1)",
        (a,),
    )
    asked = pd.DataFrame(
        {
            "key": ["by_isin", "ticker_2010", "ticker_2020", "two_owners", "nobody"],
            "ISIN": ["US000TEST001", None, None, None, "US999NOPE999"],
            "TICKER": [None, "TSTX", "TSTX", "TSTY", "NOPE"],
        }
    )
    then = match_listings(conn, asked.iloc[[1]], markets=US_MARKETS, on=pd.Timestamp("2010-06-30").date())
    assert then["listing_id"].tolist() == [old]  # in 2010 the ticker was the old company's
    now = match_listings(conn, asked, markets=US_MARKETS).set_index("key")
    assert now.loc["by_isin", "listing_id"] == a and now.loc["by_isin", "matched_by"] == "ISIN"
    assert now.loc["ticker_2020", "listing_id"] == a
    assert now.loc["two_owners", "ambiguous"] and missing(now.loc["two_owners", "listing_id"])
    assert missing(now.loc["nobody", "listing_id"]) and not now.loc["nobody", "ambiguous"]


def test_a_ticker_matches_only_listings_on_the_vendors_markets(conn: Connection) -> None:
    # London's AAL is Anglo American, Nasdaq's American Airlines: one ticker, two markets, two companies
    frame = pd.DataFrame(
        {
            "anchor": ["TSTAAL.L", "TSTAAL.OQ", "TSTBA.L"],
            "issuer_anchor": ["TSTUK1", "TSTUS1", "TSTUK2"],
            "issuer_name": ["Test Mining plc", "Test Airlines Inc", "Test Defence plc"],
            "name": ["Test Mining", "Test Airlines", "Test Defence"],
            "instrument_type": "equity",
            "exchange_mic": ["XLON", "XNAS", "XLON"],
            "currency": ["GBX", "USD", "GBX"],
        }
    )
    ids = ensure_listings(conn, frame, anchor_type="RIC", issuer_anchor_type="ISIN", vendor="refinitiv", batch_id=None)
    uk, us, uk_only = (ids.listing_ids[a] for a in ("TSTAAL.L", "TSTAAL.OQ", "TSTBA.L"))
    add_identifiers(
        conn,
        pd.DataFrame(
            {
                "entity_type": "listing",
                "entity_id": [uk, us, uk_only],
                "id_type": "TICKER",
                "value": ["TSTAAL", "TSTAAL", "TSTBA"],
                "valid_from": pd.Timestamp("1900-01-01"),  # one ticker, two markets: two codes, both recorded
                "valid_to": None,
                "vendor_id": 1,
            }
        ),
        batch_id=None,
    )
    held = conn.execute(
        sa.text("SELECT market FROM core.identifier WHERE id_type = 'TICKER' AND value = 'TSTAAL' ORDER BY market")
    ).scalars()
    assert list(held) == ["GB", "US"]
    asked = pd.DataFrame({"key": ["aal", "ba"], "TICKER": ["TSTAAL", "TSTBA"]})
    us_vendor = match_listings(conn, asked, markets=US_MARKETS).set_index("key")
    assert us_vendor.loc["aal", "listing_id"] == us and not us_vendor.loc["aal", "ambiguous"]
    assert missing(us_vendor.loc["ba", "listing_id"]) and not us_vendor.loc["ba", "ambiguous"]  # no US share of it
    anywhere = match_listings(conn, asked, markets=None).set_index("key")
    assert anywhere.loc["aal", "ambiguous"] and anywhere.loc["ba", "listing_id"] == uk_only


def test_a_call_that_cannot_name_the_company_keeps_the_one_the_listing_has(conn: Connection) -> None:
    # LSEG names most US companies by CUSIP and those registered abroad by an ISIN, in a second call: the first call,
    # which has no CUSIP for them, must not set their company to NULL only for the second to set it back
    first = ensure_listings(
        conn, _frame(), anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None
    )
    issuer = conn.exec_driver_sql(
        "SELECT issuer_id FROM core.listing WHERE listing_id = ?", (first.listing_ids["T1"],)
    ).scalar()
    unnamed = _frame().assign(issuer_anchor=None, issuer_name=None, name=["A2", "B", "Test Index"])
    ensure_listings(conn, unnamed, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=None)
    after = conn.exec_driver_sql(
        "SELECT issuer_id, name FROM core.listing WHERE listing_id = ?", (first.listing_ids["T1"],)
    ).one()
    assert tuple(after) == (issuer, "A2")  # the company kept, the other attributes still follow the source
