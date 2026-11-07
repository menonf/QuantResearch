import pandas as pd

from equity_research.ingest.vendors.sec.cik import match_issuers, sec_ticker


def test_our_tickers_in_the_secs_spelling() -> None:
    assert sec_ticker("ABBV") == "ABBV"
    assert sec_ticker("BRK.B") == "BRK-B"
    assert sec_ticker(None) is None and sec_ticker(" ") is None


def test_issuers_match_by_ticker_then_cusip_with_filer_tiebreak() -> None:
    issuers = pd.DataFrame(
        {
            "issuer_id": [1, 1, 2, 3, 4],
            "cusip6": ["037833", "037833", "00817Y", "222222", "333333"],
            "ticker": ["AAPL", "AAPL", None, "NEW", None],  # retired listings hold no current ticker
        }
    )
    tickers = pd.DataFrame(
        {"cik": [320193, 555, 1872302], "ticker": ["AAPL", "NEW", None]}  # a row without a ticker, as in the SEC file
    )
    cusip_map = pd.DataFrame(
        {
            "cik": [320193, 1013761, 1122304, 888, 999, 998],
            "cusip6": ["037833", "00817Y", "00817Y", "222222", "333333", "333333"],
        }
    )
    m = match_issuers(issuers, tickers, cusip_map, filer_ciks=[320193, 1122304, 999]).set_index("issuer_id")
    assert m.loc[1, "cik"] == 320193 and m.loc[1, "method"] == "ticker"
    assert m.loc[2, "cik"] == 1122304 and m.loc[2, "method"] == "cusip"  # the tie settled by who files XBRL
    assert m.loc[3, "cik"] == 555 and m.loc[3, "conflict"]  # ticker kept, disagreement flagged
    assert m.loc[4, "cik"] == 999  # never the empty-ticker row


def test_company_names_compare_by_the_words_that_identify_them() -> None:
    from equity_research.ingest.vendors.sec.cik import name_words, same_name  # noqa: PLC0415

    assert name_words("The Walt Disney Company") == {"walt", "disney"}
    assert name_words("DUN & BRADSTREET CORP/NW/") == {"dun", "and", "bradstreet"}
    assert same_name("The Cigna Group", "CIGNA CORP")
    assert same_name("Stericycle Inc", "STERICYCLE INC")
    assert not same_name("Stericycle Inc", "ABBOTT LABORATORIES")
    assert not same_name("Holdings Inc", "Group Corp")  # nothing left to compare: never a match


def test_a_cusip_tie_is_settled_by_name_and_a_reregistered_company_keeps_both_ciks() -> None:
    issuers = pd.DataFrame(
        {
            "issuer_id": [1, 2],
            "cusip6": ["858912", "254687"],
            "ticker": [None, "DIS"],  # Stericycle was delisted: no ticker of today names it
            "name": ["Stericycle Inc", "The Walt Disney Company"],
        }
    )
    tickers = pd.DataFrame({"cik": [1744489], "ticker": ["DIS"]})
    cusip_map = pd.DataFrame(
        {"cik": [861878, 318306, 1001039, 1744489], "cusip6": ["858912", "858912", "254687", "254687"]}
    )
    names = {
        861878: "STERICYCLE INC",
        318306: "BARON CAPITAL GROUP INC",  # a holder that filed about the shares
        1001039: "WALT DISNEY CO/",  # before the 2019 holding-company change
        1744489: "Walt Disney Co",
    }
    m = match_issuers(issuers, tickers, cusip_map, filer_ciks=list(names), names=names)
    assert m[m["issuer_id"] == 1]["cik"].tolist() == [861878]
    assert sorted(m[m["issuer_id"] == 2]["cik"].tolist()) == [1001039, 1744489]
    assert set(m["method"]) == {"cusip+name", "ticker"}
