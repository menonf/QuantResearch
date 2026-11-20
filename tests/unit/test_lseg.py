import numpy as np
import pandas as pd
import pytest

from equity_research.ingest.vendors.refinitiv import fields, rics
from equity_research.ingest.vendors.refinitiv import normalize as norm
from equity_research.ingest.vendors.refinitiv.client import LsegClient, ThrottledError
from equity_research.pipelines.load_lseg import buckets, request_key


def missing(value: object) -> bool:
    """A missing value (NaN, NaT, None) - typed ``object`` so the checker does not narrow what a frame holds."""
    return bool(pd.isna(value))  # type: ignore[call-overload]


def test_ric_spellings() -> None:
    assert rics.ticker("AAPL.OQ") == "AAPL" and rics.ticker("BAC.N^I98") == "BAC" and rics.ticker("BRKb.N") == "BRK.B"
    assert rics.share_class("BRKb.N") == "B" and rics.share_class("AAPL.OQ") is None
    assert rics.exchange_mic("AAPL.OQ") == "XNAS" and rics.exchange_mic(".SPX") is None
    assert rics.exchange_mic("AAL.L") == "XLON" and rics.exchange_mic("RDSa.L^A22") == "XLON"
    assert rics.is_retired("BAC.N^I98") and not rics.is_retired("BAC.N")
    assert rics.is_non_equity("ES", "ESc1") and not rics.is_non_equity("ES", "ES.N")  # Eversource stays
    assert rics.cusip6("037833100", None) == "037833" and rics.cusip6(None, "US0378331005") == "037833"


def test_membership_is_replayed_backwards_from_todays_list() -> None:
    snapshot = pd.DataFrame({"Instrument": ["AAA.N", "BBB.OQ", "ESc1"], "Exchange Ticker": ["AAA", "BBB", "ES"]})
    events = pd.DataFrame(
        {
            "Instrument": ".SPX",
            "Date": ["2015-03-02", "2018-06-01", "2010-01-04", "2012-05-01", "2016-12-02", "2016-12-02"],
            "Constituent RIC": ["BBB.OQ", "CCC.N^F18", "DDD.N^E12", "DDD.N^E12", "EEE.N", "EEE.N"],
            "Constituent Name": ["B", "C", "D", "D", "E", "E"],
            "Change": ["Joiner", "Leaver", "Joiner", "Leaver", "Joiner", "Leaver"],
        }
    )
    out = norm.memberships(snapshot, events, "2005-01-01").set_index("ric")
    assert out.loc["AAA.N", "start_date"] == pd.Timestamp("2005-01-01") and missing(out.loc["AAA.N", "end_date"])
    assert out.loc["BBB.OQ", "start_date"] == pd.Timestamp("2015-03-02")  # joined in the window
    assert out.loc["CCC.N^F18", "end_date"] == pd.Timestamp("2018-06-01")  # a member before the window, then left
    assert (out.loc["DDD.N^E12", "start_date"], out.loc["DDD.N^E12", "end_date"]) == (
        pd.Timestamp("2010-01-04"),
        pd.Timestamp("2012-05-01"),
    )
    assert "EEE.N" not in out.index  # joined and left the same day: an artefact
    assert "ESc1" not in out.index  # a futures code, not a company


def test_listings_share_a_company_by_cusip_and_indices_have_none() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["GOOGL.OQ", "GOOG.OQ", ".SPX", "OLD.N^A10"],
            "Company Common Name": ["Alphabet Inc", "Alphabet Inc", "Cboe Exchange Inc", "Old Co"],
            "ISIN": ["US02079K3059", "US02079K1079", "US78378X1072", "US1111111111"],
            "SEDOL": ["BYVY8G0", "BYY88Y7", "", ""],
            "CUSIP": ["02079K305", "02079K107", "", "111111111"],
            "Exchange Country Code": ["US", "US", pd.NA, "US"],
            "Exchange Ticker": ["GOOGL", "GOOG", "", "OLD"],
        }
    )
    attrs = norm.instruments(raw)
    frame = norm.listings(attrs).set_index("anchor")
    assert frame.loc["GOOGL.OQ", "issuer_anchor"] == frame.loc["GOOG.OQ", "issuer_anchor"] == "02079K"
    assert frame.loc[".SPX", "instrument_type"] == "index" and missing(frame.loc[".SPX", "issuer_anchor"])
    assert frame.loc[".SPX", "currency"] == "USD" and missing(attrs.set_index("ric").loc[".SPX", "isin"])  # not Cboe's
    codes = norm.listing_identifiers(attrs)
    tickers = codes[codes["id_type"] == "TICKER"]
    assert set(tickers["ric"]) == {"GOOGL.OQ", "GOOG.OQ"}  # no ticker for a retired RIC (tickers are reused)


def test_gics_falls_back_to_the_trbc_sector() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["A.N", "B.N^B22"],
            "GICS Sector Name": ["Energy", ""],
            "GICS Industry Group Name": ["Energy", ""],
            "GICS Industry Name": ["Oil, Gas & Consumable Fuels", ""],
            "TRBC Economic Sector Name": ["Energy", "Technology"],
        }
    )
    out = norm.classification(raw).set_index("ric")
    assert out.loc["B.N^B22", "sector"] == "Information Technology" and out.loc["B.N^B22", "industry"] is None


def test_prices_drop_the_padding_lseg_adds_after_a_retired_share_last_traded() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["OLD.N^A10"] * 4 + ["AAPL.OQ", ".SPX"],
            "Date": ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-06", "2010-01-04", "2010-01-04"],
            "TRDPRC_1": [10.5, 10.5, 10.5, 10.5, 30.5, 1132.99],
            "OPEN_PRC": [10, 10.5, 10.5, 10.5, 30, 1116.56],
            "HIGH_1": [11, 11, 10.5, 10.5, 31, 1133.87],
            "LOW_1": [9.5, 10, 10.5, 10.5, 29, 1116.56],
            "ACVOL_UNS": [1000, 500, np.nan, np.nan, 9000, np.nan],  # an index level has no volume
        }
    )
    out = norm.price_bars(raw)
    old = out[out["ric"] == "OLD.N^A10"]
    assert old["trade_date"].tolist() == [pd.Timestamp("2010-01-04"), pd.Timestamp("2010-01-05")]
    assert len(out[out["ric"] == "AAPL.OQ"]) == 1
    assert out.loc[out["ric"] == ".SPX", "close"].tolist() == [1132.99]


def test_a_column_lseg_did_not_send_is_named() -> None:
    with pytest.raises(norm.MissingColumnsError, match="TRDPRC_1"):
        norm.price_bars(pd.DataFrame({"Instrument": ["A.N"], "Date": ["2024-01-02"], "OPEN_PRC": [1.0]}))


def test_splits_are_confirmed_by_the_price_and_spinoffs_kept_apart() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["AAPL.OQ", "GE.N", "GE.N"],
            "Capital Change Effective Date": ["2020-08-28", "2021-08-02", "2023-01-04"],
            "Adjustment Factor": [0.25, 8.0, 0.95],
        }
    )
    closes = pd.DataFrame(
        {
            "ric": ["AAPL.OQ"] * 3 + ["GE.N"] * 2,
            "trade_date": pd.to_datetime(["2020-08-27", "2020-08-28", "2020-08-31", "2021-07-30", "2021-08-02"]),
            "close": [500.0, 499.0, 129.0, 13.0, 104.0],
        }
    )
    out = norm.capital_changes(raw, closes).set_index(["ric", "action_type"])
    apple = out.xs(("AAPL.OQ", "split"))
    # the vendor said the 28th; the price moved on the 31st (the next trading day): that is the ex-date
    assert apple["ex_date"] == pd.Timestamp("2020-08-31") and apple["ratio"] == 4.0 and apple["verified"]
    ge = out.xs(("GE.N", "split"))
    assert ge["ratio"] == 0.125 and ge["verified"]  # a 1-for-8 reverse split
    assert out.loc[("GE.N", "spinoff"), "ratio"] == pytest.approx(1 / 0.95)


def test_float_is_outstanding_times_the_float_percentage_known_then() -> None:
    shares = pd.DataFrame(
        {
            "Instrument": ["A.N", "A.N"],
            "Issue Default Shares Outstanding": [1000.0, 1100.0],
            "Date": ["2024-01-31", "2024-02-29"],
        }
    )
    pct = pd.DataFrame({"Instrument": ["A.N"], "Free Float (Percent)": [90.0], "Date": ["2024-01-15"]})
    out = norm.share_counts(shares, pct)
    floated = out[out["kind"] == "float"].sort_values("as_of_date")["shares"].tolist()
    assert floated == pytest.approx([900.0, 990.0])
    assert sorted(out[out["kind"] == "outstanding"]["shares"]) == [1000.0, 1100.0]


def test_statements_are_dated_by_their_announcement_or_conservatively() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["A.N", "A.N"],
            "Period End Date": ["2024-03-31", "2024-06-30"],
            **{col: [1.0, 2.0] for col, _, _ in fields.STATEMENT_ITEMS.values()},
        }
    )
    announced = norm.announcements(
        pd.DataFrame(
            {
                "Instrument": ["A.N"],
                "Period End Date": ["2024-03-31"],
                "Original Announcement Date Time": ["2024-04-25"],
            }
        )
    )
    facts = norm.statements(raw, announced)
    revenue = facts[facts["item"] == "revenue"].set_index("period_end")
    assert revenue.loc[pd.Timestamp("2024-03-31"), "published_at"] == pd.Timestamp("2024-04-25")
    assert revenue.loc[pd.Timestamp("2024-06-30"), "published_at"] == pd.Timestamp("2024-09-28")  # 90 days later
    assert revenue["period_type"].unique().tolist() == ["Q"]
    assert revenue.loc[pd.Timestamp("2024-03-31"), "period_start"] == pd.Timestamp("2024-01-01")
    assets = facts[facts["item"] == "total_assets"]
    assert assets["period_type"].unique().tolist() == ["I"] and assets["period_start"].isna().all()


def test_a_request_is_described_the_same_way_every_time() -> None:
    a = request_key("prices", ["A.N", "B.N"], ["TR.X"], {"SDate": "2020-01-01"})
    assert a == request_key("prices", ["A.N", "B.N"], ["TR.X"], {"SDate": "2020-01-01"})
    assert a != request_key("prices", ["A.N", "C.N"], ["TR.X"], {"SDate": "2020-01-01"})


def test_rics_that_name_one_security_are_one_share_line() -> None:
    attrs = pd.DataFrame(
        {
            "ric": ["WFT.N^E19", "WFRD.OQ", "AVID.OQ^B14", "AVID.OQ^K23", "HOS.N", "2037062", "LONE.N", ".SPX"],
            "isin": ["IE00BLNN3691", "IE00BLNN3691", "US05367P1003", "US05367P1003", "US42330P1075", None, None, None],
            "cusip": [None, None, None, None, "42330P107", None, None, None],
            "sedol": [None, None, None, None, "2037062", None, None, None],
        }
    )
    lines = norm.share_lines(attrs)
    assert lines["WFT.N^E19"] == lines["WFRD.OQ"] == "WFRD.OQ"  # renamed: the live RIC anchors the line
    assert lines["AVID.OQ^B14"] == lines["AVID.OQ^K23"] == "AVID.OQ^K23"  # retired twice: the latest anchors
    assert lines["2037062"] == "HOS.N"  # a SEDOL in LSEG's member feed, mapped to its RIC
    assert lines["LONE.N"] == "LONE.N" and lines[".SPX"] == ".SPX"
    assert rics.retired_on("BAC.N^I98") == pd.Timestamp("1998-09-01") and rics.retired_on("BAC.N") is None
    assert not rics.is_ric("2037062") and rics.is_ric(".SPX")


def test_a_new_ric_changes_only_its_own_bucket() -> None:
    rics_now = [f"R{i}.N" for i in range(400)]
    before = buckets(rics_now, 10)
    after = buckets([*rics_now, "NEW.N"], 10)
    changed = [b for b in after if b not in before]
    assert len(changed) == 1 and "NEW.N" in changed[0]  # every other request keeps its description
    assert sorted(r for b in after for r in b) == sorted([*rics_now, "NEW.N"])


def test_a_company_registered_abroad_is_named_by_its_isin() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["LIN.OQ", "AAPL.OQ"],
            "Company Common Name": ["Linde PLC", "Apple Inc."],
            "ISIN": ["IE000S9YS762", "US0378331005"],
            "SEDOL": ["BNZHB81", "2046251"],
            "CUSIP": ["", "037833100"],
            "Exchange Country Code": ["US", "US"],
            "Exchange Ticker": ["LIN", "AAPL"],
        }
    )
    frame = norm.listings(norm.instruments(raw)).set_index("anchor")
    assert missing(frame.loc["LIN.OQ", "issuer_anchor"]) and frame.loc["LIN.OQ", "issuer_isin"] == "IE000S9YS762"
    assert frame.loc["AAPL.OQ", "issuer_anchor"] == "037833" and missing(frame.loc["AAPL.OQ", "issuer_isin"])


def test_the_probe_reports_every_dataset_and_stops_at_the_limit() -> None:
    from typing import cast  # noqa: PLC0415

    from equity_research.ingest.vendors.refinitiv.client import LsegClient, ThrottledError  # noqa: PLC0415
    from equity_research.pipelines.load_lseg import probe  # noqa: PLC0415

    class Limited:
        calls = 0

        def get_data(self, *args: object, **kwargs: object) -> pd.DataFrame:
            self.calls += 1
            if self.calls > 2:
                raise ThrottledError("LSEG limit reached (test)")
            return pd.DataFrame({"Instrument": ["AAPL.OQ"], "Exchange Ticker": ["AAPL"]})

        def get_history(self, *args: object, **kwargs: object) -> pd.DataFrame:
            raise AssertionError("not reached")

        def allowance(self) -> dict[str, int] | None:
            return None  # a Workspace that does not say

    report = probe(cast(LsegClient, Limited()))
    assert report["dataset"].tolist()[0] == "index members"
    assert len(report) == 13 and not report["ok"].any()  # 12 datasets, then the allowance left
    assert report["detail"].str.contains("not tried").sum() == 10  # after the limit, nothing more is asked
    assert report["dataset"].iloc[-1] == "allowance left today"


def test_the_uk_probe_reports_a_broken_expectation_and_stops_at_the_limit() -> None:
    from typing import cast  # noqa: PLC0415

    from equity_research.ingest.vendors.refinitiv.client import LsegClient, ThrottledError  # noqa: PLC0415
    from equity_research.pipelines.load_lseg import probe_uk  # noqa: PLC0415

    class InDollars:
        """Answers the official closes, then prices in dollars; past four calls, LSEG's limit."""

        calls = 0

        def get_data(self, universe: list[str], fields_: list[str], *args: object) -> pd.DataFrame:
            self.calls += 1
            if self.calls == 1:  # the members' first chain: too few to be the FTSE 100
                return pd.DataFrame({"Instrument": ["VOD.L"], "Exchange Ticker": ["VOD"]})
            if self.calls > 4:
                raise ThrottledError("LSEG limit reached (test)")
            return pd.DataFrame({"Instrument": ["VOD.L"], "Date": ["2026-10-20"], "Currency": ["USD"]})

        def get_history(self, universe: list[str], *args: object) -> pd.DataFrame:
            return pd.DataFrame({"Instrument": universe, "Date": "2006-03-01", "OFF_CLOSE": 1.0})

        def allowance(self) -> dict[str, int] | None:
            return None

    report = probe_uk(cast(LsegClient, InDollars())).set_index("dataset")
    assert "members today" in str(report.loc["members: FTSE 100 / 250 / SmallCap since 2000", "detail"])
    assert report.loc["official closes (live and retired)", "ok"]
    assert not report.loc["prices in pence", "ok"] and "not pence" in str(report.loc["prices in pence", "detail"])
    assert "limit reached" in str(report.loc["dividends in pounds", "detail"])
    assert report["detail"].str.contains("not tried").sum() == 3  # after the limit, nothing more is asked
    assert len(report) == 10  # nine checks, then the allowance left


class FakeHistoryLd:
    """``ld.get_history`` as lseg-data builds it (seen live 2026-10-25): index levels answer the close only."""

    def get_history(self, universe: list[str], fields: list[str], **_: object) -> pd.DataFrame:
        days = pd.DatetimeIndex(["2009-06-01"], name="Date")
        answered = {r: ["TRDPRC_1"] if r.startswith(".") else fields for r in universe}
        if len(universe) == 1:
            return pd.DataFrame({c: [10.0] for c in answered[universe[0]]}, index=days)
        widths = {len(v) for v in answered.values()}
        if len(widths) > 1:
            raise IndexError("list assignment index out of range")  # its table builder, on mixed widths
        if widths == {1}:
            if len(fields) == 1:
                return pd.DataFrame({r: [10.0] for r in universe}, index=days)  # a column per RIC
            garbled = pd.MultiIndex.from_tuples([tuple(r.ljust(7)) for r in universe])  # each RIC cut into letters
            return pd.DataFrame([[10.0] * len(universe)], index=days, columns=garbled)
        columns = pd.MultiIndex.from_product([universe, fields])
        return pd.DataFrame([[10.0] * len(columns)], index=days, columns=columns)


def _client(monkeypatch: pytest.MonkeyPatch) -> LsegClient:
    client = LsegClient(pause_seconds=0.0)
    monkeypatch.setattr(client, "_ensure_session", FakeHistoryLd)
    monkeypatch.setattr(client, "allowance", lambda: None)  # never this computer's real Workspace
    return client


def test_a_history_lseg_data_cannot_build_is_split_until_each_part_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    out = client.get_history([".SPX", "A.N", "B.N", "C.N"], ["TRDPRC_1", "ACVOL_UNS"], "2009-06-01", "2009-06-01")
    assert sorted(out["Instrument"]) == [".SPX", "A.N", "B.N", "C.N"]
    assert client.skipped == []


def test_index_levels_asked_together_for_their_close(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    out = client.get_history([".SPX", ".MID", ".SP600"], ["TRDPRC_1"], "2009-06-01", "2009-06-01")
    assert sorted(out["Instrument"]) == [".MID", ".SP600", ".SPX"]
    assert list(out.columns) == ["Date", "Instrument", "TRDPRC_1"]
    assert client.requests == 1


def test_a_garbled_answer_is_split_not_misread(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    out = client.get_history([".SPX", ".MID"], ["TRDPRC_1", "ACVOL_UNS"], "2009-06-01", "2009-06-01")
    assert sorted(out["Instrument"]) == [".MID", ".SPX"]


def test_a_bar_whose_close_lies_outside_its_range_keeps_the_close_only() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["A.N", "A.N"],
            "Date": ["2006-10-23", "2006-10-24"],
            "TRDPRC_1": [36.0, 37.33],
            "OPEN_PRC": [35.5, 35.0],
            "HIGH_1": [36.5, 36.65],  # 2006-10-24: LSEG's range is another day's
            "LOW_1": [35.2, 34.89],
            "ACVOL_UNS": [100, 200],
        }
    )
    out = norm.price_bars(raw).set_index("trade_date")
    assert out.loc["2006-10-24", "close"] == 37.33
    assert out.loc["2006-10-24", ["open", "high", "low"]].isna().all()
    assert out.loc["2006-10-23", "high"] == 36.5


def test_a_history_costs_one_request_per_ric(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    client.get_history(["A.N", "B.N", "C.N"], ["TRDPRC_1", "ACVOL_UNS"], "2009-06-01", "2009-06-01")
    assert client.requests == 1 and client.lseg_requests == 3


def test_a_call_the_allowance_cannot_pay_for_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(client, "allowance", lambda: {"daily_request_remaining": 2})
    with pytest.raises(ThrottledError, match="needs 3 requests, 2 left"):
        client.get_history(["A.N", "B.N", "C.N"], ["TRDPRC_1"], "2009-06-01", "2009-06-01")
    assert client.requests == 0


def test_prices_are_asked_in_eleven_year_windows_from_the_build_start() -> None:
    from equity_research.pipelines.load_lseg import price_windows  # noqa: PLC0415 - the pipeline, for this one test

    windows = price_windows(pd.Timestamp("2003-11-27"), pd.Timestamp("2026-10-25"))
    assert [(lo.date().isoformat(), hi.date().isoformat()) for lo, hi in windows] == [
        ("2003-11-27", "2013-12-31"),
        ("2014-01-01", "2024-12-31"),
        ("2025-01-01", "2026-10-25"),
    ]
    assert all(len(pd.bdate_range(lo, hi)) < 3000 for lo, hi in windows)  # inside LSEG's rows per request
    assert price_windows(pd.Timestamp("2026-10-15"), pd.Timestamp("2026-10-25")) == [
        (pd.Timestamp("2026-10-15"), pd.Timestamp("2026-10-25"))
    ]  # a refresh: one short window


def test_an_announcement_made_before_its_quarter_ended_is_not_its_own() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["MOS.N"],
            "Period End Date": ["2014-03-31"],
            **{col: [1.0] for col, _, _ in fields.STATEMENT_ITEMS.values()},
        }
    )
    # LSEG lined this quarter up with the previous quarter's announcement (seen for Mosaic, 2026-10-26)
    announced = norm.announcements(
        pd.DataFrame(
            {
                "Instrument": ["MOS.N"],
                "Period End Date": ["2014-03-31"],
                "Original Announcement Date Time": ["2013-11-05"],
            }
        )
    )
    facts = norm.statements(raw, announced)
    assert set(facts["published_at"]) == {pd.Timestamp("2014-06-29")}  # 90 days after the quarter, never before it


def test_capital_changes_on_one_day_act_together_and_a_phantom_split_is_flagged() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["MSI.N", "MSI.N", "COV.N^A15", "AIG.N"],
            "Capital Change Effective Date": ["2011-01-04", "2011-01-04", "2011-03-21", "2009-07-01"],
            # a 1-for-7 reverse split and a spin-off; a "100-for-1" split; a distressed 1-for-20 reverse split
            "Adjustment Factor": [7.0, 0.585, 0.01, 20.0],
        }
    )
    closes = pd.DataFrame(
        {
            "ric": ["MSI.N"] * 2 + ["COV.N^A15"] * 2 + ["AIG.N"] * 2,
            "trade_date": pd.to_datetime(
                ["2011-01-03", "2011-01-04", "2011-03-18", "2011-03-21", "2009-06-30", "2009-07-01"]
            ),
            "close": [9.11, 39.77, 51.64, 52.98, 1.16, 18.08],  # AIG moved x15.6 on its x20 day
        }
    )
    out = norm.capital_changes(raw, closes).set_index("ric")
    assert out.loc["MSI.N", "ratio"] == pytest.approx(1 / (7.0 * 0.585))  # both, as the price moved: x4.37
    assert bool(out.loc["MSI.N", "verified"])
    assert not bool(out.loc["COV.N^A15", "verified"])  # the price never moved a hundredfold: the load leaves it out
    assert bool(out.loc["AIG.N", "verified"]) and out.loc["AIG.N", "ratio"] == pytest.approx(0.05)  # roughly, but real


def test_a_window_without_capital_changes_leaves_nothing_to_write() -> None:
    raw = pd.DataFrame(columns=["Instrument", "Capital Change Effective Date", "Adjustment Factor"])
    out = norm.capital_changes(raw, pd.DataFrame(columns=["ric", "trade_date", "close"]))
    confirmed = out[out["verified"]]  # a mask of rows, not a list of columns, even when empty
    assert confirmed.empty and "verified" in confirmed.columns


def test_a_refresh_reads_daily_prices_and_drops_a_retired_ric_padded_with_its_last_bar() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["AAA.N", "AAA.N", "LEH.N^I08", "LEH.N^I08"],
            "Date": ["2026-10-13T00:00:00Z", "2026-10-14T00:00:00Z", "2008-09-12", "2008-09-12"],
            "Price Close": [171.6, 174.0, 3.65, 3.65],
            "Volume": [566365, 683910, 0, 0],
        }
    )
    out = norm.daily_prices(raw, "2026-10-01", "2026-10-30")
    assert out["ric"].tolist() == ["AAA.N", "AAA.N"]
    assert out["close"].tolist() == [171.6, 174.0] and out["volume"].tolist() == [566365, 683910]
    assert out["trade_date"].tolist() == [pd.Timestamp("2026-10-13"), pd.Timestamp("2026-10-14")]
    assert out[["open", "high", "low"]].isna().all().all()  # close and volume only, as the histories now


class FakeBackendLd:
    """``ld.get_data`` whose backend fails any request for more than two RICs, and always for one bad RIC."""

    def __init__(self) -> None:
        self.calls = 0

    def get_data(self, universe: list[str], fields: list[str], parameters: object = None) -> pd.DataFrame:
        self.calls += 1
        if len(universe) > 2 or "BAD.N" in universe:
            raise RuntimeError("Backend error. 400 Bad Request Requested universes: ...")
        return pd.DataFrame({"Instrument": universe, "EBIT - Mean": 1.0})


def test_a_request_the_backend_keeps_failing_is_split_and_a_hopeless_ric_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeBackendLd()
    client = LsegClient(pause_seconds=0.0, max_retries=2)
    monkeypatch.setattr(client, "_ensure_session", lambda: fake)
    monkeypatch.setattr(client, "allowance", lambda: None)
    monkeypatch.setattr("equity_research.ingest.vendors.refinitiv.client.time.sleep", lambda _: None)
    out = client.get_data(["A.N", "B.N", "C.N", "D.N", "BAD.N"], ["TR.EBITMean"])
    assert sorted(out["Instrument"]) == ["A.N", "B.N", "C.N", "D.N"]
    assert client.skipped == ["BAD.N"]


def test_a_yearly_item_is_dated_by_its_years_announcement_and_payments_and_ratios_are_normalised() -> None:
    announced = pd.DataFrame(
        {"ric": ["A.N"], "period_end": [pd.Timestamp("2023-12-31")], "announced_at": [pd.Timestamp("2024-02-10 07:00")]}
    )
    capex = pd.DataFrame(
        {
            "Instrument": ["A.N", "A.N"],
            "Period End Date": ["2022-12-31", "2023-12-31"],
            "Capital Expenditures, Cumulative": [-50.0, -60.0],
        }
    )
    out = norm.annual_statement(capex, "TR.CapitalExpenditures", announced).set_index("period_end")
    assert out["value"].tolist() == [50.0, 60.0]  # paid, whatever the sign LSEG sends
    assert (out["period_type"] == "FY").all() and out.loc["2023-12-31", "period_start"] == pd.Timestamp("2023-01-01")
    assert out.loc["2023-12-31", "published_at"] == pd.Timestamp("2024-02-10 07:00")
    assert out.loc["2022-12-31", "published_at"] == pd.Timestamp("2023-03-31")  # no announcement: 90 days on
    loans = pd.DataFrame({"Instrument": ["A.N"], "Period End Date": ["2023-12-31"], "Net Loans": [900.0]})
    r = norm.annual_statement(loans, "TR.NetLoans", announced).iloc[0]
    assert (r["value"], r["unit"], r["period_type"]) == (900.0, "USD", "I")
    guess = pd.DataFrame({"Instrument": ["A.N"], "Period End Date": ["2023-12-31"], "RWA - Broker Estimate": [9.0]})
    assert norm.annual_statement(guess, "TR.RiskWeightedAssets", announced).empty  # a forecast is not a fact
    assert fields.annual_fields("TR.NetLoans")[0] == "TR.TotalAssets.periodenddate"  # banks have no revenue end
    nothing = pd.DataFrame({"Instrument": ["A.N"], "Period End Date": ["2023-12-31"]})  # LSEG had no such line
    assert norm.annual_statement(nothing, "TR.LoanLossProvision", announced).empty


def test_london_listings_take_lseg_tickers_pence_permids_and_funds() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["SJP.L", "SMT.L", "VNHq.L", ".FTSE"],
            "Company Common Name": ["St James's Place PLC", "Scottish Mortgage", "Vianet", ""],
            "ISIN": ["GB0007669376", "GB00BLDYK618", "GB00B1G55Q09", ""],
            "SEDOL": ["0766937", "BLDYK61", "B1G55Q0", ""],
            "CUSIP": ["", "", "", ""],
            "Exchange Country Code": ["GB", "GB", "GB", ""],
            "Exchange Ticker": ["STJ", "SMT", "VNH", ""],
        }
    )
    attrs = norm.instruments(raw, level_currency="GBP").set_index("ric")
    assert attrs.loc["SJP.L", "ticker"] == "STJ" and attrs.loc["SJP.L", "currency"] == "GBX"  # LSEG's, in pence
    assert attrs.loc[".FTSE", "currency"] == "GBP"
    firms = norm.companies(
        pd.DataFrame(
            {
                "Instrument": ["SJP.L", "SMT.L", "VNHq.L"],
                "Organization PermID": [4295895330, 4295895352, 4295894740],
                "Country ISO Code of Incorporation": ["GB", "GB", "JE"],
                "TRBC Industry Name": ["Investment Management", "UK Investment Trusts", "Software"],
            }
        )
    )
    frame = norm.listings(attrs.reset_index().merge(firms, on="ric", how="left")).set_index("anchor")
    assert frame.loc["SJP.L", "issuer_permid"] == "4295895330" and missing(frame.loc["SJP.L", "issuer_isin"])
    assert frame.loc["SMT.L", "instrument_type"] == "fund" and frame.loc["SJP.L", "instrument_type"] == "equity"
    assert frame.loc["VNHq.L", "incorporated"] == "JE" and missing(frame.loc["VNHq.L", "share_class"])
    assert frame.loc[".FTSE", "instrument_type"] == "index" and missing(frame.loc[".FTSE", "issuer_permid"])


def test_london_dividends_keep_their_currency() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["SHEL.L", "SHEL.L"],
            "Dividend Ex Date": ["2024-02-15", "2024-05-16"],
            "Gross Dividend Amount": [0.269, 0.2694],
            "Dividend Type": ["Final", "Interim"],
            "Currency": ["GBP", "GBP"],
        }
    )
    assert norm.dividends(raw)["currency"].tolist() == ["GBP", "GBP"]


def _statement(rows: list[tuple[str, str, float, float, str]]) -> pd.DataFrame:
    """An answer to UK_STATEMENT_FIELDS: (ric, period end, revenue, total assets, announced), other items blank."""
    out = pd.DataFrame(
        {
            "Instrument": [r[0] for r in rows],
            "Period End Date": [r[1] for r in rows],
            **{col: None for col, _, _ in fields.UK_STATEMENT_ITEMS.values()},
            "Original Announcement Date Time": [r[4] for r in rows],
        }
    )
    out["Revenue from Business Activities - Total"] = [r[2] for r in rows]
    out["Total Assets"] = [r[3] for r in rows]
    out["Capital Expenditures - Total"] = -5.0
    return out


def test_half_years_become_years_so_far_and_years_stay_as_reported() -> None:
    halves = _statement(
        [
            ("TSCO.L", "2016-08-27", 27_338.0, 47_172.0, "2016-10-05 12:00"),
            ("TSCO.L", "2017-02-25", 28_579.0, 45_853.0, "2017-04-12 12:00"),  # the second half: the year has it
        ]
    )
    years = _statement([("TSCO.L", "2017-02-25", 55_917.0, 45_853.0, "2017-04-12 12:00")])
    facts = norm.half_year_statements(halves, years, unit="GBP")
    revenue = facts[facts["item"] == "revenue"].set_index("period_type")
    assert revenue.loc["YTD", "value"] == 27_338 and revenue.loc["YTD", "period_start"] == pd.Timestamp("2016-02-28")
    assert revenue.loc["FY", "value"] == 55_917  # as reported, not the halves added up
    assert revenue.loc["FY", "published_at"] == pd.Timestamp("2017-04-12 12:00")
    assert len(revenue) == 2 and (facts["unit"] == "GBP").all()
    assets = facts[facts["item"] == "total_assets"]
    assert sorted(assets["value"]) == [45_853.0, 47_172.0] and (assets["period_type"] == "I").all()  # once per day
    assert (facts.loc[facts["item"] == "capex", "value"] > 0).all()  # paid, as an amount


def test_exchange_rates_read_as_dollars_per_unit() -> None:
    raw = pd.DataFrame(
        {"Instrument": ["GBP=", "ZAR="], "Date": ["2005-01-04", "2005-01-04"], "MID_PRICE": [1.8822, 5.8325]}
    )
    bars = norm.fx_bars(raw, {"GBP=": True, "ZAR=": False}).set_index("ric")["close"]
    assert bars["GBP="] == 1.8822 and bars["ZAR="] == pytest.approx(1 / 5.8325)  # rand per dollar, inverted
    assert rics.is_fx("ZAR=") and not rics.is_fx("ZAR.L")


def _pounds(days: pd.DatetimeIndex) -> pd.DataFrame:
    """One unit's value in pounds: pence, pounds, dollars at $1.25, euros at EUR 1.15."""
    return pd.DataFrame({"GBX": 0.01, "GBP": 1.0, "USD": 1 / 1.25, "EUR": 1 / 1.15}, index=days)


def test_a_change_of_quote_currency_is_found_by_the_exchange_rate() -> None:
    days = pd.bdate_range("2025-12-29", periods=6)
    # InterContinental Hotels: pence to dollars on 2026-01-02 (10,460p is $130.75)
    ihg = pd.DataFrame({"trade_date": days, "close": [10365.0, 10440.0, 10460.0, 138.45, 135.65, 136.55]})
    assert norm.quote_currencies(ihg, "USD", _pounds(days)).tolist() == ["GBX"] * 3 + ["USD"] * 3
    # Coca-Cola Europacific: dollars to pence ($72.80 is 5,824p)
    ccep = pd.DataFrame({"trade_date": days, "close": [72.1, 72.5, 72.8, 5990.0, 6010.0, 5980.0]})
    assert norm.quote_currencies(ccep, "GBX", _pounds(days)).tolist() == ["USD"] * 3 + ["GBX"] * 3
    # a 1-for-100 consolidation moves like pounds to pence, but a capital change explains it
    merged = pd.DataFrame({"trade_date": days, "close": [1.2, 1.1, 1.0, 101.0, 99.0, 98.0]})
    assert norm.quote_currencies(merged, "GBX", _pounds(days)).tolist() == ["GBP"] * 3 + ["GBX"] * 3
    assert set(norm.quote_currencies(merged, "GBX", _pounds(days), quiet={days[3]})) == {"GBX"}
    # a collapse is not a currency (Hammerson: 280.6p to 23p, a rights issue - no rate is 12x)
    crash = pd.DataFrame({"trade_date": days, "close": [287.0, 282.0, 280.6, 23.0, 24.1, 24.9]})
    assert set(norm.quote_currencies(crash, "GBX", _pounds(days))) == {"GBX"}


def test_a_consolidation_the_share_count_shows_is_no_change_of_quote_currency() -> None:
    # Durlacher (now Panmure Liberum), November 2002: 1.05p to 113p - pounds to pence, by the rate - but a 1-for-140
    # consolidation LSEG has no action for: its shares fell from 570m to 4m that month
    days = pd.bdate_range("2002-10-30", periods=6)
    durlacher = pd.DataFrame({"trade_date": days, "close": [1.0, 1.05, 1.05, 113.0, 115.0, 112.0]})
    assert norm.quote_currencies(durlacher, "GBX", _pounds(days)).tolist() == ["GBP"] * 3 + ["GBX"] * 3
    counts = pd.Series([569_850_595.0, 4_070_631.0], index=pd.to_datetime(["2002-10-31", "2002-11-29"]))
    assert set(norm.quote_currencies(durlacher, "GBX", _pounds(days), counts=counts)) == {"GBX"}
    # a count that does not move leaves a real change of quote currency alone (InterContinental Hotels, 2026)
    ihg_days = pd.bdate_range("2025-12-29", periods=6)
    ihg = pd.DataFrame({"trade_date": ihg_days, "close": [10365.0, 10440.0, 10460.0, 138.45, 135.65, 136.55]})
    flat = pd.Series([1.6e8, 1.6e8], index=pd.to_datetime(["2025-12-31", "2026-01-30"]))
    assert norm.quote_currencies(ihg, "USD", _pounds(ihg_days), counts=flat).tolist() == ["GBX"] * 3 + ["USD"] * 3


def test_a_jump_under_a_penny_is_no_change_of_quote_currency() -> None:
    # Quarterly High Income Trust, 2002: 0.0205 to 0.50 (pence) matches pence per lari, a currency LSEG's dividends
    # are declared in; a near-worthless line's jump is noise, and four years of pence were read as lari
    days = pd.bdate_range("2002-09-24", periods=6)
    pounds = _pounds(days).assign(GEL=0.3)
    trust = pd.DataFrame({"trade_date": days, "close": [0.02, 0.0205, 0.0205, 0.5001, 0.45, 0.5]})
    assert set(norm.quote_currencies(trust, "GBX", pounds)) == {"GBX"}


def test_count_ratios_take_the_counts_either_side_of_each_day() -> None:
    counts = pd.Series([100.0, 50.0], index=pd.to_datetime(["2020-01-31", "2020-02-28"]))
    days = pd.to_datetime(["2020-01-15", "2020-02-10", "2020-02-28", "2020-06-01"])
    out = norm.count_ratios(counts, pd.DatetimeIndex(days))
    assert np.isnan(out[0]) and np.isnan(out[3])  # no count before it; both too far from it
    assert out[1] == 0.5 and out[2] == 0.5  # a count dated on the day is the one after it


def test_london_capital_changes_go_by_their_ex_date() -> None:
    raw = pd.DataFrame(
        {"Instrument": ["RR.L"], "Capital Change Ex Date": ["2020-10-28"], "Adjustment Factor": [0.34317]}
    )
    days = pd.bdate_range("2020-10-26", periods=5)
    closes = pd.DataFrame({"ric": "RR.L", "trade_date": days, "close": [226.1, 219.0, 84.54, 72.4, 71.3]})
    out = norm.capital_changes(raw, closes, columns=fields.UK_CAPITAL_CHANGE_COLUMNS, days_after=3)
    assert out["ex_date"].tolist() == [pd.Timestamp("2020-10-28")] and out["verified"].all()
    assert norm.current_currencies(pd.DataFrame({"Instrument": ["IHG.L", "VOD.L"], "CF_CURR": ["USD", "GBp"]}))[
        "currency"
    ].tolist() == ["USD", "GBX"]


def _jump_on_day_3(
    ric: str, before: float, after: float, shares_before: float, shares_after: float
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """Six days of closes jumping on the fourth, month-end share counts around them."""
    days = pd.bdate_range("2020-01-13", periods=6)
    closes = pd.DataFrame({"ric": ric, "trade_date": days, "close": [before] * 3 + [after] * 3})
    counts = pd.DataFrame(
        {
            "ric": ric,
            "as_of_date": pd.to_datetime(["2019-12-31", "2020-01-31"]),
            "shares": [shares_before, shares_after],
        }
    )
    return days, closes, counts


def test_the_share_count_confirms_a_split_the_price_alone_could_not() -> None:
    nothing = pd.DataFrame(columns=["ric", "action_type", "ex_date", "ratio", "verified"]).astype({"verified": bool})
    # Capital & Regional's 1-for-10: LSEG dates it two months early, the price jumps x10, the shares /10
    days, closes, counts = _jump_on_day_3("CAL.L", 24.05, 244.0, 1.12e9, 1.12e8)
    out = norm.confirm_by_share_count(nothing, closes, counts)
    assert out[["action_type", "ex_date", "ratio"]].to_numpy().tolist() == [["split", days[3], 0.1]]
    # a collapse leaves the count as it was
    _, closes, counts = _jump_on_day_3("AFR.L", 17.67, 5.0, 1.1e9, 1.1e9)
    assert norm.confirm_by_share_count(nothing, closes, counts).empty
    # a crash that met an issue of shares matches the count but is no whole exchange (National City, 2008)
    _, closes, counts = _jump_on_day_3("NCC.N", 4.92, 1.81, 8.0e8, 2.176e9)
    assert norm.confirm_by_share_count(nothing, closes, counts).empty
    # a split already stored (under another of the line's codes) explains its day: nothing added again
    held = pd.DataFrame({"ric": ["CAL.L"], "ex_date": [days[3]], "ratio": [0.1]})
    days, closes, counts = _jump_on_day_3("CAL.L", 24.05, 244.0, 1.12e9, 1.12e8)
    assert norm.confirm_by_share_count(nothing, closes, counts, held=held).empty


def test_the_share_count_confirms_lsegs_split_that_came_with_cash() -> None:
    # Pinewood's 1-for-20 of 2024 with a return of cash: the price x8, not x20; the shares /20
    days, closes, counts = _jump_on_day_3("PINE.L", 39.3, 315.0, 1.4e9, 7.0e7)
    raw = pd.DataFrame(
        {"Instrument": ["PINE.L"], "Capital Change Ex Date": [days[3].date().isoformat()], "Adjustment Factor": [20.0]}
    )
    changes = norm.capital_changes(raw, closes, columns=fields.UK_CAPITAL_CHANGE_COLUMNS, days_after=3)
    assert not changes["verified"].any()  # the price alone: x8 is not within a factor of 2 of x20
    out = norm.confirm_by_share_count(changes, closes, counts, days_after=3)
    assert out[["ex_date", "ratio", "verified"]].to_numpy().tolist() == [[days[3], 0.05, True]]
    # the same split without the count's change stays left out
    _, _, flat = _jump_on_day_3("PINE.L", 39.3, 315.0, 1.4e9, 1.4e9)
    assert not norm.confirm_by_share_count(changes, closes, flat, days_after=3)["verified"].any()


def test_a_london_refresh_keeps_the_currency_each_price_is_quoted_in() -> None:
    raw = pd.DataFrame(
        {
            "Instrument": ["IHG.L", "VOD.L"],
            "Date": ["2026-10-20", "2026-10-20"],
            "Price Close": [138.45, 119.4],
            "Volume": [1000, 2000],
            "Currency": ["USD", "GBp"],
        }
    )
    bars = norm.daily_prices(raw, "2026-10-01", "2026-10-31").set_index("ric")
    assert bars.loc["IHG.L", "currency"] == "USD" and bars.loc["VOD.L", "currency"] == "GBX"
