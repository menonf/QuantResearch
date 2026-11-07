import pandas as pd

from equity_research.ingest.vendors.sec import normalize as sec


def missing(value: object) -> bool:
    """A missing value (NaN, NaT, None) - typed ``object`` so the checker does not narrow what a frame holds."""
    return bool(pd.isna(value))  # type: ignore[call-overload]


def _fact(
    start: str | None, end: str, val: float, accn: str, form: str, filed: str, fp: str = "Q1"
) -> dict[str, object]:
    f: dict[str, object] = {"end": end, "val": val, "accn": accn, "form": form, "filed": filed, "fy": 2021, "fp": fp}
    if start:
        f["start"] = start
    return f


def _companyfacts() -> dict[str, object]:
    revenues = [
        _fact("2021-01-01", "2021-03-31", 100, "q1", "10-Q", "2021-05-01"),
        _fact("2021-01-01", "2021-06-30", 210, "q2", "10-Q", "2021-08-01", "Q2"),  # YTD
        _fact("2021-01-01", "2021-12-31", 450, "k21", "10-K", "2022-02-20", "FY"),
        _fact("2021-01-01", "2021-12-31", 450, "k22", "10-K", "2023-02-20", "FY"),  # same value repeated: nothing
        _fact("2021-01-01", "2021-03-31", 105, "q1b", "10-Q", "2022-05-01"),  # restated in a later filing
        _fact("2021-01-01", "2021-03-31", 1, "x", "8-K", "2021-04-20"),  # not a 10-K / 10-Q: ignored
    ]
    lower_priority = [_fact("2021-01-01", "2021-03-31", 999, "q1", "10-Q", "2021-05-01")]
    assets = [_fact(None, "2021-03-31", 1000, "q1", "10-Q", "2021-05-01")]
    cover = [_fact(None, "2021-04-25", 50, "q1", "10-Q", "2021-05-01")]
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": revenues}},
                "SalesRevenueNet": {"units": {"USD": lower_priority}},
                "Assets": {"units": {"USD": assets}},
                "NotATag": {"units": {"USD": assets}},
            },
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover}}},
        }
    }


def test_periods_are_classified() -> None:
    start = pd.to_datetime(pd.Series([None, "2021-01-01", "2021-01-01", "2021-01-01"]))
    end = pd.to_datetime(pd.Series(["2021-03-31", "2021-03-31", "2021-06-30", "2021-12-31"]))
    assert sec.period_type(start, end).tolist() == ["I", "Q", "YTD", "FY"]


def test_facts_keep_first_reports_and_restatements_only() -> None:
    out = sec.financial_facts(sec.facts_frame(_companyfacts()))
    rev = out[out["item"] == "revenue"].sort_values(["period_type", "published_at"])
    q1 = rev[rev["period_type"] == "Q"]
    assert q1["value"].tolist() == [100, 105]  # the original, then the restatement (the lower-priority tag lost)
    assert q1["published_at"].tolist() == [pd.Timestamp("2021-05-01"), pd.Timestamp("2022-05-01")]
    fy = rev[rev["period_type"] == "FY"]
    assert fy["value"].tolist() == [450] and fy["filing_ref"].tolist() == ["k21"]  # the repeat added nothing
    assert rev[rev["period_type"] == "YTD"]["value"].tolist() == [210]
    assert out.loc[out["item"] == "shares_cover", "value"].tolist() == [50]
    assert set(out["item"]) == {"revenue", "total_assets", "shares_cover"}


def test_each_filing_is_dated_and_has_its_report_period() -> None:
    f = sec.filings(sec.facts_frame(_companyfacts())).set_index("filing_ref")
    assert f.loc["q1", "period_end"] == pd.Timestamp("2021-03-31")
    assert f.loc["q1", "published_at"] == pd.Timestamp("2021-05-01")
    assert f.loc["k21", "form"] == "10-K" and f.loc["k21", "fiscal_period"] == "FY"
    assert "x" not in f.index  # the 8-K is not a financial report


def test_a_fiscal_year_that_is_not_a_year_is_dropped() -> None:
    facts = _companyfacts()
    revenues = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]  # type: ignore[index]
    revenues[0]["fy"] = 43101  # a spreadsheet date serial, as some filers enter it
    f = sec.filings(sec.facts_frame(facts)).set_index("filing_ref")
    assert missing(f.loc["q1", "fiscal_year"])
    assert f.loc["k21", "fiscal_year"] == 2021


def test_a_fact_about_a_period_after_its_filing_is_dropped() -> None:
    debt = [
        _fact(None, "2011-09-30", 700, "q3", "10-Q", "2011-11-02"),
        _fact(None, "2041-03-15", 200, "q3", "10-Q", "2011-11-02"),  # a note's maturity tagged as a balance date
    ]
    facts = sec.facts_frame({"facts": {"us-gaap": {"NotesPayable": {"units": {"USD": debt}}}}})
    out = sec.financial_facts(facts)
    assert out["period_end"].tolist() == [pd.Timestamp("2011-09-30")]


def test_a_filing_with_share_counts_only_is_a_filing() -> None:
    cover = [_fact(None, "2023-03-07", 50, "amend", "10-K/A", "2023-03-10", "FY")]
    shares = [_fact(None, "2022-12-31", 49, "amend", "10-K/A", "2023-03-10", "FY")]
    facts = sec.facts_frame(
        {
            "facts": {
                "us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares": shares}}},
                "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover}}},
            }
        }
    )
    f = sec.filings(facts).set_index("filing_ref")
    assert f.loc["amend", "form"] == "10-K/A"
    assert f.loc["amend", "period_end"] == pd.Timestamp("2023-03-07")  # no amounts: the latest share count's date


def test_cash_flow_statements_come_through_as_year_to_date_flows() -> None:
    cash = [
        _fact("2021-01-01", "2021-03-31", 30, "q1", "10-Q", "2021-05-01"),
        _fact("2021-01-01", "2021-06-30", 70, "q2", "10-Q", "2021-08-01", "Q2"),  # cash flows are year to date
    ]
    capex = [_fact("2021-01-01", "2021-06-30", 25, "q2", "10-Q", "2021-08-01", "Q2")]
    facts = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": cash}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": capex}},
            }
        }
    }
    out = sec.financial_facts(sec.facts_frame(facts)).set_index(["item", "period_type"])["value"]
    assert out[("operating_cash_flow", "Q")] == 30 and out[("operating_cash_flow", "YTD")] == 70
    assert out[("capex", "YTD")] == 25


def test_a_capital_ratio_is_read_as_a_pure_number() -> None:
    ratio = [_fact(None, "2021-12-31", 0.135, "k21", "10-K", "2022-02-20", "FY")]
    facts = {"facts": {"us-gaap": {"TierOneRiskBasedCapitalToRiskWeightedAssets": {"units": {"pure": ratio}}}}}
    out = sec.financial_facts(sec.facts_frame(facts))
    row = out.iloc[0]
    assert len(out) == 1 and (row["item"], row["period_type"], row["value"], row["unit"]) == (
        "tier1_ratio",
        "I",
        0.135,
        "pure",
    )
