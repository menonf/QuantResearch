from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_research.research.cache import ResearchCache
from equity_research.research.data import (
    CurrencyError,
    accounts_as_of,
    check_currency,
    company_values,
    dividends_in,
    in_currency,
    in_major_units,
    in_research_currency,
    in_shares_of,
    listing_values,
    log_change,
    operating_companies,
    own_counts,
    price_actions,
    shares_a_year_before,
    stamp_duty,
    universe_currency,
    vendor_items,
)
from equity_research.research.panels import (
    membership_mask,
    split_adjusted,
    split_factor,
    total_returns,
    trading_days,
)
from equity_research.research.ratios import market_values, plausible_market_cap, ratio_panel

DAYS = pd.DatetimeIndex(pd.to_datetime(["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"]))
ACTIONS = pd.DataFrame(
    {
        "listing_id": [7, 7],
        "action_type": ["split", "cash_dividend"],
        "ex_date": pd.to_datetime(["2020-08-31", "2020-09-01"]),
        "ratio": [4.0, np.nan],
        "amount": [np.nan, 1.0],
    }
)


def _close() -> pd.DataFrame:
    return pd.DataFrame({7: [500.0, 499.0, 129.0, 130.0]}, index=DAYS)


def test_split_adjustment_expresses_old_prices_in_todays_shares() -> None:
    adj = split_adjusted(_close(), ACTIONS)[7]
    assert adj.loc["2020-08-28"] == pytest.approx(124.75)
    assert adj.loc["2020-08-31"] == 129.0


def test_total_returns_bridge_the_split_and_add_dividends() -> None:
    r = total_returns(_close(), ACTIONS)[7]
    assert r.loc["2020-08-31"] == pytest.approx(129.0 / 124.75 - 1)  # no fake -74 % on the split day
    assert r.loc["2020-09-01"] == pytest.approx((130.0 + 1.0) / 129.0 - 1)


def test_membership_end_dates_are_exclusive() -> None:
    m = pd.DataFrame(
        {"listing_id": [7], "start_date": pd.to_datetime(["2020-08-28"]), "end_date": pd.to_datetime(["2020-09-01"])}
    )
    mask = membership_mask(m, DAYS, pd.Index([7]))[7]
    assert mask.tolist() == [False, True, True, False]


def test_market_cap_uses_todays_price_and_split_adjusted_shares() -> None:
    fund = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-09-01")],
            "listing_id": [7],
            "shares": [100.0],  # reported for June 30, before the 4-for-1 split
            "known_at:shares": [pd.Timestamp("2020-06-30")],
            "net_income": [1300.0],
            "equity": [1000.0],
            "revenue": [5200.0],
            "ebit": [2000.0],
            "operating_income": [2000.0],
            "total_debt": [500.0],
            "cash": [100.0],
            "current_assets": [800.0],
            "current_liabilities": [300.0],
            "fixed_assets": [500.0],
            "total_assets": [3000.0],
            "total_liabilities": [2000.0],
            "retained_earnings": [600.0],
            "gross_profit": [2600.0],
            "interest_expense": [-50.0],
        }
    ).set_index(["date", "listing_id"])
    close = pd.DataFrame({7: [500.0, 130.0]}, index=pd.to_datetime(["2020-06-30", "2020-09-01"]))
    factor = split_factor(ACTIONS, pd.DatetimeIndex(close.index), close.columns)
    r = ratio_panel(fund, close, factor).iloc[0]
    mkt_cap = 130.0 * 400.0  # 100 shares in June = 400 after the split
    assert r["P/E"] == pytest.approx(mkt_cap / 1300.0)
    assert r["EV/EBIT"] == pytest.approx((mkt_cap + 500 - 100) / 2000.0)
    assert r["ROC"] == pytest.approx(2000.0 / (500 + 500))
    assert r["Gross Profit/TA"] == pytest.approx(2600.0 / 3000.0)
    values = market_values(fund, close, factor).iloc[0]
    assert values["shares"] == pytest.approx(400.0) and values["market_cap"] == pytest.approx(mkt_cap)
    assert r["Op. In./Interest Expense"] == pytest.approx(40.0)  # the sign of the expense does not matter


def test_a_count_filed_after_a_split_is_already_in_the_new_shares() -> None:
    inputs = ["net_income", "equity", "revenue", "ebit", "operating_income", "total_debt", "cash", "current_assets"]
    inputs += ["current_liabilities", "fixed_assets", "total_assets", "total_liabilities", "retained_earnings"]
    inputs += ["gross_profit", "interest_expense"]
    fund = pd.DataFrame(
        {"date": [pd.Timestamp("2020-09-01")], "listing_id": [7], "shares": [400.0]}
        | {c: [1300.0] for c in inputs}
        | {"known_at:shares": [pd.Timestamp("2020-08-31")]}  # the June quarter, filed after the split: restated
    ).set_index(["date", "listing_id"])
    close = pd.DataFrame({7: [500.0, 130.0, 130.0]}, index=pd.to_datetime(["2020-06-30", "2020-08-31", "2020-09-01"]))
    factor = split_factor(ACTIONS, pd.DatetimeIndex(close.index), close.columns)
    assert ratio_panel(fund, close, factor).iloc[0]["P/E"] == pytest.approx(130.0 * 400.0 / 1300.0)


def test_a_holiday_printed_only_by_an_index_is_not_a_trading_day() -> None:
    days = pd.to_datetime(["2024-07-03", "2024-07-04", "2024-07-05", "2024-07-08"])
    close = pd.DataFrame(
        {
            1: [100.0, np.nan, 101.0, 102.0],
            2: [50.0, np.nan, 51.0, np.nan],  # one stock missing a day is still a gap, not a holiday
            3: [np.nan, np.nan, 20.0, 21.0],  # listed on the 5th: not alive before
            9: [5000.0, 5000.0, 5010.0, 5020.0],  # the index level, printed on the holiday
        },
        index=days,
    )
    kept = trading_days(close)
    assert list(kept) == [days[0], days[2], days[3]]
    returns = total_returns(
        close.loc[kept], pd.DataFrame(columns=["listing_id", "action_type", "ex_date", "amount", "ratio"])
    )
    assert returns.loc[days[2], 1] == pytest.approx(0.01)  # the return across the holiday is kept


def test_a_market_value_no_company_could_have_is_dropped() -> None:
    cap = pd.Series([3.3e12, 3.3e12, 4.0e12, 3.3e12])
    revenue = pd.Series([2e9, 2e9, 100e9, 2e9])
    equity = pd.Series([4.5e9, -1e9, 70e9, 40e9])
    # a mis-scaled broker; the same with negative equity (not judged); a real giant (40x sales); one multiple fine
    assert plausible_market_cap(cap, revenue, equity).isna().tolist() == [True, False, False, False]


def test_research_refuses_other_currencies_and_leaves_out_unusable_facts() -> None:
    check_currency(pd.Series(["USD", None, "USD"]), "prices")  # dollars and unknown pass
    with pytest.raises(CurrencyError, match="EUR"):
        check_currency(pd.Series(["USD", "EUR"]), "prices")
    facts = pd.DataFrame({"issuer_id": [1, 1, 2], "value": [1.0, 2.0, 3.0], "unit": ["USD", "shares", "EUR"]})
    kept = in_research_currency(facts)
    assert kept["value"].tolist() == [1.0, 2.0] and "unit" not in kept.columns
    pounds = facts.assign(unit=["GBP", "shares", "USD"])
    assert in_research_currency(pounds, "GBP")["value"].tolist() == [1.0, 2.0]  # a London universe reads pounds


def test_pence_are_read_as_pounds() -> None:
    bars = pd.DataFrame({"close": [118.6, 7000.0, 50.0], "currency": ["GBX", "GBP", "USD"]})
    read = in_major_units(bars)
    assert read["close"].tolist() == [1.186, 7000.0, 50.0] and read["currency"].tolist() == ["GBP", "GBP", "USD"]
    check_currency(read["currency"].iloc[:2], "prices", "GBP")
    assert bars["currency"].iloc[0] == "GBX"  # the stored frame is not changed


def _items(rows: list[tuple[int, str, str, float | None, float]]) -> pd.DataFrame:
    """Accounting items ``(issuer_id, period_end)`` x revenue / net income, from (issuer, period end, known, net
    income, revenue)."""
    frame = pd.DataFrame(rows, columns=["issuer_id", "period_end", "known", "net_income", "revenue"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    known = pd.to_datetime(frame["known"])
    frame["known_at:net_income"] = known.where(frame["net_income"].notna())
    frame["known_at:revenue"] = known
    return frame.drop(columns="known").set_index(["issuer_id", "period_end"])


def test_a_companys_accounts_on_a_date_come_from_the_first_vendor_that_can_answer() -> None:
    items = {
        # the SEC's first XBRL quarter: revenue, but no trailing net income until four quarters are filed
        "sec": _items(
            [
                (1, "2009-06-27", "2009-08-05", None, 52.0),
                (1, "2010-06-26", "2010-08-04", 14.0, 60.0),
                (3, "2009-06-30", "2009-08-10", None, 7.0),  # only the SEC, still unable to answer: kept
            ]
        ),
        "refinitiv": _items(
            [
                (1, "2009-06-30", "2009-07-21", 12.0, 50.0),
                (1, "2010-06-30", "2010-07-20", 13.0, 59.0),
                (2, "2009-06-30", "2009-07-25", 3.0, 30.0),  # a company the SEC has nothing for
            ]
        ),
    }
    dates = pd.DatetimeIndex(pd.to_datetime(["2009-09-30", "2010-09-30"]))
    out = accounts_as_of(items, dates)
    assert out.loc[(dates[0], 1), "vendor"] == "refinitiv"
    assert out.loc[(dates[0], 1), "revenue"] == 50.0  # not the SEC's 52: no mixing on one date
    assert out.loc[(dates[1], 1), "vendor"] == "sec"
    assert out.loc[(dates[1], 1), "net_income"] == 14.0  # the SEC answers once it can
    assert out.loc[(dates[0], 2), "vendor"] == "refinitiv"
    assert out.loc[(dates[0], 3), "vendor"] == "sec" and out.loc[(dates[0], 3), "revenue"] == 7.0
    assert accounts_as_of({}, dates).empty


def test_each_vendor_builds_its_items_from_its_own_facts_only() -> None:
    def fy(issuer: int, vendor: str, value: float) -> dict[str, object]:
        return {
            "issuer_id": issuer,
            "vendor": vendor,
            "item": "revenue",
            "period_type": "FY",
            "period_start": pd.Timestamp("2015-01-01"),
            "period_end": pd.Timestamp("2015-12-31"),
            "value": value,
            "published_at": pd.Timestamp("2016-02-20"),
            "unit": "USD",
        }

    facts = pd.DataFrame([fy(1, "sec", 100.0), fy(1, "refinitiv", 101.0), fy(2, "simfin", 5.0)])
    items = vendor_items(facts, ("sec", "refinitiv"))
    assert list(items) == ["sec", "refinitiv"]  # preference order; a vendor not asked for is left out
    assert items["sec"]["revenue"].tolist() == [100.0] and items["refinitiv"]["revenue"].tolist() == [101.0]


def test_the_share_change_over_a_year_is_not_fooled_by_a_split() -> None:
    items = pd.DataFrame(
        {
            "shares": [100.0, 120.0],
            "known_at:shares": pd.to_datetime(["2019-08-01", "2020-08-01"]),
        },
        index=pd.MultiIndex.from_arrays(
            [[3, 3], pd.to_datetime(["2019-06-30", "2020-06-30"])], names=["issuer_id", "period_end"]
        ),
    )
    date = pd.DatetimeIndex([pd.Timestamp("2020-09-01")])
    before = shares_a_year_before({"sec": items}, date, pd.Series({7: 3}), pd.Index([7]), ACTIONS)
    assert before.loc[date[0], 7] == pytest.approx(400.0)  # the 2019 count in shares after the 4-for-1 split
    now = pd.DataFrame({7: [480.0]}, index=date)  # 120 filed before the split, in today's shares
    assert log_change(now, before).iloc[0, 0] == pytest.approx(np.log(1.2))  # 20 % more shares, not 380 %
    assert log_change(now, before * 0.0).isna().all().all()  # a zero count is a gap


def test_a_companys_value_is_divided_among_its_member_listings() -> None:
    cap = pd.DataFrame({1: [200.0], 2: [200.0], 3: [50.0]}, index=DAYS[:1])
    members = pd.DataFrame({1: [True], 2: [True], 3: [True]}, index=DAYS[:1])
    values = listing_values(cap, members, pd.Series({1: 10, 2: 10, 3: 11})).iloc[0]  # 1 and 2: one company
    assert values.tolist() == [100.0, 100.0, 50.0]


def test_a_benchmark_takes_the_actions_of_the_vendor_that_prices_it() -> None:
    actions = pd.DataFrame(
        {
            "listing_id": [7, 7, 9, 9],
            "vendor": ["refinitiv", "yahoo", "yahoo", "refinitiv"],
            "action_type": ["cash_dividend"] * 4,
            "ex_date": pd.to_datetime(["2020-03-01"] * 4),
            "amount": [1.0, 1.0, 0.5, 0.4],
            "ratio": [np.nan] * 4,
        }
    )
    bars = pd.DataFrame({"listing_id": [7, 9, 9], "vendor": ["refinitiv", "yahoo", "yahoo"]})
    kept = price_actions(actions, bars, ("refinitiv",), pd.Series([9]))
    assert "vendor" not in kept.columns
    assert kept.set_index("listing_id")["amount"].to_dict() == {7: 1.0, 9: 0.5}  # 9 is priced by Yahoo


def test_free_cash_flow_and_accruals_come_from_the_cash_flow_statement() -> None:
    base = {
        "shares": 100.0,
        "net_income": 1300.0,
        "equity": 1000.0,
        "revenue": 5200.0,
        "ebit": 2000.0,
        "operating_income": 2000.0,
        "total_debt": 500.0,
        "cash": 100.0,
        "current_assets": 800.0,
        "current_liabilities": 300.0,
        "fixed_assets": 500.0,
        "total_assets": 3000.0,
        "total_liabilities": 2000.0,
        "retained_earnings": 600.0,
        "gross_profit": 2600.0,
        "interest_expense": -50.0,
        "operating_cash_flow": 1600.0,
    }
    fund = pd.DataFrame(
        [base | {"listing_id": 7, "capex": 400.0}, base | {"listing_id": 8, "capex": np.nan}]  # 8: a bank, no capex
    ).assign(date=pd.Timestamp("2020-06-30"), **{"known_at:shares": pd.Timestamp("2020-06-30")})
    fund = fund.set_index(["date", "listing_id"])
    close = pd.DataFrame({7: [10.0], 8: [10.0]}, index=pd.to_datetime(["2020-06-30"]))
    factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    r = ratio_panel(fund, close, factor)
    ev = 10.0 * 100 + 500 - 100
    assert r.loc[(pd.Timestamp("2020-06-30"), 7), "FCF/EV"] == pytest.approx((1600 - 400) / ev)
    assert r["FCF/EV"].isna().tolist() == [False, True]  # 8 reports no capex: no free cash flow
    assert r.loc[(pd.Timestamp("2020-06-30"), 7), "Accruals/TA"] == pytest.approx((1300 - 1600) / 3000)
    # a vendor without a cash-flow statement leaves both missing instead of failing
    plain = ratio_panel(fund.drop(columns=["operating_cash_flow", "capex"]), close, factor)
    assert plain[["FCF/EV", "Accruals/TA"]].isna().all().all()


def test_a_companys_value_is_the_sum_of_its_listings_own_values() -> None:
    listed = pd.DataFrame({1: [600.0], 2: [400.0], 3: [50.0], 4: [np.nan]}, index=DAYS[:1])
    values = company_values(listed, pd.Series({1: 10, 2: 10, 3: 11, 4: 12})).iloc[0]
    assert values[[1, 2, 3]].tolist() == [1000.0, 1000.0, 50.0]  # two share classes, one company
    assert np.isnan(values[4])  # no listing of the company has a count: left to its accounts


def test_a_listings_own_value_beats_a_restated_count_from_the_accounts() -> None:
    fund = pd.DataFrame(
        {
            "date": [pd.Timestamp("2015-06-30")] * 2,
            "listing_id": [7, 8],
            # a vendor's company count carried in today's shares: twenty times the 2015 count
            "shares": [13.7e9, 100.0],
            "known_at:shares": [pd.Timestamp("2015-04-23")] * 2,
            "revenue": [6.8e10, 1000.0],
            "equity": [1.2e11, 1000.0],
            "listed_value": [500.0 * 6.85e8, np.nan],  # 7 has its own count; 8 has none
        }
    ).set_index(["date", "listing_id"])
    close = pd.DataFrame({7: [500.0], 8: [10.0]}, index=pd.to_datetime(["2015-06-30"]))
    factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    values = market_values(fund, close, factor)["market_cap"].tolist()
    assert values == [500.0 * 6.85e8, 10.0 * 100.0]  # the listing's value; the accounts only where none


def test_a_count_from_a_year_before_is_in_the_later_dates_shares() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2020-09-01")])
    earlier = pd.DataFrame({7: [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2019-09-01")]))
    assert in_shares_of(earlier, dates, ACTIONS).iloc[0, 0] == pytest.approx(400.0)  # the 4-for-1 split since


def test_own_counts_come_from_the_first_price_vendor_with_one() -> None:
    counts = pd.DataFrame(
        {
            "listing_id": [7, 7, 8],
            "vendor": ["norgate", "refinitiv", "simfin"],
            "as_of_date": pd.to_datetime(["2020-06-30"] * 3),
            "shares": [101.0, 100.0, 5.0],
        }
    )
    out = own_counts(counts, ("refinitiv", "norgate"))
    assert out.set_index("listing_id")["shares"].to_dict() == {7: 100.0}  # simfin is not a price vendor here


def test_a_london_universe_is_in_pounds_without_its_trusts_and_pays_stamp_duty(tmp_path: Path) -> None:
    def table(name: str, frame: pd.DataFrame) -> None:
        (tmp_path / name).mkdir()
        frame.to_parquet(tmp_path / name / "part-1.parquet")

    table("ref.vendor", pd.DataFrame({"vendor_id": [1], "vendor_code": ["refinitiv"], "name": ["LSEG"]}))
    table(
        "ref.index_definition",
        pd.DataFrame({"index_id": [1, 2], "code": ["FTSE100", "SP500"], "currency": ["GBP", "USD"]}),
    )
    table(
        "ref.issuer",
        pd.DataFrame({"issuer_id": [1, 2, 3, 4], "incorporation_country": ["GB", "JE", "GB", "IE"], "name": "x"}),
    )
    table(
        "ref.listing",
        pd.DataFrame(
            {
                "listing_id": [10, 20, 30, 40, 50],
                "issuer_id": [1, 2, 3, 4, 1],
                "instrument_type": ["equity", "equity", "fund", "equity", "equity"],
                "exchange_mic": ["XLON", "XLON", "XLON", "XLON", "XNYS"],
            }
        ),
    )
    cache = ResearchCache(tmp_path)
    assert universe_currency(cache, pd.Series([1])) == "GBP"
    with pytest.raises(CurrencyError):
        universe_currency(cache, pd.Series([1, 2]))  # one universe, one currency
    assert operating_companies(cache, pd.Series([10, 20, 30])) == {10, 20}  # the investment trust is left out
    duty = stamp_duty(cache, pd.Index([10, 20, 30, 40, 50]))
    # UK 0.5%, Ireland 1%; Jersey none; the same company's New York listing none
    assert duty.to_dict() == {10: 50.0, 30: 50.0, 40: 100.0}


def test_a_dividend_declared_abroad_is_read_in_the_universes_currency() -> None:
    rates = pd.DataFrame(
        {"GBP": [1.25, 1.30], "ZAR": [0.05, 0.055]}, index=pd.to_datetime(["2024-03-13", "2024-03-15"])
    ).assign(USD=1.0)  # dollars per unit
    actions = pd.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "action_type": ["cash_dividend", "cash_dividend", "cash_dividend", "split", "cash_dividend"],
            "ex_date": pd.to_datetime(["2024-03-14", "2024-03-15", "2024-03-15", "2024-03-15", "2024-03-15"]),
            "amount": [0.41, 10.0, 0.30, None, 0.2],
            "ratio": [None, None, None, 2.0, None],
            "currency": ["USD", "ZAR", "GBP", None, "JPY"],
        }
    )
    out = dividends_in(actions, rates, "GBP").set_index("listing_id")
    assert out.loc[1, "amount"] == pytest.approx(0.41 / 1.25)  # the day before's rate: none on the ex-date
    assert out.loc[2, "amount"] == pytest.approx(10.0 * 0.055 / 1.30)
    assert out.loc[3, "amount"] == 0.30 and out.loc[4, "ratio"] == 2.0  # already in pounds; a split untouched
    assert 5 not in out.index and "currency" not in out.columns  # no rate for yen: left out, not read as pounds


def test_bars_quoted_in_another_currency_are_read_in_the_universes() -> None:
    rates = pd.DataFrame({"GBP": [1.25], "USD": [1.0]}, index=pd.to_datetime(["2026-01-02"]))
    bars = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-02"]),
            "close": [10460.0, 138.45, 5.0],
            "currency": ["GBX", "USD", "JPY"],
        }
    )
    out = in_currency(bars, rates, "GBP")
    assert out["close"].iloc[0] == pytest.approx(104.6) and out["close"].iloc[1] == pytest.approx(138.45 / 1.25)
    assert out["currency"].tolist() == ["GBP", "GBP", "JPY"]  # no rate for yen: left to check_currency to name
