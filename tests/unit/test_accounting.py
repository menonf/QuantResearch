import numpy as np
import pandas as pd

from equity_research.research.accounting import (
    accounting_items,
    as_of,
    first_reported,
    implausible_share_counts,
    quarters,
    trailing_twelve_months,
)


def _f(item: str, ptype: str, start: str | None, end: str, value: float, published: str) -> dict[str, object]:
    return {
        "issuer_id": 1,
        "item": item,
        "period_type": ptype,
        "period_start": pd.Timestamp(start) if start else pd.NaT,
        "period_end": pd.Timestamp(end),
        "value": value,
        "published_at": pd.Timestamp(published),
    }


def _facts() -> pd.DataFrame:
    rows = [
        # 2021: Q1 and Q3 as 3-month figures, Q2 only as YTD, Q4 only inside the 10-K
        _f("revenue", "Q", "2021-01-01", "2021-03-31", 100, "2021-05-01"),
        _f("revenue", "YTD", "2021-01-01", "2021-06-30", 210, "2021-08-01"),
        _f("revenue", "Q", "2021-07-01", "2021-09-30", 120, "2021-11-01"),
        _f("revenue", "YTD", "2021-01-01", "2021-09-30", 330, "2021-11-01"),
        _f("revenue", "FY", "2021-01-01", "2021-12-31", 450, "2022-02-20"),
        _f("revenue", "FY", "2021-01-01", "2021-12-31", 999, "2023-02-20"),  # restated later: never used
        _f("revenue", "Q", "2022-01-01", "2022-03-31", 130, "2022-05-01"),
        _f("operating_income", "FY", "2021-01-01", "2021-12-31", 50, "2022-02-20"),
        _f("pretax_income", "FY", "2021-01-01", "2021-12-31", 40, "2022-02-20"),
        _f("interest_expense", "FY", "2021-01-01", "2021-12-31", -5, "2022-02-20"),
        _f("total_assets", "I", None, "2021-12-31", 1000, "2022-02-20"),
        _f("equity_parent", "I", None, "2021-12-31", 400, "2022-02-20"),
        _f("equity_total", "I", None, "2021-12-31", 450, "2022-02-20"),
        _f("lt_debt_noncurrent", "I", None, "2021-12-31", 200, "2022-02-20"),
        _f("debt_current", "I", None, "2021-12-31", 30, "2022-02-20"),
        _f("notes_payable", "I", None, "2021-12-31", 999, "2022-02-20"),  # not used: the non-current path exists
    ]
    return pd.DataFrame(rows)


def test_quarters_are_derived_from_year_to_date_figures() -> None:
    facts = _facts()
    q = quarters(facts[facts["item"] == "revenue"]).set_index("period_end")["value"]
    assert q.loc["2021-06-30"] == 110  # 6M - Q1
    assert q.loc["2021-12-31"] == 120  # FY - 9M (the first-reported FY, not the restatement)
    assert q.loc["2022-03-31"] == 130


def test_trailing_twelve_months_and_when_it_became_known() -> None:
    facts = _facts()
    ttm = trailing_twelve_months(facts[facts["item"] == "revenue"]).set_index("period_end")
    assert ttm.loc["2021-12-31", "value"] == 450  # the reported year
    assert ttm.loc["2022-03-31", "value"] == 110 + 120 + 120 + 130
    assert ttm.loc["2022-03-31", "known_at"] == pd.Timestamp("2022-05-01")  # when its latest part was published
    assert pd.Timestamp("2021-09-30") not in ttm.index  # only three quarters by then


def test_a_half_year_reporters_trailing_year() -> None:
    # Tesco-like: fiscal years to late February, first halves to late August, no quarters
    facts = pd.DataFrame(
        [
            _f("revenue", "FY", "2015-03-01", "2016-02-27", 54_433, "2016-04-13"),
            _f("revenue", "YTD", "2015-03-01", "2015-08-29", 27_227, "2015-10-07"),
            _f("revenue", "YTD", "2016-02-28", "2016-08-27", 27_338, "2016-10-05"),
            _f("revenue", "FY", "2016-02-28", "2017-02-25", 55_917, "2017-04-12"),
        ]
    )
    ttm = trailing_twelve_months(facts).set_index("period_end")
    assert ttm.loc["2016-08-27", "value"] == 54_433 + 27_338 - 27_227
    assert ttm.loc["2016-08-27", "known_at"] == pd.Timestamp("2016-10-05")
    assert ttm.loc["2017-02-25", "value"] == 55_917  # the reported year
    assert pd.Timestamp("2015-08-29") not in ttm.index  # no year before it to build on


def test_derived_items_follow_their_fallbacks() -> None:
    items = accounting_items(_facts()).xs((1, pd.Timestamp("2021-12-31")))
    assert items["ebit"] == 50  # operating income first
    assert items["equity"] == 400  # attributable to the parent first
    assert items["total_debt"] == 230  # non-current + current; the instruments path is not added on top
    facts = _facts()
    no_operating = accounting_items(facts[facts["item"] != "operating_income"]).xs((1, pd.Timestamp("2021-12-31")))
    assert no_operating["ebit"] == 45  # pretax 40 + interest 5


def test_as_of_sees_only_what_was_published_before_the_date() -> None:
    items = accounting_items(_facts())
    panel = as_of(items, pd.to_datetime(["2022-02-20", "2022-02-21", "2022-05-02", "2023-06-01"]))
    revenue = panel["revenue"].xs(1, level="issuer_id")
    assert pd.isna(revenue.get(pd.Timestamp("2022-02-20"), np.nan))  # the 10-K's own day: not yet visible
    assert revenue.loc["2022-02-21"] == 450  # the day after the 10-K
    assert revenue.loc["2022-05-02"] == 480  # the TTM to Q1 2022, published 2022-05-01
    assert pd.Timestamp("2023-06-01") not in revenue.dropna().index  # Q1 2022 is over 400 days old by then


def test_a_quarter_reported_again_with_another_start_counts_once() -> None:
    rows = [
        _f("shares_weighted_basic", "Q", "2021-07-01", "2021-09-30", 50, "2021-11-01"),
        _f("shares_weighted_basic", "Q", "2021-07-02", "2021-09-30", 51, "2022-11-01"),  # next year's comparative
        _f("total_assets", "I", None, "2021-09-30", 1000, "2021-11-01"),
    ]
    items = accounting_items(pd.DataFrame(rows))
    assert items.loc[(1, pd.Timestamp("2021-09-30")), "shares"] == 50
    assert items.loc[(1, pd.Timestamp("2021-09-30")), "known_at:shares"] == pd.Timestamp("2021-11-01")


def test_the_cover_page_share_count_is_the_freshest() -> None:
    rows = [
        _f("shares_outstanding", "I", None, "2021-12-31", 100, "2022-02-20"),
        _f("shares_cover", "I", None, "2022-02-10", 98, "2022-02-20"),  # after a buyback, on the 10-K's cover
    ]
    items = accounting_items(pd.DataFrame(rows))
    known = as_of(items, [pd.Timestamp("2022-03-01")], with_period_end=True)
    assert known["shares"].tolist() == [98]
    assert known["period_end:shares"].tolist() == [pd.Timestamp("2022-02-10")]
    assert known["known_at:shares"].tolist() == [pd.Timestamp("2022-02-20")]  # in the shares of the filing day


def test_a_share_count_off_by_a_scale_factor_is_dropped_but_a_split_is_kept() -> None:
    counts = [76e6, 76.1e6, 76.2e6, 76.4e12, 76.5e6, 76.6e6, 76.6, 76.7e6]  # a "trillions" and a "millions" typo
    counts += [767e6, 768e6, 769e6, 770e6]  # then a 10-for-1 split: real
    rows = [
        _f(
            "shares_cover",
            "I",
            None,
            str(pd.Timestamp("2015-01-31") + pd.DateOffset(months=3 * i))[:10],
            v,
            "2030-01-01",
        )
        for i, v in enumerate(counts)
    ]
    first = first_reported(pd.DataFrame(rows))
    bad = implausible_share_counts(first)
    assert sorted(first.loc[bad, "value"].tolist()) == [76.6, 76.4e12]


def test_the_accounts_do_not_depend_on_the_order_the_facts_arrive_in() -> None:
    facts = _facts()
    # several facts published together (one filing): their order decides what neighbouring-row checks see
    facts = pd.concat(
        [
            facts,
            pd.DataFrame(
                [
                    _f("shares_cover", "I", None, "2022-01-31", 50, "2022-02-20"),
                    _f("shares_outstanding", "I", None, "2021-12-31", 49, "2022-02-20"),
                    _f("net_income", "FY", "2021-01-01", "2021-12-31", 30, "2022-02-20"),
                ]
            ),
        ],
        ignore_index=True,
    )
    shuffled = facts.sample(frac=1.0, random_state=1).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_reported(facts), first_reported(shuffled))
    pd.testing.assert_frame_equal(accounting_items(facts), accounting_items(shuffled))


def test_an_earnings_surprise_is_a_beat_against_the_companys_own_pattern() -> None:
    from equity_research.research.accounting import earnings_surprise  # noqa: PLC0415

    ends = pd.to_datetime(
        # a 52-53 week year: the quarter ends drift a few days from one year to the next
        [
            *("2018-03-31", "2018-06-30", "2018-09-29", "2018-12-29", "2019-03-30", "2019-06-29", "2019-09-28"),
            *("2019-12-28", "2020-03-28", "2020-06-27", "2020-09-26", "2020-12-26", "2021-03-27"),
        ]
    )
    season = [10.0, 20.0, 15.0, 30.0]  # the same quarter is alike every year
    noise = np.random.default_rng(2).normal(0, 0.5, len(ends))
    values = [season[i % 4] + i * 0.5 + noise[i] for i in range(len(ends))]  # steady growth: no surprise
    values[-1] += 20.0  # the last quarter beats last year's by far more than usual
    q = pd.DataFrame(
        {
            "issuer_id": 1,
            "item": "net_income",
            "period_end": ends,
            "value": values,
            "known_at": ends + pd.Timedelta(days=40),
        }
    )
    s = earnings_surprise(q).set_index("period_end")["value"]
    assert s.index.min() == ends[8]  # the first four year-on-year changes only build the expectation
    assert s.iloc[-1] > 2.0  # the beat stands out
    assert s.iloc[:-1].abs().max() < s.iloc[-1] / 2
    assert earnings_surprise(q.iloc[:0]).empty


def test_cash_flow_quarters_are_derived_from_year_to_date_figures() -> None:
    cash = [
        _f("operating_cash_flow", "Q", "2021-01-01", "2021-03-31", 30, "2021-05-01"),
        _f("operating_cash_flow", "YTD", "2021-01-01", "2021-06-30", 70, "2021-08-01"),
        _f("operating_cash_flow", "YTD", "2021-01-01", "2021-09-30", 110, "2021-11-01"),
        _f("operating_cash_flow", "FY", "2021-01-01", "2021-12-31", 160, "2022-02-20"),
        _f("operating_cash_flow", "Q", "2022-01-01", "2022-03-31", 45, "2022-05-01"),
        _f("capex", "FY", "2021-01-01", "2021-12-31", 40, "2022-02-20"),
    ]
    items = accounting_items(pd.DataFrame(cash))
    assert items.loc[(1, pd.Timestamp("2021-12-31")), "operating_cash_flow"] == 160  # the reported year
    # a year later by a quarter: 2021's last three quarters (40 + 40 + 50) and 2022's first (45)
    assert items.loc[(1, pd.Timestamp("2022-03-31")), "operating_cash_flow"] == 175
    assert items.loc[(1, pd.Timestamp("2021-12-31")), "capex"] == 40


def test_the_new_items_derive_unusual_expense_interest_income_and_risk_weighted_assets() -> None:
    rows = [
        _f("net_income", "FY", "2021-01-01", "2021-12-31", 90, "2022-02-20"),
        _f("restructuring_charges", "FY", "2021-01-01", "2021-12-31", 30, "2022-02-20"),
        _f("impairment_charges", "FY", "2021-01-01", "2021-12-31", 20, "2022-02-20"),
        _f("disposal_gains", "FY", "2021-01-01", "2021-12-31", 15, "2022-02-20"),
        _f("net_interest_income", "FY", "2021-01-01", "2021-12-31", 60, "2022-02-20"),
        _f("interest_expense", "FY", "2021-01-01", "2021-12-31", -25, "2022-02-20"),
        _f("tier1_capital", "I", None, "2021-12-31", 120, "2022-02-20"),
        _f("tier1_ratio", "I", None, "2021-12-31", 0.12, "2022-02-20"),
        _f("dividends_paid", "FY", "2021-01-01", "2021-12-31", 10, "2022-02-20"),
    ]
    items = accounting_items(pd.DataFrame(rows)).iloc[0]
    assert items["unusual_expense"] == 35  # charges less gains: the SEC has no one line
    assert items["interest_income"] == 85  # net interest income + interest expense, whatever its sign
    assert items["risk_weighted_assets"] == 1000  # tier-1 capital / the tier-1 ratio
    assert items["tier1_ratio"] == 0.12 and items["dividends_paid"] == 10
    no_gains = accounting_items(pd.DataFrame([r for r in rows if r["item"] != "disposal_gains"])).iloc[0]
    assert no_gains["unusual_expense"] == 50  # a gain nobody reported is none
