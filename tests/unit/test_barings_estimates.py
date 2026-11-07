from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from equity_research.research import barings_estimates as be

DAY = pd.Timestamp("2015-06-30")
DATES = pd.DatetimeIndex([DAY])
SPLIT = pd.Timestamp("2015-06-15")  # a 2-for-1 split of listing 1 in the middle of June


Row = tuple[int, str, str, str, str, float, float, int]


def _consensus(rows: Sequence[Row]) -> pd.DataFrame:
    """``(listing, measure, period, as_of, fiscal end, mean, std, estimates)`` rows as stored."""
    cols = ["listing_id", "measure", "period", "as_of_date", "fiscal_period_end", "mean", "std_dev", "estimates"]
    out = pd.DataFrame(rows, columns=cols)
    return out.assign(
        as_of_date=pd.to_datetime(out["as_of_date"]), fiscal_period_end=pd.to_datetime(out["fiscal_period_end"])
    )


def _inputs(
    consensus: pd.DataFrame,
    revisions: pd.DataFrame | None = None,
    actuals: pd.DataFrame | None = None,
) -> be.EstimateInputs:
    days = pd.bdate_range("2014-12-01", "2015-07-31")
    basis = pd.DataFrame({1: np.where(days < SPLIT, 2.0, 1.0), 2: 1.0}, index=days)  # splits after each day
    panel = pd.DataFrame({1: [50.0], 2: [20.0]}, index=DATES)
    return be.EstimateInputs(
        dates=DATES,
        consensus=consensus,
        revisions=revisions
        if revisions is not None
        else pd.DataFrame(columns=["listing_id", "measure", "period", "as_of_date", "up", "down"]),
        actuals=actuals
        if actuals is not None
        else pd.DataFrame(columns=["listing_id", "measure", "period_end", "value", "announced_at"]),
        basis=basis,
        close=panel,
        value=panel * 1000,
        shares=panel * 0 + 10.0,
        total_assets=panel * 0 + 500.0,
    )


def test_next_twelve_months_weighs_fy1_and_fy2_by_the_year_after_the_date() -> None:
    rows = [
        (2, "eps", "FY1", "2015-06-30", "2015-12-31", 2.0, 0.1, 5),
        (2, "eps", "FY2", "2015-06-30", "2016-12-31", 3.0, 0.1, 5),
    ]
    ntm = be.Estimates(_inputs(_consensus(rows))).ntm("eps").loc[DAY, 2]
    # a year on is 2016-06-30: 184 of FY2's 366 days still to come in FY1's place, 182 into FY2
    assert ntm == pytest.approx(184 / 366 * 2.0 + 182 / 366 * 3.0)


def test_a_snapshot_before_a_split_is_put_into_the_shares_of_the_date() -> None:
    rows = [
        (1, "eps", "FY1", "2015-05-31", "2015-12-31", 4.0, 0.2, 5),  # quoted before the 2-for-1 split
        (1, "eps", "FY1", "2015-06-30", "2015-12-31", 2.0, 0.1, 5),
        (1, "revenue", "FY1", "2015-05-31", "2015-12-31", 100.0, 1.0, 5),
    ]
    e = be.Estimates(_inputs(_consensus(rows)))
    assert e.get("eps", "FY1", "mean", 1).loc[DAY, 1] == pytest.approx(2.0)  # 4.0 then is 2.0 in June's shares
    assert e.get("eps", "FY1", "mean").loc[DAY, 1] == pytest.approx(2.0)
    assert e.get("revenue", "FY1", "mean", 1).loc[DAY, 1] == 100.0  # a total is not per share


def test_three_months_ago_is_the_same_fiscal_year_under_its_label_then() -> None:
    rows = [
        (2, "eps", "FY1", "2015-03-31", "2015-03-31", 1.0, 0.1, 5),  # the year then ending
        (2, "eps", "FY2", "2015-03-31", "2016-03-31", 1.8, 0.1, 5),  # this year's FY1, three months ago
        (2, "eps", "FY1", "2015-06-30", "2016-03-31", 2.0, 0.1, 5),
    ]
    e = be.Estimates(_inputs(_consensus(rows)))
    assert e.same_year_mean("eps", "FY1", 3).loc[DAY, 2] == pytest.approx(1.8)


def test_an_actual_counts_only_once_announced_and_in_the_dates_shares() -> None:
    actuals = pd.DataFrame(
        {
            "listing_id": [1, 1],
            "measure": ["eps", "eps"],
            "period_end": pd.to_datetime(["2014-12-31", "2015-03-31"]),
            "value": [3.0, 1.0],
            "announced_at": pd.to_datetime(["2015-02-10", "2015-07-15"]),  # the second after the date
        }
    )
    value, end = be.Estimates(_inputs(_consensus([]), actuals=actuals)).actual("eps")
    assert value.loc[DAY, 1] == pytest.approx(1.5)  # FY2014's 3.0, quoted before the split
    assert end.loc[DAY, 1] == (pd.Timestamp("2014-12-31") - pd.Timestamp(0)).days


def test_fewer_than_two_estimates_leave_a_factor_missing_and_breadth_is_bounded() -> None:
    rows = [
        (1, "eps", "FY1", "2015-06-30", "2015-12-31", 2.0, 0.1, 1),  # one estimate: gated
        (1, "eps", "FY2", "2015-06-30", "2016-12-31", 2.5, 0.1, 1),
        (2, "eps", "FY1", "2015-06-30", "2015-12-31", 2.0, 0.4, 4),
        (2, "eps", "FY2", "2015-06-30", "2016-12-31", 2.5, 0.1, 4),
    ]
    revisions = pd.DataFrame(
        {
            "listing_id": [2, 2],
            "measure": "eps",
            "period": ["FY1", "FY2"],
            "as_of_date": pd.to_datetime(["2015-06-30", "2015-06-30"]),
            "up": [4, 3],
            "down": [0, 0],
        }
    )
    f = be.eps_factors(be.Estimates(_inputs(_consensus(rows), revisions=revisions)))
    assert np.isnan(f["eps_uncertainty"].to_numpy(dtype=float)[0, 0]) and f["eps_uncertainty"].loc[
        DAY, 2
    ] == pytest.approx(0.2)
    assert f["eps_growth_fy1_to_fy2"].loc[DAY, 2] == pytest.approx(0.25)
    assert f["earnings_yield_fy1"].loc[DAY, 2] == pytest.approx(0.1)  # 2.0 over a 20.0 close
    assert f["earnings_revisions_1_mth"].loc[DAY, 2] == pytest.approx(0.5 * 4 / 4 + 0.5 * 3 / 4)
    assert np.isnan(
        f["earnings_revision_persistence_6mth"].to_numpy(dtype=float)[0, 1]
    )  # earlier months unknown: Spark's null


def test_the_revisions_composite_needs_six_months_of_estimates() -> None:
    rows = [
        (lid, "eps", p, str(end.date()), "2015-12-31", 2.0, 0.1, 4)
        for lid in (1, 2)
        for p in ("FY1", "FY2")
        for end in be.month_ends(DATES, 0).append([be.month_ends(DATES, k) for k in range(1, 6)])
        if not (lid == 1 and end == be.month_ends(DATES, 5)[0])  # listing 1 lacks the sixth month
    ]
    members = pd.DataFrame({1: [True], 2: [True]}, index=DATES)
    out = be.revisions_composite(_inputs(_consensus(rows)), members)
    assert np.isnan(out.to_numpy(dtype=float)[0, 0])
