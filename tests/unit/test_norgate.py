import pandas as pd
import pytest

from equity_research.ingest.vendors.norgate import normalize as norm

DAYS = pd.to_datetime(["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"])


def _prices() -> pd.DataFrame:
    """Split-adjusted bars with the real close beside them: a 4-for-1 split on the 31st, a dividend on the 28th."""
    return pd.DataFrame(
        {
            "Open": [125.0, 126.0, 127.0, 132.0],
            "High": [126.0, 127.0, 131.0, 134.0],
            "Low": [124.0, 125.0, 126.0, 130.0],
            "Close": [125.0, 125.5, 129.0, 134.0],
            "Volume": [400.0, 600.0, 900.0, 800.0],
            "Unadjusted Close": [500.0, 502.0, 129.0, 134.0],
            "Dividend": [0.0, 0.205, 0.0, 0.0],
        },
        index=pd.DatetimeIndex(DAYS, name="Date"),
    )


def test_symbols_name_the_ticker_and_the_delisting_month() -> None:
    assert norm.parse_symbol("LEH-200809") == ("LEH", pd.Timestamp("2008-09-01"))
    assert norm.parse_symbol("BRK.B") == ("BRK.B", None)
    assert norm.parse_symbol("BF-B") == ("BF-B", None)  # a hyphen alone is not a delisting


def test_bars_are_put_back_in_each_days_shares() -> None:
    out = norm.bars(_prices())
    assert out["close"].tolist() == [500.0, 502.0, 129.0, 134.0]
    assert out["open"].tolist() == [500.0, 504.0, 127.0, 132.0]  # adjusted x 4 before the split
    assert out["volume"].tolist() == [100, 150, 900, 800]  # adjusted / 4 before the split


def test_the_split_and_the_dividend_are_read_from_the_prices() -> None:
    out = norm.actions(_prices()).set_index("action_type")
    assert out.loc["split", "ex_date"] == pd.Timestamp("2020-08-31") and out.loc["split", "ratio"] == pytest.approx(4.0)
    assert out.loc["cash_dividend", "amount"] == pytest.approx(0.82)  # 0.205 in today's shares = 0.82 then
    assert bool(out["ratio"].isna()["cash_dividend"])


def test_a_small_capital_change_is_a_spinoff_not_a_split() -> None:
    prices = _prices().assign(**{"Unadjusted Close": [131.25, 131.775, 129.0, 134.0]})  # factor 1.05 -> 1.0
    out = norm.actions(prices)
    assert out.loc[out["action_type"] == "spinoff", "ratio"].tolist() == [pytest.approx(1.05)]


def test_membership_runs_become_intervals() -> None:
    days = pd.bdate_range("2024-01-01", periods=8)
    series = pd.DataFrame({"Index Constituent": [0, 1, 1, 0, 0, 1, 1, 1]}, index=days)
    out = norm.memberships(series)
    assert out["start_date"].tolist() == [days[1], days[5]]
    assert out["end_date"].iloc[0] == days[3] and pd.isna(out["end_date"].iloc[1])  # still a member at the end


def test_a_missing_price_column_is_named() -> None:
    with pytest.raises(KeyError, match="Unadjusted Close"):
        norm.bars(_prices().drop(columns="Unadjusted Close"))


def test_an_index_level_is_its_own_real_close() -> None:
    level = _prices().drop(columns=["Unadjusted Close", "Dividend"])
    out = norm.bars(norm.as_level(level))
    assert out["close"].tolist() == [125.0, 125.5, 129.0, 134.0]
    assert norm.actions(norm.as_level(level)).empty
