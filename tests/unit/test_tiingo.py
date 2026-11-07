import io
import json
import zipfile

import pandas as pd

from equity_research.ingest.vendors.tiingo import normalize as norm
from equity_research.pipelines.load_tiingo import OVERLAP_DAYS, plan


def _body() -> bytes:
    """Apple around its 2020 split, as Tiingo answers: real prices, the dividend and the split on their days."""
    days = ["2020-08-06", "2020-08-07", "2020-08-28", "2020-08-31"]
    close = [455.61, 444.45, 499.23, 129.04]
    return json.dumps(
        [
            {
                "date": f"{d}T00:00:00.000Z",
                "open": c,
                "high": c + 1,
                "low": c - 1,
                "close": c,
                "volume": 1000.0,
                "adjClose": c / 4,
                "divCash": 0.82 if d == "2020-08-07" else 0.0,
                "splitFactor": 4.0 if d == "2020-08-31" else 1.0,
            }
            for d, c in zip(days, close, strict=True)
        ]
    ).encode()


def test_tickers_are_written_with_a_hyphen_for_the_share_class() -> None:
    assert norm.tiingo_ticker("BRK.B") == "BRK-B"
    assert norm.tiingo_ticker("bf/b") == "BF-B"
    assert norm.tiingo_ticker("AAPL") == "AAPL"


def test_bars_are_the_real_prices_of_each_day() -> None:
    bars = norm.bars(norm.frame(_body()))
    assert bars["close"].tolist() == [455.61, 444.45, 499.23, 129.04]
    assert bars["trade_date"].iloc[0] == pd.Timestamp("2020-08-06")
    assert str(bars["volume"].dtype) == "Int64"


def test_the_dividend_and_the_split_are_read_on_their_days() -> None:
    acts = norm.actions(norm.frame(_body())).set_index("action_type")
    assert acts.loc["cash_dividend", "ex_date"] == pd.Timestamp("2020-08-07")
    assert acts.loc["cash_dividend", "amount"] == 0.82
    assert acts.loc["split", "ex_date"] == pd.Timestamp("2020-08-31")
    assert acts.loc["split", "ratio"] == 4.0


def test_an_empty_answer_gives_no_rows() -> None:
    assert norm.bars(norm.frame(b"[]")).empty
    assert norm.actions(norm.frame(b"[]")).empty


def test_supported_tickers_keep_the_latest_holder_of_a_ticker() -> None:
    csv = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        "abc,NYSE,Stock,USD,1990-01-02,2001-05-31\n"
        "abc,NASDAQ,Stock,USD,2015-03-02,2026-10-23\n"
        "aapl,NASDAQ,Stock,USD,1980-12-12,2026-10-23\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("supported_tickers.csv", csv)
    out = norm.supported(buffer.getvalue()).set_index("ticker")
    assert sorted(out.index) == ["AAPL", "ABC"]
    assert out.loc["ABC", "first_day"] == pd.Timestamp("2015-03-02")


def test_closes_agreeing_with_another_vendor_are_the_same_stock() -> None:
    days = pd.bdate_range("2024-01-01", periods=10)
    ours = pd.DataFrame({"trade_date": days, "close": 100.0})
    assert norm.same_stock(ours, ours.assign(close=100.5)) is True
    assert norm.same_stock(ours, ours.assign(close=30.0)) is False  # another company under the ticker
    assert norm.same_stock(ours, ours.iloc[:3]) is None  # too few common days to tell


def test_the_plan_continues_from_the_newest_bar_and_skips_what_tiingo_lacks() -> None:
    today = pd.Timestamp("2026-10-23")  # a Friday
    listings = pd.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "ticker": ["NEW", "HALF", "DONE", "GONE", "OLD"],
            "first": pd.to_datetime(["2005-01-01"] * 5),
            "last": [today, today, today, today, pd.Timestamp("2001-05-31")],
            "newest": pd.to_datetime(pd.Series([None, "2020-06-30", "2026-10-22", None, None])),
        }
    )
    supported = pd.DataFrame(
        {
            "ticker": ["NEW", "HALF", "DONE", "OLD"],
            "first_day": pd.to_datetime(["1999-01-04", "1999-01-04", "1999-01-04", "2015-03-02"]),
            "last_day": [today] * 4,
        }
    )
    todo, skipped = plan(listings, supported)
    assert todo["ticker"].tolist() == ["NEW", "HALF"]
    assert todo.set_index("ticker").loc["HALF", "start"] == pd.Timestamp("2020-06-30") - pd.Timedelta(days=OVERLAP_DAYS)
    assert skipped == {"complete": ["DONE"], "not_at_tiingo": ["GONE"], "outside_tiingo_dates": ["OLD"]}
