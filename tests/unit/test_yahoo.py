import datetime as dt
import json

import pandas as pd
import pytest

from equity_research.ingest.vendors.yahoo.normalize import parse_chart


def _stamp(day: str) -> int:
    """The session open in New York (14:30 UTC in winter), as Yahoo stamps a daily bar."""
    return int(dt.datetime.fromisoformat(f"{day}T14:30:00+00:00").timestamp())


def _body() -> bytes:
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    return json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {"currency": "USD", "exchangeTimezoneName": "America/New_York"},
                        "timestamp": [_stamp(d) for d in days],
                        "events": {
                            "splits": {"s": {"date": _stamp("2024-01-04"), "numerator": 4.0, "denominator": 1.0}},
                            "dividends": {"d": {"date": _stamp("2024-01-03"), "amount": 0.25}},
                        },
                        "indicators": {
                            "quote": [
                                {
                                    "open": [25.0, 25.5, 26.0, None],
                                    "high": [26.0, 26.5, 27.0, None],
                                    "low": [24.5, 25.0, 25.5, None],
                                    "close": [25.5, 26.0, 26.5, None],  # the 5th: no session (a holiday print)
                                    "volume": [400, 800, 100, None],
                                }
                            ]
                        },
                    }
                ],
            }
        }
    ).encode()


def test_prices_volumes_and_dividends_go_back_to_their_own_days_shares() -> None:
    bars, actions = parse_chart(_body())
    assert bars["trade_date"].tolist() == [pd.Timestamp(d) for d in ("2024-01-02", "2024-01-03", "2024-01-04")]
    assert bars["close"].tolist() == [102.0, 104.0, 26.5]  # before the 4-for-1 split: four times Yahoo's figure
    assert bars["volume"].tolist() == [100, 200, 100]
    split = actions[actions["action_type"] == "split"].iloc[0]
    assert split["ex_date"] == pd.Timestamp("2024-01-04") and split["ratio"] == 4.0
    dividend = actions[actions["action_type"] == "cash_dividend"].iloc[0]
    assert dividend["ex_date"] == pd.Timestamp("2024-01-03") and dividend["amount"] == pytest.approx(1.0)


def test_a_yahoo_error_is_raised() -> None:
    with pytest.raises(ValueError, match="Not Found"):
        parse_chart(json.dumps({"chart": {"result": None, "error": {"code": "Not Found"}}}))
