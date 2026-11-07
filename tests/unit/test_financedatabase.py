import pandas as pd

from equity_research.ingest.countries import country_code
from equity_research.ingest.vendors.financedatabase.normalize import ticker, us_equities


def missing(value: object) -> bool:
    """A missing value (NaN, NaT, None) - typed ``object`` so the checker does not narrow what a frame holds."""
    return bool(pd.isna(value))  # type: ignore[call-overload]


def _raw() -> pd.DataFrame:
    rows = {
        "AAPL": [
            "Apple Inc.",
            "USD",
            "Information Technology",
            "Hardware",
            "Computers",
            "XNAS",
            "United States",
            "US0378331005",
            "037833100",
            "BBG000B9Y5X2",
        ],
        "BRK-B": [
            "Berkshire Hathaway Inc.",
            "USD",
            "Financials",
            "Insurance",
            "Insurance",
            "XNYS",
            "United States",
            "US0846707026",
            "084670702",
            "BBG000DWG505",
        ],
        "LIN": [
            "Linde plc",
            "USD",
            "Materials",
            "Basic Materials",
            "Chemicals",
            "XNAS",
            "United Kingdom",
            "IE000S9YS762",
            "",
            "BBG01FND0CC1",
        ],
        "SAP.DE": [
            "SAP SE",
            "EUR",
            "Information Technology",
            "Software",
            "Software",
            "XETR",
            "Germany",
            "DE0007164600",
            None,
            None,
        ],
    }
    cols = ["name", "currency", "sector", "industry_group", "industry", "mic", "country", "isin", "cusip", "figi"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols).rename_axis("symbol")


def test_us_listings_with_our_codes_and_country_codes() -> None:
    out = us_equities(_raw()).set_index("symbol")
    assert list(out.index) == ["AAPL", "BRK-B", "LIN"]  # SAP in Frankfurt is not a US listing
    assert out.loc["BRK-B", "ticker"] == "BRK.B" and out.loc["LIN", "domicile"] == "GB"
    assert missing(out.loc["LIN", "CUSIP"])  # a blank code is no code
    assert out.loc["AAPL", "ISIN"] == "US0378331005" and out.loc["AAPL", "FIGI"] == "BBG000B9Y5X2"


def test_country_names_become_iso_codes() -> None:
    assert country_code("United States") == "US" and country_code("Turkey") == "TR" and country_code("Macau") == "MO"
    assert country_code("Atlantis") is None and country_code(None) is None
    assert ticker("BF-B") == "BF.B"
