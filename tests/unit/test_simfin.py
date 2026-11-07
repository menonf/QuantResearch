import io
import zipfile
from pathlib import Path

import pandas as pd

from equity_research.ingest.vendors.simfin import normalize as norm
from equity_research.ingest.vendors.simfin.client import DATASETS


def _income() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "EUR"],
            "SimFinId": [11, 11, 22],
            "Currency": ["USD", "USD", "EUR"],
            "Fiscal Year": [2024, 2024, 2024],
            "Fiscal Period": ["Q1", "Q2", "Q1"],
            "Report Date": ["2024-03-31", "2024-06-30", "2024-03-31"],
            "Publish Date": ["2024-05-02", "2024-08-01", "2024-05-10"],
            "Restated Date": ["2024-05-02", "2025-08-01", "2024-05-10"],
            "Revenue": [1000.0, 1100.0, 50.0],
            "Cost of Revenue": [-600.0, -650.0, -30.0],
            "Net Income": [100.0, None, 5.0],
            "Shares (Basic)": [10.0, 10.0, 1.0],
        }
    )


def test_flows_are_quarters_with_costs_positive() -> None:
    facts = norm.facts(_income(), norm.INCOME_ITEMS)
    q1 = facts[facts["period_end"] == pd.Timestamp("2024-03-31")].set_index("item")
    assert q1.loc["revenue", "value"] == 1000.0
    assert q1.loc["cost_of_revenue", "value"] == 600.0  # SimFin writes it -600
    assert q1.loc["revenue", "period_type"] == "Q"
    assert q1.loc["revenue", "period_start"] == pd.Timestamp("2024-01-01")
    assert q1.loc["shares_weighted_basic", "unit"] == "shares"
    assert q1.loc["revenue", "unit"] == "USD"


def test_a_figure_is_known_when_last_restated() -> None:
    facts = norm.facts(_income(), norm.INCOME_ITEMS)
    q2 = facts[(facts["period_end"] == pd.Timestamp("2024-06-30")) & (facts["item"] == "revenue")]
    assert q2["published_at"].tolist() == [pd.Timestamp("2025-08-01")]  # never earlier than SimFin had it
    assert "net_income" not in set(facts.loc[facts["period_end"] == pd.Timestamp("2024-06-30"), "item"])


def test_only_dollar_statements_are_kept_and_the_rest_counted() -> None:
    facts = norm.facts(_income(), norm.INCOME_ITEMS)
    assert set(facts["simfin_id"]) == {"11"}
    assert norm.currencies(_income()) == {"USD": 2, "EUR": 1}


def test_balances_are_instants_and_missing_columns_are_skipped() -> None:
    balance = pd.DataFrame(
        {
            "SimFinId": [11],
            "Report Date": ["2024-03-31"],
            "Publish Date": ["2024-05-02"],
            "Total Assets": [5000.0],
            "Total Deposits": [10.0],  # a bank's line: not one of ours
        }
    )
    facts = norm.facts(balance, norm.BALANCE_ITEMS)
    assert facts[["item", "period_type", "value"]].to_numpy().tolist() == [["total_assets", "I", 5000.0]]
    assert facts["period_start"].isna().all()
    assert facts["published_at"].tolist() == [pd.Timestamp("2024-05-02")]  # no restated date: first published


def test_companies_codes_with_and_without_cik() -> None:
    with_cik = norm.companies(
        pd.DataFrame({"SimFinId": [11.0], "Ticker": ["AAA"], "CIK": [320193.0], "ISIN": [" US1 "]})
    )
    assert with_cik.loc[0, "simfin_id"] == "11"
    assert with_cik.loc[0, "cik"] == "320193"
    assert with_cik.loc[0, "isin"] == "US1"
    without = norm.companies(pd.DataFrame({"SimFinId": [11], "Ticker": ["AAA"], "IndustryId": [101001]}))
    assert without[["cik", "isin"]].isna().all(axis=None)
    assert without.loc[0, "industry_id"] == 101001


def test_a_landed_zip_is_read_as_semicolon_separated(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("us-companies.csv", "Ticker;SimFinId;Company Name\nAAA;11;Triple A Inc\n")
    path = tmp_path / "companies.zip"
    path.write_bytes(buffer.getvalue())
    assert norm.read_table(path).to_dict(orient="records") == [
        {"Ticker": "AAA", "SimFinId": 11, "Company Name": "Triple A Inc"}
    ]


def test_each_dataset_asks_only_for_what_it_has() -> None:
    assert DATASETS["industries"].params == {"dataset": "industries"}
    assert DATASETS["income_quarterly"].params == {"dataset": "income", "variant": "quarterly", "market": "us"}
    assert DATASETS["companies"].params == {"dataset": "companies", "market": "us"}


def test_a_figure_the_sec_first_reported_the_same_is_dated_by_its_first_publication() -> None:
    simfin = pd.DataFrame(
        {
            "issuer_id": [1, 1, 1],
            "item": ["revenue", "net_income", "total_assets"],
            "period_type": ["Q", "Q", "I"],
            "period_end": pd.to_datetime(["2024-06-30"] * 3),
            "value": [1000.0, 100.0, 5000.0],
            "published_at": pd.to_datetime(["2025-08-01"] * 3),  # re-dated a year later by SimFin
            "first_published": pd.to_datetime(["2024-08-02"] * 3),
        }
    )
    sec = pd.DataFrame(
        {
            "issuer_id": [1, 1],
            "item": ["revenue", "net_income"],
            "period_type": ["Q", "Q"],
            "period_end": pd.to_datetime(["2024-06-29", "2024-06-29"]),  # Apple-style week ending
            "value": [1003.0, 90.0],  # revenue within 0.5%; net income restated since
            "published_at": pd.to_datetime(["2024-08-03", "2024-08-03"]),
        }
    )
    out = norm.verified_dates(simfin, sec).set_index("item")
    assert out.loc["revenue", "published_at"] == pd.Timestamp("2024-08-03")  # the later of the two first reports
    assert out.loc["revenue", "verified"] is True
    assert out.loc["net_income", "published_at"] == pd.Timestamp("2025-08-01")
    assert out.loc["net_income", "verified"] is False
    assert out.loc["total_assets", "published_at"] == pd.Timestamp("2025-08-01")  # no SEC figure: stays late
    assert pd.isna(out.loc["total_assets", "verified"])
