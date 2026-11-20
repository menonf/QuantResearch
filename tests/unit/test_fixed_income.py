import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bond_files import BOND, SECOND, write_file
from fixed_income_research.analytics import replication as rp
from fixed_income_research.ingest import fred, ice


def test_the_file_name_gives_the_index_and_the_month_end() -> None:
    assert ice.parse_name("H0A0-01312019.xlsx") == ("H0A0", dt.date(2019, 1, 31))
    assert ice.parse_name("H0A0_20100228_H0A0-02282010.xlsx") == ("H0A0", dt.date(2010, 2, 28))
    with pytest.raises(ice.LayoutError, match="not a month end"):
        ice.parse_name("H0A0-01302019.xlsx")
    with pytest.raises(ice.LayoutError, match="expected"):
        ice.parse_name("notes.xlsx")


def test_a_cusip_gets_its_check_digit() -> None:
    assert ice.cusip9("013822AC") == "013822AC5"  # as inside its ISIN, US 013822AC5 4
    assert ice.cusip9("03783310") == "037833100"


def test_a_file_reads_the_same_with_or_without_its_title(tmp_path: Path) -> None:
    titled = tmp_path / "titled"
    titled.mkdir()
    with_title = write_file(titled / "H0A0-01312019.xlsx", [BOND, SECOND], "H0A0 for 01/31/2019  ( February Universe )")
    bare = write_file(tmp_path / "H0A0-01312019.xlsx", [BOND, SECOND])
    a, b = ice.read_constituents(with_title), ice.read_constituents(bare)
    assert list(a["ISIN number"]) == list(b["ISIN number"]) == ["US013822AC54", "US013822AA98"]  # no small print
    assert ice.parse_title("H0A0 for 01/31/2019  ( February Universe )") == ("H0A0", dt.date(2019, 1, 31), "February")

    wrong = write_file(tmp_path / "H0A0-02282019.xlsx", [BOND], "H0A0 for 01/31/2019  ( February Universe )")
    with pytest.raises(ice.LayoutError, match="its title says"):
        ice.read_constituents(wrong)
    short = tmp_path / "H0A0-03312019.xlsx"
    pd.DataFrame([["Cusip", "ISIN number"], ["013822AC", "US013822AC54"]]).to_excel(short, header=False, index=False)
    with pytest.raises(ice.LayoutError, match="columns missing"):
        ice.read_constituents(short)


def test_an_issuer_is_the_ticker_and_takes_the_industry_of_its_largest_weight(tmp_path: Path) -> None:
    other = SECOND | {"Industry Lvl 4 Desc": "Steel Producers/Products", "Description": "Alcoa Corp."}
    month = ice.normalize(ice.read_constituents(write_file(tmp_path / "H0A0-01312019.xlsx", [BOND, other])))
    assert list(month.bonds["cusip"]) == ["013822AC5", "013822AA9"]
    assert month.bonds.loc[0, "maturity_date"] == dt.date(2028, 5, 15)
    assert month.bonds[["weight_pct", "total_return_mtd_pct", "oas_bp"]].iloc[0].tolist() == [60.0, 6.226, 312.0]
    assert "sub_industry" not in month.bonds.columns  # the industry levels are the issuer's
    issuer = month.issuers.set_index("ticker").loc["AA"]
    assert (issuer["name"], issuer["sub_industry"], issuer["country"]) == (
        "Alcoa Nederland Holding B.V.",
        "Metals/Mining Excluding Steel",
        "US",
    )
    assert month.issuers_disagreeing == 1

    with pytest.raises(ice.LayoutError, match="listed twice"):
        ice.normalize(ice.read_constituents(write_file(tmp_path / "H0A0-02282019.xlsx", [BOND, BOND])))


def test_only_finished_months_have_an_official_level() -> None:
    payload = (
        b"observation_date,BAMLHYH0A0HYM2TRIV\n2026-08-28,100.0\n2026-08-31,101.0\n2026-09-29,.\n2026-09-30,103.0\n"
        b"2026-10-20,104.0\n2026-11-12,105.0\n"
    )
    levels = fred.month_end_levels(payload, dt.date(2026, 11, 15))
    # October's last level is eleven days before its end; November is not finished
    assert levels["as_of_date"].tolist() == [dt.date(2026, 8, 31), dt.date(2026, 9, 30)]
    assert levels["total_return_level"].tolist() == [101.0, 103.0]


def test_the_rebuilt_return_says_how_much_of_the_index_it_covers() -> None:
    months = pd.to_datetime(["2019-01-31", "2019-02-28", "2019-03-31", "2019-05-31"])
    weights = pd.DataFrame(
        {1: [0.6, 0.5, 0.5, 0.5], 2: [0.4, np.nan, np.nan, np.nan], 3: [np.nan, 0.5, 0.5, 0.5]}, months
    )
    returns = pd.DataFrame(
        {1: [0.0, 0.02, 0.01, 0.03], 2: [0.0] + [np.nan] * 3, 3: [np.nan, np.nan, 0.03, 0.01]}, months
    )
    rebuilt = rp.index_returns(weights, returns)
    assert list(rebuilt.index) == list(months[1:3])  # January has no month before it; May's (April) has no file
    feb = rebuilt.loc["2019-02-28"]  # bond 2 left at the February rebalance: 40 % of the index has no return
    assert (feb["return"], feb["covered"], feb["leavers"], feb["bonds"]) == pytest.approx((0.02, 0.6, 0.4, 2))
    assert rebuilt.loc["2019-03-31", "return"] == pytest.approx(0.02) and rebuilt.loc["2019-03-31", "covered"] == 1.0

    levels = pd.Series([100.0, 102.0, 105.0], index=months[[0, 1, 3]])
    assert rp.official_returns(levels).to_dict() == {months[1]: pytest.approx(0.02)}


def test_an_index_described_by_its_weights() -> None:
    months = pd.to_datetime(["2019-01-31", "2019-02-28"])
    weights = pd.DataFrame({1: [0.6, 0.5], 2: [0.4, 0.5]}, months)
    oas = pd.DataFrame({1: [300.0, 200.0], 2: [np.nan, 400.0]}, months)
    assert rp.weighted_average(weights, oas).tolist() == pytest.approx([300.0, 300.0])  # over the bonds that have it
    rating = pd.DataFrame({1: ["BB1", "BB1"], 2: ["B2", "BB1"]}, months)
    by_rating = rp.weight_by_group(weights, rating)
    assert by_rating.loc["2019-01-31"].to_dict() == {"B2": 0.4, "BB1": 0.6}
    assert by_rating.loc["2019-02-28", "BB1"] == 1.0
