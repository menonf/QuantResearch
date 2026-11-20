from pathlib import Path

import pytest

from equity_research.ingest.corrections import CorrectionError, distribution_ratio, read_corrections

HEADER = "ric,action_type,ex_date,ratio,distributed_ric,per_share,note\n"


def test_the_repositorys_corrections_are_well_formed() -> None:
    table = read_corrections()
    assert {"GOOGL.OQ", "CHRW.OQ"} <= set(table["ric"])
    assert (table.loc[table["action_type"] == "split", "ratio"] > 0).all()


@pytest.mark.parametrize(
    ("ric", "before", "after"),
    [
        ("CCL.L^E26", 4.60, 15.47),  # P&O Princess -> Carnival plc, 22 April 2003
        ("SHEL.L", 36.49, 17.665),  # Royal Dutch -> Royal Dutch Shell A, 20 July 2005
        ("RDSb.L^A22", 5.28, 18.065),  # Shell Transport -> Royal Dutch Shell B, 20 July 2005
    ],
)
def test_a_london_share_exchange_leaves_an_ordinary_days_return(ric: str, before: float, after: float) -> None:
    # LSEG carries the old company's price on the new line unadjusted: without the exchange ratio the day books
    # +236 %, -52 % and +242 %; with it, the day's move is the market's (within a few percent)
    table = read_corrections()
    ratio = float(table.loc[(table["ric"] == ric) & (table["action_type"] == "split"), "ratio"].iloc[0])
    assert abs(after / (before / ratio) - 1) < 0.05


def test_a_distribution_is_worth_the_parent_and_the_line_handed_out() -> None:
    # Alphabet, 3 April 2014: class A closed at 571.50, the class C handed out (one per share) at 569.74
    ratio = distribution_ratio(571.50, 569.74, 1.0)
    assert ratio == pytest.approx(1.99692, abs=1e-5)
    assert 571.50 * ratio / 1135.10 - 1 == pytest.approx(0.0054, abs=1e-4)  # the day's return: +0.5 %, not -50 %
    with pytest.raises(CorrectionError):
        distribution_ratio(571.50, float("nan"), 1.0)


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        ("X.N,merger,2020-01-02,,,,a note", "unknown action"),
        ("X.N,split,2020-01-02,,,,a note", "positive ratio"),
        ("X.N,spinoff,2020-01-02,,,1,a note", "names the line handed out"),
        ("X.N,split,2020-01-02,2,,,", "says what happened"),
    ],
)
def test_a_badly_stated_correction_is_refused(tmp_path: Path, row: str, problem: str) -> None:
    path = tmp_path / "corporate_actions.csv"
    path.write_text(HEADER + row + "\n", encoding="utf-8")
    with pytest.raises(CorrectionError, match=problem):
        read_corrections(path)
