"""SEC filing indexes and the text measures of annual reports: pure functions, no network."""

from __future__ import annotations

import datetime as dt
import gzip
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from equity_research.ingest.vendors.sec import documents as docs
from equity_research.ingest.vendors.sec import text
from equity_research.pipelines import load_sec_text as pipe

INDEX: dict[str, Any] = {
    "cik": "320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000123", "0000320193-24-000100", "0000320193-24-000090", "x-1"],
            "filingDate": ["2024-11-01", "2024-10-31", "2024-08-02", "2002-01-01"],
            "reportDate": ["2024-09-28", "2024-10-31", "", "2001-09-30"],
            "acceptanceDateTime": [
                "2024-11-01T20:30:00.000Z",
                "2024-10-31T20:31:00.000Z",
                "",
                "2002-01-01T10:00:00.000Z",
            ],
            "form": ["10-K", "8-K", "4", "10-K"],
            "items": ["", "2.02,9.01", "", ""],
            "primaryDocument": ["aapl-20240928.htm", "a8k.htm", "form4.xml", "old.txt"],
        },
        "files": [{"name": "CIK0000320193-submissions-001.json", "filingCount": 3}],
    },
}

LM_CSV = (
    b"Word,Seq_num,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining\n"
    b"LOSS,1,2009,0,0,0,0,0,0\n"
    b"GAIN,2,0,2009,0,0,0,0,0\n"
    b"MAY,3,0,0,2009,0,0,2009,0\n"
    b"CRISIS,4,-2020,0,0,0,0,0,0\n"
)


def _report(mdna: str, risk: str, *, toc: bool = True) -> bytes:
    contents = (
        "<p>Item 1A. Risk Factors</p><p>Item 7. Management's Discussion and Analysis</p>"
        "<p>Item 8. Financial Statements</p>"
    )
    body = (
        "<html><head><style>p {color: red}</style></head><body>"
        '<div style="display:none"><ix:header><ix:hidden>LOSS LOSS LOSS</ix:hidden></ix:header></div>'
        + (contents if toc else "")
        + f"<p>Item 1A. Risk Factors</p><p>{risk}</p><p>Item 1B. Unresolved Staff Comments</p><p>None.</p>"
        + f"<p>ITEM 7 &#8212; MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS</p><p>{mdna}</p>"
        + "<p>Item 7A. Quantitative and Qualitative Disclosures</p><p>Rates.</p>"
        + "<p>Item 8. Financial Statements</p><table><tr><td>1,234</td></tr></table></body></html>"
    )
    return body.encode()


def test_filings_keeps_reports_since_a_date_with_acceptance_instants() -> None:
    rows = docs.filings(INDEX, 320193, since="2003-01-01")
    assert list(rows["form"]) == ["10-K", "8-K"]  # a Form 4 is not a report; 2002 is before ``since``
    first = rows.iloc[0]
    assert first["published_at"] == pd.Timestamp("2024-11-01 20:30:00")
    assert first["report_date"] == dt.date(2024, 9, 28) and first["filing_date"] == dt.date(2024, 11, 1)
    assert rows.iloc[1]["items"] == "2.02,9.01" and pd.isna(rows.iloc[0]["items"])
    assert docs.older_pages(INDEX) == ["CIK0000320193-submissions-001.json"]
    page = {k: v[:1] for k, v in INDEX["filings"]["recent"].items()}  # an older page: the columns at the top
    assert len(docs.filings(page, 320193)) == 1
    url = docs.document_url(320193, "0000320193-24-000123", "aapl-20240928.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"


def test_a_missing_acceptance_instant_is_the_end_of_the_filing_day() -> None:
    recent = {k: list(v) for k, v in INDEX["filings"]["recent"].items()}
    recent["form"][2] = "10-Q"
    rows = docs.filings({"filings": {"recent": recent}}, 320193)
    assert rows.loc[rows["form"] == "10-Q", "published_at"].iloc[0] == pd.Timestamp("2024-08-02 23:59:00")


def test_plain_text_drops_hidden_xbrl_styles_and_markup() -> None:
    plain = text.plain_text(_report("Revenue rose.", "Risks."))
    assert "LOSS" not in plain and "color" not in plain
    assert "ITEM 7 " + chr(0x2014) + " MANAGEMENT" + chr(0x2019) + "S DISCUSSION AND ANALYSIS" in plain.splitlines()


def test_sections_skip_the_table_of_contents() -> None:
    mdna = " ".join(["Revenue grew while costs may rise"] * 60)
    risk = " ".join(["A loss may occur"] * 100)
    counts = text.report_counts(_report(mdna, risk))
    assert sum(counts["mdna"].values()) == 360 and counts["mdna"]["REVENUE"] == 60
    assert sum(counts["risk"].values()) == 300  # "A" is one letter: not a word
    short = text.report_counts(_report("Brief.", risk))
    assert "mdna" not in short and "risk" in short  # a section under 250 words is a heading or a pointer


def test_dictionary_counts_and_removed_words() -> None:
    lm = text.lm_dictionary(LM_CSV)
    assert lm["Negative"] == frozenset({"LOSS"})  # CRISIS was removed (a negative year)
    counts = text.dictionary_counts(Counter({"LOSS": 3, "GAIN": 1, "MAY": 2, "CRISIS": 5, "THE": 9}), lm)
    assert counts["words"] == 20 and counts["negative"] == 3 and counts["positive"] == 1
    assert counts["uncertainty"] == 2 and counts["weak_modal"] == 2 and counts["litigious"] == 0


def test_similarity_of_identical_disjoint_and_partial_reports() -> None:
    a = Counter({"SALES": 2, "ROSE": 1})
    assert text.similarity(a, a) == {"sim_cosine": pytest.approx(1.0), "sim_jaccard": 1.0}
    assert text.similarity(a, Counter({"LOSS": 1})) == {"sim_cosine": 0.0, "sim_jaccard": 0.0}
    part = text.similarity(a, Counter({"SALES": 2, "FELL": 1}))
    assert part["sim_cosine"] == pytest.approx(4 / 5) and part["sim_jaccard"] == pytest.approx(1 / 3)
    assert math.isnan(text.similarity(Counter(), Counter())["sim_cosine"])


def test_measures_compare_with_the_previous_report_only_when_given() -> None:
    lm = text.lm_dictionary(LM_CSV)
    now = {"all": Counter({"LOSS": 1, "SALES": 3})}
    first = text.measures(now, None, lm)
    assert {m for _, m, _, _ in first} == {"words", *[c.lower() for c in text.CATEGORIES]}
    second = text.measures(now, {"all": Counter({"SALES": 3})}, lm)
    sims = {m: v for s, m, method, v in second if method == text.SIMILARITY_METHOD}
    assert set(sims) == {"sim_cosine", "sim_jaccard"} and sims["sim_jaccard"] == pytest.approx(0.5)


def test_sample_sentences_takes_prose_spread_through_the_text() -> None:
    prose = [f"Sentence number {i} explains how revenue changed this year." for i in range(100)]
    table = "Revenue 1,234 5,678 9,012 3,456 7,890 1,234 5,678 9,012."
    picked = text.sample_sentences(" ".join([*prose[:50], table, *prose[50:]]), 10)
    assert len(picked) == 10 and picked[0] == prose[0] and picked[5] == prose[50]
    assert table not in picked
    assert text.sample_sentences("Too short. Also short.", 5) == []


def test_company_measures_compares_each_report_with_the_one_a_year_before(tmp_path: Path) -> None:
    lm_path = tmp_path / "lm.csv"
    lm_path.write_bytes(LM_CSV)
    pipe._load_dictionary(str(lm_path))
    mdna = " ".join(["Revenue grew while costs may rise"] * 60)
    paths = []
    for n, risk_word in enumerate(("loss", "gain", "gain")):
        p = tmp_path / f"{n}.htm.gz"
        p.write_bytes(gzip.compress(_report(mdna, " ".join([f"A {risk_word} may occur"] * 100))))
        paths.append(str(p))
    reports = [
        ("r2", pd.Timestamp("2022-03-01"), paths[2], 3),  # out of order: sorted by publication
        ("r0", pd.Timestamp("2018-03-01"), paths[0], 1),
        ("r1", pd.Timestamp("2021-03-01"), paths[1], 2),
    ]
    rows = pd.DataFrame(
        pipe.company_measures(reports), columns=["filing_ref", "section", "measure", "method", "value", "batch_id"]
    )
    sim = rows[(rows["measure"] == "sim_cosine") & (rows["section"] == "risk")].set_index("filing_ref")["value"]
    assert list(sim.index) == ["r2"]  # r0 has no report before; r1 came three years after r0 - too long
    assert sim["r2"] == pytest.approx(1.0)
    neg = rows[(rows["measure"] == "negative") & (rows["section"] == "risk")].set_index("filing_ref")["value"]
    assert neg.to_dict() == {"r0": 100.0, "r1": 0.0, "r2": 0.0}
    assert set(rows.loc[rows["filing_ref"] == "r1", "batch_id"]) == {2}
