"""Text features as known on decision dates: no database."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_research.research import text_features as tf


def _missing(value: object) -> bool:
    """Whether a cell is missing (``pd.isna``, typed for any cell)."""
    return bool(pd.isna(value))  # type: ignore[call-overload]


def _rows() -> pd.DataFrame:
    rows = []
    for issuer, ref, when, words, neg, pos, sim in (
        (1, "a1", "2019-02-20 21:00", 10_000, 100, 50, None),
        (1, "a2", "2020-02-20 21:00", 12_000, 240, 60, 0.9),
        (2, "b1", "2016-03-01 12:00", 5_000, 25, 25, None),
    ):
        base = {"issuer_id": issuer, "filing_ref": ref, "published_at": pd.Timestamp(when), "section": "all"}
        rows += [base | {"measure": "words", "value": words}, base | {"measure": "negative", "value": neg}]
        rows.append(base | {"measure": "positive", "value": pos})
        if sim is not None:
            rows.append(base | {"measure": "sim_cosine", "value": sim})
    rows.append({"issuer_id": 1, "filing_ref": "a2", "published_at": pd.Timestamp("2020-02-20 21:00"),
                 "section": "mdna", "measure": "finbert_positive", "value": 0.3})  # fmt: skip
    rows.append({"issuer_id": 1, "filing_ref": "a2", "published_at": pd.Timestamp("2020-02-20 21:00"),
                 "section": "mdna", "measure": "finbert_negative", "value": 0.1})  # fmt: skip
    return pd.DataFrame(rows)


LISTINGS = pd.DataFrame({"listing_id": [10, 11, 20], "issuer_id": [1, 1, 2]})  # company 1 has two share lines


def test_report_features_shares_tone_and_change_since_the_previous_report() -> None:
    per = tf.report_features(tf.pivot_measures(_rows())).set_index("filing_ref")
    assert per.loc["a1", "txt_negative"] == pytest.approx(0.01)
    assert per.loc["a2", "txt_negative"] == pytest.approx(0.02)
    assert per.loc["a2", "txt_net_tone"] == pytest.approx((60 - 240) / 12_000)
    assert per.loc["a2", "txt_d_negative"] == pytest.approx(0.01)
    assert per.loc["a2", "txt_d_log_words"] == pytest.approx(np.log(1.2))
    assert _missing(per.loc["a1", "txt_d_negative"])  # nothing before it
    assert per.loc["a2", "txt_sim_all"] == 0.9 and _missing(per.loc["a1", "txt_sim_all"])
    assert per.loc["a2", "txt_finbert_tone"] == pytest.approx(0.2)
    assert _missing(per.loc["b1", "txt_mdna_negative"])  # no MD&A measured: missing, not zero


def test_as_of_takes_the_latest_report_published_strictly_before_and_recent() -> None:
    per = tf.report_features(tf.pivot_measures(_rows()))
    dates = pd.DatetimeIndex(["2019-01-31", "2020-02-20", "2020-02-21", "2021-06-30", "2016-12-30", "2017-12-29"])
    panels = tf.as_of(per, dates, LISTINGS)
    neg = panels["txt_negative"]
    assert [int(c) for c in neg.columns] == [10, 11, 20]
    assert _missing(neg.loc["2019-01-31", 10])  # no report yet
    assert neg.loc["2020-02-20", 10] == pytest.approx(0.01)  # the 2020 report came that evening: not yet known
    assert neg.loc["2020-02-21", 10] == pytest.approx(0.02) and neg.loc["2020-02-21", 11] == pytest.approx(0.02)
    assert _missing(neg.loc["2021-06-30", 10])  # 16 months on: the company stopped filing
    assert neg.loc["2016-12-30", 20] == pytest.approx(0.005)
    assert _missing(neg.loc["2017-12-29", 20])


def test_event_panels_count_items_in_the_quarter_before() -> None:
    docs = pd.DataFrame(
        {
            "issuer_id": [1, 1, 1, 2],
            "published_at": pd.to_datetime(
                ["2020-01-15 12:00", "2020-03-01 12:00", "2020-03-31 22:00", "2019-01-10 12:00"]
            ),
            "items": ["5.02,9.01", "2.02,9.01", "5.02", "2.02"],
        }
    )
    dates = pd.DatetimeIndex(["2019-12-31", "2020-03-31", "2020-06-30"])
    panels = tf.event_panels(docs, dates, LISTINGS)
    officer, count = panels["ev_officer_change"], panels["ev_8k_count"]
    assert officer.loc["2020-03-31", 10] == 1.0  # the 5.02 of the evening of the 31st is not yet known
    assert count.loc["2020-03-31", 10] == 2.0
    assert officer.loc["2020-06-30", 10] == 1.0 and count.loc["2020-06-30", 10] == 1.0
    assert _missing(count.loc["2019-12-31", 10])  # before its first 8-K, a company is not in the index yet
    assert count.loc["2019-12-31", 20] == 0.0  # known, and quiet that quarter
    assert panels["ev_results"].loc["2020-03-31", 11] == 1.0  # every share line of the company


def test_event_gap_is_the_members_with_the_event_less_the_rest() -> None:
    dates = pd.DatetimeIndex(["2020-01-31", "2020-02-28"])
    feature = pd.DataFrame({1: [1.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 2.0], 4: [0.0, float("nan")]}, index=dates)
    returns = pd.DataFrame({1: [0.01, 0.02], 2: [0.03, 0.04], 3: [0.05, -0.02], 4: [0.50, 0.50]}, index=dates)
    members = pd.DataFrame({1: [True, True], 2: [True, True], 3: [True, True], 4: [False, True]}, index=dates)
    gap = tf.event_gap(feature, returns, members)
    assert gap.loc["2020-01-31"] == pytest.approx(0.01 - 0.04)  # member 4 is not in the index in January
    assert gap.loc["2020-02-28"] == pytest.approx(-0.02 - 0.03)  # no feature for member 4: on neither side
