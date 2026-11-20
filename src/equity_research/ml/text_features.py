"""Features from filing text, as known on each decision date: the tone and change of annual reports, 8-K events.

    measures = read_measures(cache)                     # one row per annual report, its text measures as columns
    per_report = report_features(measures)              # shares of each category, tone, change from the year before
    panels = as_of(per_report, dates, listings)         # feature -> date x listing_id, the latest report known
    events = event_panels(read_documents(cache), dates, listings)   # 8-K items over the last quarter
    gap = event_gap(events["ev_acquisition"], monthly, members)      # members with the event against the rest

Steps:

1. **Measures** - the counts of ``equity.text_measure`` (Loughran-McDonald categories per section, similarity to the
   previous report, FinBERT's tone of the MD&A) pivoted to one row per report, with its publication instant.
2. **Report features** - per report: each category's share of the words (Loughran and McDonald's proportions);
   net tone ((positive - negative) / words); the change of the negative share and of the length (log words) since
   the company's previous annual report; the similarities ("Lazy Prices": Cohen, Malloy and Nguyen find that firms
   whose reports change most do worse); FinBERT's net tone (positive - negative probability).
3. **As of a date** - each company's latest report published before the decision date (strictly: a report accepted
   on the day itself is not used), and only within :data:`MAX_REPORT_AGE_DAYS` - a company that stopped filing has
   no text. A company's features go to each of its listings.
4. **Events** - the number of 8-Ks reporting each item of :data:`EVENT_ITEMS` in the :data:`EVENT_WINDOW_DAYS` before
   the decision date, and of all 8-Ks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_research.data.cache import ResearchCache

#: A report older than this (15 months) is no longer the company's latest: it stopped filing, or its filing is late.
MAX_REPORT_AGE_DAYS = 456
EVENT_WINDOW_DAYS = 91
#: 8-K items that announce news (the 2004 numbering), by the feature name they become.
EVENT_ITEMS = {
    "ev_officer_change": "5.02",  # a director or officer leaves or is appointed
    "ev_agreement": "1.01",  # a material agreement is entered
    "ev_agreement_ended": "1.02",
    "ev_acquisition": "2.01",  # an acquisition or disposal completed
    "ev_restructuring": "2.05",  # costs of exit or disposal
    "ev_impairment": "2.06",
    "ev_auditor_change": "4.01",
    "ev_restatement": "4.02",  # earlier statements can no longer be relied on
    "ev_delisting_notice": "3.01",
    "ev_results": "2.02",  # results announced
}
CATEGORIES = ("negative", "positive", "uncertainty", "litigious", "constraining", "weak_modal", "strong_modal")


def read_measures(cache: ResearchCache) -> pd.DataFrame:
    """One row per annual report (``issuer_id``, ``filing_ref``, ``published_at``), a column per ``section:measure``."""
    rows = cache.query(
        "SELECT m.issuer_id, m.filing_ref, m.section, m.measure, m.value, d.published_at "
        "FROM equity.text_measure m JOIN equity.document d "
        "ON d.issuer_id = m.issuer_id AND d.vendor_id = m.vendor_id AND d.filing_ref = m.filing_ref"
    )
    return pivot_measures(rows)


def pivot_measures(rows: pd.DataFrame) -> pd.DataFrame:
    """``issuer_id, filing_ref, published_at, section, measure, value`` rows to one row per report."""
    rows = rows.assign(name=rows["section"] + ":" + rows["measure"])
    wide = rows.pivot_table(
        index=["issuer_id", "filing_ref", "published_at"], columns="name", values="value", aggfunc="last"
    )
    wide.columns.name = None
    out = wide.reset_index()
    out["published_at"] = pd.to_datetime(out["published_at"])
    return out.sort_values(["issuer_id", "published_at"]).reset_index(drop=True)


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)


def report_features(measures: pd.DataFrame) -> pd.DataFrame:
    """Per report: shares, tone, changes since the company's previous report, similarities, FinBERT's tone."""
    m = measures
    out = m[["issuer_id", "filing_ref", "published_at"]].copy()
    words = _col(m, "all:words").where(lambda w: w > 0)
    for c in CATEGORIES:
        out[f"txt_{c}"] = _col(m, f"all:{c}") / words
    out["txt_net_tone"] = (_col(m, "all:positive") - _col(m, "all:negative")) / words
    out["txt_log_words"] = np.log(words)
    for section in ("mdna", "risk"):
        sec_words = _col(m, f"{section}:words").where(lambda w: w > 0)
        out[f"txt_{section}_negative"] = _col(m, f"{section}:negative") / sec_words
        out[f"txt_{section}_uncertainty"] = _col(m, f"{section}:uncertainty") / sec_words
    out["txt_risk_log_words"] = np.log(_col(m, "risk:words").where(lambda w: w > 0))
    for section in ("all", "mdna", "risk"):
        out[f"txt_sim_{section}"] = _col(m, f"{section}:sim_cosine")
    out["txt_jaccard_all"] = _col(m, "all:sim_jaccard")
    out["txt_finbert_tone"] = _col(m, "mdna:finbert_positive") - _col(m, "mdna:finbert_negative")
    # the change since the company's previous report (one row back: rows are sorted by company and publication)
    previous = out.groupby("issuer_id")[["txt_negative", "txt_log_words", "txt_risk_log_words"]].shift(1)
    gap = out.groupby("issuer_id")["published_at"].diff().dt.days
    recent = gap <= MAX_REPORT_AGE_DAYS
    out["txt_d_negative"] = (out["txt_negative"] - previous["txt_negative"]).where(recent)
    out["txt_d_log_words"] = (out["txt_log_words"] - previous["txt_log_words"]).where(recent)
    out["txt_d_risk_log_words"] = (out["txt_risk_log_words"] - previous["txt_risk_log_words"]).where(recent)
    return out


def as_of(
    per_report: pd.DataFrame,
    dates: pd.DatetimeIndex,
    listings: pd.DataFrame,
    *,
    max_age_days: int = MAX_REPORT_AGE_DAYS,
) -> dict[str, pd.DataFrame]:
    """``feature -> date x listing_id``: each company's latest report published before each date, if recent."""
    features = [c for c in per_report.columns if c.startswith("txt_")]
    reports = per_report.sort_values("published_at")
    grid = pd.DataFrame({"date": pd.DatetimeIndex(dates)}).sort_values("date")
    parts = []
    for issuer_id, g in reports.groupby("issuer_id"):
        hit = pd.merge_asof(
            grid,
            g[["published_at", *features]],
            left_on="date",
            right_on="published_at",
            direction="backward",
            allow_exact_matches=False,
        )
        fresh = (hit["date"] - hit["published_at"]).dt.days <= max_age_days
        hit.loc[~fresh, features] = np.nan
        parts.append(hit.assign(issuer_id=int(str(issuer_id))))
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "issuer_id", *features])
    return _to_listings(long, features, dates, listings)


def _to_listings(
    long: pd.DataFrame, features: list[str], dates: pd.DatetimeIndex, listings: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Spread ``date, issuer_id, features`` rows to each of the company's listings: ``feature -> date x listing``."""
    pairs = listings[["listing_id", "issuer_id"]].drop_duplicates().astype(int)
    merged = long.merge(pairs, on="issuer_id", how="inner")
    columns = sorted(pairs["listing_id"].unique())
    return {
        f: merged.pivot_table(index="date", columns="listing_id", values=f, aggfunc="last").reindex(
            index=dates, columns=columns
        )
        for f in features
    }


def read_documents(cache: ResearchCache) -> pd.DataFrame:
    """Every 8-K: ``issuer_id``, ``published_at``, ``items``."""
    rows = cache.query("SELECT issuer_id, published_at, items FROM equity.document WHERE form IN ('8-K', '8-K/A')")
    rows["published_at"] = pd.to_datetime(rows["published_at"])
    return rows


def event_panels(
    documents: pd.DataFrame,
    dates: pd.DatetimeIndex,
    listings: pd.DataFrame,
    *,
    window_days: int = EVENT_WINDOW_DAYS,
) -> dict[str, pd.DataFrame]:
    """``feature -> date x listing_id``: 8-Ks reporting each item (and all 8-Ks) in the window before each date.

    A company with no 8-K in the window counts 0 - an absence of news is information, not a gap - but only from
    the first 8-K it ever filed (before that, its filings are not in the index).
    """
    docs = documents.dropna(subset=["published_at"]).copy()
    items = docs["items"].fillna("").astype(str).str.split(",")
    flags = pd.DataFrame(
        {name: items.map(lambda xs, code=code: code in [x.strip() for x in xs]) for name, code in EVENT_ITEMS.items()},
        index=docs.index,
    ).astype(float)
    flags["ev_8k_count"] = 1.0
    flags["issuer_id"] = docs["issuer_id"].astype(int)
    flags["published_at"] = docs["published_at"]
    names = [*EVENT_ITEMS, "ev_8k_count"]
    rows = []
    first = flags.groupby("issuer_id")["published_at"].min()
    published = flags["published_at"].to_numpy(dtype="datetime64[ns]")
    span = np.timedelta64(window_days, "D")
    for date in pd.DatetimeIndex(dates):
        now = date.to_datetime64().astype("datetime64[ns]")
        window = flags[(published < now) & (published >= now - span)]
        counts = window.groupby("issuer_id")[names].sum()
        known = first[first < date].index
        counts = counts.reindex(known, fill_value=0.0)
        rows.append(counts.assign(date=date).reset_index())
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "issuer_id", *names])
    return _to_listings(long, names, dates, listings)


def event_gap(feature: pd.DataFrame, returns: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Each date's mean next return of the members with the event (``feature > 0``) less that of the members without.

    An event count is zero for most members, so it cannot be cut into fifths; this is its long-short spread with
    equal weights. Dates with no member on either side are left out.
    """
    held = members.reindex(index=feature.index, columns=feature.columns).fillna(False).astype(bool)
    r = returns.reindex(index=feature.index, columns=feature.columns).where(held)
    gap = r.where(feature > 0).mean(axis=1) - r.where(feature == 0).mean(axis=1)
    return gap.dropna().rename("event gap")
