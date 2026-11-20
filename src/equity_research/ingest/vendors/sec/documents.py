"""A company's EDGAR submissions index, normalised: the filings it lists and when each became public.

The index (``data.sec.gov/submissions/CIK##########.json``) lists a company's latest 1,000 or so filings under
``filings.recent`` and points to older pages (``filings.files``); an older page holds the same columns at its top
level. Each filing has its accession number, form, filing date, the instant EDGAR accepted it (UTC: the moment it
became public, which a filing date alone would place up to a day early), the period or event it reports, an 8-K's
item codes and its primary document's file name.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from equity_research.data.schema import DOCUMENT_FORMS

#: Original annual reports - the documents whose text is measured (an amendment often restates a part only).
ANNUAL_FORMS = ("10-K", "10-K405", "10-KT")


def older_pages(payload: dict[str, Any]) -> list[str]:
    """File names of the older pages a company's index points to."""
    return [str(f["name"]) for f in payload.get("filings", {}).get("files", []) if f.get("name")]


def _columns(payload: dict[str, Any]) -> dict[str, Any]:
    """The filing columns of a main index (under ``filings.recent``) or of an older page (at the top)."""
    return dict(payload["filings"]["recent"]) if "filings" in payload else dict(payload)


def filings(payload: dict[str, Any], cik: int, *, since: str | pd.Timestamp = "2003-01-01") -> pd.DataFrame:
    """The reports and current reports of one index page filed on or after ``since``.

    Columns: ``filing_ref`` (accession number), ``cik``, ``form``, ``filing_date``, ``report_date`` (missing when not
    given), ``published_at`` (acceptance, UTC, naive), ``items`` (an 8-K's codes), ``primary_document``.
    """
    cols = _columns(payload)
    n = len(cols.get("accessionNumber", []))
    if n == 0:
        return pd.DataFrame(
            columns=[
                "filing_ref",
                "cik",
                "form",
                "filing_date",
                "report_date",
                "published_at",
                "items",
                "primary_document",
            ]
        )

    def col(name: str) -> list[Any]:
        values = cols.get(name)
        return list(values) if values is not None and len(values) == n else [None] * n

    frame = pd.DataFrame(
        {
            "filing_ref": col("accessionNumber"),
            "cik": int(cik),
            "form": col("form"),
            "filing_date": pd.to_datetime(pd.Series(col("filingDate")), errors="coerce"),
            "report_date": pd.to_datetime(pd.Series(col("reportDate")).replace("", None), errors="coerce"),
            "published_at": pd.to_datetime(
                pd.Series(col("acceptanceDateTime")), errors="coerce", utc=True, format="ISO8601"
            ),
            "items": pd.Series(col("items")).replace("", None),
            "primary_document": pd.Series(col("primaryDocument")).replace("", None),
        }
    )
    frame["published_at"] = frame["published_at"].dt.tz_convert(None)
    # an acceptance instant missing (rare, very old filings) is the end of the filing day: never earlier than known
    fallback = frame["filing_date"] + pd.to_timedelta(23 * 60 + 59, unit="min")
    frame["published_at"] = frame["published_at"].fillna(fallback)
    keep = frame["form"].isin(DOCUMENT_FORMS) & frame["filing_date"].notna()
    keep &= frame["filing_date"] >= pd.Timestamp(since)
    frame = frame[keep].copy()
    frame["filing_date"] = frame["filing_date"].dt.date
    frame["report_date"] = frame["report_date"].dt.date.where(frame["report_date"].notna(), None)
    frame["items"] = (
        frame["items"].where(frame["items"].notna(), None).map(lambda v: None if v is None else str(v)[:96])
    )
    return frame.reset_index(drop=True)


def document_url(cik: int, filing_ref: str, primary_document: str) -> str:
    """Where EDGAR keeps a filing's primary document."""
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{filing_ref.replace('-', '')}/{primary_document}"
