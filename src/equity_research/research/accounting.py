"""Vendor-neutral accounting: canonical financial facts -> point-in-time items for each decision date.

    items = accounting_items(facts)          # per (issuer, period end): items, each with when it became known
    panel = as_of(items, dates)              # what was known on each decision date

The same formulas apply to any vendor's canonical facts.

* **As first reported.** Every value is its first published version; a derived value - a quarter from year-to-date
  figures, a trailing twelve months, EBIT from its parts - is known when the *latest* of its parts was published. A
  restatement published later never reaches back in time.
* **Flows** are trailing twelve months: the fiscal-year figure at a fiscal-year end, else the sum of the last four
  quarters ending within :data:`TTM_MAX_SPAN_DAYS`. A quarter not reported as a 3-month figure is derived from
  year-to-date figures with the same start (6M - 3M, 9M - 6M, FY - 9M) or as FY minus its three quarters. A company
  reporting half-years has no quarters: at its first half, last year + the half - last year's first half.
* **Balances** are the instants at the period end.
* **Earnings surprise** is each quarter's net income against the same quarter a year before, scaled by how much
  those year-on-year changes usually vary (:func:`earnings_surprise`); known when the quarter is.
* **Point in time.** On a decision date an issuer's latest period whose value was known strictly before the date is
  used, unless that period ended more than :data:`MAX_AGE_DAYS` earlier (a company that stopped reporting).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TTM_MAX_SPAN_DAYS = 300
MAX_AGE_DAYS = 400
QUARTER_DAYS = (80, 100)

FLOW_ITEMS = (
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "pretax_income",
    "net_income",
    "interest_expense",
    "operating_cash_flow",
    "capex",
    "depreciation",
    "financing_cash_flow",
    "dividends_paid",
    "stock_issued",
    "stock_repurchased",
    "st_debt_change",
    "lt_debt_issued",
    "lt_debt_repaid",
    "discontinued_operations",
    "unusual_expense",
    "restructuring_charges",
    "impairment_charges",
    "disposal_gains",
    "net_interest_income",
    "noninterest_income",
    "noninterest_expense",
    "loan_loss_provision",
)
BALANCE_ITEMS = (
    "total_assets",
    "total_liabilities",
    "liabilities_and_equity",
    "equity_parent",
    "equity_total",
    "current_assets",
    "current_liabilities",
    "cash_and_st_investments",
    "cash",
    "st_investments",
    "ppe_net",
    "ppe_and_lease_assets",
    "retained_earnings",
    "lt_debt_noncurrent",
    "lt_debt_total",
    "debt_current",
    "lt_debt_current",
    "st_borrowings",
    "commercial_paper",
    "notes_payable",
    "secured_debt",
    "convertible_notes",
    "credit_lines",
    "other_borrowings",
    "shares_outstanding",
    "shares_cover",
    "debt_total",  # a vendor's reported total (LSEG); the SEC has no single tag for it
    "inventory",
    "loans_net",
    "deposits",
    "nonperforming_loans",
    "tier1_capital",
    "risk_weighted_assets",
    "tier1_ratio",  # pure numbers (0.125), not dollars
    "total_capital_ratio",
)
#: Items :func:`accounting_items` returns.
ITEMS = (
    "revenue",
    "gross_profit",
    "operating_income",
    "ebit",
    "net_income",
    "interest_expense",
    "operating_cash_flow",
    "capex",
    "total_assets",
    "total_liabilities",
    "equity",
    "current_assets",
    "current_liabilities",
    "cash",
    "total_debt",
    "current_debt",
    "fixed_assets",
    "retained_earnings",
    "shares_outstanding",
    "shares",
    "earnings_surprise",
    "depreciation",
    "inventory",
    "financing_cash_flow",
    "dividends_paid",
    "stock_issued",
    "stock_repurchased",
    "st_debt_change",
    "lt_debt_issued",
    "lt_debt_repaid",
    "discontinued_operations",
    "unusual_expense",
    "net_interest_income",
    "interest_income",
    "noninterest_income",
    "noninterest_expense",
    "loan_loss_provision",
    "loans_net",
    "deposits",
    "nonperforming_loans",
    "risk_weighted_assets",
    "tier1_ratio",
    "total_capital_ratio",
)
KEY = ["issuer_id", "period_end"]
DATE_COLUMNS = ("period_start", "period_end", "published_at")


def _ns(values: pd.Series | pd.Index) -> pd.Series:
    """Dates at nanosecond resolution: pandas may hold them in us or s, and mixed units break joins and arithmetic."""
    return pd.Series(pd.to_datetime(values)).astype("datetime64[ns]")


# ---------------------------------------------------------------- facts -> flows and balances


def first_reported(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (issuer, item, period): the earliest published version, in one fixed row order.

    The order is fixed whatever order the facts arrive in (a DuckDB scan returns rows in no set order): later steps
    look at neighbouring rows (the share-count check's rolling median), so an arbitrary order would make two loads of
    the same data give different accounts.
    """
    key = ["issuer_id", "item", "period_type", "period_start", "period_end"]
    facts = facts.assign(**{c: _ns(facts[c]).to_numpy() for c in DATE_COLUMNS})
    facts = facts.sort_values(["published_at", *key, "value"], kind="stable")  # every tie broken: one order
    return facts.drop_duplicates(key, keep="first").reset_index(drop=True)


SHARE_ITEMS = ("shares_cover", "shares_outstanding", "shares_weighted_basic")
#: A share count this many times above or below its issuer's nearby counts is a filing error, not a split: filers
#: sometimes tag a count in thousands or millions as if in units (76 trillion Whirlpool shares on one cover page).
#: Real splits stay far below it (10-for-1, 20-for-1).
SHARE_OUTLIER_FACTOR = 30.0
SHARE_NEIGHBOURS = 9


def implausible_share_counts(first: pd.DataFrame) -> pd.Series:
    """True for share-count facts far from the median of the issuer's nearby share counts (see above)."""
    shares = first[first["item"].isin(SHARE_ITEMS) & (first["value"] > 0)].sort_values(["issuer_id", "period_end"])
    local = shares.groupby("issuer_id")["value"].transform(
        lambda v: v.rolling(SHARE_NEIGHBOURS, center=True, min_periods=1).median()
    )
    ratio = shares["value"] / local
    bad = (ratio > SHARE_OUTLIER_FACTOR) | (ratio < 1 / SHARE_OUTLIER_FACTOR)
    out = pd.Series(False, index=first.index)
    out[bad[bad].index] = True
    out[first["item"].isin(SHARE_ITEMS) & (first["value"] <= 0)] = True
    return out


@dataclass(frozen=True)
class _Duration:
    """One first-reported duration fact."""

    issuer_id: int
    item: str
    start: pd.Timestamp
    end: pd.Timestamp
    value: float
    published_at: pd.Timestamp
    period_type: str

    @property
    def days(self) -> int:
        return int((self.end - self.start).days)


def _durations(first: pd.DataFrame) -> list[_Duration]:
    f = first[first["period_start"].notna()]
    return [
        _Duration(int(i), str(it), pd.Timestamp(s), pd.Timestamp(e), float(v), pd.Timestamp(k), str(t))
        for i, it, s, e, v, k, t in zip(
            f["issuer_id"].tolist(),
            f["item"].tolist(),
            f["period_start"].tolist(),
            f["period_end"].tolist(),
            f["value"].tolist(),
            f["published_at"].tolist(),
            f["period_type"].tolist(),
            strict=True,
        )
    ]


def quarters(first: pd.DataFrame) -> pd.DataFrame:
    """``[issuer_id, item, period_end, value, known_at]``: the 3-month value of each flow quarter.

    ``first``: first-reported duration facts (``Q``, ``YTD``, ``FY``) of flow items.
    """
    facts = _durations(first)
    # (issuer, item) -> {quarter end: (3-month value, known_at)}
    out: dict[tuple[int, str], dict[pd.Timestamp, tuple[float, pd.Timestamp]]] = {}
    for f in facts:
        if f.period_type == "Q":
            out.setdefault((f.issuer_id, f.item), {}).setdefault(f.end, (f.value, f.published_at))
    # year-to-date differences: two figures with the same start, a quarter apart
    same_start: dict[tuple[int, str, pd.Timestamp], list[_Duration]] = {}
    for f in facts:
        same_start.setdefault((f.issuer_id, f.item, f.start), []).append(f)
    for (issuer, item, _), group in same_start.items():
        known = out.setdefault((issuer, item), {})
        group.sort(key=lambda d: d.end)
        for i, f in enumerate(group):
            if f.end in known or f.days <= QUARTER_DAYS[1]:
                continue
            prev = next((p for p in reversed(group[:i]) if QUARTER_DAYS[0] <= f.days - p.days <= QUARTER_DAYS[1]), None)
            if prev is not None:
                known[f.end] = (f.value - prev.value, max(f.published_at, prev.published_at))
    # a fiscal-year end without a 9-month figure: the year minus its three quarters
    for f in facts:
        if f.period_type != "FY":
            continue
        known = out.setdefault((f.issuer_id, f.item), {})
        if f.end in known:
            continue
        inside = [(v, k) for e, (v, k) in known.items() if f.start < e < f.end]
        if len(inside) == 3:
            known[f.end] = (f.value - sum(v for v, _ in inside), max([f.published_at] + [k for _, k in inside]))
    rows = [(i, it, e, v, k) for (i, it), known in out.items() for e, (v, k) in known.items()]
    out_frame = pd.DataFrame(rows, columns=["issuer_id", "item", "period_end", "value", "known_at"])
    # typed even when empty (an issuer with balances only), so date arithmetic downstream still works
    return out_frame.astype({"period_end": "datetime64[ns]", "value": float, "known_at": "datetime64[ns]"})


def trailing_twelve_months(first: pd.DataFrame, q: pd.DataFrame | None = None) -> pd.DataFrame:
    """``[issuer_id, item, period_end, value, known_at]``: the fiscal-year figure, else four quarters summed.

    ``q``: the :func:`quarters` of ``first`` when the caller already has them.
    """
    q = (quarters(first) if q is None else q).sort_values(["issuer_id", "item", "period_end"]).reset_index(drop=True)
    g = q.groupby(["issuer_id", "item"], sort=False)
    contiguous = (q["period_end"] - g["period_end"].shift(3)).dt.days <= TTM_MAX_SPAN_DAYS
    summed = q.assign(
        value=g["value"].transform(lambda s: s.rolling(4, min_periods=4).sum()),
        # rolling() cannot take dates: the maximum of the nanosecond values, converted back
        known_at=pd.to_datetime(
            _ns(q["known_at"])
            .astype("int64")
            .groupby([q["issuer_id"], q["item"]], sort=False)
            .transform(lambda v: v.rolling(4, min_periods=4).max()),
            unit="ns",
        ),
    )[contiguous]
    fy = first[(first["period_type"] == "FY") & first["period_start"].notna()].rename(
        columns={"published_at": "known_at"}
    )
    cols = ["issuer_id", "item", "period_end", "value", "known_at"]
    # a reported year beats four quarters, and four quarters beat a year rebuilt from year-to-date figures
    parts = [fy[cols].assign(_rank=0), summed[cols].assign(_rank=1), from_year_to_date(first).assign(_rank=2)]
    both = pd.concat([p for p in parts if not p.empty] or parts[:1])
    both = both.sort_values([*cols[:3], "_rank"]).drop_duplicates(cols[:3], keep="first")
    return both.drop(columns="_rank").dropna(subset=["value"]).reset_index(drop=True)


def from_year_to_date(first: pd.DataFrame) -> pd.DataFrame:
    """``[issuer_id, item, period_end, value, known_at]``: twelve months to a year-to-date end without quarters.

    A company reporting half-years (most UK companies) has no quarters to sum: at its first half the trailing year is
    the last fiscal year, plus this year so far, less the same part of last year. The fiscal year is the one ending
    the day before the year-to-date figure starts; last year's part ends about a year earlier and is as long (both
    within :data:`YEAR_TOLERANCE_DAYS`: 52-53 week years move the ends). Known when the latest of the three is.
    """
    cols = ["issuer_id", "item", "period_end", "value", "known_at"]
    empty = pd.DataFrame(columns=cols).astype({"period_end": "datetime64[ns]", "known_at": "datetime64[ns]"})
    d = first[first["period_start"].notna()]
    d = d.assign(**{c: _ns(d[c]).to_numpy() for c in ("period_start", "period_end", "published_at")})
    d = d.assign(days=(d["period_end"] - d["period_start"]).dt.days)
    ytd = d[(d["period_type"] == "YTD") & (d["days"] > QUARTER_DAYS[1])]
    fy = d[d["period_type"] == "FY"]
    if ytd.empty or fy.empty:
        return empty
    key = ["issuer_id", "item"]
    tolerance = pd.Timedelta(days=YEAR_TOLERANCE_DAYS)
    now = ytd.assign(
        year_end=ytd["period_start"] - pd.Timedelta(days=1), prior_end=ytd["period_end"] - pd.Timedelta(days=365)
    )
    years = fy[[*key, "period_end", "value", "published_at"]].rename(
        columns={"period_end": "fy_end", "value": "year", "published_at": "year_known"}
    )
    now = pd.merge_asof(
        now.sort_values("year_end"),
        years.sort_values("fy_end"),
        left_on="year_end",
        right_on="fy_end",
        by=key,
        direction="nearest",
        tolerance=tolerance,
    )
    before = ytd[[*key, "period_end", "value", "published_at", "days"]].rename(
        columns={"period_end": "before_end", "value": "before", "published_at": "before_known", "days": "before_days"}
    )
    now = pd.merge_asof(
        now.sort_values("prior_end"),
        before.sort_values("before_end"),
        left_on="prior_end",
        right_on="before_end",
        by=key,
        direction="nearest",
        tolerance=tolerance,
    )
    now = now.dropna(subset=["year", "before"])
    now = now[(now["days"] - now["before_days"]).abs() <= YEAR_TOLERANCE_DAYS]
    if now.empty:
        return empty
    known = _latest(_ns(now["published_at"]), _ns(now["year_known"]), _ns(now["before_known"]))
    out = now.assign(value=now["year"] + now["value"] - now["before"], known_at=known.to_numpy())
    return out[cols].reset_index(drop=True)


#: The earlier year-on-year changes an earnings surprise is measured against (two years of quarters), and the fewest
#: it needs.
SURPRISE_QUARTERS = 8
SURPRISE_MIN_QUARTERS = 4
#: A quarter "a year before" ends within this many days of 365 earlier (52-53 week years move the ends).
YEAR_TOLERANCE_DAYS = 20


def earnings_surprise(q: pd.DataFrame) -> pd.DataFrame:
    """``[issuer_id, item, period_end, value, known_at]``: standardised unexpected earnings for each quarter.

    The change from the same quarter a year before (which carries the seasons) less the company's usual change - the
    mean of its previous :data:`SURPRISE_QUARTERS` changes, so steady growth is no surprise - divided by those
    changes' standard deviation (Bernard and Thomas's seasonal random walk with drift). A company beating its own
    pattern by much has tended to drift up for months after the report. ``q`` is :func:`quarters`' output; the
    surprise is known when both quarters are.
    """
    cols = ["issuer_id", "item", "period_end", "value", "known_at"]
    ni = q[q["item"] == "net_income"][["issuer_id", "period_end", "value", "known_at"]].dropna(subset=["value"])
    if ni.empty:
        return pd.DataFrame(columns=cols).astype({"period_end": "datetime64[ns]", "known_at": "datetime64[ns]"})
    prior = ni.rename(columns={"period_end": "prior_end", "value": "prior", "known_at": "prior_known"})
    target = ni.assign(target=ni["period_end"] - pd.Timedelta(days=365)).sort_values("target")
    paired = pd.merge_asof(
        target,
        prior.sort_values("prior_end"),
        left_on="target",
        right_on="prior_end",
        by="issuer_id",
        direction="nearest",
        tolerance=pd.Timedelta(days=YEAR_TOLERANCE_DAYS),
    ).dropna(subset=["prior"])
    paired = paired.sort_values(["issuer_id", "period_end"], kind="stable").reset_index(drop=True)
    change = paired["value"] - paired["prior"]
    earlier = change.groupby(paired["issuer_id"]).shift(1)  # the previous changes only: a beat cannot dampen itself
    by_issuer = earlier.groupby(paired["issuer_id"])
    usual = by_issuer.transform(lambda c: c.rolling(SURPRISE_QUARTERS, min_periods=SURPRISE_MIN_QUARTERS).mean())
    scale = by_issuer.transform(lambda c: c.rolling(SURPRISE_QUARTERS, min_periods=SURPRISE_MIN_QUARTERS).std())
    return pd.DataFrame(
        {
            "issuer_id": paired["issuer_id"],
            "item": "earnings_surprise",
            "period_end": paired["period_end"],
            # changes that never vary give no scale (a rounding error is not a standard deviation)
            "value": (change - usual) / scale.where(scale > 1e-9 * (paired["value"].abs() + paired["prior"].abs())),
            "known_at": _latest(_ns(paired["known_at"]), _ns(paired["prior_known"])),
        }
    ).dropna(subset=["value"])[cols]


# ---------------------------------------------------------------- items with their known dates


@dataclass(frozen=True)
class Known:
    """An item for every (issuer, period end): its values and when each value became known."""

    value: pd.Series
    known_at: pd.Series

    @classmethod
    def missing(cls, index: pd.Index) -> Known:
        """All missing."""
        return cls(pd.Series(np.nan, index=index, dtype=float), pd.Series(pd.NaT, index=index, dtype="datetime64[ns]"))

    def __sub__(self, other: Known) -> Known:
        both = self.value.notna() & other.value.notna()
        return Known((self.value - other.value).where(both), _latest(self.known_at, other.known_at).where(both))

    def abs(self) -> Known:
        """Absolute values (some filers report an expense as a negative number)."""
        return Known(self.value.abs(), self.known_at)


def _latest(*known: pd.Series) -> pd.Series:
    return pd.concat(known, axis=1).max(axis=1)


def first(*candidates: Known) -> Known:
    """Row by row, the first candidate with a value."""
    value, known_at = candidates[0].value.copy(), candidates[0].known_at.copy()
    for c in candidates[1:]:
        take = value.isna() & c.value.notna()
        value, known_at = value.where(~take, c.value), known_at.where(~take, c.known_at)
    return Known(value, known_at)


def total(*parts: Known, require: Known | None = None) -> Known:
    """Sum of the parts present (missing when none is, or when ``require`` is missing); known when the latest is."""
    value = pd.concat([p.value for p in parts], axis=1).sum(axis=1, min_count=1)
    known_at = _latest(*(p.known_at.where(p.value.notna()) for p in parts))
    if require is not None:
        value, known_at = value.where(require.value.notna()), known_at.where(require.value.notna())
    return Known(value, known_at)


def _zero(*present: Known) -> Known:
    """Zero wherever any of ``present`` has a value (a gain nobody reported is none), known when they are."""
    t = total(*present)
    return Known(t.value * 0.0, t.known_at)


def _ratio(numerator: Known, denominator: Known) -> Known:
    """``numerator / denominator`` (missing where the denominator is zero), known when the later of the two is."""
    both = numerator.value.notna() & denominator.value.notna() & (denominator.value != 0)
    value = (numerator.value / denominator.value.where(denominator.value != 0)).where(both)
    return Known(value, _latest(numerator.known_at, denominator.known_at).where(both))


def _by_item(long: pd.DataFrame, index: pd.MultiIndex) -> dict[str, Known]:
    out = {}
    for item, grp in long.groupby("item"):
        g = grp.set_index(KEY)
        out[str(item)] = Known(g["value"].reindex(index), g["known_at"].reindex(index))
    return out


def accounting_items(facts: pd.DataFrame) -> pd.DataFrame:
    """Items the ratios use, per ``(issuer_id, period_end)``: ``<item>`` and ``known_at:<item>`` columns.

    Derived items:

    * ``ebit`` - operating income, else pretax income + interest expense;
    * ``gross_profit`` - reported, else revenue - cost of revenue;
    * ``equity`` - equity attributable to the parent, else total equity;
    * ``cash`` - cash and short-term investments, else cash + short-term investments;
    * ``total_debt`` - a reported total (LSEG); else non-current long-term debt + current debt; else a long-term
      total incl. current maturities + short-term borrowings and commercial paper; else the reported borrowings
      (notes, secured, convertible, credit lines, other) + current debt;
    * ``current_debt`` - debt due within a year: reported, else current maturities of long-term debt + short-term
      borrowings and commercial paper (a vendor that reports only a total, LSEG, leaves it missing);
    * ``fixed_assets`` - net PP&E, else PP&E including lease assets;
    * ``total_liabilities`` - reported, else liabilities and equity - total equity;
    * ``shares`` - the cover-page count (dated near the filing day, the freshest), else shares outstanding at the
      period end, else the quarter's weighted-average basic shares;
    * ``operating_cash_flow`` and ``capex`` (cash paid for property, plant and equipment) - trailing twelve months
      like every flow; a vendor that does not report them leaves them missing;
    * ``earnings_surprise`` - :func:`earnings_surprise` of the quarterly net income;
    * ``unusual_expense`` - reported net (LSEG), else restructuring and impairment charges less gains on disposals (the
      SEC has no one line; the net is missing when none of the three is reported);
    * ``interest_income`` - net interest income + interest expense (a bank's gross interest income);
    * ``risk_weighted_assets`` - reported (LSEG), else tier-1 capital / the tier-1 ratio (the SEC's two tags);
    * depreciation, financing flows, discontinued operations and the bank items as reported, flows over the trailing
      twelve months; ``tier1_ratio`` and ``total_capital_ratio`` are pure numbers (0.125 for 12.5 %).
    """
    reported = first_reported(facts)
    reported = reported[~implausible_share_counts(reported)]
    flow_facts = reported[reported["item"].isin(FLOW_ITEMS)]
    q = quarters(flow_facts)
    flows = trailing_twelve_months(flow_facts, q)
    balances = reported[(reported["period_type"] == "I") & reported["item"].isin(BALANCE_ITEMS)]
    # weighted-average shares: only a reported 3-month figure (averages cannot be differenced like flows); a quarter
    # re-reported as next year's comparative may carry a start a day or two off - the first known wins, as for flows
    weighted = reported[(reported["item"] == "shares_weighted_basic") & (reported["period_type"] == "Q")]
    weighted = weighted.drop_duplicates(["issuer_id", "item", "period_end"], keep="first")  # sorted by published_at
    parts = [
        flows,
        balances.rename(columns={"published_at": "known_at"})[flows.columns],
        weighted.rename(columns={"published_at": "known_at"})[flows.columns],
        earnings_surprise(q)[flows.columns],
    ]
    long = pd.concat([p for p in parts if len(p)] or parts[:1], ignore_index=True).astype(
        {"issuer_id": "int64"}  # an empty part (no flows at all) would otherwise leave the ids untyped
    )
    index = pd.MultiIndex.from_frame(long[KEY].drop_duplicates().sort_values(KEY))
    raw = _by_item(long, index)
    nothing = Known.missing(index)

    def get(name: str) -> Known:
        return raw.get(name, nothing)

    current_debt = first(
        get("debt_current"), total(get("lt_debt_current"), get("st_borrowings"), get("commercial_paper"))
    )
    items = {
        "revenue": get("revenue"),
        "gross_profit": first(get("gross_profit"), get("revenue") - get("cost_of_revenue")),
        "operating_income": get("operating_income"),
        "ebit": first(
            get("operating_income"),
            total(get("pretax_income"), get("interest_expense").abs(), require=get("pretax_income")),
        ),
        "net_income": get("net_income"),
        "operating_cash_flow": get("operating_cash_flow"),
        "capex": get("capex"),
        "interest_expense": get("interest_expense"),
        "total_assets": get("total_assets"),
        "total_liabilities": first(
            get("total_liabilities"), get("liabilities_and_equity") - first(get("equity_total"), get("equity_parent"))
        ),
        "equity": first(get("equity_parent"), get("equity_total")),
        "current_assets": get("current_assets"),
        "current_liabilities": get("current_liabilities"),
        "cash": first(get("cash_and_st_investments"), total(get("cash"), get("st_investments"), require=get("cash"))),
        "total_debt": first(
            get("debt_total"),
            total(get("lt_debt_noncurrent"), current_debt, require=get("lt_debt_noncurrent")),
            total(get("lt_debt_total"), get("st_borrowings"), get("commercial_paper"), require=get("lt_debt_total")),
            total(
                get("notes_payable"),
                get("secured_debt"),
                get("convertible_notes"),
                get("credit_lines"),
                get("other_borrowings"),
                current_debt,
            ),
        ),
        "current_debt": current_debt,
        "fixed_assets": first(get("ppe_net"), get("ppe_and_lease_assets")),
        "retained_earnings": get("retained_earnings"),
        "shares_outstanding": get("shares_outstanding"),
        "shares": first(get("shares_cover"), get("shares_outstanding"), get("shares_weighted_basic")),
        "earnings_surprise": get("earnings_surprise"),
        **{
            name: get(name)
            for name in (
                "depreciation",
                "inventory",
                "financing_cash_flow",
                "dividends_paid",
                "stock_issued",
                "stock_repurchased",
                "st_debt_change",
                "lt_debt_issued",
                "lt_debt_repaid",
                "discontinued_operations",
                "net_interest_income",
                "noninterest_income",
                "noninterest_expense",
                "loan_loss_provision",
                "loans_net",
                "deposits",
                "nonperforming_loans",
                "tier1_ratio",
                "total_capital_ratio",
            )
        },
        "unusual_expense": first(
            get("unusual_expense"),
            total(get("restructuring_charges"), get("impairment_charges"))
            - first(get("disposal_gains"), _zero(get("restructuring_charges"), get("impairment_charges"))),
        ),
        "interest_income": total(
            get("net_interest_income"), get("interest_expense").abs(), require=get("net_interest_income")
        ),
        "risk_weighted_assets": first(get("risk_weighted_assets"), _ratio(get("tier1_capital"), get("tier1_ratio"))),
    }
    out = pd.DataFrame(index=index)
    for name in ITEMS:
        out[name] = items[name].value
        out[f"known_at:{name}"] = items[name].known_at
    return out


# ---------------------------------------------------------------- point in time


def as_of(
    items: pd.DataFrame,
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    max_age_days: int = MAX_AGE_DAYS,
    *,
    with_period_end: bool = False,
) -> pd.DataFrame:
    """What was known on each decision date: ``(date, issuer_id)`` x items.

    Each item comes from the issuer's latest period end whose value was known strictly before the date, if that
    period ended at most ``max_age_days`` before the date. ``with_period_end`` adds ``period_end:<item>`` and
    ``known_at:<item>`` columns per item: a share count is expressed in the shares of its publication date (a split
    between a quarter's end and the filing is already in the filed numbers), so splits after ``known_at`` adjust it.
    """
    names = [c for c in items.columns if not c.startswith("known_at:")]
    days = pd.DatetimeIndex(_ns(pd.Index(list(dates))).drop_duplicates().sort_values())
    issuers = items.index.get_level_values("issuer_id").unique()
    grid = pd.MultiIndex.from_product([days, issuers], names=["date", "issuer_id"]).to_frame(index=False)
    grid = grid.sort_values("date").reset_index(drop=True)
    index = pd.MultiIndex.from_frame(grid[["date", "issuer_id"]])
    columns: dict[str, Any] = {}  # built whole at the end: sixty-odd items inserted one by one fragment a frame
    never = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    for name in names:
        col = items[[name, f"known_at:{name}"]].dropna().reset_index()
        col.columns = ["issuer_id", "period_end", "value", "known_at"]
        col = col.assign(known_at=_ns(col["known_at"]).to_numpy(), period_end=_ns(col["period_end"]).to_numpy())
        if col.empty:
            columns[name] = np.full(len(grid), np.nan)
            if with_period_end:
                columns[f"period_end:{name}"] = never
                columns[f"known_at:{name}"] = never
            continue
        # only rows that move an issuer's latest period end forward can ever be "the latest known"
        col = col.sort_values(["issuer_id", "known_at", "period_end"])
        col = col[col["period_end"] >= col.groupby("issuer_id")["period_end"].cummax()].sort_values("known_at")
        hit = pd.merge_asof(grid, col, left_on="date", right_on="known_at", by="issuer_id", allow_exact_matches=False)
        stale = hit["period_end"] < hit["date"] - pd.Timedelta(days=max_age_days)
        columns[name] = hit["value"].where(~stale).to_numpy()
        if with_period_end:
            columns[f"period_end:{name}"] = hit["period_end"].where(~stale).to_numpy()
            columns[f"known_at:{name}"] = hit["known_at"].where(~stale).to_numpy()
    return pd.DataFrame(columns, index=index).dropna(how="all")
