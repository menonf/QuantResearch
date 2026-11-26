"""The GEMS estimate-based factors (I/B/E/S), rebuilt as coded on our point-in-time consensus history.

    inputs = load_estimate_inputs(cache, dates, close, actions, value=..., shares=..., total_assets=...)
    panels = estimate_factors(inputs)                     # {factor: date x listing_id}
    revisions = revisions_composite(inputs, members)      # zscore_earnings_revisions_4_mth_composite

After ``PSA_Crisp_Metrics``' ``base_ntm`` / ``base_ltm`` and the ``ibes_raw_*`` classes (``metrics.py`` 4315-6857),
and ``zscore_earnings_revisions_4_mth_composite`` (8661-8928), as reconstructed in ``GEMS_Factor_Reconstruction.py``:

* **Next twelve months** blends FY1 and FY2 by how much of each fiscal year falls in the year after the date;
  **last twelve months** blends the last reported year with FY1 (:meth:`Estimates.ntm`, :meth:`Estimates.ltm`).
* **Gates**: a factor is missing where fewer than two estimates stand behind it (production's ``n < 2``; a missing
  count keeps the value, as Spark's null comparison does).
* **Growth** from one figure to another follows production's sign rules
  (:func:`~equity_research.strategies.qgarp_math.growth_piecewise`; the two-year FY1 to FY3 rate is compounded).
* **Revision breadth** is half FY1's and half FY2's (raised - lowered) / estimates; over 4 months its months are
  weighted 0.4 / 0.3 / 0.2 / 0.1, and its persistence counts the signs of the last six months.

What is ours, not production's:

* **When.** A decision date (a month's last trading day) reads the consensus of its own month end - a month ending
  on a weekend carries the last trading day's consensus - and the month ends before it for revisions and changes; a
  reported actual only once it was announced (strictly before the date). Production reads weekly snapshots.
* **Shares.** Per-share figures are stored in the shares of their own date and put into the shares of the decision
  date here, so a split between two snapshots is not a change (production's history is restated to one basis).
* **Prices.** A per-share yield divides by the listing's close on the date; a total (revenue, EBIT) by the
  company's market value - production's ``currency_units`` scaling is not needed (LSEG answers in units).
* **Not built**: ``sales_revisions_1_mth`` - only EPS revision counts are loaded.
* **Revision counts are live only** (:func:`live_revisions`). LSEG answers ``TR.NumEstRevisingUp`` / ``Down`` with
  the counts of the day it is asked, whatever date the request names: in the build of 2026-11-17, 99.6 % of the
  stored counts from 2000 on equal each listing's latest (GE: 21 raised, 0 lowered every June from 2000 to 2026),
  and a company since delisted answers 0 and 0. A count asked long after its date is therefore today's count under
  an old date - look-ahead in a backtest - so research reads only counts asked within
  :data:`REVISION_LIVE_DAYS` of their month end (the refresh asks each month end within days). Backtests before
  the first live month have no breadth; a screen of today's companies has their real counts.
* **Consensus revisions** (:func:`consensus_revision_factors`, not in the library): the revision of the consensus
  itself. The consensus is a true history (LSEG keeps each month's mean, for departed companies too); its
  month-on-month change, for the same fiscal year, over the price, is the classic measure of earnings revisions
  (Chan, Jegadeesh and Lakonishok, 1996). It stands in for breadth in every backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from equity_research.data import schema as _schema
from equity_research.data.cache import ResearchCache
from equity_research.data.panels import split_factor
from equity_research.data.reads import id_list
from equity_research.strategies.qgarp_math import div, growth_piecewise, rows_on

PER_SHARE = frozenset({"eps", "dps", "cfps"})
#: Months of snapshots before the first decision date the factors read (revisions over six months).
MONTHS_BEFORE = 6
#: An actual without an announcement time is used this long after its fiscal year ended - late, never early.
FALLBACK_REPORTING_DAYS = 90
#: A revision count is that month end's only when it was asked within this many days of it (:func:`live_revisions`).
REVISION_LIVE_DAYS = _schema.REVISION_LIVE_DAYS

EPS_FACTORS = [
    "earnings_revisions_1_mth",
    "earnings_revisions_4_mth",
    "earnings_revision_persistence_6mth",
    "eps_growth_fy0_to_fy1",
    "eps_growth_fy1_to_fy2",
    "eps_growth_fy1_to_fy3",
    "eps_growth_fy2_to_fy3",
    "eps_growth_ltm_to_ntm",
    "eps_uncertainty",
    "standardised_unexpected_earnings",
    "earnings_yield_ntm",
    "earnings_yield_ltm",
    "earnings_yield_fy1",
    "earnings_yield_fy2",
    "earnings_yield_fy3",
]
EBIT_FACTORS = [
    "ebit_yield_fy1",
    "ebit_yield_fy2",
    "ebit_yield_fy3",
    "ebit_yield_ntm",
    "ebit_yield_ltm",
    "cash_flow_growth_ltm_to_ntm",
    "ebit_growth_ltm_to_ntm",
    "ebit_growth_fy1_to_fy2",
    "ebit_growth_fy2_to_fy3",
    "ebit_growth_fy1_to_fy3",
]
REVENUE_FACTORS = [
    "sales_yield_ntm",
    "sales_growth_ltm_to_ntm",
    "sales_growth_fy1_to_fy3",
    "return_on_equity_ntm",
    "sustainable_growth_ntm",
]
OTHER_FACTORS = ["dividend_yield_ntm", "eps_ntm_1mth_chg", "eps_ntm_3mth_chg"]
ESTIMATE_FACTORS = EPS_FACTORS + EBIT_FACTORS + REVENUE_FACTORS + OTHER_FACTORS
#: Ours, not the library's (see the module docstring): breadth's survivorship-free stand-in, for backtests.
CONSENSUS_REVISION_FACTORS = ["consensus_revisions_1_mth", "consensus_revisions_4_mth"]
REVISIONS_COMPOSITE = "zscore_earnings_revisions_4_mth_composite"


@dataclass(frozen=True)
class EstimateInputs:
    """What the estimate factors read; panels are ``date x listing_id`` on the decision dates.

    ``consensus`` / ``revisions`` / ``actuals`` are long tables as stored (``fiscal_period_end`` and ``period_end`` as
    dates); ``basis`` is the daily split factor (the product of split ratios after each day) that moves a per-share
    figure between the shares of two dates; ``close`` the listing's unadjusted close on each date; ``value``,
    ``shares`` and ``total_assets`` the company's market value, share count (in the date's shares) and assets.
    """

    dates: pd.DatetimeIndex
    consensus: pd.DataFrame
    revisions: pd.DataFrame
    actuals: pd.DataFrame
    basis: pd.DataFrame
    close: pd.DataFrame
    value: pd.DataFrame
    shares: pd.DataFrame
    total_assets: pd.DataFrame

    @property
    def listings(self) -> pd.Index:
        """The listings of every panel."""
        return self.close.columns


def load_estimate_inputs(
    cache: ResearchCache,
    dates: pd.DatetimeIndex,
    close: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    value: pd.DataFrame,
    shares: pd.DataFrame,
    total_assets: pd.DataFrame,
) -> EstimateInputs:
    """The estimates of ``close``'s listings from the research cache, from :data:`MONTHS_BEFORE` before the first date.

    ``close`` is the daily unadjusted close panel (its days carry the split factor), ``actions`` the corporate
    actions of the price vendors (their splits move per-share figures between dates).
    """
    listings = close.columns
    first = (dates.min() - pd.DateOffset(months=MONTHS_BEFORE + 1)).date().isoformat()
    consensus = cache.query(
        "SELECT listing_id, measure, period, as_of_date, fiscal_period_end, mean, std_dev, estimates "
        f"FROM equity.estimate WHERE listing_id IN ({id_list(listings)}) AND as_of_date >= ?",
        [first],
    )
    revisions = live_revisions(
        cache.query(
            "SELECT listing_id, measure, period, as_of_date, up, down, loaded_at FROM equity.estimate_revision "
            f"WHERE listing_id IN ({id_list(listings)}) AND as_of_date >= ?",
            [first],
        )
    )
    actuals = cache.query(
        "SELECT listing_id, measure, period_end, value, announced_at FROM equity.estimate_actual "
        f"WHERE listing_id IN ({id_list(listings)})"
    )
    days = pd.DatetimeIndex(close.index)
    return EstimateInputs(
        dates=dates,
        consensus=consensus,
        revisions=revisions,
        actuals=actuals,
        basis=split_factor(actions, days, listings),
        close=close.reindex(dates),
        value=value,
        shares=shares,
        total_assets=total_assets,
    )


def live_revisions(rows: pd.DataFrame, live_days: int = REVISION_LIVE_DAYS) -> pd.DataFrame:
    """The revision counts asked from their month end to ``live_days`` after it; ``loaded_at`` dropped.

    LSEG answers a revision count with the day's count whatever date is asked (see the module docstring), so a count
    loaded years after its ``as_of_date`` is a snapshot of the day it was loaded, not of its date.
    """
    as_of = pd.to_datetime(rows["as_of_date"])
    loaded = pd.to_datetime(rows["loaded_at"])
    live = (loaded >= as_of) & (loaded <= as_of + np.timedelta64(live_days, "D"))
    return rows.loc[live].drop(columns="loaded_at").reset_index(drop=True)


def month_ends(dates: pd.DatetimeIndex, months_back: int = 0) -> pd.DatetimeIndex:
    """The calendar month end of each date's month, ``months_back`` months earlier."""
    ends = pd.DatetimeIndex(dates) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex(ends - pd.DateOffset(months=months_back)) + pd.offsets.MonthEnd(0)


def _days(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Dates as day numbers (floats): spans subtract and compare as numbers, missing dates as NaN."""
    stamps = pd.to_datetime(pd.Series(dates))
    return np.asarray((stamps - pd.Timestamp(0)).dt.days.astype(float))


def gate(value: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    """``when(n < 2, null).otherwise(value)``: missing where fewer than two estimates; a missing count keeps it."""
    return value.where(~(n < 2))


def gate2(value: pd.DataFrame, n_a: pd.DataFrame, n_b: pd.DataFrame) -> pd.DataFrame:
    """``when(greatest(n_a, n_b) < 2, null)``: Spark's ``greatest`` skips a missing count."""
    with np.errstate(invalid="ignore"):
        most = pd.DataFrame(
            np.fmax(n_a.to_numpy(dtype=float), n_b.to_numpy(dtype=float)), index=value.index, columns=value.columns
        )
    return value.where(~(most < 2))


def clip1(value: pd.DataFrame) -> pd.DataFrame:
    """Bounded to [-1, 1] (a breadth of revisions)."""
    return value.clip(-1.0, 1.0)


def _sign(value: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.sign(value.to_numpy(dtype=float)), index=value.index, columns=value.columns)


@dataclass
class Estimates:
    """Lookups of :class:`EstimateInputs` on the decision dates, every per-share figure in the shares of the date."""

    inputs: EstimateInputs
    _wide: dict[tuple[str, ...], pd.DataFrame] = field(default_factory=dict)

    def _table(self, source: str, measure: str, period: str, column: str) -> pd.DataFrame:
        key = (source, measure, period, column)
        if key not in self._wide:
            rows = self.inputs.consensus if source == "consensus" else self.inputs.revisions
            rows = rows[(rows["measure"] == measure) & (rows["period"] == period)]
            values = rows[column]
            if column == "fiscal_period_end":
                values = pd.Series(_days(rows[column]), index=rows.index)
            frame = rows.assign(_v=values.astype(float), as_of_date=pd.to_datetime(rows["as_of_date"]))
            frame = frame.drop_duplicates(["as_of_date", "listing_id"])
            self._wide[key] = frame.pivot(index="as_of_date", columns="listing_id", values="_v")  # noqa: PD010
        return self._wide[key]

    def to_date_shares(self, then: pd.DatetimeIndex) -> pd.DataFrame:
        """Multipliers moving a per-share figure quoted in the shares of ``then`` (row by row) into the date's."""
        at_date = rows_on(self.inputs.basis, self.inputs.dates)
        at_then = rows_on(self.inputs.basis, then)
        return pd.DataFrame(at_date.to_numpy() / at_then.to_numpy(), index=self.inputs.dates, columns=at_date.columns)

    def get(self, measure: str, period: str, column: str, months_back: int = 0) -> pd.DataFrame:
        """``column`` of the consensus of ``measure`` / ``period``, at each date's month end ``months_back`` before."""
        then = month_ends(self.inputs.dates, months_back)
        wide = self._table("consensus", measure, period, column).reindex(index=then, columns=self.inputs.listings)
        out = pd.DataFrame(wide.to_numpy(), index=self.inputs.dates, columns=self.inputs.listings)
        if measure in PER_SHARE and column in ("mean", "std_dev"):
            out = out * self.to_date_shares(then)
        return out

    def revision(self, period: str, column: str, months_back: int = 0) -> pd.DataFrame:
        """EPS revision counts (``up`` / ``down``) of ``period`` at each date's month end ``months_back`` before."""
        then = month_ends(self.inputs.dates, months_back)
        wide = self._table("revisions", "eps", period, column).reindex(index=then, columns=self.inputs.listings)
        return pd.DataFrame(wide.to_numpy(), index=self.inputs.dates, columns=self.inputs.listings)

    def same_year_mean(self, measure: str, period: str, months_back: int) -> pd.DataFrame:
        """The mean ``months_back`` months before, for the same fiscal year as ``period`` now.

        I/B/E/S' "mean three months ago": a year that was FY2 then and is FY1 now is found under FY2 then.
        """
        now_end = self.get(measure, period, "fiscal_period_end")
        out = pd.DataFrame(np.nan, index=now_end.index, columns=now_end.columns)
        for then_period in ("FY1", "FY2", "FY3"):
            mean = self.get(measure, then_period, "mean", months_back)
            end = self.get(measure, then_period, "fiscal_period_end", months_back)
            out = out.where(out.notna(), mean.where(end == now_end))
        return out

    def actual(self, measure: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``(value, period end in days)`` of the latest fiscal year reported strictly before each date."""
        a = self.inputs.actuals
        a = a[a["measure"] == measure]
        a = a.assign(
            listing_id=a["listing_id"].astype("int64"),  # typed even when empty: the as-of join needs one key type
            period_end=pd.to_datetime(a["period_end"]),
            announced_at=pd.to_datetime(a["announced_at"]),
            value=a["value"].astype(float),
        )
        known = a["announced_at"].fillna(a["period_end"] + pd.Timedelta(days=FALLBACK_REPORTING_DAYS))
        # the as-of join takes the last row known before the date: years known at the same moment (LSEG back-filled
        # some companies' early years with one announcement time) are put in fiscal order, so the latest year wins
        # whatever order the cache returned the rows in
        a = a.assign(known=known.astype("datetime64[ns]")).sort_values(["known", "period_end"], kind="mergesort")
        pairs = pd.MultiIndex.from_product([self.inputs.dates, self.inputs.listings], names=["date", "listing_id"])
        grid = pairs.to_frame(index=False).astype({"date": "datetime64[ns]"}).sort_values("date")
        hit = pd.merge_asof(grid, a, left_on="date", right_on="known", by="listing_id", allow_exact_matches=False)
        hit = hit.assign(end_days=_days(hit["period_end"]))
        value = hit.pivot(index="date", columns="listing_id", values="value").reindex(  # noqa: PD010
            index=self.inputs.dates, columns=self.inputs.listings
        )
        end_days = hit.pivot(index="date", columns="listing_id", values="end_days").reindex(  # noqa: PD010
            index=self.inputs.dates, columns=self.inputs.listings
        )
        if measure in PER_SHARE:
            ends = hit.pivot(index="date", columns="listing_id", values="period_end").reindex(  # noqa: PD010
                index=self.inputs.dates, columns=self.inputs.listings
            )
            value = value * self._to_date_shares_from(ends)
        return value, end_days

    def _to_date_shares_from(self, ends: pd.DataFrame) -> pd.DataFrame:
        """Multipliers from the shares of each cell's own date (``ends``) into the decision date's."""
        b = self.inputs.basis
        at_date = rows_on(b, self.inputs.dates).to_numpy()
        days = pd.DatetimeIndex(b.index)
        out = np.full(ends.shape, np.nan)
        for j, listing in enumerate(ends.columns):
            when = pd.to_datetime(ends[listing])
            known = when.notna().to_numpy()
            if not known.any() or listing not in b.columns:
                continue
            pos = days.searchsorted(pd.DatetimeIndex(when[known]), side="right") - 1
            col = b[listing].to_numpy()
            then = np.where(pos >= 0, col[np.clip(pos, 0, None)], col[0])
            out[known, j] = at_date[known, j] / then
        return pd.DataFrame(out, index=ends.index, columns=ends.columns)

    def ntm(self, measure: str, months_back: int = 0) -> pd.DataFrame:
        """Next twelve months: FY1 and FY2 means weighted by how much of each falls in the year after the date.

        ``(FY2 end - D+1y) / (FY2 end - FY1 end) x FY1 + (D+1y - FY1 end) / (FY2 end - FY1 end) x FY2``, the date D
        ``months_back`` months back; the weights are not clipped (production's ``base_ntm``).
        """
        m1, m2 = self.get(measure, "FY1", "mean", months_back), self.get(measure, "FY2", "mean", months_back)
        e1 = self.get(measure, "FY1", "fiscal_period_end", months_back)
        e2 = self.get(measure, "FY2", "fiscal_period_end", months_back)
        year_on = _days(
            pd.DatetimeIndex(self.inputs.dates) - pd.DateOffset(months=months_back) + pd.DateOffset(years=1)
        )
        horizon = pd.DataFrame(np.repeat(year_on[:, None], m1.shape[1], axis=1), index=m1.index, columns=m1.columns)
        span = e2 - e1
        return div(e2 - horizon, span) * m1 + div(horizon - e1, span) * m2

    def ltm(self, measure: str) -> pd.DataFrame:
        """Last twelve months: the last reported year and FY1, by how much of each lies in the year before the date.

        Production's ``base_ltm``: ``(FY1 end - D) / (FY1 end - FY0 end) x FY0 + (D - FY0 end) / (...) x FY1``.
        """
        a0, e0 = self.actual(measure)
        m1, e1 = self.get(measure, "FY1", "mean"), self.get(measure, "FY1", "fiscal_period_end")
        today = _days(self.inputs.dates)
        now = pd.DataFrame(np.repeat(today[:, None], m1.shape[1], axis=1), index=m1.index, columns=m1.columns)
        span = e1 - e0
        return div(e1 - now, span) * a0 + div(now - e0, span) * m1

    def consensus_revision(self, period: str, months_back: int) -> pd.DataFrame:
        """How much ``period``'s EPS consensus moved in the month ending ``months_back`` months before the date.

        The mean at that month end less the mean a month earlier *for the same fiscal year* (a year that was FY2 then
        is found under FY2), both in the shares of the date - so neither a new fiscal year nor a split is a revision.
        """
        end = self.get("eps", period, "fiscal_period_end", months_back)
        before = pd.DataFrame(np.nan, index=end.index, columns=end.columns)
        for then_period in ("FY1", "FY2", "FY3"):
            mean = self.get("eps", then_period, "mean", months_back + 1)
            then_end = self.get("eps", then_period, "fiscal_period_end", months_back + 1)
            before = before.where(before.notna(), mean.where(then_end == end))
        return self.get("eps", period, "mean", months_back) - before

    def consensus_breadth(self, months_back: int) -> pd.DataFrame:
        """Half FY1's and half FY2's consensus revision over the date's price - breadth's stand-in, from consensus."""
        price = self.inputs.close
        return (
            div(self.consensus_revision("FY1", months_back), price)
            + div(self.consensus_revision("FY2", months_back), price)
        ) * 0.5

    def breadth(self, months_back: int) -> pd.DataFrame:
        """Half FY1's and half FY2's (raised - lowered) / estimates, at the month end ``months_back`` before."""
        parts = []
        for period in ("FY1", "FY2"):
            up, down = self.revision(period, "up", months_back), self.revision(period, "down", months_back)
            parts.append(div(up - down, self.get("eps", period, "estimates", months_back)) * 0.5)
        return parts[0] + parts[1]


def estimate_factors(inputs: EstimateInputs) -> dict[str, pd.DataFrame]:
    """Every factor of :data:`ESTIMATE_FACTORS` on the decision dates (see the module docstring)."""
    e = Estimates(inputs)
    return eps_factors(e) | ebit_factors(e) | revenue_factors(e) | other_factors(e)


def eps_factors(e: Estimates) -> dict[str, pd.DataFrame]:
    """``ibes_raw_eps_metrics`` (4315-5763)."""
    n1, n2, n3 = (e.get("eps", p, "estimates") for p in ("FY1", "FY2", "FY3"))
    m1, m2, m3 = (e.get("eps", p, "mean") for p in ("FY1", "FY2", "FY3"))
    sd1, sd2 = e.get("eps", "FY1", "std_dev"), e.get("eps", "FY2", "std_dev")
    price = e.inputs.close
    ntm, ltm = e.ntm("eps"), e.ltm("eps")
    a0, _ = e.actual("eps")
    surprise = (div(m1 - e.same_year_mean("eps", "FY1", 3), sd1) + div(m2 - e.same_year_mean("eps", "FY2", 3), sd2)) / 2
    b = [e.breadth(k) for k in range(6)]
    four = b[0] * 0.4 + b[1] * 0.3 + b[2] * 0.2 + b[3] * 0.1
    persistence = sum((_sign(x) for x in b[1:]), start=_sign(b[0]))
    return {
        "earnings_revisions_1_mth": clip1(gate2(b[0], n1, n2)),
        "earnings_revisions_4_mth": clip1(gate2(four, n1, n2)),
        "earnings_revision_persistence_6mth": gate2(persistence, n1, n2),
        "eps_growth_fy0_to_fy1": gate(growth_piecewise(m1, a0), n1),
        "eps_growth_fy1_to_fy2": gate2(growth_piecewise(m2, m1), n1, n2),
        "eps_growth_fy1_to_fy3": gate2(growth_piecewise(m3, m1, 2), n1, n3),
        "eps_growth_fy2_to_fy3": gate2(growth_piecewise(m3, m2), n2, n3),
        "eps_growth_ltm_to_ntm": gate(growth_piecewise(ntm, ltm), n1),
        "eps_uncertainty": gate2(div(sd1, m1.abs()), n1, n2),
        "standardised_unexpected_earnings": gate2(surprise, n1, n2),
        "earnings_yield_ntm": gate2(div(ntm, price), n1, n2),
        "earnings_yield_ltm": gate(div(ltm, price), n1),
        "earnings_yield_fy1": gate(div(m1, price), n1),
        "earnings_yield_fy2": gate(div(m2, price), n2),
        "earnings_yield_fy3": gate(div(m3, price), n3),
    }


def ebit_factors(e: Estimates) -> dict[str, pd.DataFrame]:
    """``ibes_raw_ebit_metrics`` (5765-6434): totals over the company's value; cash-flow growth over its assets."""
    n1, n2, n3 = (e.get("ebit", p, "estimates") for p in ("FY1", "FY2", "FY3"))
    m1, m2, m3 = (e.get("ebit", p, "mean") for p in ("FY1", "FY2", "FY3"))
    value, shares, assets = e.inputs.value, e.inputs.shares, e.inputs.total_assets
    ntm, ltm = e.ntm("ebit"), e.ltm("ebit")
    cash_flow = div(e.ntm("cfps") * shares, assets) - div(e.ltm("cfps") * shares, assets)
    return {
        "ebit_yield_fy1": gate(div(m1, value), n1),
        "ebit_yield_fy2": gate(div(m2, value), n2),
        "ebit_yield_fy3": gate(div(m3, value), n3),
        "ebit_yield_ntm": gate(div(ntm, value), n1),
        "ebit_yield_ltm": gate(div(ltm, value), n1),
        "cash_flow_growth_ltm_to_ntm": gate(cash_flow, n1),
        "ebit_growth_ltm_to_ntm": gate2(growth_piecewise(ntm, ltm), n1, n2),
        "ebit_growth_fy1_to_fy2": gate2(growth_piecewise(m2, m1), n1, n2),
        "ebit_growth_fy2_to_fy3": gate2(growth_piecewise(m3, m2), n2, n3),
        "ebit_growth_fy1_to_fy3": gate2(growth_piecewise(m3, m1, 2), n1, n3),
    }


def revenue_factors(e: Estimates) -> dict[str, pd.DataFrame]:
    """``ibes_raw_revenue_metrics`` (6436-6705); its gates count revenue estimates, ROE's too (as coded)."""
    n1, n3 = e.get("revenue", "FY1", "estimates"), e.get("revenue", "FY3", "estimates")
    m1, m3 = e.get("revenue", "FY1", "mean"), e.get("revenue", "FY3", "mean")
    ntm, ltm = e.ntm("revenue"), e.ltm("revenue")
    roe = e.ntm("roe")
    two_year = div(m3 - m1, m1.abs()) + 1
    with np.errstate(invalid="ignore"):
        root = pd.DataFrame(np.power(two_year.to_numpy(dtype=float), 0.5), index=m1.index, columns=m1.columns) - 1
    return {
        "sales_yield_ntm": gate(div(ntm, e.inputs.value), n1),
        "sales_growth_ltm_to_ntm": gate(div(ntm - ltm, ltm.abs()), n1),
        "sales_growth_fy1_to_fy3": gate2(root, n1, n3),
        "return_on_equity_ntm": gate(roe / 100, n1),
        "sustainable_growth_ntm": gate(roe * (1 - div(e.ntm("dps"), e.ntm("eps"))) / 100, n1),
    }


def other_factors(e: Estimates) -> dict[str, pd.DataFrame]:
    """``ibes_raw_dividends_metrics`` and ``ibes_raw_ntm_change_metrics`` (6707-6857)."""
    n_dps, n_eps = e.get("dps", "FY1", "estimates"), e.get("eps", "FY1", "estimates")
    now, month, quarter = e.ntm("eps"), e.ntm("eps", 1), e.ntm("eps", 3)
    return {
        "dividend_yield_ntm": gate(div(e.ntm("dps"), e.inputs.close), n_dps),
        "eps_ntm_1mth_chg": gate(div(now - month, month.abs()), n_eps),
        "eps_ntm_3mth_chg": gate(div(now - quarter, quarter.abs()), n_eps),
    }


def consensus_revision_factors(inputs: EstimateInputs) -> dict[str, pd.DataFrame]:
    """:data:`CONSENSUS_REVISION_FACTORS`: the 1- and 4-month consensus revisions, gated as breadth is.

    The 4-month figure weighs its months 0.4 / 0.3 / 0.2 / 0.1, as production's ``earnings_revisions_4_mth`` does.
    Not clipped: a revision over the price has no natural bound, and the z-scores QGARP reads are clipped at +/-3.
    """
    e = Estimates(inputs)
    n1, n2 = e.get("eps", "FY1", "estimates"), e.get("eps", "FY2", "estimates")
    b = [e.consensus_breadth(k) for k in range(4)]
    four = b[0] * 0.4 + b[1] * 0.3 + b[2] * 0.2 + b[3] * 0.1
    return {
        "consensus_revisions_1_mth": gate2(b[0], n1, n2),
        "consensus_revisions_4_mth": gate2(four, n1, n2),
    }


def _sample_z(panel: pd.DataFrame) -> pd.DataFrame:
    """Per date: (x - mean) / sample standard deviation over the row's values, clipped at +/-3."""
    z = panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1).replace(0.0, np.nan), axis=0)
    return z.clip(-3.0, 3.0)


def revisions_composite(inputs: EstimateInputs, members: pd.DataFrame) -> pd.DataFrame:
    """``zscore_earnings_revisions_4_mth_composite`` (8661-8928) among each date's members.

    Each of the last four months' breadth (missing where both FY1 and FY2 have fewer than two estimates) is a
    z-score clipped at +/-3; their 0.4 / 0.3 / 0.2 / 0.1 mix is standardised and clipped again. As coded, a member
    without FY1 and FY2 estimates at all six month ends (the composite reads six, uses four) has none.
    """
    e = Estimates(inputs)
    inside = members.reindex(index=inputs.dates, columns=inputs.listings).fillna(False).astype(bool)
    for k in range(6):
        for period in ("FY1", "FY2"):
            inside &= e.get("eps", period, "estimates", k).notna()
    rev = []
    for k in range(4):
        n1, n2 = e.get("eps", "FY1", "estimates", k), e.get("eps", "FY2", "estimates", k)
        rev.append(_sample_z(gate2(e.breadth(k), n1, n2).where(inside)))
    mix = rev[0] * 0.4 + rev[1] * 0.3 + rev[2] * 0.2 + rev[3] * 0.1
    return _sample_z(mix)
