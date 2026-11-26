"""Everything research needs of one bond index, read from the local research cache as monthly panels.

    data = load_index_data(cache, "H0A0")
    data.weights                # month end x bond: the weight set at that rebalance - held over the month that follows
    data.total_returns          # month end x bond: the return over the month ending that day
    data.analytics["oas_bp"]    # month end x bond: what was known of the bond on that day
    data.industry               # month end x bond: the provider's industry (level 3) of the bond's issuer that month
    data.until("2022-06-30")    # the index as a decision on that day could have read it

Timing, as the provider's files have it: the row of month end *t* is the index **after** its rebalance on *t*. So
``weights.loc[t]`` is what the index holds during month *t+1*, while ``total_returns.loc[t]`` is what each of those
bonds earned during month *t*. A bond that left at *t* is not in the row of *t*: its return over its last month is
missing, which :mod:`fixed_income_research.analytics.replication` measures instead of hiding.

Panels are ``date x listing_id`` (the bond's instrument id), fractions for weights and returns (the tables keep the
provider's percent), every other number in the provider's unit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from research_core.data.cache import ResearchCache

#: The per-bond measures read as panels, by their column in ``fixed_income.bond_analytics``.
MEASURES = (
    "price",
    "accrued_interest",
    "coupon_pct",
    "oas_bp",
    "asset_swap_spread_bp",
    "effective_yield_pct",
    "yield_to_worst_pct",
    "yield_to_maturity_pct",
    "effective_duration",
    "modified_duration_to_worst",
    "spread_duration",
    "effective_convexity",
)
#: ICE's composite ratings, best first: investment grade ends at BBB3, the high yield index runs from BB1 down.
RATINGS = (
    "AAA", "AA1", "AA2", "AA3", "A1", "A2", "A3", "BBB1", "BBB2", "BBB3",
    "BB1", "BB2", "BB3", "B1", "B2", "B3", "CCC1", "CCC2", "CCC3", "CC", "C", "D",
)  # fmt: skip
#: The broad rating bands, best first: a notch's letters, with everything below B together (CCC to D: distressed).
BANDS = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC and below")


@dataclass(frozen=True)
class IndexData:
    """One index's months: ``date x listing_id`` panels on its month ends, and who each bond is."""

    code: str
    weights: pd.DataFrame
    total_returns: pd.DataFrame
    excess_returns: pd.DataFrame
    analytics: dict[str, pd.DataFrame]
    rating: pd.DataFrame
    maturity: pd.DataFrame
    issuer: pd.DataFrame
    industry: pd.DataFrame
    names: pd.Series
    official: pd.Series  # the provider's own total-return level on each month end it is known for

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The month ends the index has a file for."""
        return pd.DatetimeIndex(self.weights.index)

    def until(self, end: str | pd.Timestamp) -> IndexData:
        """The index as known on ``end``: every panel cut there, so what is computed from it can use nothing later.

        A signal computed from this and from the whole history must agree on ``end``; if it does not, it reads the
        future (``fixed_income_research.strategies.factor_strategy.point_in_time_check``).
        """
        cut = pd.Timestamp(end)

        def upto(panel: pd.DataFrame) -> pd.DataFrame:
            return panel.loc[panel.index <= cut]

        return replace(
            self,
            weights=upto(self.weights),
            total_returns=upto(self.total_returns),
            excess_returns=upto(self.excess_returns),
            analytics={m: upto(p) for m, p in self.analytics.items()},
            rating=upto(self.rating),
            maturity=upto(self.maturity),
            issuer=upto(self.issuer),
            industry=upto(self.industry),
            official=self.official.loc[pd.DatetimeIndex(self.official.index) <= cut],
        )


def broad_rating(rating: pd.DataFrame) -> pd.DataFrame:
    """Each notch (``BB2``) as its band of :data:`BANDS` (``BB``); a rating the provider does not grade is missing."""
    band = {r: r.rstrip("123") for r in RATINGS[:16]} | dict.fromkeys(RATINGS[16:], BANDS[-1])
    return pd.DataFrame(rating.apply(lambda column: column.map(band)))


def load_index_data(
    cache: ResearchCache, code: str, start: str | None = None, end: str | None = None, *, vendor: str = "ice"
) -> IndexData:
    """The panels of index ``code`` over ``start..end`` (all its months by default), from ``vendor``'s files."""
    found = cache.query("SELECT index_id FROM fixed_income.index_definition WHERE code = ?", [code])
    if found.empty:
        raise KeyError(f"no bond index {code!r} in the cache: load its files and sync the cache")
    index_id = int(found["index_id"].iloc[0])
    window = ("AND c.as_of_date >= ? " if start else "") + ("AND c.as_of_date <= ? " if end else "")
    params: list[object] = [index_id, vendor, *([start] if start else []), *([end] if end else [])]
    rows = cache.query(
        "SELECT c.as_of_date, c.listing_id, c.weight_pct, a.issuer_id, a.rating, a.maturity_date, k.industry, "
        "a.total_return_mtd_pct, a.excess_return_mtd_pct, " + ", ".join(f"a.{m}" for m in MEASURES) + " "
        "FROM fixed_income.index_constituent c "
        "LEFT JOIN fixed_income.bond_analytics a ON a.listing_id = c.listing_id AND a.as_of_date = c.as_of_date "
        "AND a.vendor = c.vendor "
        # the issuer's industry that month, as the same file gave it
        "LEFT JOIN fixed_income.issuer_classification k ON k.issuer_id = a.issuer_id "
        "AND k.as_of_date = a.as_of_date AND k.vendor = a.vendor "
        f"WHERE c.index_id = ? AND c.vendor = ? {window}",
        params,
    )
    if rows.empty:
        raise KeyError(f"index {code!r} has no constituents in the cache for that period")
    rows["as_of_date"] = pd.to_datetime(rows["as_of_date"])

    def panel(column: str) -> pd.DataFrame:
        # one row per month end and bond: pivot, which refuses a duplicate, where pivot_table would average it
        return rows.pivot(index="as_of_date", columns="listing_id", values=column).sort_index()  # noqa: PD010

    names = cache.query("SELECT listing_id, name FROM core.listing WHERE instrument_type = 'bond'")
    levels = cache.query(
        "SELECT as_of_date, total_return_level FROM fixed_income.index_level WHERE index_id = ? ORDER BY as_of_date",
        [index_id],
    )
    official = pd.Series(
        levels["total_return_level"].to_numpy(dtype=float), index=pd.to_datetime(levels["as_of_date"]), name="official"
    )
    weights = panel("weight_pct") / 100.0
    return IndexData(
        code=code,
        weights=weights,
        total_returns=panel("total_return_mtd_pct") / 100.0,
        excess_returns=panel("excess_return_mtd_pct") / 100.0,
        analytics={m: panel(m) for m in MEASURES},
        rating=panel("rating"),
        maturity=panel("maturity_date"),
        issuer=panel("issuer_id"),
        industry=panel("industry"),
        names=names.set_index("listing_id")["name"].reindex(weights.columns),
        official=official,
    )
