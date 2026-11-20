"""Two vendors' accounts, one set of formulas: do the ratios agree, period by period?

    sec = period_ratios(accounting_items(sec_facts), listing_issuer, close, factor)          # at each period end
    lseg = period_ratios(accounting_items(refinitiv_facts), listing_issuer, close, factor)
    pairs = pair_ratios(sec, lseg)
    agreement(pairs)          # per ratio: level and cross-sectional rank agreement

Both sides go through the same code, so a difference is a difference in the vendors' numbers (or in which version of
a restated figure they keep), never in the maths. Ratios are dated by fiscal period end and priced at that date here
only - research reads them at decision dates. The rank correlation within a quarter is what matters for a factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_research.data.ratios import ratio_panel

TOLERANCE_DAYS = 7


def period_ratios(
    items: pd.DataFrame, listing_issuer: pd.Series, close: pd.DataFrame, factor: pd.DataFrame
) -> pd.DataFrame:
    """Ratios of each issuer's listings at each of its period ends: ``(period_end, listing_id)`` x RATIOS."""
    keep = [c for c in items.columns if not c.startswith("known_at:") or c == "known_at:shares"]
    values = items[keep].reset_index()
    mapping = listing_issuer.dropna().astype(int).rename("issuer_id").rename_axis("listing_id").reset_index()
    fund = values.merge(mapping, on="issuer_id").assign(date=lambda d: d["period_end"])
    fund = fund.drop(columns=["issuer_id", "period_end"]).set_index(["date", "listing_id"]).sort_index()
    fund = fund[fund.index.get_level_values("listing_id").isin(close.columns)]
    out = ratio_panel(fund, close, factor)
    out.index = out.index.set_names(["period_end", "listing_id"])
    return out


def pair_ratios(left: pd.DataFrame, right: pd.DataFrame, tolerance_days: int = TOLERANCE_DAYS) -> pd.DataFrame:
    """``[listing_id, metric, period_end, left, right]``: each ``right`` ratio with ``left``'s nearest period end.

    Both are ``(period_end, listing_id)`` x ratio panels (:func:`period_ratios`); vendors date a quarter a few days
    apart now and then, hence the tolerance.
    """

    def long(panel: pd.DataFrame, name: str) -> pd.DataFrame:
        out = panel.reset_index().melt(id_vars=["period_end", "listing_id"], var_name="metric", value_name=name)
        out = out.dropna(subset=[name])
        out["period_end"] = pd.to_datetime(out["period_end"]).astype("datetime64[ns]")
        out["key"] = out["listing_id"].astype(int).astype(str) + "|" + out["metric"]
        return out

    a, b = long(left, "left"), long(right, "right")
    paired = pd.merge_asof(
        b.sort_values("period_end"),
        a[["key", "period_end", "left"]]
        .rename(columns={"period_end": "left_period_end"})
        .sort_values("left_period_end"),
        left_on="period_end",
        right_on="left_period_end",
        by="key",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    cols = ["listing_id", "metric", "period_end", "left", "right"]
    return paired.dropna(subset=["left"])[cols].reset_index(drop=True)


def agreement(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per ratio: pairs, median relative difference, share within 5 %, and the mean rank correlation per quarter."""
    rows = []
    for metric, g in pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=["left", "right"]).groupby("metric"):
        rel = (g["left"] - g["right"]).abs() / g[["left", "right"]].abs().max(axis=1).replace(0, np.nan)
        by_quarter = [
            q["left"].rank().corr(q["right"].rank())
            for _, q in g.groupby(pd.to_datetime(g["period_end"]).dt.to_period("Q"))
            if len(q) >= 20
        ]
        rows.append(
            {
                "ratio": metric,
                "pairs": len(g),
                "median_rel_diff": float(rel.median()),
                "within_5pct": float((rel <= 0.05).mean()),
                "rank_corr_by_quarter": float(np.nanmean(by_quarter)) if by_quarter else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("ratio").sort_values("rank_corr_by_quarter") if rows else pd.DataFrame()
