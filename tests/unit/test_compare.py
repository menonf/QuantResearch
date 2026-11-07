import numpy as np
import pandas as pd

from equity_research.research.compare import agreement, pair_ratios
from equity_research.research.ratios import RATIOS

METRIC = next(iter(RATIOS))


def _panel(rows: dict[tuple[str, int], float]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([(pd.Timestamp(d), lid) for d, lid in rows], names=["period_end", "listing_id"])
    return pd.DataFrame({METRIC: list(rows.values())}, index=index)


def test_ratios_pair_with_the_nearest_period_end_within_tolerance() -> None:
    sec = _panel({("2024-03-31", 1): 10.0, ("2024-06-30", 1): 12.0})
    lseg = _panel({("2024-03-29", 1): 11.0, ("2024-09-30", 1): 13.0, ("2024-03-31", 2): 5.0})
    pairs = pair_ratios(sec, lseg)
    # only listing 1 in March: September has no SEC period within 7 days, listing 2 none at all
    assert pairs[["listing_id", "left", "right"]].to_numpy().tolist() == [[1, 10.0, 11.0]]


def test_agreement_measures_levels_and_ranks_per_quarter() -> None:
    rng = np.random.default_rng(0)
    right = rng.lognormal(size=30)
    pairs = pd.DataFrame(
        {
            "listing_id": range(30),
            "metric": METRIC,
            "period_end": pd.Timestamp("2024-03-31"),
            "left": right * 1.02,  # 2 % off everywhere, same order
            "right": right,
        }
    )
    row = agreement(pairs).loc[METRIC]
    assert row["pairs"] == 30
    assert row["within_5pct"] == 1.0
    assert abs(row["median_rel_diff"] - 0.02 / 1.02) < 1e-9
    assert row["rank_corr_by_quarter"] == 1.0
