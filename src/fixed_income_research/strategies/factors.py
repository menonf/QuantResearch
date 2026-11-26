"""Four factors of a bond index's members, month end by month end: value, momentum, size and low risk.

    years = years_to_maturity(data.maturity)
    cheap = oas_value(data.analytics["oas_bp"], years, data.rating, data.industry)
    mom = momentum(data.excess_returns)
    size = issuer_weight(data.weights, data.issuer)         # the strategy prefers the smaller issuers
    risk = riskiness(years, data.rating)                     # ... and the safer bonds
    blend = composite({"value": cheap, "momentum": mom}, {"value": 1, "momentum": 1}, members)

Credit factors in the manner of Houweling and van Zundert (*Factor Investing in the Corporate Bond Market*, 2017),
measured from the index provider's month-end files:

* **Value** - the bond's spread against the spread of bonds like it. Each month the option-adjusted spread is
  regressed across the members on the years to maturity, the change in the bond's spread over the last three months,
  and dummies for its rating notch and its issuer's industry; the signal is how far the bond's spread is above the
  fitted one, in percent of it. Above the fit is cheap.
* **Momentum** - the bond's excess return (over duration-matched government bonds) compounded over six months,
  ending a month before the decision (the latest month is skipped: a month's winners tend to give some of it back).
* **Size** - the issuer's total weight in the index, all its bonds together: smaller issuers are the factor's
  preference (less followed, a liquidity premium).
* **Low risk** - half the bond's maturity percentile among the members, half its rating's place on the scale: short,
  well-rated bonds are the safe ones, which the factor prefers (more return per unit of risk than the market pays).

Every function returns a ``date x bond`` panel on the index's month ends; the factors are oriented as measured
(:func:`composite` is given them oriented so that higher is better). Each uses only files dated on or before its date
(the tests delete the later months and check nothing moves). Months are counted on the calendar, not in files: a
month end without a file (April 2025 for H0A0) is a gap in a window, never a shorter window.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

#: Rating notches on the high-yield scale, safest first: the crossover BBB3 down to C (D, in default, is off it).
HIGH_YIELD = ("BBB3", "BB1", "BB2", "BB3", "B1", "B2", "B3", "CCC1", "CCC2", "CCC3", "CC", "C")
#: ... and on the investment-grade one (ICE's C0A0).
INVESTMENT_GRADE = ("AAA", "AA1", "AA2", "AA3", "A1", "A2", "A3", "BBB1", "BBB2", "BBB3")
RATING_SCALES = {"high_yield": HIGH_YIELD, "investment_grade": INVESTMENT_GRADE}


def calendar_shift(panel: pd.DataFrame, months: int) -> pd.DataFrame:
    """Each bond's value ``months`` calendar months earlier (later when negative), on ``panel``'s own dates.

    Missing where that month end has no file or the bond was not in it: a shift by rows would reach across a missing
    month end and compare two months that are further apart than they claim.
    """
    periods = pd.DatetimeIndex(panel.index).to_period("M")
    calendar = pd.period_range(periods.min(), periods.max(), freq="M")
    on_calendar = pd.DataFrame(panel.to_numpy(), index=periods, columns=panel.columns).reindex(calendar)
    shifted = on_calendar.shift(months).reindex(periods)
    return pd.DataFrame(shifted.to_numpy(), index=panel.index, columns=panel.columns)


def years_to_maturity(maturity: pd.DataFrame) -> pd.DataFrame:
    """Per month end and bond: the years until the bond matures (a quarter-day year)."""
    due = maturity.to_numpy(dtype="datetime64[ns]")
    today = pd.DatetimeIndex(maturity.index).to_numpy(dtype="datetime64[ns]")[:, None]
    years = (due - today) / np.timedelta64(1, "D") / 365.25
    return pd.DataFrame(years, index=maturity.index, columns=maturity.columns)


def cheapness(month: pd.DataFrame) -> pd.Series:
    """For one month end's bonds: how far each spread is above the fitted one, in percent of the fitted spread.

    ``month`` has the columns ``oas``, ``years``, ``change``, ``rating`` and ``industry`` (complete rows only). The fit
    is least squares on a constant, the two numbers and a dummy per rating notch and per industry. A bond whose
    fitted spread is not positive has no cheapness (a percentage of nothing), and a month with no more bonds than
    regressors has none at all.
    """
    x = pd.concat(
        [
            pd.Series(1.0, index=month.index, name="constant"),
            month[["years", "change"]].astype(float),
            pd.get_dummies(month["rating"], prefix="rating", dtype=float),
            pd.get_dummies(month["industry"], prefix="industry", dtype=float),
        ],
        axis=1,
    )
    if len(x) <= x.shape[1]:
        return pd.Series(np.nan, index=month.index)
    y = month["oas"].to_numpy(dtype=float)
    # each kind's dummies sum to the constant: the minimum-norm solution fits the same values as dropping one of them
    beta, *_ = np.linalg.lstsq(x.to_numpy(), y, rcond=None)
    fitted = x.to_numpy() @ beta
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(fitted > 0, (y - fitted) / fitted * 100.0, np.nan)
    return pd.Series(gap, index=month.index)


def oas_value(
    oas: pd.DataFrame,
    years: pd.DataFrame,
    rating: pd.DataFrame,
    industry: pd.DataFrame,
    *,
    change_months: int = 3,
) -> pd.DataFrame:
    """Per month end and bond: :func:`cheapness` against the bonds of the same month (higher is cheaper).

    The spread change is over ``change_months`` calendar months: a bond without a spread that long ago (newly issued)
    has none, so no value score that month.
    """
    change = oas - calendar_shift(oas, change_months)
    rows = {}
    for d in oas.index:
        month = pd.DataFrame(
            {
                "oas": oas.loc[d],
                "years": years.loc[d],
                "change": change.loc[d],
                "rating": rating.loc[d],
                "industry": industry.loc[d],
            }
        ).dropna()
        rows[d] = cheapness(month)
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(index=oas.index, columns=oas.columns)
    return out.astype(float)


def momentum(returns: pd.DataFrame, *, months: int = 6, skip: int = 1) -> pd.DataFrame:
    """Per month end and bond: its return compounded over ``months`` calendar months, ending ``skip`` months before.

    Every month of the window is needed: a bond that was out of the index for one of them (or a month end without a
    file) has no momentum. ``skip`` 1 leaves out the month just ended, which is known at the decision but tends to
    reverse.
    """
    periods = pd.DatetimeIndex(returns.index).to_period("M")
    calendar = pd.period_range(periods.min(), periods.max(), freq="M")
    logs = pd.DataFrame(np.log1p(returns.to_numpy(dtype=float)), index=periods, columns=returns.columns)
    window = logs.reindex(calendar).rolling(months, min_periods=months).sum().shift(skip).reindex(periods)
    return pd.DataFrame(np.expm1(window.to_numpy()), index=returns.index, columns=returns.columns)


def issuer_weight(weights: pd.DataFrame, issuer: pd.DataFrame) -> pd.DataFrame:
    """Per month end and bond: the index weight of its issuer, all the issuer's bonds in the index together.

    A bond without an issuer counts as its own (its own weight).
    """
    rows = {}
    for d in weights.index:
        w = weights.loc[d].dropna()
        who = issuer.loc[d].reindex(w.index).astype(object)
        alone = pd.Series([f"bond {c}" for c in w.index], index=w.index, dtype=object)
        rows[d] = w.groupby(who.where(who.notna(), alone)).transform("sum")
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(index=weights.index, columns=weights.columns)
    return out.astype(float)


def rating_position(rating: pd.DataFrame, scale: tuple[str, ...] = HIGH_YIELD) -> pd.DataFrame:
    """Each rating's place on ``scale``, from 1 / len(scale) (the safest notch) to 1 (the riskiest); off it missing."""
    place = {notch: (i + 1) / len(scale) for i, notch in enumerate(scale)}
    return pd.DataFrame(rating.apply(lambda column: column.map(place)), dtype=float)


def riskiness(
    years: pd.DataFrame,
    rating: pd.DataFrame,
    *,
    scale: tuple[str, ...] = HIGH_YIELD,
    maturity_share: float = 0.5,
) -> pd.DataFrame:
    """Per month end and bond, 0 safest to 1 riskiest: its maturity percentile and its rating's place, blended.

    ``maturity_share`` of the blend is the maturity's percentile among the month's bonds (longer is riskier), the
    rest :func:`rating_position`. A bond missing either part takes the month's median of that part, so it is scored
    on what is known of it rather than dropped.
    """
    present = years.notna() | rating.notna()
    longer = years.rank(axis=1, pct=True)
    worse = rating_position(rating, scale).where(present)
    parts = [p.T.fillna(p.median(axis=1)).T.where(present) for p in (longer, worse)]
    return maturity_share * parts[0] + (1 - maturity_share) * parts[1]


def percentile(score: pd.DataFrame) -> pd.DataFrame:
    """Per month end: each bond's percentile rank among the bonds scored (towards 1 the highest)."""
    return score.rank(axis=1, pct=True)


def composite(scores: Mapping[str, pd.DataFrame], weights: Mapping[str, float], members: pd.DataFrame) -> pd.DataFrame:
    """The weighted mean of the scores' percentiles, on the members; a member a score misses counts as average (0.5).

    ``scores`` must already be oriented so that higher is better. A bond is scored on everything known of it, so a
    new issue without three months of spreads is still ranked by the other factors rather than dropped.
    """
    on = members.astype(bool)
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("the composite needs a positive weight")
    blend = pd.DataFrame(0.0, index=members.index, columns=members.columns)
    for name, w in weights.items():
        ranked = percentile(scores[name].reindex_like(members).where(on))
        blend += w * ranked.fillna(0.5)
    return (blend / total).where(on)
