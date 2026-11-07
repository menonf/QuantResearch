"""Raw substyles for a fundamental risk model, after BlackRock's BFRE (Appendix 8.3). Pure functions.

    raw = substyles(inputs)                      # {substyle: date x listing_id}, on the exposure dates
    exposures = risk_model.style_exposures(raw, inputs, STYLES)

A style is the average of a few standardised substyles - different measures of one theme (size from market value,
assets and sales) - which is steadier than any one of them. Each substyle here is signed so that a higher value means
more of the style (``amihud`` is minus the illiquidity ratio, ``balance_sheet_cash`` minus the cash share). Measures
BFRE takes from data we do not hold are left out: analyst estimates (broker coverage, predicted sales growth), foreign
sales and assets, and the macro betas (emerging markets, oil, the VIX).

Where the paper leaves a choice open, the one made here is said where it is made: weekly returns run Friday to
Friday; a return's excess over the risk-free rate is not taken (one rate for every share on a day moves only the
market factor); liquidity is measured in dollars traded against market value, so a split inside the window cannot
change it.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from equity_research.research.factors import dividend_yield, monthly_returns

#: ``style -> substyles``, each already signed so that higher means more of the style (BFRE's North America styles,
#: less what needs data we do not hold). Mid-Cap is not a combination: :func:`mid_cap` gives it directly.
STYLES: dict[str, tuple[str, ...]] = {
    "Size": ("log_market_value", "log_assets", "log_sales"),
    "Volatility": ("beta", "sigma", "beta_sigma", "range_12m", "volatility_1y"),
    "Momentum": ("relative_strength", "alpha"),
    "Reversal": ("reversal",),
    "Liquidity": ("turnover_3m", "turnover_6m", "turnover_12m", "amihud"),
    "Value": ("book_to_price", "sales_to_price", "cash_flow_to_price"),
    "Earnings Yield": ("earnings_to_price", "ebit_to_ev"),
    "Dividend Yield": ("dividend_yield",),
    "Profitability": (
        "return_on_equity",
        "return_on_assets",
        "interest_coverage",
        "operating_margin",
        "cash_flow_to_liabilities",
    ),
    "Growth": ("asset_growth", "sales_growth", "change_in_assets"),
    "Leverage": ("debt_to_assets", "market_leverage", "book_leverage", "balance_sheet_cash"),
}

#: One month of trading days, as BFRE counts it.
MONTH_DAYS = 22


@dataclass(frozen=True)
class Accounts:
    """Accounting items on the exposure dates, now and in each of the years before.

    ``years[k][item]`` is a ``date x listing_id`` panel as known ``k`` years before each date (``years[0]``: as known
    on the date). Flows are trailing twelve months.
    """

    years: Sequence[Mapping[str, pd.DataFrame]]

    def item(self, name: str, years_back: int = 0) -> pd.DataFrame:
        """``name`` as known ``years_back`` years before each date."""
        return self.years[years_back][name]


@dataclass(frozen=True)
class RiskInputs:
    """What the substyles and the regressions read; daily panels are ``date x listing_id`` over all trading days.

    ``dates`` are the exposure dates (month ends); ``listing_value`` and ``company_value`` are on them - a listing's
    own share of its company's market value (it weights the regressions) and the company's whole value (it measures
    size and prices the ratios). ``industry`` is each listing's industry (missing where none is known).
    """

    returns: pd.DataFrame
    adjusted: pd.DataFrame
    members: pd.DataFrame
    dollar_volume: pd.DataFrame
    dates: pd.DatetimeIndex
    listing_value: pd.DataFrame
    company_value: pd.DataFrame
    accounts: Accounts
    industry: pd.Series
    listing_issuer: pd.Series


# ---------------------------------------------------------------- market data


def cap_weighted_market(returns: pd.DataFrame, listing_value: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """The daily return of the members weighted by market value - BFRE's market index for betas.

    Each day uses the values of the latest exposure date strictly before it, so a day's own move does not weight it.
    """
    weights = listing_value.reindex(returns.index.union(listing_value.index)).ffill().shift(1).reindex(returns.index)
    weights = weights.reindex(columns=returns.columns).where(members.reindex_like(returns).fillna(False).astype(bool))
    weights = weights.where(returns.notna())
    total = weights.sum(axis=1)
    out: pd.Series = (weights * returns).sum(axis=1) / total.where(total > 0)
    return out


def weekly(returns: pd.DataFrame, min_days: int = 3) -> pd.DataFrame:
    """Daily returns compounded to weeks ending on Friday; a week with fewer than ``min_days`` returns is missing."""
    grown = (1.0 + returns.fillna(0.0)).resample("W-FRI").prod() - 1.0
    return grown.where(returns.notna().resample("W-FRI").sum() >= min_days)


def decay_weights(n: int, half_life: float) -> np.ndarray:
    """Weights for ``n`` observations, oldest first, halving every ``half_life`` observations back from the newest."""
    return np.power(0.5, np.arange(n - 1, -1, -1) / half_life)


def historical_beta(
    returns: pd.DataFrame,
    market: pd.Series,
    dates: pd.DatetimeIndex,
    *,
    weeks: int = 260,
    half_life: float = 52.0,
    min_weeks: int = 104,
) -> dict[str, pd.DataFrame]:
    """``{"beta", "sigma", "alpha"}``: each share's weekly returns regressed on the market's (equation 14).

    Up to ``weeks`` whole weeks ending on or before each date (the week holding a Wednesday month end would reach two
    days past it), weighted to halve every ``half_life`` weeks back; a share with fewer than ``min_weeks`` weeks has
    none. Sigma is the residuals' plain standard deviation, alpha the intercept (a weekly rate) - BFRE's Historical
    Alpha, a momentum substyle.
    """
    wr = weekly(returns)
    wm = weekly(market.to_frame("m"))["m"].reindex(wr.index)
    out = {k: pd.DataFrame(np.nan, index=dates, columns=returns.columns) for k in ("beta", "sigma", "alpha")}
    for d, hi in zip(dates, _rows_up_to(pd.DatetimeIndex(wr.index), dates), strict=True):
        lo = max(0, hi - weeks)
        y = wr.iloc[lo:hi].to_numpy(dtype=float)
        x = wm.iloc[lo:hi].to_numpy(dtype=float)[:, None]
        valid = ~np.isnan(y) & ~np.isnan(x)
        w = decay_weights(len(y), half_life)[:, None] * valid
        n = valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sw = w.sum(axis=0)
            xs, ys = np.where(valid, x, 0.0), np.where(valid, y, 0.0)
            mx, my = (w * xs).sum(axis=0) / sw, (w * ys).sum(axis=0) / sw
            beta = (w * (xs - mx) * (ys - my)).sum(axis=0) / (w * (xs - mx) ** 2).sum(axis=0)
            alpha = my - beta * mx
            resid = np.where(valid, ys - alpha - beta * xs, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a share with no weeks yet
                sigma = np.nanstd(resid, axis=0, ddof=1)
        enough = n >= min_weeks
        out["beta"].loc[d] = np.where(enough, beta, np.nan)
        out["sigma"].loc[d] = np.where(enough, sigma, np.nan)
        out["alpha"].loc[d] = np.where(enough, alpha, np.nan)
    return out


def _rows_up_to(index: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> np.ndarray:
    """For each date, one past the position of the last row on or before it."""
    return np.asarray(index.searchsorted(dates, side="right"))


def ew_volatility(
    returns: pd.DataFrame, dates: pd.DatetimeIndex, *, half_life: float = 180.0, window: int = 360, min_obs: int = 120
) -> pd.DataFrame:
    """The standard deviation of daily returns over ``window`` days, weighted to halve every ``half_life`` days."""
    out = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
    values = returns.to_numpy(dtype=float)
    for d, end in zip(dates, _rows_up_to(pd.DatetimeIndex(returns.index), dates), strict=True):
        y = values[max(0, end - window) : end]
        valid = ~np.isnan(y)
        w = decay_weights(len(y), half_life)[:, None] * valid
        ys = np.where(valid, y, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sw = w.sum(axis=0)
            mean = (w * ys).sum(axis=0) / sw
            var = (w * (ys - mean) ** 2).sum(axis=0) / sw
        out.loc[d] = np.where(valid.sum(axis=0) >= min_obs, np.sqrt(var), np.nan)
    return out


def cumulative_range(returns: pd.DataFrame, dates: pd.DatetimeIndex, months: int = 12) -> pd.DataFrame:
    """The highest less the lowest cumulative log return, summed back from each date over ``months`` (equation 15)."""
    monthly = monthly_returns(returns)
    logs = pd.DataFrame(np.log1p(monthly.to_numpy(dtype=float)), index=monthly.index, columns=monthly.columns)
    out = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
    periods = pd.PeriodIndex(logs.index)
    for d in dates:
        k = int(periods.searchsorted(pd.PeriodIndex([pd.Timestamp(d).to_period("M")]), side="right")[0])
        block = logs.iloc[max(0, k - months) : k].to_numpy(dtype=float)
        if len(block) == 0:
            continue
        back = np.nancumsum(block[::-1], axis=0)  # from the latest month backwards: s = 0 .. months - 1
        enough = (~np.isnan(block)).sum(axis=0) >= months * 3 // 4
        out.loc[d] = np.where(enough, back.max(axis=0) - back.min(axis=0), np.nan)
    return out


def log_return(
    returns: pd.DataFrame, dates: pd.DatetimeIndex, *, days: int, skip: int = 0, min_share: float = 2 / 3
) -> pd.DataFrame:
    """The log return over ``days`` trading days ending ``skip`` days before each date (missing days count as flat).

    Momentum is 11 months ending a month ago (equation 19), reversal the last month (equation 20).
    """
    logs = np.log1p(returns.to_numpy(dtype=float))
    out = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
    for d, end in zip(dates, _rows_up_to(pd.DatetimeIndex(returns.index), dates), strict=True):
        block = logs[max(0, end - skip - days) : max(0, end - skip)]
        if len(block) == 0:
            continue
        enough = (~np.isnan(block)).sum(axis=0) >= days * min_share
        out.loc[d] = np.where(enough, np.nansum(block, axis=0), np.nan)
    return out


def turnover(dollar_volume: pd.DataFrame, value: pd.DataFrame, *, days: int) -> pd.DataFrame:
    """``ln`` of the average daily dollars traded over ``days`` per dollar of market value on each date (equation 26).

    In dollars rather than shares, so a split inside the window changes neither side.
    """
    dates = pd.DatetimeIndex(value.index)
    mean = dollar_volume.rolling(days, min_periods=days // 2).mean()
    ratio = mean.reindex(dates).reindex(columns=value.columns) / value.where(value > 0)
    return pd.DataFrame(np.log(ratio.where(ratio > 0).to_numpy(dtype=float)), index=dates, columns=value.columns)


def amihud(
    returns: pd.DataFrame, dollar_volume: pd.DataFrame, dates: pd.DatetimeIndex, days: int = 360
) -> pd.DataFrame:
    """Minus ``ln`` of the average absolute daily return per dollar traded over ``days`` (equation 23, negated).

    A large move on little trading marks an illiquid share, so higher here means more liquid.
    """
    impact = returns.abs() / dollar_volume.where(dollar_volume > 0).reindex_like(returns)
    mean = impact.rolling(days, min_periods=days // 4).mean().reindex(dates)
    return pd.DataFrame(-np.log(mean.where(mean > 0).to_numpy(dtype=float)), index=dates, columns=returns.columns)


# ---------------------------------------------------------------- accounts


def _positive(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.where(panel > 0)


def _log(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.log(_positive(panel).to_numpy(dtype=float)), index=panel.index, columns=panel.columns)


def _mean_of(points: Sequence[pd.DataFrame], min_points: int = 3) -> pd.DataFrame:
    """The average of the yearly points each date has, if it has at least ``min_points``."""
    stack = np.stack([p.to_numpy(dtype=float) for p in points])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    enough = (~np.isnan(stack)).sum(axis=0) >= min_points
    return pd.DataFrame(np.where(enough, mean, np.nan), index=points[0].index, columns=points[0].columns)


def trend(points: Sequence[pd.DataFrame], min_points: int = 3) -> pd.DataFrame:
    """The yearly slope of a regression of the points on time, per unit of their average (equations 38 and 42).

    ``points[k]`` is the value ``k`` years back. Missing unless at least ``min_points`` years are known and the
    average is positive.
    """
    stack = np.stack([p.to_numpy(dtype=float) for p in points])
    t = -np.arange(len(points), dtype=float)[:, None, None] * np.ones_like(stack)
    valid = ~np.isnan(stack)
    n = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ys, ts = np.where(valid, stack, 0.0), np.where(valid, t, 0.0)
        my, mt = ys.sum(axis=0) / n, ts.sum(axis=0) / n
        slope = (valid * (ts - mt) * (ys - my)).sum(axis=0) / (valid * (ts - mt) ** 2).sum(axis=0)
        out = slope / my
    ok = (n >= min_points) & (my > 0)
    return pd.DataFrame(np.where(ok, out, np.nan), index=points[0].index, columns=points[0].columns)


def account_substyles(accounts: Accounts, company_value: pd.DataFrame, years: int = 5) -> dict[str, pd.DataFrame]:
    """The substyles built from the accounts (and the company's market value), on the exposure dates."""
    value = _positive(company_value)

    def item(name: str, k: int = 0) -> pd.DataFrame:
        return accounts.item(name, k).reindex(index=value.index, columns=value.columns)

    assets = [item("total_assets", k) for k in range(years)]
    sales = [item("revenue", k) for k in range(years)]
    cash_flows = [item("operating_cash_flow", k) for k in range(years)]
    equity, equity_before = item("equity"), item("equity", 1)
    debt, cash = item("total_debt"), item("cash")
    income, operating = item("net_income"), item("operating_income")
    average_equity = _positive((equity + equity_before) / 2)
    interest = item("interest_expense").abs()
    return {
        "log_assets": _log(_mean_of(assets)),
        "log_sales": _log(_mean_of(sales)),
        "book_to_price": equity / value,
        "sales_to_price": sales[0] / value,
        "cash_flow_to_price": cash_flows[0] / value,
        "earnings_to_price": income / value,
        "ebit_to_ev": item("ebit") / _positive(value + debt.fillna(0.0) - cash.fillna(0.0)),
        "return_on_equity": income / average_equity,
        "return_on_assets": income / _positive(assets[0]),
        "interest_coverage": operating / interest.where(interest > 0),
        "operating_margin": operating / _positive(sales[0]),
        "cash_flow_to_liabilities": _mean_of(cash_flows) / _positive(item("current_liabilities")),
        "asset_growth": trend(assets),
        "sales_growth": trend(sales),
        "change_in_assets": _log(assets[0]) - _log(assets[1]),
        "debt_to_assets": debt / _positive(assets[0]),
        "market_leverage": debt / value,
        "book_leverage": debt / _positive(equity),
        "balance_sheet_cash": -cash / _positive(assets[0]),
    }


# ---------------------------------------------------------------- all together


def mid_cap(company_value: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """BFRE's Mid-Cap exposure (equation 50): 1 around the 60th size percentile, falling away on either side.

    Ranks are by company value among the members, 1 the largest; the curve is centred at rank ``0.6 N`` and worth
    0.7 at rank ``0.54 N`` - deciles 6 and 7 in all. Not standardised: a mid-cap share has exposure near one.
    """
    value = company_value.where(members.reindex_like(company_value).fillna(False).astype(bool))
    rank = value.rank(axis=1, ascending=False)
    n = value.notna().sum(axis=1).to_numpy(dtype=float)[:, None]
    mu = np.log1p(0.60 * n)
    scale2 = -((np.log1p(0.54 * n) - mu) ** 2) / np.log(0.7)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.exp(-((np.log1p(rank.to_numpy(dtype=float)) - mu) ** 2) / scale2)
    return pd.DataFrame(out, index=value.index, columns=value.columns)


def substyles(inputs: RiskInputs) -> dict[str, pd.DataFrame]:
    """Every raw substyle :data:`STYLES` names, ``date x listing_id`` on ``inputs.dates``."""
    dates, returns = inputs.dates, inputs.returns
    market = cap_weighted_market(returns, inputs.listing_value, inputs.members)
    betas = historical_beta(returns, market, dates)
    value = inputs.company_value.reindex(columns=returns.columns)
    raw = {
        "log_market_value": _log(value),
        "beta": betas["beta"],
        "sigma": betas["sigma"],
        "beta_sigma": betas["beta"] * betas["sigma"],
        "alpha": betas["alpha"],
        "range_12m": cumulative_range(returns, dates),
        "volatility_1y": ew_volatility(returns, dates),
        "relative_strength": log_return(returns, dates, days=11 * MONTH_DAYS, skip=MONTH_DAYS),
        "reversal": log_return(returns, dates, days=MONTH_DAYS),
        "turnover_3m": turnover(inputs.dollar_volume, value, days=63),
        "turnover_6m": turnover(inputs.dollar_volume, value, days=126),
        "turnover_12m": turnover(inputs.dollar_volume, value, days=252),
        "amihud": amihud(returns, inputs.dollar_volume, dates),
        "dividend_yield": dividend_yield(returns, inputs.adjusted).reindex(dates),
        **account_substyles(inputs.accounts, value),
    }
    return {k: v.reindex(index=dates, columns=returns.columns) for k, v in raw.items()}
