"""Barings' GEMS QGARP score, rebuilt from its production code on our point-in-time data.

    result = run_barings_garp(cache, BaringsGarpConfig(start="2010-01-01", end="2024-12-31"), engine=engine)

After Barings' ``PSA_Crisp_Metrics`` (the GEMS Factor Construction Library), as reconstructed from Quantlake in
``GEMS_Factor_Reconstruction.py``: each factor and composite keeps its production formula, quirks included. On each
month end, among the universe's members of the day:

1. **Raw factors** - the Worldscope-based ones from our accounts (:func:`account_factors`), the price-based ones
   from daily prices (:func:`price_factors`), the I/B/E/S ones from analysts' consensus history
   (:mod:`~equity_research.research.barings_estimates`). A factor whose inputs we do not hold yet
   (:data:`NOT_YET_BUILT`) is missing, never guessed.
2. **Cleaning** (:func:`clean`) - a value outside its factor's limits is set missing, not clipped (Barings'
   ``settings.metric_rules``, passed as ``rules``; without them nothing is removed) and financials are left out of
   factors flagged so.
3. **Composites** as coded (:func:`composites`): value, growth and quality (with its alpha and defensive halves) as
   percentile ranks of averaged ranks, by industry group; momentum as clipped z-scores of 1-12 month returns; then
   ``qgarp = -(0.2 alpha + 0.1 defensive + 0.3 growth + 0.3 value + 0.1 revisions)`` of each input's rank, a missing
   rank counting as 50.5 (the middle), and ``pct_rank_qgarp`` its percentile rank (1 = best). The weights are
   production's unless a run sets ``qgarp_weights`` - a variant, which may weigh any composite (momentum included).
4. **Book** - long the best ``enter`` fraction by QGARP, held while inside the best ``exit_`` (a buffer), rebalanced
   monthly, equally weighted or by market value.

What differs from production, by data rather than choice:

* **Our items for Worldscope's lines** (``accounting.py``): EBITDA is EBIT + depreciation; unusual items are one net
  figure (LSEG's, or the SEC's restructuring and impairment charges less gains on disposals) split into a charge and
  a credit by its sign; interest income is net interest income + interest expense; risk-weighted assets are tier-1
  capital over the tier-1 ratio where not reported. A company without a reported inventory holds none. LSEG reports
  share issues, buybacks, borrowing and discontinued operations not at all: where LSEG answers for a company (before
  the SEC's XBRL accounts), shareholder yield is zero as coded (every flow missing counts as zero).
* ``estimates=False`` runs the first version, before the I/B/E/S estimates were loaded (QGARP then had no growth and
  no revisions: quality and value only).
* **Accounts as known on the date** (dated by publication) - production reads a period as soon as it has ended.
  Trailing twelve months stand in for Worldscope's annual figures in the "annuals" factors, and its 1- and 3-year
  staleness guards compare our periods' ends.
* **Our items for Worldscope's**: market value from each share line's own count; long-term debt is total debt less
  debt due within a year; interest coverage divides by the interest expense's size (filers sign it either way);
  dividends per share come from the dividend record, and a payout from dividends over net income.
* **Industry**: Worldscope's general industry classification from GICS (:func:`worldscope_industry`) - banks,
  insurers, other financials (real estate included), utilities, transport, industrial for the rest.
* **Universe**: an S&P index's members, in US dollars (local and USD returns are the same); production ranks across
  its global ``qwd_global`` universe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from equity_research.research.backtest import BacktestResult, run_backtest
from equity_research.research.barings_estimates import (
    ESTIMATE_FACTORS,
    REVISIONS_COMPOSITE,
    EstimateInputs,
    estimate_factors,
    load_estimate_inputs,
    revisions_composite,
)
from equity_research.research.cache import ResearchCache
from equity_research.research.classification import classification_on
from equity_research.research.data import (
    UniverseData,
    accounts_as_of,
    company_values,
    in_shares_of,
    load_universe_data,
    own_counts,
    vendor_items,
)
from equity_research.research.factors import dividend_yield
from equity_research.research.gems_math import div as _div
from equity_research.research.gems_math import growth_piecewise
from equity_research.research.gems_math import rows_on as _on
from equity_research.research.panels import by_listing, latest_counts
from equity_research.research.portfolio import quantile_book, weighted_by
from equity_research.research.reads import (
    as_vendors,
    corporate_actions,
    financial_facts,
    listing_issuers,
    outstanding_counts,
)
from equity_research.research.registry import fence
from equity_research.research.strategy import WEIGHTINGS, finish_run
from equity_research.settings import vendor_preferences

# ---------------------------------------------------------------- what is built

CURRENT_FACTORS = [
    "asset_turnover",
    "book_yield",
    "cash_flow_conversion",
    "cash_flow_yield",
    "current_ratio",
    "debt_to_assets",
    "debt_to_equity",
    "dividend_growth_hist",
    "dividend_yield",
    "earnings_yield",
    "ebit_margin",
    "ebit_yield",
    "interest_coverage",
    "market_capitalization",
    "net_income_margin",
    "operating_margin",
    "payout_ratio",
    "sales_growth_hist",
    "sales_to_ev",
    "sales_yield",
    "short_term_debt_to_total_assets",
    "short_term_debt_to_total_debt",
    "ebitda_to_ev",
    "ebitda_to_price",
    "capex_to_depreciation",
    "quick_ratio",
    "capital_adequacy_ratio_tier_1_capital",
    "capital_adequacy_ratio_total_capital",
    "loans_to_deposits",
]
ANNUAL_FACTORS = [
    "cash_flow_growth_1_year_change",
    "cash_flow_growth_3_year_change",
    "operating_cash_flow_growth_1_year_change",
    "operating_cash_flow_growth_3_year_change",
    "free_cash_flow_yield",
    "free_cash_flow_conversion",
    "cash_flow_from_financing_to_total_assets",
    "extraordinary_credit_to_operating_income",
    "extraordinary_charge_to_operating_income",
    "unusual_items_to_operating_income",
    "shareholder_yield_growth_hist",
    "shareholder_yield_growth_3_year_hist",
    "net_income_to_risk_weighted_assets",
    "net_interest_income_to_net_loan",
    "interest_income_to_interest_expense",
    "non_interest_income_to_non_interest_expense",
    "loan_loss_provisions_to_net_loans",
    "non_performing_loans_to_equity_and_loan_loss_provision",
]
OVER_TIME_FACTORS = [
    "return_on_equity",
    "return_on_equity_5_year_vol",
    "return_on_invested_capital",
    "return_on_invested_capital_5_year_vol",
    "return_on_assets",
    "return_on_assets_1_year_change",
    "return_on_assets_5_year_vol",
    "accruals",
    "accruals_year_on_year_change",
    "sustainable_growth",
    "sustainable_growth_3_year_average",
    "eps_growth_1_year_hist",
    "eps_growth_3_year_hist_cagr",
    "eps_growth_5_year_hist_cagr",
    "earnings_yield_3_year_hist",
    "earnings_yield_5_year_hist",
    "net_income_margin_5_year_vol",
    "net_income_5_year_stability",
    "ebit_growth_3_year_hist_cagr",
    "ebit_growth_5_year_hist_cagr",
    "ebit_yield_3_year_hist",
    "ebit_yield_5_year_hist",
    "operating_leverage",
]
RETURN_HORIZONS = (1, 3, 6, 9, 12)
PRICE_FACTORS = [
    *(f"{m}_mth_price_return_{c}" for m in RETURN_HORIZONS for c in ("local", "usd")),
    "regression_momentum_1_year",
    "1_year_volatility",
    "price_return_stability",
]
ACCOUNT_FACTORS = CURRENT_FACTORS + ANNUAL_FACTORS + OVER_TIME_FACTORS
BUILT = ACCOUNT_FACTORS + PRICE_FACTORS + ESTIMATE_FACTORS + [REVISIONS_COMPOSITE]
COMPOSITES = [
    "pct_rank_value_composite",
    "pct_rank_value_composite_fins_neutral",
    "pct_rank_growth_composite",
    "pct_rank_quality_composite",
    "pct_rank_quality_alpha",
    "pct_rank_quality_defensive",
    "zscore_momentum_composite",
    REVISIONS_COMPOSITE,  # built with the factors (it needs the members of each month), passed through
    "pct_rank_qgarp",
]

#: The library's factors not built yet, with the data each needs (none is guessed from something else).
NOT_YET_BUILT: dict[str, str] = {
    "sales_revisions_1_mth": "revenue revision counts (only EPS revision counts are loaded)",
}

#: Accounting items read on each date and on each of the five yearly snapshots before it.
ITEMS_USED = (
    "net_income",
    "revenue",
    "ebit",
    "operating_income",
    "interest_expense",
    "operating_cash_flow",
    "capex",
    "total_assets",
    "equity",
    "total_debt",
    "current_debt",
    "cash",
    "current_assets",
    "current_liabilities",
)
#: Items only the "current" and "annual" factors read: on the date, and a year and three years before it (the
#: 1- and 3-year changes of shareholder yield).
EXTRA_ITEMS = (
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
EXTRA_SNAPSHOTS = (0, 1, 3)
SNAPSHOTS = 6  # the date and five years before it (five yearly ratios need six balances)
#: Price history before the first decision date: three years of monthly returns, and dividends two years back.
HISTORY_DAYS = 3 * 366 + 60


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True)
class GemsInputs:
    """What the factors read, on the decision dates (``date x listing_id`` panels).

    ``accounts[k]`` holds the :data:`ITEMS_USED` known ``k`` years before each date (and ``period_end:<item>``,
    in days);
    ``shares[k]`` the company's share count then, in the shares of the date (splits since adjust it);
    ``value[k]`` the company's market value then; ``dividend_yield[k]`` the trailing year's dividends over the price
    then; ``adjusted`` the daily split-adjusted closes (price factors); ``industry`` Worldscope's general industry;
    ``estimates`` the analysts' estimates (None for a run without them).
    """

    dates: pd.DatetimeIndex
    accounts: list[dict[str, pd.DataFrame]]
    shares: list[pd.DataFrame]
    value: list[pd.DataFrame]
    dividend_yield: list[pd.DataFrame]
    adjusted: pd.DataFrame
    industry: pd.DataFrame
    estimates: EstimateInputs | None = None


def _years_before(dates: pd.DatetimeIndex, years: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates - pd.DateOffset(years=years))


def accounts_years_before(
    items: Mapping[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    listing_issuer: pd.Series,
    listings: pd.Index,
    years: int,
    *,
    names: Sequence[str] = ITEMS_USED,
) -> dict[str, pd.DataFrame]:
    """``{item: date x listing_id}`` known ``years`` before each date (labelled with the date), with period ends.

    Like :func:`~equity_research.research.data.items_years_before`, plus ``period_end:<item>`` panels (in days since
    1970, so they compare and subtract as numbers): the "annual" factors are missing where two snapshots read the
    same period (production's staleness guards).
    """
    earlier = pd.Series(_years_before(dates, years), index=dates)
    known = by_listing(
        accounts_as_of(items, pd.DatetimeIndex(earlier.unique()), only=list(names)), listing_issuer
    ).reset_index()
    known = known[known["listing_id"].isin(listings)]
    label = pd.DataFrame({"earlier": earlier.to_numpy(), "label": earlier.index})
    known = known.merge(label, left_on="date", right_on="earlier").drop_duplicates(["label", "listing_id"])
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        for col in (name, f"period_end:{name}"):
            if col not in known.columns:
                out[col] = pd.DataFrame(np.nan, index=dates, columns=listings)
                continue
            values = known[col]
            if col.startswith("period_end:"):
                values = (pd.to_datetime(values) - pd.Timestamp(0)).dt.days.astype(float)
            wide = known.assign(_v=values).pivot(index="label", columns="listing_id", values="_v")  # noqa: PD010
            out[col] = wide.reindex(index=dates, columns=listings)
    return out


#: Worldscope's general industry classification, from a GICS industry group (else its sector).
GICS_INDUSTRY = {"Banks": "BANK", "Insurance": "INSURANCE", "Transportation": "TRANSPORTATION"}
GICS_SECTOR = {"Financials": "OTHER FINANCIAL", "Real Estate": "OTHER FINANCIAL", "Utilities": "UTILITY"}


def worldscope_industry(industry_group: pd.Series, sector: pd.Series) -> pd.Series:
    """BANK / INSURANCE / OTHER FINANCIAL / UTILITY / TRANSPORTATION / INDUSTRIAL; missing without a sector.

    The composites rank within these groups and treat financials apart. Worldscope files real-estate trusts under
    other financials; a company GICS knows only by sector takes the sector's group (a bank known only as
    "Financials" becomes an other financial).
    """
    by_group = industry_group.map(GICS_INDUSTRY)
    by_sector = sector.map(GICS_SECTOR)
    out = by_group.where(by_group.notna(), by_sector)
    return out.where(out.notna() | sector.isna(), "INDUSTRIAL")


def industry_panel(cache: ResearchCache, dates: pd.DatetimeIndex, listing_issuer: pd.Series) -> pd.DataFrame:
    """``date x listing_id``: each listing's Worldscope industry on each date (the company's GICS of the day)."""
    rows = {}
    for day in dates:
        gics = classification_on(cache, day)
        labels = worldscope_industry(gics["industry_group"], gics["sector"])
        rows[day] = listing_issuer.map(labels)
    return pd.DataFrame(rows).T.reindex(index=dates)


def load_gems_inputs(
    cache: ResearchCache,
    data: UniverseData,
    *,
    price_vendors: tuple[str, ...] | None = None,
    fundamentals_vendor: str | tuple[str, ...] | None = None,
    with_estimates: bool = True,
) -> GemsInputs:
    """The factors' inputs for ``data``'s listings on its rebalance dates (``data`` loaded with :data:`HISTORY_DAYS`).

    Vendors as everywhere in research: accounts from the first preferred vendor that answers for the company on each
    (earlier) date, in the universe's currency; share counts and corporate actions from the price vendors.
    """
    prefs = vendor_preferences()
    price_vendors = price_vendors or prefs.prices
    vendors = as_vendors(fundamentals_vendor or prefs.fundamentals)
    listings = data.close.columns
    dates = data.rebalances
    listing_issuer = listing_issuers(cache, listings)
    items = vendor_items(financial_facts(cache, listing_issuer, vendors), vendors, data.currency)
    actions = corporate_actions(cache, listings)
    actions = actions[actions["vendor"].isin(price_vendors)].drop(columns="vendor").reset_index(drop=True)
    counts = own_counts(outstanding_counts(cache, listings), price_vendors)
    yields = dividend_yield(data.returns, data.adjusted)
    accounts, shares, value, div = [], [], [], []
    for k in range(SNAPSHOTS):
        then = _years_before(dates, k)
        names = ITEMS_USED + EXTRA_ITEMS if k in EXTRA_SNAPSHOTS else ITEMS_USED
        accounts.append(accounts_years_before(items, dates, listing_issuer, listings, k, names=names))
        count_then = latest_counts(counts, actions, then, listings)
        shares.append(company_values(in_shares_of(count_then, dates, actions, years=k), listing_issuer))
        listed = _on(data.close, then) * count_then
        value.append(company_values(listed.set_axis(dates), listing_issuer))
        div.append(_on(yields, then).set_axis(dates))
    estimates = (
        load_estimate_inputs(
            cache,
            dates,
            data.close,
            actions,
            value=value[0],
            shares=shares[0],
            total_assets=accounts[0]["total_assets"],
        )
        if with_estimates
        else None
    )
    return GemsInputs(
        dates=dates,
        accounts=accounts,
        shares=shares,
        value=value,
        dividend_yield=div,
        adjusted=data.adjusted,
        industry=industry_panel(cache, dates, listing_issuer).reindex(columns=listings),
        estimates=estimates,
    )


# ---------------------------------------------------------------- raw factors: accounts


def _neg_pop_std(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Minus the population standard deviation of ``parts``, missing if any part is (Spark column arithmetic)."""
    mean = sum(parts, start=parts[0] * 0) / len(parts)
    var = sum(((x - mean) ** 2 for x in parts), start=parts[0] * 0) / len(parts)
    return pd.DataFrame(-np.sqrt(var.to_numpy(dtype=float)), index=mean.index, columns=mean.columns)


@dataclass(frozen=True)
class _Snapshots:
    """The accounts, counts and values of :class:`GemsInputs` with the lookups every factor group uses."""

    inputs: GemsInputs

    def a(self, item: str, k: int = 0) -> pd.DataFrame:
        """``item`` as known ``k`` years before each date."""
        return self.inputs.accounts[k][item]

    def ltd(self, k: int) -> pd.DataFrame:
        """Long-term debt: total less due within a year; a company that reports no debt has none (as in EV)."""
        return self.a("total_debt", k).fillna(0.0) - self.a("current_debt", k).fillna(0.0)

    def dividends(self, k: int) -> pd.DataFrame:
        """Dividends paid over the trailing year, ``k`` years before: the yield then times the value then."""
        return self.inputs.dividend_yield[k] * self.inputs.value[k]

    def dps(self, k: int) -> pd.DataFrame:
        """Dividends per share ``k`` years before, in split-adjusted shares (comparable across years)."""
        then = _years_before(self.inputs.dates, k)
        return self.inputs.dividend_yield[k] * _on(self.inputs.adjusted, then).set_axis(self.inputs.dates)

    def eps(self, k: int) -> pd.DataFrame:
        """Trailing earnings per share ``k`` years before, in the shares of the date (a split is not growth)."""
        return _div(self.a("net_income", k), self.inputs.shares[k])

    def roe(self, k: int) -> pd.DataFrame:
        """Net income over the average of the year's opening and closing equity."""
        return _div(self.a("net_income", k), (self.a("equity", k) + self.a("equity", k + 1)) / 2)


def current_factors(s: _Snapshots) -> dict[str, pd.DataFrame]:
    """``worldscope_raw_current_metrics`` (538-886): yields, margins, leverage and one-year growth on the date."""
    a, mcap = s.a, s.inputs.value[0]
    revenue, ni, ebit, cfo = a("revenue"), a("net_income"), a("ebit"), a("operating_cash_flow")
    ev = mcap + a("total_debt").fillna(0.0) - a("cash").fillna(0.0)
    ebitda = ebit + a("depreciation")
    # a company that reports no inventory holds none (Worldscope's 0): the quick ratio is then the current ratio
    inventory = a("inventory").where(a("inventory").notna() | a("current_assets").isna(), 0.0)
    return {
        "ebitda_to_ev": _div(ebitda, ev),
        "ebitda_to_price": _div(ebitda, mcap),
        "capex_to_depreciation": _div(a("capex"), a("depreciation")),
        "quick_ratio": _div(a("current_assets") - inventory, a("current_liabilities")),
        # pure numbers already (production divides Worldscope's percentages by 100)
        "capital_adequacy_ratio_tier_1_capital": a("tier1_ratio"),
        "capital_adequacy_ratio_total_capital": a("total_capital_ratio"),
        "loans_to_deposits": _div(-a("loans_net"), a("deposits")),
        "sales_yield": _div(revenue, mcap),
        "earnings_yield": _div(ni, mcap),
        "book_yield": _div(a("equity"), mcap),
        "dividend_yield": s.inputs.dividend_yield[0],
        "cash_flow_yield": _div(cfo, mcap),
        "debt_to_equity": _div(a("total_debt"), a("equity")),
        "sales_growth_hist": _div(revenue, a("revenue", 1)) - 1,
        "market_capitalization": mcap,
        "ebit_margin": _div(ebit, revenue),
        "net_income_margin": _div(ni, revenue),
        "payout_ratio": _div(s.dividends(0), ni),
        "sales_to_ev": _div(revenue, ev),
        "asset_turnover": _div(revenue, (a("total_assets") + a("total_assets", 1)) / 2),
        "cash_flow_conversion": _div(cfo, a("equity")),
        "current_ratio": _div(a("current_assets"), a("current_liabilities")),
        "debt_to_assets": _div(a("total_debt"), a("total_assets")),
        "ebit_yield": _div(ebit, mcap),
        "dividend_growth_hist": _div(s.dps(0), s.dps(1)) - 1,
        "interest_coverage": _div(ebit, a("interest_expense").abs()),
        "operating_margin": _div(a("operating_income"), revenue),
        "short_term_debt_to_total_assets": _div(a("current_debt"), a("total_assets")),
        "short_term_debt_to_total_debt": _div(a("current_debt"), a("total_debt")),
    }


def annual_factors(s: _Snapshots) -> dict[str, pd.DataFrame]:
    """``worldscope_raw_annuals_metrics`` (888-1470): profit and cash-flow changes, unusual items, the bank ratios.

    Also cash-flow quality and shareholder yield.
    A 1-year change is missing where both snapshots read the same period, a 3-year one where the periods end under
    700 days apart (production's staleness guards). Trailing twelve months stand in for Worldscope's fiscal years.
    """
    ta = [s.a("total_assets", k).where(s.a("total_assets", k) != 0) for k in range(4)]
    end = [s.a("period_end:net_income", k) for k in range(4)]
    same_year = end[0] == end[1]
    too_close = (end[0] - 700) < end[3]
    out = {}
    for name, item in (("cash_flow_growth", "net_income"), ("operating_cash_flow_growth", "operating_cash_flow")):
        per_asset = [s.a(item, k) / ta[k] for k in range(4)]
        out[f"{name}_1_year_change"] = (per_asset[0] - per_asset[1]).mask(same_year)
        out[f"{name}_3_year_change"] = (per_asset[0] - per_asset[3]).mask(too_close)
    a, mcap = s.a, s.inputs.value[0]
    free_cash_flow = a("net_income") + a("depreciation") - a("capex")
    operating = a("operating_income").abs()
    # Worldscope splits extraordinary items into credits and charges; ours are one net figure (expense positive)
    unusual = a("unusual_expense")
    charge, credit = unusual.clip(lower=0.0), (-unusual).clip(lower=0.0)
    is_bank = s.inputs.industry.reindex_like(ta[0]).isin(["BANK"])
    shareholder_yield = {k: _shareholder_yield(s, k, ta[k], is_bank) for k in (0, 1, 3)}
    equity = a("equity").where(a("equity") != 0)
    return out | {
        "free_cash_flow_yield": _div(free_cash_flow, mcap),
        "free_cash_flow_conversion": _div(free_cash_flow, equity),
        "cash_flow_from_financing_to_total_assets": _div(a("financing_cash_flow"), (ta[0] + ta[1]) / 2),
        "extraordinary_credit_to_operating_income": _div(credit.abs(), operating),
        "extraordinary_charge_to_operating_income": _div(charge.abs(), operating),
        "unusual_items_to_operating_income": _div((charge - credit).abs(), operating),
        "shareholder_yield_growth_hist": (shareholder_yield[0] - shareholder_yield[1]).mask(same_year),
        "shareholder_yield_growth_3_year_hist": (shareholder_yield[0] - shareholder_yield[3]).mask(too_close),
        # banks seldom report an operating income: their net income stands in (the factor's own name); a company that
        # reports no discontinued operations has none
        "net_income_to_risk_weighted_assets": _div(
            a("operating_income").where(a("operating_income").notna(), a("net_income"))
            - a("discontinued_operations").fillna(0.0),
            a("risk_weighted_assets"),
        ),
        "net_interest_income_to_net_loan": _div(a("net_interest_income"), a("loans_net")),
        "interest_income_to_interest_expense": _div(a("interest_income"), a("interest_expense").abs()),
        "non_interest_income_to_non_interest_expense": _div(a("noninterest_income"), a("noninterest_expense")),
        "loan_loss_provisions_to_net_loans": _div(-a("loan_loss_provision"), a("loans_net")),
        "non_performing_loans_to_equity_and_loan_loss_provision": _div(
            -a("nonperforming_loans"), equity + a("loan_loss_provision")
        ),
    }


def _shareholder_yield(s: _Snapshots, k: int, total_assets: pd.DataFrame, is_bank: pd.DataFrame) -> pd.DataFrame:
    """Cash returned to holders ``k`` years before, over total assets (``metrics.py`` 1286-1418, as coded).

    ``(dividends - (shares issued - repurchased) - [not banks: short-term borrowing + long-term borrowing - repaid])
    / |total assets|``, every flow counted as zero where it is not reported (production's ``fillna(0)``).
    """

    def z(item: str) -> pd.DataFrame:
        return s.a(item, k).fillna(0.0)

    debt = (z("st_debt_change") + z("lt_debt_issued") - z("lt_debt_repaid")).mask(is_bank, 0.0)
    returned = z("dividends_paid") - (z("stock_issued") - z("stock_repurchased")) - debt
    return returned / total_assets.abs()


def over_time_factors(s: _Snapshots) -> dict[str, pd.DataFrame]:
    """``worldscope_raw_over_time_metrics`` (2708-4221) with ``_over_time_ebit`` and ``_operating_leverage``.

    Yearly returns on the average of a balance and the one a year before; their volatility over five years (minus
    the population standard deviation); accruals; sustainable growth; EPS and EBIT growth with production's sign rules.
    """
    a, mcap = s.a, s.inputs.value[0]
    roe = [s.roe(k) for k in range(5)]
    roic = [
        _div(a("net_income", k), (a("equity", k) + s.ltd(k) + a("equity", k + 1) + s.ltd(k + 1)) / 2) for k in range(5)
    ]
    roa = [_div(a("net_income", k), (a("total_assets", k) + a("total_assets", k + 1)) / 2) for k in range(5)]
    accruals = [
        _div(
            a("net_income", k) - (a("operating_cash_flow", k) - a("capex", k)),
            (a("total_assets", k) + a("total_assets", k + 1)) / 2,
        )
        for k in range(2)
    ]
    sg = [roe[k] * (1 - _div(s.dividends(k), a("net_income", k))) for k in range(3)]
    eps = [s.eps(k) for k in range(SNAPSHOTS)]
    price = _div(mcap, s.inputs.shares[0])
    profits = [a("net_income", k) for k in range(5)]
    # as coded: deviations are measured from the size of the mean, not from the mean
    mean_profit = (sum(profits, start=profits[0] * 0) / 5).abs()
    spread = np.sqrt((sum(((x - mean_profit) ** 2 for x in profits), start=profits[0] * 0) / 5).to_numpy(dtype=float))

    def leverage(k: int) -> pd.DataFrame:
        e0, e1, s0, s1 = a("ebit", k), a("ebit", k + 1), a("revenue", k), a("revenue", k + 1)
        return _div(_div(e0 - e1, e1.abs()), _div(s0 - s1, s1.abs()))

    recent, before = leverage(0), leverage(1)
    return {
        "return_on_equity": roe[0],
        "return_on_equity_5_year_vol": _neg_pop_std(roe),
        "return_on_invested_capital": roic[0],
        "return_on_invested_capital_5_year_vol": _neg_pop_std(roic),
        "return_on_assets": roa[0],
        "return_on_assets_1_year_change": roa[0] - roa[1],
        "return_on_assets_5_year_vol": _neg_pop_std(roa),
        "accruals": accruals[0],
        "accruals_year_on_year_change": -(_div(accruals[0], accruals[1].abs()) - 1),
        "sustainable_growth": sg[0],
        "sustainable_growth_3_year_average": (sg[0] + sg[1] + sg[2]) / 3,
        "eps_growth_1_year_hist": growth_piecewise(eps[0], eps[1]),
        "eps_growth_3_year_hist_cagr": growth_piecewise(eps[0], eps[3], 3),
        "eps_growth_5_year_hist_cagr": growth_piecewise(eps[0], eps[5], 5),
        "earnings_yield_3_year_hist": _div(eps[3], price),
        "earnings_yield_5_year_hist": _div(eps[5], price),
        "net_income_margin_5_year_vol": _neg_pop_std([_div(a("net_income", k), a("revenue", k)) for k in range(5)]),
        "net_income_5_year_stability": _div(
            -pd.DataFrame(spread, index=mean_profit.index, columns=mean_profit.columns), mean_profit
        ),
        "ebit_growth_3_year_hist_cagr": growth_piecewise(a("ebit"), a("ebit", 3), 3),
        "ebit_growth_5_year_hist_cagr": growth_piecewise(a("ebit"), a("ebit", 5), 5),
        "ebit_yield_3_year_hist": _div(a("ebit", 3), mcap),
        "ebit_yield_5_year_hist": _div(a("ebit", 5), mcap),
        # the latest year's leverage where positive, else the year before's where positive
        "operating_leverage": recent.where(recent > 0, before.where(before > 0)),
    }


def account_factors(inputs: GemsInputs) -> dict[str, pd.DataFrame]:
    """The Worldscope-based factors our accounts can build (:data:`ACCOUNT_FACTORS`), formulas as production codes.

    A yield divides by the company's market value on the date; per-share amounts are in the shares of the date.
    """
    s = _Snapshots(inputs)
    built = current_factors(s) | annual_factors(s) | over_time_factors(s)
    return {name: built[name] for name in ACCOUNT_FACTORS}


# ---------------------------------------------------------------- raw factors: prices


def _month_end(day: pd.Timestamp, months_back: int) -> pd.Timestamp:
    """``day - relativedelta(months=n, day=31)``: the last calendar day of the month ``n`` months before."""
    return pd.Timestamp(day - pd.DateOffset(months=months_back)) + pd.offsets.MonthEnd(0)


def _day_number(day: pd.Timestamp) -> float:
    return float((day - pd.Timestamp(0)).days)


def _t_stats(log_prices: pd.DataFrame) -> pd.Series:
    """Per column: the t-statistic of the slope of the valid values on 0..n-1 (scipy's ``linregress``).

    As coded in production, the missing prices are dropped and x counts the ones left (x is not the week's position).
    """
    y = log_prices.to_numpy(dtype=float)
    valid = ~np.isnan(y)
    x = np.where(valid, np.cumsum(valid, axis=0) - 1, np.nan)
    n = valid.sum(axis=0).astype(float)
    with np.errstate(all="ignore"):  # sums over the valid values: a listing with none gives nan, not a warning
        mx, my = np.where(valid, x, 0.0).sum(axis=0) / n, np.where(valid, y, 0.0).sum(axis=0) / n
        dx, dy = np.where(valid, x - mx, 0.0), np.where(valid, y - my, 0.0)
        sxx, syy, sxy = (dx * dx).sum(axis=0) / n, (dy * dy).sum(axis=0) / n, (dx * dy).sum(axis=0) / n
        r_den = np.sqrt(sxx * syy)
        r = np.clip(np.where(r_den == 0, 0.0, sxy / np.where(r_den == 0, 1.0, r_den)), -1.0, 1.0)
        slope = sxy / sxx
        stderr = np.where(n == 2, 0.0, np.sqrt((1 - r**2) * syy / sxx / (n - 2)))
        t = slope / stderr
    return pd.Series(np.where(n < 2, np.nan, t), index=log_prices.columns)


def _returns_between_valid(prices: pd.DataFrame) -> pd.DataFrame:
    """Each valid price over the previous valid one, less 1 (production's rows exist only where a price does)."""
    previous = prices.ffill().shift(1)
    return (prices / previous - 1).where(prices.notna())


def price_factors(adjusted: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Price momentum and risk on each date (``worldscope_raw_price_metrics``, ``metrics.py`` 1472-1918).

    * returns over 1-12 months: the month's last price over the last price of the month ``m`` months before (each at
      most 30 days older than its month end), split-adjusted, without dividends;
    * ``regression_momentum_1_year``: the slope's t-statistic of log weekly prices over the year, divided by 100;
    * ``1_year_volatility``: the standard deviation of weekly returns times the root of the number of weeks;
    * ``price_return_stability``: minus the standard deviation of monthly returns in percent over three years.

    A listing without a price in the 30 days before the date has none of them (production's entity filter).
    """
    days = pd.DatetimeIndex(adjusted.index)
    months = days.to_period("M")
    day_number = (days - pd.Timestamp(0)).days.to_numpy(dtype=float)  # dates as numbers: they compare as floats
    stamp = pd.DataFrame(
        np.repeat(day_number[:, None], adjusted.shape[1], axis=1), index=days, columns=adjusted.columns
    )
    monthly = adjusted.groupby(months).last()
    observed = stamp.where(adjusted.notna()).groupby(months).last()  # the day of each month's last price
    iso = days.isocalendar()
    weeks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    position = {month: i for i, month in enumerate(monthly.index)}  # every month has trading days: no gaps

    rows: dict[str, dict[pd.Timestamp, pd.Series]] = {name: {} for name in PRICE_FACTORS}
    for day in dates:
        month = day.to_period("M")
        if month not in position:
            continue
        i = position[month]
        latest, latest_at = monthly.iloc[i], observed.iloc[i]
        alive = latest_at >= _day_number(day) - 30
        for m in RETURN_HORIZONS:
            ref_month = month - m
            if ref_month not in position:
                continue
            ref = ref_month.to_timestamp(how="end").normalize()
            j = position[ref_month]
            then = monthly.iloc[j].where(observed.iloc[j] >= _day_number(ref) - 30)
            ret = (latest / then - 1).where(alive)
            rows[f"{m}_mth_price_return_local"][day] = ret
            rows[f"{m}_mth_price_return_usd"][day] = ret
        year = (days >= _month_end(day, 12)) & (days <= day)
        weekly = adjusted[year].groupby(weeks[year]).last()
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = pd.DataFrame(np.log(weekly.to_numpy(dtype=float)), index=weekly.index, columns=weekly.columns)
        rows["regression_momentum_1_year"][day] = (_t_stats(logs) / 100).where(alive)
        weekly_returns = _returns_between_valid(weekly)
        rows["1_year_volatility"][day] = (weekly_returns.std() * np.sqrt(weekly.notna().sum())).where(alive)
        start = _month_end(day, 36)
        first = max(i - 36, 0)
        window = monthly.iloc[first : i + 1].where(observed.iloc[first : i + 1] >= _day_number(start))
        rows["price_return_stability"][day] = (-(_returns_between_valid(window) * 100).std()).where(alive)
    return {
        name: pd.DataFrame(by_day).T.reindex(index=dates, columns=adjusted.columns) for name, by_day in rows.items()
    }


# ---------------------------------------------------------------- one table of factors


def factor_table(panels: Mapping[str, pd.DataFrame], members: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    """``[date, listing_id, industry, <factor>...]``: one row per member on each date (the composites' population).

    Infinities become missing, as Spark's division by zero would have left them.
    """
    mask = members.to_numpy(dtype=bool)
    r, c = np.nonzero(mask)
    columns: dict[str, np.ndarray] = {
        "date": members.index.to_numpy()[r],
        "listing_id": members.columns.to_numpy()[c],
        "industry": industry.reindex_like(members).to_numpy(dtype=object)[r, c],
    }
    for name, panel in panels.items():  # gathered first: a hundred-odd factors inserted one by one fragment a frame
        values = panel.reindex(index=members.index, columns=members.columns).to_numpy(dtype=float)[r, c]
        columns[name] = np.where(np.isinf(values), np.nan, values)
    return pd.DataFrame(columns)


FINANCIALS = ("BANK", "INSURANCE", "OTHER FINANCIAL")


def clean(table: pd.DataFrame, rules: pd.DataFrame | None = None) -> pd.DataFrame:
    """``table`` with values outside each factor's limits set missing (``generate_clean_factors``).

    ``rules``: ``[barings_metric_name, lower_limit, upper_limit, exclude_financials]`` (Barings' ``metric_rules``;
    missing limits are not enforced). ``exclude_financials`` 1 leaves out banks, insurers and other financials, 2 banks
    and insurers only. Several rules for one factor keep a value any of them keeps. Without rules nothing changes.
    """
    if rules is None or rules.empty:
        return table
    out = table.copy()
    fin1 = out["industry"].isin(FINANCIALS).to_numpy()
    fin2 = out["industry"].isin(FINANCIALS[:2]).to_numpy()
    for name, its in rules.groupby("barings_metric_name"):
        if name not in out.columns:
            continue
        x = out[name].to_numpy(dtype=float)
        keep = np.zeros(len(x), dtype=bool)
        for lo, hi, ex in zip(its["lower_limit"], its["upper_limit"], its["exclude_financials"], strict=True):
            with np.errstate(invalid="ignore"):
                drop = np.zeros(len(x), dtype=bool)
                if pd.notna(lo):
                    drop |= x < float(lo)
                if pd.notna(hi):
                    drop |= x > float(hi)
            if pd.notna(ex) and float(ex) == 1:
                drop |= fin1
            if pd.notna(ex) and float(ex) == 2:
                drop |= fin2
            keep |= ~drop
        out[name] = np.where(keep, x, np.nan)
    return out


# ---------------------------------------------------------------- composites (metrics.py 6919-9100, as coded)

_DD = "date"


def _pct_rank(df: pd.DataFrame, col: str, keys: list[str], ascending: bool = False) -> np.ndarray:
    """Production's percentile rank: ``ceil(rank(pct) * 100)`` within ``keys``; 1 = the highest value by default."""
    r = df.groupby(keys)[col].rank(pct=True, ascending=ascending)
    inner = np.where(r == 0, 1, np.ceil(r * 100))
    return np.where(df[col].notna(), inner, np.nan).astype(float)


def _count(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.asarray(df[cols].notna().sum(axis=1).to_numpy())


def _neg_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return -df[cols].mean(axis=1)


def _pick(group: pd.Series, by_group: Mapping[int, pd.Series | np.ndarray]) -> np.ndarray:
    out = np.full(len(group), np.nan)
    g = group.to_numpy()
    for k, values in by_group.items():
        out = np.where(g == k, np.asarray(values, dtype=float), out)
    return out


def _groups(industry: pd.Series, mapping: Mapping[str, int]) -> pd.Series:
    return industry.map(mapping).astype(float)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """A factor column, all missing where it is not built yet (production would need it; it is null here)."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


VALUE_INPUTS = [
    "earnings_yield_ntm",
    "earnings_yield",
    "book_yield",
    "cash_flow_yield",
    "free_cash_flow_yield",
    "sales_yield",
    "ebitda_to_ev",
]


def value_composite(table: pd.DataFrame) -> pd.DataFrame:
    """``pct_rank_value_composite`` and its financials-neutral version (``pct_rank_value_composite``, 6919-7236).

    Financials score on earnings (NTM, else trailing) and book yield; the others on six yields (four needed).
    """
    df = pd.DataFrame({_DD: table[_DD], **{c: _column(table, c) for c in VALUE_INPUTS}})
    df["industry_group"] = _groups(
        table["industry"], {**dict.fromkeys(FINANCIALS, 1), "INDUSTRIAL": 2, "UTILITY": 2, "TRANSPORTATION": 2}
    )
    df["earnings_yield_combined"] = df["earnings_yield_ntm"].where(
        df["earnings_yield_ntm"].notna(), df["earnings_yield"]
    )
    for f in ("cash_flow_yield", "free_cash_flow_yield", "sales_yield", "ebitda_to_ev"):
        df[f"{f}_ex_fins"] = df[f].where(df["industry_group"] != 1)
    factors = [
        "earnings_yield_combined",
        "book_yield",
        "cash_flow_yield_ex_fins",
        "free_cash_flow_yield_ex_fins",
        "sales_yield_ex_fins",
        "ebitda_to_ev_ex_fins",
    ]
    for f in factors:
        df[f"r_{f}"] = _pct_rank(df, f, [_DD])
    fin_cols = ["r_earnings_yield_combined", "r_book_yield"]
    all_cols = [f"r_{f}" for f in factors]
    df["fin"] = np.where(_count(df, fin_cols) > 0, _neg_mean(df, fin_cols), np.nan)
    df["non_fin"] = np.where(_count(df, all_cols) > 3, _neg_mean(df, all_cols), np.nan)
    for f in ("fin", "non_fin"):
        df[f"r_{f}"] = _pct_rank(df, f, [_DD])
        r_all = df.groupby([_DD])[f].rank(pct=True, ascending=False)
        r_grp = df.groupby([_DD, "industry_group"])[f].rank(pct=True, ascending=False)
        # as coded: the ==0 test reads the whole day's rank, the value the industry group's
        df[f"n_{f}"] = np.where(df[f].notna(), np.where(r_all == 0, 1, np.ceil(r_grp * 100)), np.nan)
    g = df["industry_group"]
    return pd.DataFrame(
        {
            "pct_rank_value_composite": _pick(g, {1: df["r_fin"], 2: df["r_non_fin"]}),
            "pct_rank_value_composite_fins_neutral": _pick(g, {1: df["n_fin"], 2: df["n_non_fin"]}),
        },
        index=table.index,
    )


GROWTH_INPUTS = [
    "cash_flow_growth_ltm_to_ntm",
    "cash_flow_growth_1_year_change",
    "cash_flow_growth_3_year_change",
    "eps_growth_ltm_to_ntm",
    "eps_growth_1_year_hist",
    "eps_growth_3_year_hist_cagr",
    "shareholder_yield_growth_hist",
    "shareholder_yield_growth_3_year_hist",
    "sustainable_growth_ntm",
    "sustainable_growth",
    "sustainable_growth_3_year_average",
]


def growth_composite(table: pd.DataFrame) -> pd.DataFrame:
    """``pct_rank_growth_composite`` (``pct_rank_growth_composite``, 7239-7578).

    Ranks within banks & insurers, and the rest; cash-flow, EPS, shareholder-yield and sustainable growth scores.
    The sustainable-growth score needs the NTM term (as coded, even its two-term fallback), so without estimates the
    composite cannot form for anyone: banks need two of three scores, the rest three of four.
    """
    df = pd.DataFrame({_DD: table[_DD], **{c: _column(table, c) for c in GROWTH_INPUTS}})
    df["industry_group"] = _groups(
        table["industry"],
        {"BANK": 1, "INSURANCE": 1, "INDUSTRIAL": 2, "UTILITY": 2, "TRANSPORTATION": 2, "OTHER FINANCIAL": 2},
    )
    keys = [_DD, "industry_group"]
    for f in GROWTH_INPUTS:
        df[f"r_{f}"] = _pct_rank(df, f, keys)

    def r(*names: str) -> list[str]:
        return [f"r_{n}" for n in names]

    df["cf"] = _neg_mean(
        df, r("cash_flow_growth_ltm_to_ntm", "cash_flow_growth_1_year_change", "cash_flow_growth_3_year_change")
    )
    df["eps"] = _neg_mean(df, r("eps_growth_ltm_to_ntm", "eps_growth_1_year_hist", "eps_growth_3_year_hist_cagr"))
    df["sy"] = _neg_mean(df, r("shareholder_yield_growth_hist", "shareholder_yield_growth_3_year_hist"))
    ntm, ltm, avg = (
        df["r_sustainable_growth_ntm"],
        df["r_sustainable_growth"],
        df["r_sustainable_growth_3_year_average"],
    )
    three, two = ntm * 0.33 + ltm * 0.33 + avg * 0.33, ntm * 0.33 + ltm * 0.33
    df["sgr"] = np.where(three.notna(), -three, -two)
    for f in ("cf", "eps", "sy", "sgr"):
        df[f"r_{f}"] = _pct_rank(df, f, keys)
    bi, non_bi = r("eps", "sy", "sgr"), r("cf", "eps", "sy", "sgr")
    df["bi"] = np.where(_count(df, bi) > 1, _neg_mean(df, bi), np.nan)
    df["non_bi"] = np.where(_count(df, non_bi) > 2, _neg_mean(df, non_bi), np.nan)
    for f in ("bi", "non_bi"):
        df[f"r_{f}"] = _pct_rank(df, f, keys)
    return pd.DataFrame(
        {"pct_rank_growth_composite": _pick(df["industry_group"], {1: df["r_bi"], 2: df["r_non_bi"]})},
        index=table.index,
    )


QUALITY_INPUTS = [
    "return_on_equity",
    "return_on_equity_5_year_vol",
    "return_on_assets",
    "return_on_assets_5_year_vol",
    "return_on_invested_capital",
    "return_on_invested_capital_5_year_vol",
    "net_income_margin",
    "net_income_margin_5_year_vol",
    "free_cash_flow_conversion",
    "unusual_items_to_operating_income",
    "cash_flow_from_financing_to_total_assets",
    "debt_to_assets",
    "interest_coverage",
    "short_term_debt_to_total_debt",
    "eps_uncertainty",
    "net_income_5_year_stability",
    "interest_income_to_interest_expense",
    "loan_loss_provisions_to_net_loans",
    "loans_to_deposits",
    "net_income_to_risk_weighted_assets",
    "net_interest_income_to_net_loan",
    "non_interest_income_to_non_interest_expense",
    "non_performing_loans_to_equity_and_loan_loss_provision",
    "capital_adequacy_ratio_tier_1_capital",
    "capital_adequacy_ratio_total_capital",
    "accruals",
    "accruals_year_on_year_change",
]


def quality_composite(table: pd.DataFrame) -> pd.DataFrame:
    """``pct_rank_quality_composite`` with its alpha and defensive halves (``pct_rank_quality_composite``, 7581-8491).

    Banks, insurers & other financials, and the rest, each scored on its own inputs; as coded, the volatility, loan
    loss, loans-to-deposits, non-performing-loan and accruals-change inputs are ranked high-is-good (not negated).
    """
    df = pd.DataFrame({_DD: table[_DD], **{c: _column(table, c) for c in QUALITY_INPUTS}})
    df["industry_group"] = _groups(
        table["industry"],
        {"BANK": 1, "INSURANCE": 2, "OTHER FINANCIAL": 2, "INDUSTRIAL": 3, "UTILITY": 3, "TRANSPORTATION": 3},
    )
    keys = [_DD, "industry_group"]
    for f in (
        "unusual_items_to_operating_income",
        "eps_uncertainty",
        "cash_flow_from_financing_to_total_assets",
        "debt_to_assets",
        "accruals",
    ):
        df[f] = -df[f]
    df["interest_coverage"] = np.where(df["debt_to_assets"] == 0, 1000, df["interest_coverage"])
    df["short_term_debt_to_total_debt"] = np.where(df["debt_to_assets"] == 0, 0, -df["short_term_debt_to_total_debt"])

    def rank(*names: str) -> None:
        for f in names:
            df[f"r_{f}"] = _pct_rank(df, f, keys)

    def r(*names: str) -> list[str]:
        return [f"r_{n}" for n in names]

    rank(*QUALITY_INPUTS)
    bank_alpha = r(
        "net_income_to_risk_weighted_assets",
        "net_interest_income_to_net_loan",
        "non_interest_income_to_non_interest_expense",
        "capital_adequacy_ratio_tier_1_capital",
        "capital_adequacy_ratio_total_capital",
    )
    bank_defensive = r(
        "interest_income_to_interest_expense",
        "loan_loss_provisions_to_net_loans",
        "loans_to_deposits",
        "non_performing_loans_to_equity_and_loan_loss_provision",
    )
    df["banks_quality_alpha_score"] = _neg_mean(df, bank_alpha)
    df["banks_quality_defensive_score"] = _neg_mean(df, bank_defensive)
    df["stable_prof_score"] = _neg_mean(
        df,
        r(
            "return_on_equity_5_year_vol",
            "return_on_assets_5_year_vol",
            "return_on_invested_capital_5_year_vol",
            "net_income_margin_5_year_vol",
            "net_income_5_year_stability",
        ),
    )
    rank("banks_quality_alpha_score", "banks_quality_defensive_score", "stable_prof_score")
    df["stable_prod_eps_uncertainty_score"] = _neg_mean(df, r("stable_prof_score", "eps_uncertainty"))
    df["debt_quality_score"] = _neg_mean(df, r("debt_to_assets", "interest_coverage", "short_term_debt_to_total_debt"))
    profitability = r(
        "return_on_equity",
        "return_on_assets",
        "return_on_invested_capital",
        "net_income_margin",
        "unusual_items_to_operating_income",
    )
    df["profitability_score"] = _neg_mean(df, profitability)
    df["cashflow_quality_score"] = _neg_mean(
        df, r("free_cash_flow_conversion", "cash_flow_from_financing_to_total_assets")
    )
    df["accruals_quality_score"] = _neg_mean(df, r("accruals", "accruals_year_on_year_change"))
    rank(
        "stable_prod_eps_uncertainty_score",
        "debt_quality_score",
        "profitability_score",
        "cashflow_quality_score",
        "accruals_quality_score",
    )
    banks_p1 = [*profitability, *r("free_cash_flow_conversion")]
    fin_alpha = [*profitability, *r("cash_flow_from_financing_to_total_assets")]
    df["banks_only_quality_alpha_score_p1"] = df[banks_p1].mean(axis=1)  # not negated, as coded
    df["banks_only_quality_alpha_score"] = _neg_mean(
        df, ["banks_only_quality_alpha_score_p1", *r("banks_quality_alpha_score")]
    )
    df["financials_quality_alpha_score"] = _neg_mean(df, fin_alpha)
    df["non_financials_quality_alpha_score"] = _neg_mean(
        df, r("profitability_score", "cashflow_quality_score", "accruals_quality_score")
    )
    defensive_pair = r("stable_prod_eps_uncertainty_score", "debt_quality_score")
    df["banks_only_quality_defensive_score"] = _neg_mean(
        df, r("stable_prod_eps_uncertainty_score", "banks_quality_defensive_score")
    )
    df["financials_quality_defensive_score"] = _neg_mean(df, defensive_pair)
    df["non_financials_quality_defensive_score"] = _neg_mean(df, defensive_pair)
    halves = [
        "banks_only_quality_alpha_score",
        "banks_only_quality_defensive_score",
        "financials_quality_alpha_score",
        "financials_quality_defensive_score",
        "non_financials_quality_alpha_score",
        "non_financials_quality_defensive_score",
    ]
    rank(*halves)
    df["banks_quality_composite"] = np.where(
        (_count(df, banks_p1) > 0)
        & (_count(df, bank_alpha) > 0)
        & (_count(df, bank_defensive) > 0)
        & (_count(df, r("stable_prof_score", "eps_uncertainty")) > 0),
        _neg_mean(df, r("banks_only_quality_alpha_score", "banks_only_quality_defensive_score")),
        np.nan,
    )
    df["financials_quality_composite"] = np.where(
        (_count(df, fin_alpha) > 1) & (_count(df, defensive_pair) > 0),
        _neg_mean(df, r("financials_quality_alpha_score", "financials_quality_defensive_score")),
        np.nan,
    )
    df["non_financials_quality_composite"] = np.where(
        (_count(df, r("profitability_score", "cashflow_quality_score", "accruals_quality_score")) > 1)
        & (_count(df, defensive_pair) > 0),
        _neg_mean(df, r("non_financials_quality_alpha_score", "non_financials_quality_defensive_score")),
        np.nan,
    )
    rank("banks_quality_composite", "financials_quality_composite", "non_financials_quality_composite")
    g = df["industry_group"]
    comp = _pick(
        g,
        {
            1: df["r_banks_quality_composite"],
            2: df["r_financials_quality_composite"],
            3: df["r_non_financials_quality_composite"],
        },
    )
    alpha = _pick(
        g,
        {
            1: df["r_banks_only_quality_alpha_score"],
            2: df["r_financials_quality_alpha_score"],
            3: df["r_non_financials_quality_alpha_score"],
        },
    )
    defensive = _pick(
        g,
        {
            1: df["r_banks_only_quality_defensive_score"],
            2: df["r_financials_quality_defensive_score"],
            3: df["r_non_financials_quality_defensive_score"],
        },
    )
    return pd.DataFrame(
        {
            "pct_rank_quality_composite": comp,
            "pct_rank_quality_alpha": np.where(np.isnan(comp), np.nan, alpha),
            "pct_rank_quality_defensive": np.where(np.isnan(comp), np.nan, defensive),
        },
        index=table.index,
    )


def population_z(table: pd.DataFrame, col: str) -> pd.Series:
    """``(x - mean) / sqrt(mean(x^2) - mean^2)`` per date (Spark's averages: the population standard deviation)."""
    x = _column(table, col)
    by_day = x.groupby(table[_DD])
    mean = by_day.transform("mean")
    sd = pd.Series(np.sqrt((x * x).groupby(table[_DD]).transform("mean") - mean * mean), index=x.index)
    return pd.Series((x - mean) / sd.where(sd > 0), index=x.index)


def _clip3(z: pd.Series) -> pd.Series:
    return z.clip(-3, 3)


def _sample_z(x: pd.Series, by: pd.Series) -> pd.Series:
    g = x.groupby(by)
    return pd.Series((x - g.transform("mean")) / g.transform("std"), index=x.index)


def momentum_composite(table: pd.DataFrame) -> pd.Series:
    """``zscore_momentum_composite`` (8494-8658): the 1-12 month returns combined.

    Each return's population z-score is standardised again (sample standard deviation) and clipped at +/-3; the
    average of those, where three or more exist, is standardised and clipped the same way.
    """
    limited = pd.DataFrame(
        {h: _clip3(_sample_z(population_z(table, f"{h}_mth_price_return_local"), table[_DD])) for h in RETURN_HORIZONS}
    )
    average = limited.mean(axis=1).where(limited.notna().sum(axis=1) > 2)
    return _clip3(_sample_z(average, table[_DD])).rename("zscore_momentum_composite")


REVISIONS_1_MTH = "zscore_earnings_revisions_1_mth"
#: QGARP's inputs and weights as coded in production.
QGARP_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("pct_rank_quality_alpha", 0.2),
    ("pct_rank_quality_defensive", 0.1),
    ("pct_rank_growth_composite", 0.3),
    ("pct_rank_value_composite", 0.3),
    (REVISIONS_1_MTH, 0.1),
)
#: What QGARP may weigh: production's inputs first (in production's order, so its sums are unchanged), then the
#: other composites - momentum among them - for variants.
QGARP_INPUTS = tuple(dict.fromkeys([name for name, _ in QGARP_WEIGHTS] + COMPOSITES[:-1]))

QgarpWeights = Mapping[str, float] | Sequence[tuple[str, float]]


def qgarp_weights(weights: QgarpWeights) -> tuple[tuple[str, float], ...]:
    """``weights`` as ``((input, weight), ...)`` in :data:`QGARP_INPUTS` order, zero weights dropped.

    One spelling per set of weights, so the same weights always hash to the same registered config. Only their ratios
    matter (the weighted sum is ranked), so they need not add up to one.
    """
    pairs = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    given = dict(pairs)
    if len(given) < len(pairs):
        raise ValueError("an input is weighted twice")
    unknown = sorted(set(given) - set(QGARP_INPUTS))
    if unknown:
        raise ValueError(f"unknown QGARP inputs {unknown}; choose from {list(QGARP_INPUTS)}")
    if any(w < 0 for w in given.values()) or not any(w > 0 for w in given.values()):
        raise ValueError("QGARP weights must be zero or more, and at least one above zero")
    return tuple((name, float(given[name])) for name in QGARP_INPUTS if given.get(name, 0.0) > 0)


def qgarp(
    comp: pd.DataFrame,
    dates: pd.Series,
    *,
    weights: QgarpWeights = QGARP_WEIGHTS,
    fix_revisions_direction: bool = False,
) -> pd.Series:
    """``pct_rank_qgarp`` (8931-9100): 1 = best.

    Each input is re-ranked so 1 = best - a percentile rank ascending (1 = best already), a z-score (higher = better)
    descending - a missing rank counts as 50.5, and the weighted sum is ranked again. As coded, the 1-month revisions
    z-score is ranked ascending too, which rewards the worst revisions; ``fix_revisions_direction`` ranks it descending
    like the other z-scores (not production). ``weights`` defaults to production's (:data:`QGARP_WEIGHTS`).
    """
    df = pd.DataFrame({_DD: dates})
    total = pd.Series(0.0, index=comp.index)
    for name, weight in qgarp_weights(weights):
        df[name] = _column(comp, name)
        if name.startswith("pct_rank_"):
            ascending = True
        elif name == REVISIONS_1_MTH:
            ascending = not fix_revisions_direction
        else:
            ascending = False
        ranked = pd.Series(_pct_rank(df, name, [_DD], ascending=ascending), index=comp.index)
        total += weight * ranked.fillna(50.5)
    df["qgarp"] = -total
    return pd.Series(_pct_rank(df, "qgarp", [_DD]), index=comp.index, name="pct_rank_qgarp")


def composites(
    table: pd.DataFrame, *, weights: QgarpWeights = QGARP_WEIGHTS, fix_revisions_direction: bool = False
) -> pd.DataFrame:
    """``[date, listing_id, <composite>...]`` for every row of a cleaned :func:`factor_table`."""
    out = pd.concat(
        [
            table[[_DD, "listing_id"]],
            value_composite(table),
            growth_composite(table),
            quality_composite(table),
            momentum_composite(table),
        ],
        axis=1,
    )
    out[REVISIONS_1_MTH] = population_z(table, "earnings_revisions_1_mth")
    out[REVISIONS_COMPOSITE] = _column(table, REVISIONS_COMPOSITE)
    out["pct_rank_qgarp"] = qgarp(out, table[_DD], weights=weights, fix_revisions_direction=fix_revisions_direction)
    return out[[_DD, "listing_id", *COMPOSITES]]


def score_panel(comp: pd.DataFrame, name: str) -> pd.DataFrame:
    """``date x listing_id`` of one composite with higher meaning better: a percentile rank (1 = best) is negated."""
    wide = comp.pivot(index=_DD, columns="listing_id", values=name)  # noqa: PD010 - one row each
    return -wide if name.startswith("pct_rank_") else wide


def coverage_over_time(table: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """``date x name``: the share of the day's members with a value."""
    return table[list(names)].notna().groupby(table[_DD]).mean()


def coverage(table: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Per factor: the share of members with a value, on average over dates, and at the first and last date."""
    present = coverage_over_time(table, names)
    return pd.DataFrame(
        {"average": present.mean(), "first date": present.iloc[0], "last date": present.iloc[-1]}
    ).rename_axis("factor")


# ---------------------------------------------------------------- the run


@dataclass(frozen=True)
class BaringsGarpConfig:
    """Every setting of a run (hashed into its registry entry); the factor rules are production's, fixed.

    ``qgarp_weights`` (production's by default) takes a mapping or pairs of any :data:`QGARP_INPUTS`; it is stored
    in :func:`qgarp_weights`' one spelling, so equal weights are one config and every change is a new trial.
    """

    family: str = "barings_qgarp"
    universe: str = "SP1500"
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    enter: float = 0.2
    exit_: float = 0.3
    weighting: str = "equal"
    qgarp_weights: QgarpWeights = QGARP_WEIGHTS
    fix_revisions_direction: bool = False
    estimates: bool = True  # False: the first version, before the I/B/E/S estimates were loaded
    price_vendors: tuple[str, ...] = field(default_factory=lambda: vendor_preferences().prices)
    fundamentals_vendor: str | tuple[str, ...] = field(default_factory=lambda: vendor_preferences().fundamentals)
    open_holdout: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "qgarp_weights", qgarp_weights(self.qgarp_weights))  # checked before any loading


@dataclass(frozen=True)
class BaringsGarpResult:
    """The raw and cleaned factors, the composites, the book, the backtest and the summary metrics of a run."""

    config: BaringsGarpConfig
    factors: pd.DataFrame
    composites: pd.DataFrame
    score: pd.DataFrame
    book: pd.DataFrame
    backtest: BacktestResult
    benchmark: pd.Series
    members: pd.DataFrame
    returns: pd.DataFrame
    metrics: dict[str, Any]
    run_id: int | None = None


def gems_factors(
    cache: ResearchCache,
    data: UniverseData,
    *,
    rules: pd.DataFrame | None = None,
    price_vendors: tuple[str, ...] | None = None,
    fundamentals_vendor: str | tuple[str, ...] | None = None,
    with_estimates: bool = True,
) -> pd.DataFrame:
    """The cleaned :func:`factor_table` of ``data``'s members on its rebalance dates."""
    inputs = load_gems_inputs(
        cache, data, price_vendors=price_vendors, fundamentals_vendor=fundamentals_vendor, with_estimates=with_estimates
    )
    members = data.members.reindex(inputs.dates).fillna(False).astype(bool)
    panels = account_factors(inputs) | price_factors(inputs.adjusted, inputs.dates)
    if inputs.estimates is not None:
        panels |= estimate_factors(inputs.estimates)
        panels[REVISIONS_COMPOSITE] = revisions_composite(inputs.estimates, members)
    return clean(factor_table(panels, members, inputs.industry), rules)


def qgarp_scores(
    cache: ResearchCache, data: UniverseData, config: BaringsGarpConfig, *, rules: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(factors, composites, score)`` on ``data``'s rebalance dates; ``score`` is QGARP with higher meaning better."""
    table = gems_factors(
        cache,
        data,
        rules=rules,
        price_vendors=config.price_vendors,
        fundamentals_vendor=config.fundamentals_vendor,
        with_estimates=config.estimates,
    )
    comp = composites(table, weights=config.qgarp_weights, fix_revisions_direction=config.fix_revisions_direction)
    score = score_panel(comp, "pct_rank_qgarp").reindex(index=data.rebalances, columns=data.close.columns)
    return table, comp, score


def qgarp_book(
    data: UniverseData, config: BaringsGarpConfig, score: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """The best QGARP names on ``dates`` (with the enter / exit buffer), weighted as ``config`` says."""
    book = quantile_book(score, dates, enter=config.enter, exit_=config.exit_, long_only=True)
    return weighted_by(book, data.market_cap) if config.weighting == "market_cap" else book


def run_barings_garp(
    cache: ResearchCache,
    config: BaringsGarpConfig,
    *,
    engine: Engine | None = None,
    rules: pd.DataFrame | None = None,
) -> BaringsGarpResult:
    """Fence, load, build the factors and composites, hold the best QGARP names, backtest, and register the run.

    ``rules`` (Barings' cleaning limits) changes the scores, so a run with them should get its own ``family``.
    """
    if config.weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {config.weighting!r}; choose from {WEIGHTINGS}")
    holdout = fence(config.end, open_holdout=config.open_holdout)
    data = load_universe_data(
        cache,
        config.universe,
        config.start,
        config.end,
        price_vendors=config.price_vendors,
        fundamentals_vendor=config.fundamentals_vendor,
        history_days=HISTORY_DAYS,
    )
    table, comp, score = qgarp_scores(cache, data, config, rules=rules)
    book = qgarp_book(data, config, score, data.rebalances)
    returns = data.returns[data.returns.index <= pd.Timestamp(config.end)]
    result = run_backtest(
        book, returns, cost_bps=data.cost_bps, borrow_bps=data.borrow_bps, buy_tax_bps=data.buy_tax_bps
    )
    metrics, run_id = finish_run(
        result, data.benchmark, asdict(config), family=config.family, engine=engine, holdout=holdout
    )
    members = data.members.reindex(data.rebalances).fillna(False).astype(bool)
    return BaringsGarpResult(
        config, table, comp, score, book, result, data.benchmark, members, returns, metrics, run_id
    )
