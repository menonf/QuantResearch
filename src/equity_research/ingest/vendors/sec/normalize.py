"""SEC XBRL company facts -> canonical filings and financial facts. Pure functions.

What is stored is what was **reported**, with every version:

* each 10-K / 10-Q (and amendment) becomes a ``equity.filing`` (accession number, form, report period, fiscal year /
  period, filing date as ``published_at``); a filing's report period is the latest ``end`` among its USD facts, or
  among its share counts for a filing that reports no amounts (an amendment of the cover page);
* each XBRL fact becomes a ``equity.financial_fact`` of a canonical ``item`` (see :data:`ITEM_TAGS`) with its period:
  ``I`` an instant (balance sheet), ``Q`` about three months, ``FY`` about a year, ``YTD`` any other duration
  (6 or 9 months) - quarters missing as three-month figures are derived from these when read, not here;
* a fact whose period ends after the filing that reports it is dropped (a filer's tagging slip);
* the first report of a value is kept, and every later filing that reports a *different* value for the same period
  adds a row (a restatement) dated by its own filing; repeats of the same value add nothing.

When several tags of one item report the same period in one filing, the tag listed first wins.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

FORMS = frozenset({"10-K", "10-Q", "10-KT", "10-QT", "10-K/A", "10-Q/A", "10-KT/A", "10-QT/A"})

#: Canonical item -> XBRL tags in priority order (``us-gaap`` unless prefixed ``dei:``). Items stay separate where
#: definitions differ (parent vs total equity; PP&E vs PP&E plus lease assets) - the research formulas choose.
ITEM_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenuesNetOfInterestExpense",
        "InterestAndDividendIncomeOperating",
    ),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "pretax_income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    "interest_expense": (
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ),
    # the cash-flow statement: reported year to date in 10-Qs, so quarters come from differences when read
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "liabilities_and_equity": ("LiabilitiesAndStockholdersEquity",),
    "equity_parent": ("StockholdersEquity",),
    "equity_total": ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "cash_and_st_investments": ("CashCashEquivalentsAndShortTermInvestments",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "st_investments": (
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ),
    "ppe_net": ("PropertyPlantAndEquipmentNet",),
    "ppe_and_lease_assets": (
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ),
    "retained_earnings": ("RetainedEarningsAccumulatedDeficit",),
    "lt_debt_noncurrent": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermNotesPayable"),
    "lt_debt_total": (
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "DebtAndCapitalLeaseObligations",
        "DebtLongtermAndShorttermCombinedAmount",
    ),
    "debt_current": ("DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    "lt_debt_current": ("LongTermDebtCurrent",),
    "st_borrowings": ("ShortTermBorrowings",),
    "commercial_paper": ("CommercialPaper",),
    "notes_payable": ("NotesPayable", "SeniorNotes", "UnsecuredDebt"),
    "secured_debt": ("SecuredDebt",),
    "convertible_notes": ("ConvertibleNotesPayable",),
    "credit_lines": ("LineOfCredit", "LongTermLineOfCredit"),
    "other_borrowings": ("OtherBorrowings", "LoansPayable"),
    "shares_outstanding": ("CommonStockSharesOutstanding",),
    "shares_cover": ("dei:EntityCommonStockSharesOutstanding",),
    "shares_weighted_basic": ("WeightedAverageNumberOfSharesOutstandingBasic",),
    "shares_weighted_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    # added 2026-10-31 for the GEMS factors: depreciation, inventories, financing flows, unusual items, banks
    "depreciation": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ),
    "inventory": ("InventoryNet",),
    "financing_cash_flow": (
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ),
    "dividends_paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "stock_issued": ("ProceedsFromIssuanceOfCommonStock", "ProceedsFromIssuanceOrSaleOfEquity"),
    "stock_repurchased": ("PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"),
    "st_debt_change": ("ProceedsFromRepaymentsOfShortTermDebt", "ProceedsFromRepaymentsOfCommercialPaper"),
    "lt_debt_issued": ("ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt"),
    "lt_debt_repaid": ("RepaymentsOfLongTermDebt", "RepaymentsOfDebt"),
    "discontinued_operations": (
        "IncomeLossFromDiscontinuedOperationsNetOfTax",
        "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
    ),
    # unusual items: the SEC has no one line - charges (restructuring, impairments) less gains on disposals
    "restructuring_charges": (
        "RestructuringCharges",
        "RestructuringCosts",
        "RestructuringCostsAndAssetImpairmentCharges",
    ),
    "impairment_charges": (
        "AssetImpairmentCharges",
        "GoodwillAndIntangibleAssetImpairment",
        "GoodwillImpairmentLoss",
        "ImpairmentOfLongLivedAssetsHeldForUse",
    ),
    "disposal_gains": (
        "GainLossOnDispositionOfAssets",
        "GainLossOnDispositionOfAssets1",
        "GainLossOnSaleOfPropertyPlantEquipment",
        "GainLossOnSaleOfBusiness",
    ),
    # banks
    "net_interest_income": ("InterestIncomeExpenseNet",),
    "noninterest_income": ("NoninterestIncome",),
    "noninterest_expense": ("NoninterestExpense",),
    "loan_loss_provision": (
        "ProvisionForLoanLeaseAndOtherLosses",
        "ProvisionForLoanAndLeaseLosses",
        "ProvisionForLoanLossesExpensed",
        "FinancingReceivableCreditLossExpenseReversal",
    ),
    "loans_net": (
        "LoansAndLeasesReceivableNetReportedAmount",
        "LoansAndLeasesReceivableNetOfDeferredIncome",
        "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
    ),
    "deposits": ("Deposits",),
    "nonperforming_loans": (
        "FinancingReceivableExcludingAccruedInterestNonaccrual",
        "FinancingReceivableRecordedInvestmentNonaccrualStatus",
        "LoansAndLeasesReceivableNonperformingNonaccrualOfInterest",
    ),
    "tier1_capital": ("TierOneRiskBasedCapital",),
    # regulatory capital ratios: unit "pure" (0.125 for 12.5 %)
    "tier1_ratio": ("TierOneRiskBasedCapitalToRiskWeightedAssets",),
    "total_capital_ratio": ("CapitalToRiskWeightedAssets",),
}

#: The units read: amounts in dollars, share counts, and pure numbers (regulatory capital ratios).
UNITS = ("USD", "shares", "pure")
QUARTER_DAYS = (80, 100)
YEAR_DAYS = (350, 380)
FACT_COLUMNS = ["tag", "unit", "start", "end", "val", "accn", "fy", "fp", "form", "filed"]


def _tag_priority() -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    for item, tags in ITEM_TAGS.items():
        for rank, tag in enumerate(tags):
            out[tag] = (item, rank)
    return out


TAG_ITEM = _tag_priority()


def facts_frame(companyfacts: dict[str, Any], tags: Iterable[str] | None = None) -> pd.DataFrame:
    """Every wanted fact from 10-K / 10-Q filings: ``[tag, unit, start, end, val, accn, fy, fp, form, filed]``.

    ``tag`` is prefixed ``dei:`` for cover-page facts. Units: USD, shares and pure numbers (the capital ratios).
    """
    wanted = set(tags if tags is not None else TAG_ITEM)
    rows = []
    for taxonomy, prefix in (("us-gaap", ""), ("dei", "dei:")):
        for tag, body in companyfacts.get("facts", {}).get(taxonomy, {}).items():
            name = prefix + tag
            if name not in wanted:
                continue
            for unit, facts in body.get("units", {}).items():
                if unit not in UNITS:
                    continue
                for f in facts:
                    if f.get("form") in FORMS:
                        rows.append(
                            (
                                name,
                                unit,
                                f.get("start"),
                                f.get("end"),
                                f.get("val"),
                                f.get("accn"),
                                f.get("fy"),
                                f.get("fp"),
                                f.get("form"),
                                f.get("filed"),
                            )
                        )
    df = pd.DataFrame(rows, columns=FACT_COLUMNS)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce").astype(float)
    return df.dropna(subset=["end", "val", "filed", "accn"]).reset_index(drop=True)


FISCAL_YEARS = (1900, 2100)


def _fiscal_year(fy: pd.Series) -> pd.Series:
    """The filer-entered ``fy`` as a year, or NULL when it is not one (some filers enter a spreadsheet date serial)."""
    year = pd.to_numeric(fy, errors="coerce")
    return year.where(year.between(*FISCAL_YEARS)).astype("Int64")


def filings(facts: pd.DataFrame) -> pd.DataFrame:
    """``[filing_ref, form, period_end, fiscal_year, fiscal_period, published_at]`` - one row per accession."""
    if facts.empty:
        return pd.DataFrame(
            columns=["filing_ref", "form", "period_end", "fiscal_year", "fiscal_period", "published_at"]
        )
    facts = facts[pd.to_datetime(facts["end"], errors="coerce") <= pd.to_datetime(facts["filed"], errors="coerce")]
    g = facts.groupby("accn")
    # the report period: from the amounts where there are any (a cover page's share count is dated later)
    ends = facts[facts["unit"] == "USD"].groupby("accn")["end"].max()
    out = pd.DataFrame(
        {
            "form": g["form"].first(),
            "period_end": ends.reindex(g.size().index).fillna(g["end"].max()),
            "fiscal_year": _fiscal_year(g["fy"].max()),
            "fiscal_period": g["fp"].last(),
            "published_at": g["filed"].min(),
        }
    )
    return out.rename_axis("filing_ref").reset_index()


def period_type(start: pd.Series, end: pd.Series) -> pd.Series:
    """``I`` (instant), ``Q`` (80-100 days), ``FY`` (350-380 days) or ``YTD`` (any other duration)."""
    days = (end - start).dt.days
    out = pd.Series("YTD", index=start.index, dtype=object)
    out[start.isna()] = "I"
    out[days.between(*QUARTER_DAYS)] = "Q"
    out[days.between(*YEAR_DAYS)] = "FY"
    return out


def financial_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Canonical facts with every reported version (see the module docstring).

    Returns ``[item, period_type, period_start, period_end, value, unit, published_at, filing_ref]``.
    """
    cols = ["item", "period_type", "period_start", "period_end", "value", "unit", "published_at", "filing_ref"]
    if facts.empty:
        return pd.DataFrame(columns=cols)
    f = facts[facts["tag"].isin(TAG_ITEM)].copy()
    # a report cannot describe a period that ends after it was filed: such a fact is a tagging slip (a note's maturity
    # tagged as the balance-sheet date, "debt at 2041-03-15" in a 2011 10-K) and would pass for the newest figure
    f = f[pd.to_datetime(f["end"], errors="coerce") <= pd.to_datetime(f["filed"], errors="coerce")]
    mapped = f["tag"].map(TAG_ITEM)
    f["item"] = mapped.str[0]
    f["rank"] = mapped.str[1]
    f["period_type"] = period_type(f["start"], f["end"])
    period = ["item", "period_type", "start", "end"]
    # one value per (item, period, filing): the highest-priority tag
    f = f.sort_values([*period, "accn", "rank"], kind="stable").drop_duplicates([*period, "accn"], keep="first")
    # versions: the first report, then each filing that changes the value
    f = f.sort_values([*period, "filed", "accn"], kind="stable")
    previous = f.groupby(period, dropna=False)["val"].shift(1)
    changed = previous.isna() | ~np.isclose(f["val"], previous, rtol=1e-9, atol=0.0)
    f = f[changed]
    out = pd.DataFrame(
        {
            "item": f["item"],
            "period_type": f["period_type"],
            "period_start": f["start"],
            "period_end": f["end"],
            "value": f["val"],
            "unit": f["unit"],
            "published_at": f["filed"],
            "filing_ref": f["accn"],
        }
    )
    # two filings on the same day (a 10-K and its amendment): the later accession number wins
    key = ["item", "period_type", "period_end", "published_at"]
    return out.sort_values([*key, "filing_ref"]).drop_duplicates(key, keep="last").reset_index(drop=True)
