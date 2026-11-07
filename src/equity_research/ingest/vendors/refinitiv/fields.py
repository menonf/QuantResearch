"""Every ``TR.*`` field the adapter asks LSEG for, with the column name LSEG answers with.

``ld.get_data`` returns columns by display name, not by field code, so the two live side by side here. Knowledge
carried over from the previous project (checked 2026-10-19..23):

* ``0#.SPX(<date>)`` ignores the date and returns *today's* members: history comes from replaying the joiner /
  leaver events (``TR.IndexJLConstituent*``) backwards from today's list.
* ``TR.IssueSharesOutstanding`` is the actual share count per share class, monthly, retired RICs included, not
  restated for later splits; times ``TR.FreeFloatPct`` it is the float index providers weight by. Company-level
  counts (``TR.F.ComShrOutsTot``) give Alphabet's total to both GOOG and GOOGL, and Berkshire in class-A units - and
  they *are* restated for later splits: Alphabet's 2015 count comes in post-2022 shares (13.7 billion, not 687
  million), Broadcom's 2014 count in post-2024 shares. Research therefore values a listing from its own count
  (checked 2026-10-26).
* ``TR.CAAdjustmentFactor`` is the price factor of every capital change: 0.25 for a 4-for-1 split, but also ~0.95
  for a spin-off.
* ``TR.Currency`` answers nothing for US listings; an index RIC's attributes are its publisher's (``.SPX`` comes back
  as Cboe Exchange Inc with Cboe's ISIN) - not used. Checked live 2026-10-24.
* Statements (``TR.TotalRevenue`` and co., ``Period=FQ0, Frq=FQ``) are LSEG's latest (restated) figures, dated here by
  ``TR.F.OriginalAnnouncementDate`` - not "as first reported" the way the SEC facts are.
"""

from __future__ import annotations

# ---------------------------------------------------------------- index members
#: Which LSEG chain lists each index's members (our index code -> chain RIC).
MEMBERSHIP_CHAINS = {"SP500": ".SPX", "SP400": ".MID", "SP600": ".SPCY"}
SNAPSHOT_FIELDS = ["TR.ExchangeTicker"]
EVENT_FIELDS = [
    "TR.IndexJLConstituentChangeDate",
    "TR.IndexJLConstituentRIC",
    "TR.IndexJLConstituentName",
    "TR.IndexJLConstituentituentChange",  # sic: the field is spelled this way
]
EVENT_COLUMNS = {"Date": "date", "Constituent RIC": "ric", "Constituent Name": "name", "Change": "change"}

# ---------------------------------------------------------------- listings and their codes
INSTRUMENT_FIELDS = [
    "TR.CommonName",
    "TR.ISIN",
    "TR.SEDOL",
    "TR.CUSIP",
    "TR.ExchangeCountryCode",
    "TR.ExchangeTicker",
]
INSTRUMENT_COLUMNS = {
    "Company Common Name": "name",
    "ISIN": "isin",
    "SEDOL": "sedol",
    "CUSIP": "cusip",
    "Exchange Country Code": "country",
    "Exchange Ticker": "ticker",
}
#: Where the company is run (headquarters country, ISO code) - asked on its own, so an answer LSEG cannot give spoils
#: nothing else. Its one answer column is taken whatever LSEG names it.
DOMICILE_FIELDS = ["TR.HQCountryCode"]
CLASSIFICATION_FIELDS = ["TR.GICSSector", "TR.GICSIndustryGroup", "TR.GICSIndustry", "TR.TRBCEconomicSector"]
CLASSIFICATION_COLUMNS = {
    "GICS Sector Name": "sector",
    "GICS Industry Group Name": "industry_group",
    "GICS Industry Name": "industry",
    "TRBC Economic Sector Name": "trbc_sector",
}
#: TRBC economic sector -> closest GICS sector, where GICS is blank (some retired RICs)
TRBC_TO_GICS = {
    "Basic Materials": "Materials",
    "Consumer Cyclicals": "Consumer Discretionary",
    "Consumer Non-Cyclicals": "Consumer Staples",
    "Energy": "Energy",
    "Financials": "Financials",
    "Healthcare": "Health Care",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Technology": "Information Technology",
    "Telecommunications Services": "Communication Services",
    "Utilities": "Utilities",
}

# ---------------------------------------------------------------- daily prices (unadjusted)
#: ``ld.get_history`` real-time fields, ``adjustments=["unadjusted"]``. ``get_data``'s ``TR.CLOSEPRICE`` fails a whole
#: request when one RIC has no prices in the window (a listing not yet trading) - get_history leaves it blank.
#: Close and volume only: research uses nothing else, and LSEG's daily allowance stretches further (2026-10-24).
HISTORY_FIELDS = ["TRDPRC_1", "ACVOL_UNS"]
#: Index levels have no volume: asked together with stocks, lseg-data cannot build the answer (one column for them,
#: two for each stock), so they are asked apart, all in one request, for their close only.
LEVEL_HISTORY_FIELDS = ["TRDPRC_1"]
#: The fields asked for before that: an answer landed with them holds everything HISTORY_FIELDS asks for.
HISTORY_FIELDS_FULL = ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS"]
HISTORY_COLUMNS = {
    "TRDPRC_1": "close",
    "OPEN_PRC": "open",
    "HIGH_1": "high",
    "LOW_1": "low",
    "ACVOL_UNS": "volume",
    "OFF_CLOSE": "close",  # London's official close: asked instead of TRDPRC_1 there, never beside it
    "MID_PRICE": "close",  # an exchange rate's (currencies have no trades)
}
#: A refresh's few days by ``get_data`` instead: it costs one request a call where a history costs one per RIC
#: (2026-10-30: 50 RICs over 13 days cost 1 request; closes and volumes equal TRDPRC_1 / ACVOL_UNS bar for bar, and
#: retired RICs in the call did not fail it). Too large an answer for a build's 11-year windows.
DAILY_FIELDS = ["TR.PriceClose.date", "TR.PriceClose", "TR.Volume"]
DAILY_PARAMS = {"Frq": "D", "Adjusted": 0}  # unadjusted, as the histories
DAILY_COLUMNS = {"Date": "trade_date", "Price Close": "close", "Volume": "volume"}

# ---------------------------------------------------------------- dividends and capital changes
DIVIDEND_FIELDS = ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivType"]
DIVIDEND_COLUMNS = {"Dividend Ex Date": "ex_date", "Gross Dividend Amount": "amount", "Dividend Type": "kind"}
CAPITAL_CHANGE_FIELDS = ["TR.CAEffectiveDate", "TR.CAAdjustmentFactor"]
CAPITAL_CHANGE_COLUMNS = {"Capital Change Effective Date": "ex_date", "Adjustment Factor": "factor"}

# ---------------------------------------------------------------- share counts (monthly)
SHARES_FIELDS = ["TR.IssueSharesOutstanding", "TR.IssueSharesOutstanding.Date"]
SHARES_COLUMNS = {"Issue Default Shares Outstanding": "shares", "Date": "as_of_date"}
FREE_FLOAT_FIELDS = ["TR.FreeFloatPct", "TR.FreeFloatPct.Date"]
FREE_FLOAT_COLUMNS = {"Free Float (Percent)": "float_pct", "Date": "as_of_date"}
MONTHLY = {"Frq": "M"}

# ---------------------------------------------------------------- quarterly statements and when they were announced
QUARTERLY = {"Period": "FQ0", "Frq": "FQ"}
#: Statements converted by LSEG to US dollars at its own rates: without ``Curn`` a figure comes in the company's
#: reporting currency (euros, say) and would be read as dollars.
STATEMENT_PARAMS = {**QUARTERLY, "Curn": "USD"}
ANNOUNCEMENT_FIELDS = ["TR.TotalRevenue.periodenddate", "TR.F.OriginalAnnouncementDate"]
ANNOUNCEMENT_COLUMNS = {"Period End Date": "period_end", "Original Announcement Date Time": "announced_at"}
#: field -> (column LSEG answers with, our canonical item, ``flow`` (the quarter's amount) or ``balance`` (on the day))
STATEMENT_ITEMS = {
    "TR.TotalRevenue": ("Total Revenue", "revenue", "flow"),
    "TR.GrossProfit": ("Gross Profit", "gross_profit", "flow"),
    "TR.OperatingIncome": ("Operating Income", "operating_income", "flow"),
    "TR.NetIncome": ("Net Income Incl Extra Before Distributions", "net_income", "flow"),
    "TR.InterestExpense": ("Interest Expense", "interest_expense", "flow"),
    "TR.TotalAssets": ("Total Assets", "total_assets", "balance"),
    "TR.TotalCurrentAssets": ("Total Current Assets", "current_assets", "balance"),
    "TR.CurrentLiabilities": ("Current Liabilities", "current_liabilities", "balance"),
    "TR.TotalLiabilities": ("Total Liabilities", "total_liabilities", "balance"),
    "TR.TotalEquity": ("Total Equity", "equity_total", "balance"),
    "TR.RetainedEarnings": ("Retained Earnings (Accumulated Deficit)", "retained_earnings", "balance"),
    "TR.CashAndSTInvestments": ("Cash and Short Term Investments", "cash_and_st_investments", "balance"),
    "TR.PropertyPlantEquipmentTotalNet": ("Property/Plant/Equipment, Total - Net", "ppe_net", "balance"),
    "TR.TotalDebt": ("Total Debt", "debt_total", "balance"),
    "TR.F.ComShrOutsTot": ("Common Shares - Outstanding - Total", "shares_outstanding", "balance"),
}
STATEMENT_FIELDS = ["TR.TotalRevenue.periodenddate", *STATEMENT_ITEMS]

# ---------------------------------------------------------------- yearly statement items, one field per request
# Checked live 2026-10-31. These answer by fiscal year (``Period=FY0, Frq=FY``), one field per request beside revenue's
# period end: asked two or more together LSEG answers "Backend error" or "unexpected error during parsing", and
# ``.periodenddate`` on some of them is unknown. Their quarters are year to date ("Capital Expenditures, Cumulative"),
# so the year's figure is asked. Share issues and buybacks, debt issued and repaid, short-term borrowing and
# discontinued operations answer nothing, and LSEG's tier-1 and total capital ratios are brokers' estimates ("Total
# Capital Ratio - Broker Estimate"): those come from the SEC only. Each answer's rows are dated by a period end asked
# beside the item: revenue's for most, total assets' for the bank lines - LSEG gives banks no revenue period end, and
# beside total assets' end it answers depreciation empty.
#: field -> (our item, ``flow`` (the fiscal year's amount) or ``balance`` (at its end), the field dating its rows)
ANNUAL_ITEMS = {
    "TR.CashFromOperatingActivities": ("operating_cash_flow", "flow", "TR.TotalRevenue.periodenddate"),
    "TR.CapitalExpenditures": ("capex", "flow", "TR.TotalRevenue.periodenddate"),
    "TR.CashFromFinancingActivities": ("financing_cash_flow", "flow", "TR.TotalRevenue.periodenddate"),
    "TR.DepreciationAmort": ("depreciation", "flow", "TR.TotalRevenue.periodenddate"),
    "TR.CashDividendsPaid": ("dividends_paid", "flow", "TR.TotalRevenue.periodenddate"),
    "TR.UnusualExpense": ("unusual_expense", "flow", "TR.TotalRevenue.periodenddate"),
    "TR.TotalInventory": ("inventory", "balance", "TR.TotalRevenue.periodenddate"),
    "TR.NetInterestIncome": ("net_interest_income", "flow", "TR.TotalAssets.periodenddate"),
    "TR.NonInterestIncome": ("noninterest_income", "flow", "TR.TotalAssets.periodenddate"),
    "TR.NonInterestExpense": ("noninterest_expense", "flow", "TR.TotalAssets.periodenddate"),
    "TR.LoanLossProvision": ("loan_loss_provision", "flow", "TR.TotalAssets.periodenddate"),
    "TR.NetLoans": ("loans_net", "balance", "TR.TotalAssets.periodenddate"),
    "TR.TotalDeposits": ("deposits", "balance", "TR.TotalAssets.periodenddate"),
    "TR.RiskWeightedAssets": ("risk_weighted_assets", "balance", "TR.TotalAssets.periodenddate"),
    "TR.NonPerformingLoans": ("nonperforming_loans", "balance", "TR.TotalAssets.periodenddate"),
}
#: A value column LSEG names with this is a forecast, never stored as a reported fact.
ESTIMATE_MARKER = "Estimate"
#: Payments, stored as the amount paid whatever sign LSEG sends (capital spending comes negative, dividends positive).
PAID = frozenset({"capex", "dividends_paid"})
YEARLY_STATEMENT_PARAMS = {"Period": "FY0", "Frq": "FY", "Curn": "USD"}


def annual_fields(field: str) -> list[str]:
    """One yearly item's request: the period end dating its rows (:data:`ANNUAL_ITEMS`), and the item."""
    return [ANNUAL_ITEMS[field][2], field]


# ---------------------------------------------------------------- I/B/E/S estimates
# Checked live 2026-10-30:
# * consensus history (``SDate``/``EDate``/``Frq=M`` with one ``Period``) answers one row per month end (``Calc Date``)
#   in one call - 170 RICs x 20 years in ~70 s, 4 requests;
# * per-share figures (EPS, DPS, cash flow per share) are restated to *today's* shares: Apple's FY2015 EPS comes back
#   as 2.305, its 9.22 over the 2020 4-for-1 split; totals (revenue, EBIT) and ROE are not;
# * the revision counts (``TR.NumEstRevisingUp`` / ``Down``) never answer as a history: lseg-data re-sends the
#   request about every 15 s, each costing one request (~1,000 lost one afternoon) - they are asked one date at a time;
# * a reported actual carries the time it was reported (``TR.EPSActReportDate``, exchange time).
#: our measure -> LSEG's field stem
ESTIMATE_STEMS = {"eps": "EPS", "revenue": "Revenue", "ebit": "EBIT", "dps": "DPS", "cfps": "CFPS", "roe": "ROE"}
#: Measures quoted per share, which LSEG restates for later splits.
PER_SHARE_MEASURES = frozenset({"eps", "dps", "cfps"})
#: The forecast periods asked per measure: what Barings' GEMS factors read (FY3 for growth to the third year,
#: FY1 and FY2 for the next twelve months).
ESTIMATE_PERIODS = {
    "eps": ("FY1", "FY2", "FY3"),
    "revenue": ("FY1", "FY2", "FY3"),
    "ebit": ("FY1", "FY2", "FY3"),
    "dps": ("FY1", "FY2"),
    "cfps": ("FY1", "FY2"),
    "roe": ("FY1", "FY2"),
}
#: Measures whose reported actuals are asked (the last twelve months blend the last actual with FY1).
ACTUAL_MEASURES = ("eps", "revenue", "ebit", "dps", "cfps")
#: Answer column endings -> our names (LSEG prefixes each with the measure's name: "Revenue - Mean").
CONSENSUS_COLUMNS = {
    " - Mean": "mean",
    "Calc Date": "as_of_date",
    "Period End Date": "fiscal_period_end",
    " - Median": "median",
    " - Standard Deviation": "std_dev",
    " - Number of Included Estimates": "estimates",
}
ACTUAL_COLUMNS = {" - Actual": "value", "Period End Date": "period_end", "Report Date": "announced_at"}
REVISION_FIELDS = ["TR.NumEstRevisingUp", "TR.NumEstRevisingDown"]
REVISION_COLUMNS = {"Number of Analysts Revising Up": "up", "Number of Analysts Revising Down": "down"}
#: The forecast periods whose revisions are asked (GEMS' earnings revisions read FY1 and FY2).
REVISION_PERIODS = ("FY1", "FY2")
MONTHLY_HISTORY = {"Frq": "M"}
YEARLY_ACTUALS = {"Frq": "FY", "Period": "FY0"}


# ---------------------------------------------------------------- London: the FTSE UK indices
# Checked live 2026-11-01 (``eqr probe-lseg-uk`` repeats the checks):
# * ``0#.FTSE`` / ``0#.FTMC`` / ``0#.FTSC`` list today's FTSE 100 / 250 / SmallCap (100 + 251 + 174: the All-Share's
#   525); their joiner / leaver events reach back to January 2005 - 1,379 RICs since;
# * London's ``TRDPRC_1`` is the last trade, often after the closing auction (Vodafone 2026-10-23: 118.6866), not the
#   official close ``OFF_CLOSE`` (118.2) that ``TR.PriceClose`` answers - histories ask ``OFF_CLOSE``, which answers
#   back to 2006, for retired RICs too, and a whole 11-year window in one request;
# * prices are in pence (``GBp``); a London line's dividends in pounds (``GBP``), also for companies declaring them in
#   dollars (Shell, HSBC);
# * no CUSIP - ISIN and SEDOL; a RIC's root is not the ticker for 118 of the 525 (``SJP.L`` is STJ, ``VNHq.L`` is no
#   share class Q): ``TR.ExchangeTicker`` is;
# * investment trusts have no GICS; their TRBC industry is "UK Investment Trusts". ``TR.RegCountryCode`` is where a
#   company is incorporated (Glencore: JE - stamp duty follows it), ``TR.OrganizationID`` its PermID;
# * most companies report half-years: ``Frq=FQ`` answers only the revenue of trading updates (Tesco: nothing),
#   ``Frq=FS`` six-month figures (the second half is the year less the first). ``TR.F.*`` fields with
#   ``ReportingState=Orig`` answer figures as first reported (Tesco's FY2013 net income 124m, restated 28m), revenue
#   for banks too (``TR.TotalRevenue`` is blank for them);
# * ``Curn=GBP`` converts statements and estimates at each period's or month's own rate (Shell's EPS consensus of
#   January 2015 in pounds is its dollars / 1.507, that month's rate); without it per-share estimates come in the
#   reporting currency (Vodafone euros, Shell dollars);
# * total-return levels: the All-Share ``.FTASXGBPT`` (from 2003), the All-Share ex investment trusts ``.FTASXXMGBPT``
#   (from 2024-09 only, and its closes do not line up with the All-Share's day by day: kept, never a benchmark).
#: Our index code -> the chain listing its members.
UK_MEMBERSHIP_CHAINS = {"FTSE100": ".FTSE", "FTSE250": ".FTMC", "FTSESC": ".FTSC"}
UK_HISTORY_FIELDS = ["OFF_CLOSE", "ACVOL_UNS"]
PRICE_CURRENCY_FIELDS = [*DAILY_FIELDS, "TR.PriceClose.currency"]
#: The company's PermID, country of incorporation and TRBC industry (which names investment trusts).
UK_COMPANY_FIELDS = ["TR.OrganizationID", "TR.RegCountryCode", "TR.TRBCIndustry"]
UK_COMPANY_COLUMNS = {
    "Organization PermID": "permid",
    "Country ISO Code of Incorporation": "incorporated",
    "TRBC Industry Name": "trbc_industry",
}
INVESTMENT_TRUST_INDUSTRY = "UK Investment Trusts"
DIVIDEND_CURRENCY_FIELDS = [*DIVIDEND_FIELDS, "TR.DivUnadjustedGross.currency"]
#: Half-years, as first reported, in pounds at each period's rate.
UK_STATEMENT_PARAMS = {"Period": "FS0", "Frq": "FS", "Curn": "GBP", "ReportingState": "Orig"}
UK_YEARLY_PARAMS = {"Period": "FY0", "Frq": "FY", "Curn": "GBP", "ReportingState": "Orig"}
#: ``TR.F.*`` field -> (column LSEG answers with, our item, ``flow`` or ``balance``), checked live 2026-11-01. A name
#: LSEG does not know is left out of the answer without an error (``TR.F.IntrExpnTot``), so the columns are checked.
#: One request answers them all, half-yearly or yearly. A year is asked as reported, not as its halves added: Lloyds'
#: FY2016 operating profit (5,117m) is not its first-reported halves' sum (2,907m + 2,043m).
UK_STATEMENT_ITEMS = {
    "TR.F.TotRevenue": ("Revenue from Business Activities - Total", "revenue", "flow"),
    "TR.F.GrossProfIndPropTot": ("Gross Profit - Industrials/Property - Total", "gross_profit", "flow"),
    "TR.F.OpProfBefNonRecurIncExpn": (
        "Operating Profit before Non-Recurring Income/Expense",
        "operating_income",
        "flow",
    ),
    "TR.F.NetIncAfterMinIntr": ("Net Income after Minority Interest", "net_income", "flow"),
    "TR.F.IntrExpn": ("Interest Expense", "interest_expense", "flow"),
    "TR.F.IncBefTax": ("Income before Taxes", "pretax_income", "flow"),
    "TR.F.NetCashFlowOp": ("Net Cash Flow from Operating Activities", "operating_cash_flow", "flow"),
    "TR.F.CAPEXTot": ("Capital Expenditures - Total", "capex", "flow"),
    "TR.F.NetCashFlowFin": ("Net Cash Flow from Financing Activities", "financing_cash_flow", "flow"),
    "TR.F.DeprDeplAmortTot": ("Depreciation Depletion & Amortization - Total", "depreciation", "flow"),
    "TR.F.TotAssets": ("Total Assets", "total_assets", "balance"),
    "TR.F.TotCurrAssets": ("Total Current Assets", "current_assets", "balance"),
    "TR.F.TotCurrLiab": ("Total Current Liabilities", "current_liabilities", "balance"),
    "TR.F.TotLiab": ("Total Liabilities", "total_liabilities", "balance"),
    "TR.F.TotShHoldEq": ("Total Shareholders' Equity incl Minority Intr & Hybrid Debt", "equity_total", "balance"),
    "TR.F.ShHoldEqCom": ("Shareholders Equity - Common", "equity_parent", "balance"),
    "TR.F.RetainedEarnTot": ("Retained Earnings - Total", "retained_earnings", "balance"),
    "TR.F.CashSTInvstTot": ("Cash & Short Term Investments - Total", "cash_and_st_investments", "balance"),
    "TR.F.PPENetTot": ("Property Plant & Equipment - Net - Total", "ppe_net", "balance"),
    "TR.F.DebtTot": ("Debt - Total", "debt_total", "balance"),
    "TR.F.InvntTot": ("Inventories - Total", "inventory", "balance"),
    "TR.F.ComShrOutsTot": ("Common Shares - Outstanding - Total", "shares_outstanding", "balance"),
    "TR.F.LoansNetTot": ("Loans - Net - Total", "loans_net", "balance"),
    "TR.F.DeposTot": ("Deposits - Total", "deposits", "balance"),
}
UK_STATEMENT_FIELDS = ["TR.F.TotRevenue.periodenddate", *UK_STATEMENT_ITEMS, "TR.F.OriginalAnnouncementDate"]
UK_LEVELS = {"FTSEAS_TR": ".FTASXGBPT", "FTSEAS_XIT_TR": ".FTASXXMGBPT"}

# ---------------------------------------------------------------- exchange rates
# Checked live 2026-11-01: a currency's history answers MID_PRICE (BID, ASK too; TRDPRC_1 is refused) from 2005, a
# whole 11-year window in one request. Some dividends of London lines are declared in other currencies (Anglo American
# dollars, Shell euros until 2022, Old Mutual rand); asked converted (``Curn=GBP``), LSEG's dividend amounts come back
# shifted against their ex-dates, so they are kept as declared and converted in research with these rates.
FX_HISTORY_FIELDS = ["MID_PRICE"]

# ---------------------------------------------------------------- London: capital changes and quote currencies
# Checked live 2026-11-01. A London rights issue's "effective date" is weeks after its ex-date (Rolls-Royce: ex
# 2020-10-28, when the price fell from 219p to 85p; effective 2020-11-12), so London asks the ex-date. Some London lines
# change the currency they are quoted in (Coca-Cola Europacific: $72.80 to 5,990p on 2024-10-21; InterContinental
# Hotels: 10,460p to $138.45 on 2026-01-02); a history answers prices as quoted with no currency (``CURRENCY``,
# ``CF_CURR`` and the like are ignored there), and ``TR.PriceClose`` answers the whole history in today's currency,
# converted - so the currency is today's (``CF_CURR``) and, walking back, changes where the price jumps by the exchange
# rate (:func:`~.normalize.quote_currencies`).
UK_CAPITAL_CHANGE_FIELDS = ["TR.CAExDate", "TR.CAAdjustmentFactor"]
UK_CAPITAL_CHANGE_COLUMNS = {"Capital Change Ex Date": "ex_date", "Adjustment Factor": "factor"}
QUOTE_CURRENCY_FIELDS = ["CF_CURR"]
#: LSEG's currency codes -> ours: pence as GBX (``GBp`` reads as pounds to a case-insensitive database)
CURRENCY_CODES = {"GBp": "GBX"}
#: currency -> (its RIC, True when quoted in dollars per unit - else in units per dollar, which the adapter inverts)
FX_RICS = {
    "GBP": ("GBP=", True),
    "EUR": ("EUR=", True),
    "AUD": ("AUD=", True),
    "NZD": ("NZD=", True),
    "ZAR": ("ZAR=", False),
    "JPY": ("JPY=", False),
    "CHF": ("CHF=", False),
    "CAD": ("CAD=", False),
    "SEK": ("SEK=", False),
    "NOK": ("NOK=", False),
    "DKK": ("DKK=", False),
    "HKD": ("HKD=", False),
    "GEL": ("GEL=", False),  # Georgian lari: London-listed Georgian banks' dividends (checked live 2026-11-01)
}


def consensus_fields(measure: str) -> list[str]:
    """The consensus fields of one measure: mean (with its date and period end), median, spread and count."""
    s = ESTIMATE_STEMS[measure]
    return [
        f"TR.{s}Mean",
        f"TR.{s}Mean.calcdate",
        f"TR.{s}Mean.periodenddate",
        f"TR.{s}Median",
        f"TR.{s}StdDev",
        f"TR.{s}NumIncEstimates",
    ]


def actual_fields(measure: str) -> list[str]:
    """One measure's reported actual, its fiscal period end and when it was reported."""
    s = ESTIMATE_STEMS[measure]
    return [f"TR.{s}ActValue", f"TR.{s}ActValue.periodenddate", f"TR.{s}ActReportDate"]
