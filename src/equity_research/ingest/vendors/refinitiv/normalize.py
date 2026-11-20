"""LSEG answers -> canonical rows. Pure functions: no session, no database.

Every function takes the frame LSEG returned (columns as LSEG names them, see :mod:`.fields`) and returns rows keyed
by RIC; the pipeline maps RICs to listing ids. A column LSEG did not send raises :class:`MissingColumnsError` naming
it, so a vendor-side rename is found on the first request, not as silently empty tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import fields as f
from . import rics

#: An adjustment factor this far from 1 is a split (or a reverse split); nearer, a spin-off or rights issue.
SPLIT_FACTOR_BAND = (0.85, 1 / 1.2)
#: How closely the price must move by the split ratio on the ex-date to confirm it (log difference): within a factor of
#: 2. A distressed company's reverse split moves its price only roughly by the ratio (AIG's 1-for-20 of 2009: x15.6;
#: Chesapeake's 1-for-200 of 2020: x125); a phantom split misses by its whole ratio (Covidien's "100-for-1": x1.03).
SPLIT_TOLERANCE = float(np.log(2.0))
#: A one-day move of the price this large, either way (log): a split's, or a collapse, or a bad price (the quality
#: gate ``unexplained_price_jumps`` uses the same 2.5x).
JUMP = float(np.log(2.5))
#: How closely the shares outstanding must move by a split's ratio to confirm it (log difference): within 10%. London's
#: missed consolidations match within 3% (Capital & Regional's 1-for-10 of 2020: x10.000; Allied Gold's 1-for-6 of 2011:
#: x5.99); the nearest other jump misses by 38% (Keurig Dr Pepper's merger of 2018), a collapse by its whole move.
SHARE_COUNT_TOLERANCE = float(np.log(1.1))
#: Share counts are monthly: the last before a day and the first after it count if no further from it than this.
SHARE_COUNT_DAYS = 45
#: A split found by the count alone exchanges whole shares: within 3% of n-for-1, 1-for-n or one of these. A crash
#: that met an issue of shares can match the count as closely (National City, 2008-09-29: price /2.73, preferreds
#: converted, shares x2.72) but not by a whole exchange; Shell's unification of 2005 (0.287 new per old) is lost too.
WHOLE_EXCHANGES = (3 / 2, 2 / 3, 4 / 3, 3 / 4, 5 / 4, 4 / 5, 5 / 2, 2 / 5, 5 / 3, 3 / 5)
WHOLE_TOLERANCE = float(np.log(1.03))
#: Statements without an announcement date are dated this long after their period end (late, never early).
FALLBACK_REPORTING_LAG_DAYS = 90
#: Pence sterling. LSEG writes ``GBp``; stored as that, a case-insensitive comparison would read it as pounds.
PENCE = "GBX"


class MissingColumnsError(KeyError):
    """LSEG's answer lacks columns the adapter relies on."""


def _take(raw: pd.DataFrame, columns: Mapping[str, str], what: str) -> pd.DataFrame:
    """``raw``'s ``Instrument`` plus ``columns`` renamed; raises when LSEG did not send one of them."""
    if raw.empty:
        return pd.DataFrame(columns=["ric", *columns.values()])
    missing = [c for c in ["Instrument", *columns] if c not in raw.columns]
    if missing:
        raise MissingColumnsError(f"LSEG {what}: no column(s) {missing}; it sent {list(raw.columns)}")
    out = raw[["Instrument", *columns]].copy()
    out.columns = ["ric", *columns.values()]
    return out


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()


# ---------------------------------------------------------------- index members


@dataclass
class _Interval:
    ric: str
    ticker: str | None
    start: pd.Timestamp
    end: pd.Timestamp | None = None


def memberships(snapshot: pd.DataFrame, events: pd.DataFrame, start: str) -> pd.DataFrame:
    """``[ric, ticker, start_date, end_date]`` membership intervals (``end_date`` = first day out, NaT = member).

    ``snapshot`` is today's member list (``0#<chain>``); ``events`` the joiner / leaver events since ``start``.
    The events are replayed backwards from today: a member today that joined in the window starts on its join
    date; one with no join event was a member before ``start``; a leaver closes the interval it was in.
    """
    today = _take(snapshot, {"Exchange Ticker": "ticker"}, "index members")
    ev = _take(events, {"Date": "date", "Constituent RIC": "member", "Change": "change"}, "index events")
    ev = ev.assign(date=_dates(ev["date"]), ric=ev["member"]).dropna(subset=["ric", "date"])
    # a joiner and a leaver for one RIC on one day is a corporate-action artefact with no net effect
    both = ev.groupby(["ric", "date"])["change"].transform("nunique") > 1
    ev = ev[~both].sort_values("date", kind="stable")
    origin = pd.Timestamp(start)
    intervals = [
        _Interval(str(r), None if pd.isna(t) else str(t), origin)
        for r, t in zip(today["ric"].tolist(), today["ticker"].tolist(), strict=True)
    ]
    for ric, date, change in zip(ev["ric"].tolist(), ev["date"].tolist(), ev["change"].tolist(), strict=True):
        day = pd.Timestamp(date)
        open_ = [i for i in intervals if i.ric == ric and i.end is None]
        if change == "Joiner":
            seed = [i for i in open_ if i.start == origin]
            if seed:
                seed[0].start = day
            else:
                intervals.append(_Interval(str(ric), None, day))
        elif change == "Leaver":
            started = [i for i in open_ if i.start <= day]
            if started:
                max(started, key=lambda i: i.start).end = day
            else:
                intervals.append(_Interval(str(ric), None, origin, day))
    keep = [i for i in intervals if not rics.is_non_equity(i.ticker, i.ric)]
    out = pd.DataFrame(
        {
            "ric": [i.ric for i in keep],
            "ticker": [i.ticker for i in keep],
            "start_date": pd.to_datetime([i.start for i in keep]),
            "end_date": pd.to_datetime(pd.Series([i.end for i in keep], dtype="object")),
        }
    )
    return out.sort_values(["ric", "start_date"]).reset_index(drop=True)


def members_today(snapshot: pd.DataFrame) -> list[str]:
    """The RICs in today's member list (``0#<chain>``): an open spell of one ended with no leave event only if not."""
    return [str(r) for r in _take(snapshot, {}, "index members")["ric"].dropna()]


# ---------------------------------------------------------------- listings


def instruments(raw: pd.DataFrame, *, level_currency: str = "USD") -> pd.DataFrame:
    """``[ric, name, isin, sedol, cusip, country, currency, ticker]``, one row per RIC, blanks as None.

    An index RIC's attributes describe its publisher (``.SPX``: Cboe Exchange Inc, with Cboe's ISIN), so they are
    blanked and its currency is ``level_currency``. A US ticker is the RIC's own (``BRKb.N`` -> ``BRK.B``, where LSEG
    writes ``BRK B``) and trades in dollars; a London ticker is LSEG's (``SJP.L`` is STJ: London RICs do not spell
    their tickers) and trades in pence - ``GBX``, never ``GBp``, which a case-blind database reads as pounds.
    """
    out = _take(raw, f.INSTRUMENT_COLUMNS, "instruments")
    for col in out.columns[1:]:
        out[col] = rics.blank_to_none(out[col])
    index = out["ric"].map(rics.is_index)
    for col in ("name", "isin", "sedol", "cusip", "ticker"):
        out[col] = out[col].where(~index, None)
    london = out["ric"].map(rics.exchange_mic).eq("XLON")
    out["ticker"] = out["ticker"].where(london, out["ric"].map(rics.ticker)).where(~index, None)
    currency = pd.Series(None, index=out.index, dtype="object")
    out["currency"] = currency.mask(out["country"].eq("US"), "USD").mask(london, PENCE).mask(index, level_currency)
    return out.drop_duplicates("ric").reset_index(drop=True)


def domicile(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, domicile]``: the company's country (ISO code) from a one-field answer, whatever LSEG calls the column."""
    if raw.empty or "Instrument" not in raw.columns or len(raw.columns) < 2:
        return pd.DataFrame(columns=["ric", "domicile"])
    value = rics.blank_to_none(raw[next(c for c in raw.columns if c != "Instrument")])
    out = pd.DataFrame({"ric": raw["Instrument"], "domicile": value})
    out["domicile"] = out["domicile"].where(out["domicile"].astype(str).str.fullmatch(r"[A-Z]{2}"), None)
    return out.drop_duplicates("ric").reset_index(drop=True)


def classification(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, sector, industry_group, industry]``: GICS, else the TRBC sector mapped to the nearest GICS sector."""
    out = _take(raw, f.CLASSIFICATION_COLUMNS, "classification")
    for col in out.columns[1:]:
        out[col] = rics.blank_to_none(out[col])
    trbc = out["trbc_sector"].map(f.TRBC_TO_GICS)
    has_gics = out["sector"].notna()
    out["sector"] = out["sector"].where(has_gics, trbc)
    out["industry_group"] = out["industry_group"].where(has_gics, None)
    out["industry"] = out["industry"].where(has_gics, None)
    return out.drop(columns="trbc_sector").drop_duplicates("ric").reset_index(drop=True)


def companies(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, permid, incorporated, trbc_industry]`` (:data:`~.fields.UK_COMPANY_FIELDS`), blanks as None."""
    out = _take(raw, f.UK_COMPANY_COLUMNS, "companies")
    for col in out.columns[1:]:
        out[col] = rics.blank_to_none(out[col])
    out["permid"] = out["permid"].map(lambda v: None if v is None else str(v).split(".")[0])
    return out.drop_duplicates("ric").reset_index(drop=True)


def listings(attrs: pd.DataFrame, classes: pd.DataFrame | None = None) -> pd.DataFrame:
    """The frame :func:`~research_core.data.identity.ensure_listings` takes, one row per RIC.

    A share's company is its CUSIP's first six characters (GOOG and GOOGL, BRK.A and BRK.B share one) in
    ``issuer_anchor``. A company registered abroad has no CUSIP (Linde, Aon: Irish ISINs), so ``issuer_isin``
    names it by its share's ISIN instead - the pipeline creates those companies under that code. Where ``attrs``
    carries :func:`companies`' columns (London), ``issuer_permid`` names a company without a CUSIP by LSEG's PermID -
    one company whatever its share lines' ISINs - and an investment trust is a ``fund``, not an equity. An index
    level has no company.
    """
    a = attrs.merge(classes, on="ric", how="left") if classes is not None else attrs
    index = a["ric"].map(rics.is_index)
    cusips = [None if i else rics.cusip6(c, s) for i, c, s in zip(index, a["cusip"], a["isin"], strict=True)]
    nothing = pd.Series(None, index=a.index, dtype="object")
    permid = (a["permid"] if "permid" in a.columns else nothing).where(~index & pd.Series(cusips).isna().to_numpy())
    trust = (a["trbc_industry"] if "trbc_industry" in a.columns else nothing).eq(f.INVESTMENT_TRUST_INDUSTRY)
    kind = np.where(index, "index", np.where(trust, "fund", "equity"))
    return pd.DataFrame(
        {
            "anchor": a["ric"],
            "issuer_anchor": cusips,
            "issuer_permid": permid.where(permid.notna(), None),
            "issuer_isin": [
                isin if not i and c is None and not isinstance(p, str) and isinstance(isin, str) else None
                for i, c, p, isin in zip(index, cusips, permid, a["isin"], strict=True)
            ],
            "incorporated": a["incorporated"] if "incorporated" in a.columns else nothing,
            "issuer_name": a["name"],
            "issuer_domicile": (a["domicile"] if "domicile" in a.columns else pd.Series(None, index=a.index)).where(
                ~index, None
            ),
            "name": a["name"].fillna(a["ric"]).astype(str).str[:256],
            "instrument_type": kind,
            "share_class": a["ric"].map(rics.share_class),
            "exchange_mic": a["ric"].map(rics.exchange_mic),
            "currency": a["currency"],
            **(
                {"sector": a["sector"], "industry_group": a["industry_group"], "industry": a["industry"]}
                if classes is not None
                else {}
            ),
        }
    )


def share_lines(attrs: pd.DataFrame) -> dict[str, str]:
    """``{code: the RIC that anchors its share line}`` for every code in ``attrs``.

    One security is one listing, whatever LSEG called it over the years: RICs with the same ISIN (else CUSIP) are one
    share line - a ticker change (``HGR.N^B16`` -> ``HNGR.N^J22``) or a delisting and relisting retires a RIC, not
    the shares. The line is anchored by its live RIC, else the most recently retired one. A code that is not a RIC
    (LSEG's member feed has carried SEDOLs) maps to the RIC sharing its SEDOL, ISIN or CUSIP, or is left out.
    """
    shaped = attrs[attrs["ric"].map(rics.is_ric)]
    out: dict[str, str] = {}
    key = shaped["isin"].fillna(shaped["cusip"])
    for _, group in shaped.assign(_key=key).groupby("_key", dropna=True):
        ranked = sorted(
            group["ric"],
            key=lambda r: (not rics.is_retired(r), rics.retired_on(r) or pd.Timestamp.min, r),
        )
        out |= dict.fromkeys(group["ric"], ranked[-1])
    for ric in shaped["ric"]:
        out.setdefault(ric, ric)  # no ISIN or CUSIP: its own line
    for _, row in attrs[~attrs["ric"].map(rics.is_ric)].iterrows():
        code = str(row["ric"])
        hits = shaped[shaped["sedol"].eq(code)]  # the code itself is a SEDOL
        for col in ("isin", "cusip", "sedol"):
            if hits.empty and pd.notna(row[col]):
                hits = shaped[shaped[col].eq(row[col])]
        if not hits.empty:
            out[code] = out[str(hits["ric"].iloc[0])]
    return out


def listing_identifiers(attrs: pd.DataFrame) -> pd.DataFrame:
    """``[ric, id_type, value]``: ISIN, CUSIP, SEDOL and - for live RICs only, tickers being reused - the ticker."""
    rows = []
    for id_type, col in (("ISIN", "isin"), ("CUSIP", "cusip"), ("SEDOL", "sedol"), ("TICKER", "ticker")):
        part = attrs[["ric", col]].dropna().rename(columns={col: "value"}).assign(id_type=id_type)
        if id_type == "TICKER":
            part = part[~part["ric"].map(rics.is_retired) & ~part["ric"].map(rics.is_index)]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)[["ric", "id_type", "value"]]


# ---------------------------------------------------------------- prices


def price_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, trade_date, open, high, low, close, volume]`` from a long ``get_history`` answer - unadjusted.

    Open, high, low and volume may be missing (an index level has no volume); the close is required - the last trade
    (``TRDPRC_1``), or London's official close (``OFF_CLOSE``). LSEG can pad a retired RIC's series past its end by
    repeating the last bar without volume: rows after a retired share's last day with volume are dropped.
    """
    if raw.empty:
        return pd.DataFrame(columns=["ric", "trade_date", "open", "high", "low", "close", "volume"])
    close = next((c for c in ("OFF_CLOSE", "MID_PRICE") if c in raw.columns), "TRDPRC_1")
    missing = [c for c in ("Instrument", "Date", close) if c not in raw.columns]
    if missing:
        raise MissingColumnsError(f"LSEG prices: no column(s) {missing}; it sent {list(raw.columns)}")
    out = raw.rename(columns={"Instrument": "ric", "Date": "trade_date", **f.HISTORY_COLUMNS})
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = np.nan
    return _clean_bars(out[["ric", "trade_date", "open", "high", "low", "close", "volume"]])


def fx_bars(raw: pd.DataFrame, in_dollars: Mapping[str, bool]) -> pd.DataFrame:
    """``[ric, trade_date, close]``: US dollars per unit of each currency, from a ``MID_PRICE`` history.

    ``in_dollars``: RIC -> whether LSEG quotes it in dollars per unit (``GBP=``: 1.27); one quoted in units per dollar
    (``ZAR=``: 18.2) is inverted, so every rate reads the same way.
    """
    bars = price_bars(raw)
    per_dollar = ~bars["ric"].map(in_dollars).fillna(True).astype(bool)
    bars.loc[per_dollar, "close"] = 1.0 / bars.loc[per_dollar, "close"]
    return bars[bars["close"] > 0][["ric", "trade_date", "close"]].reset_index(drop=True)


def current_currencies(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, currency]``: the currency each RIC is quoted in today (:data:`~.fields.QUOTE_CURRENCY_FIELDS`), ours."""
    out = _take(raw, {"CF_CURR": "currency"}, "quote currencies")
    out["currency"] = rics.blank_to_none(out["currency"]).map(
        lambda c: None if c is None else f.CURRENCY_CODES.get(c, c)
    )
    return out.dropna().drop_duplicates("ric").reset_index(drop=True)


#: A change of quote currency shows as a move of more than this (log) ...
QUOTE_SWITCH_MIN_MOVE = float(np.log(5.0))
#: ... that the exchange rate explains to within this (log): the price in pounds the day before and the day after.
QUOTE_SWITCH_TOLERANCE = float(np.log(1.3))
#: Below a penny a day's 5x move is a near-worthless line's noise, not a change of quote currency (Quarterly High
#: Income Trust, 2002: 0.02 to 0.50 matched pence per lari, and four years of pence were read as lari).
QUOTE_SWITCH_MIN_POUNDS = 0.01
#: A move the line's own share count explains to within this (log) is a capital change, whatever the exchange rate
#: says: Durlacher's (now Panmure Liberum) 1-for-140 consolidation of November 2002, which LSEG lacks, moved its price
#: x108 - pounds to pence - while its shares fell from 570m to 4m. Counts are monthly, hence the width.
QUOTE_COUNT_TOLERANCE = float(np.log(2.0))


def count_ratios(counts: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    """Per day: the first share count on or after it over the last before it, each within :data:`SHARE_COUNT_DAYS`.

    ``counts``: shares outstanding by ``as_of_date``. NaN where either side has no count.
    """
    out = np.full(len(days), np.nan)
    s = counts[counts > 0].sort_index()
    if s.empty:
        return out
    dates = pd.DatetimeIndex(pd.to_datetime(s.index))
    values = s.to_numpy(dtype=float)
    after = dates.searchsorted(days, side="left")  # first count on or after the day
    before = after - 1  # last count before it
    near = pd.Timedelta(days=SHARE_COUNT_DAYS)
    ok = (before >= 0) & (after < len(dates))
    i = np.flatnonzero(ok)
    a, b = after[i], before[i]
    close_enough = (dates[a] - days[i] <= near) & (days[i] - dates[b] <= near)
    out[i[close_enough]] = values[a[close_enough]] / values[b[close_enough]]
    return out


def quote_currencies(
    bars: pd.DataFrame,
    last: str,
    pounds_per_unit: pd.DataFrame,
    quiet: set[pd.Timestamp] | None = None,
    counts: pd.Series | None = None,
) -> pd.Series:
    """Each bar's currency, for one London listing whose history LSEG answers as quoted, with no currency.

    ``bars``: ``[trade_date, close]`` in date order; ``last``: the latest bar's currency (today's quote, ``GBX`` when
    LSEG has none - a retired line); ``pounds_per_unit``: ``date x currency`` (``GBX``, ``GBP``, ``USD``...) - the value
    in pounds of one unit; ``quiet``: days a capital change explains (a 1-for-100 consolidation is no switch to pence);
    ``counts``: the line's shares outstanding by date, which explain the capital changes LSEG has no action for.
    Walking back from the latest bar, a day whose move is more than 5x and is the exchange rate between two currencies
    to within 30% starts the earlier currency's run - the price in pounds was the same either side - unless the price
    is under a penny (:data:`QUOTE_SWITCH_MIN_POUNDS`) or the share count moved the other way to match
    (:data:`QUOTE_COUNT_TOLERANCE`).
    """
    quiet = quiet or set()
    days = pd.DatetimeIndex(pd.to_datetime(bars["trade_date"]))
    close = bars["close"].to_numpy(dtype=float)
    rates = pounds_per_unit.sort_index().reindex(days, method="ffill")
    shares = count_ratios(counts, days) if counts is not None else np.full(len(days), np.nan)
    out = [last] * len(close)
    current = last
    for i in range(len(close) - 1, 0, -1):
        out[i] = current
        if close[i] <= 0 or close[i - 1] <= 0 or days[i] in quiet or current not in rates.columns:
            continue
        move = float(np.log(close[i] / close[i - 1]))
        if abs(move) <= QUOTE_SWITCH_MIN_MOVE:
            continue
        here = rates.iloc[i]
        if not close[i] * here[current] >= QUOTE_SWITCH_MIN_POUNDS:
            continue
        if shares[i] > 0 and abs(move + float(np.log(shares[i]))) < QUOTE_COUNT_TOLERANCE:
            continue
        best: tuple[float, str] | None = None
        for before in rates.columns:
            if before == current or not (here[before] > 0 and here[current] > 0):
                continue
            miss = abs(move - float(np.log(here[before] / here[current])))
            if miss < QUOTE_SWITCH_TOLERANCE and (best is None or miss < best[0]):
                best = (miss, str(before))
        if best is not None:
            current = best[1]
    if len(out):
        out[0] = current
    return pd.Series(out, index=bars.index, dtype="object")


def daily_prices(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """``[ric, trade_date, open, high, low, close, volume]`` from a refresh's ``get_data`` answer - close and volume.

    A retired RIC comes back as its last bar repeated, dated on its last day (Lehman's 2008-09-12 in a 2026 window):
    only rows inside ``[start, end]`` are kept. An answer with the price's currency (London's,
    :data:`~.fields.PRICE_CURRENCY_FIELDS`: today's quote) adds a ``currency`` column, pence as ``GBX``.
    """
    quoted = "Currency" in raw.columns
    columns = {**f.DAILY_COLUMNS, **({"Currency": "currency"} if quoted else {})}
    out = _take(raw, columns, "daily prices").assign(open=np.nan, high=np.nan, low=np.nan)
    keep = ["ric", "trade_date", "open", "high", "low", "close", "volume"]
    bars = _clean_bars(out[[*keep, *(["currency"] if quoted else [])]])
    if quoted:
        bars["currency"] = rics.blank_to_none(bars["currency"]).map(
            lambda c: None if c is None else f.CURRENCY_CODES.get(str(c), str(c))
        )
    inside = (bars["trade_date"] >= pd.Timestamp(start)) & (bars["trade_date"] <= pd.Timestamp(end))
    return bars[inside].reset_index(drop=True)


def _clean_bars(out: pd.DataFrame) -> pd.DataFrame:
    out = out.assign(trade_date=_dates(out["trade_date"]))
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["trade_date", "close"])
    out = out[out["close"] > 0].drop_duplicates(["ric", "trade_date"], keep="first").sort_values(["ric", "trade_date"])
    # a range the close lies outside (LSEG's bars of 2006-10-24 mix two days): the close is kept, the range is not
    torn = (out["high"] < out["low"]) | (out["close"] > out["high"] * 1.0001) | (out["close"] < out["low"] * 0.9999)
    out.loc[torn, ["open", "high", "low"]] = np.nan
    retired = out["ric"].map(rics.is_retired)
    traded = out["volume"].fillna(0) > 0
    last = out[traded].groupby("ric")["trade_date"].max()
    padded = retired & (out["trade_date"] > out["ric"].map(last))
    out = out[~padded]
    out["volume"] = out["volume"].round().astype("Int64")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- dividends and splits


def dividends(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, ex_date, amount]``: cash per share as of the ex-date; one day's several payments summed.

    With a ``currency`` column when the answer has one (:data:`~.fields.DIVIDEND_CURRENCY_FIELDS`: London's are in
    pounds while its prices are in pence).
    """
    columns = {**f.DIVIDEND_COLUMNS, **({"Currency": "currency"} if "Currency" in raw.columns else {})}
    out = _take(raw, columns, "dividends")
    out = out.assign(ex_date=_dates(out["ex_date"]), amount=pd.to_numeric(out["amount"], errors="coerce"))
    out = out.dropna(subset=["ex_date", "amount"])
    out = out[out["amount"] > 0]
    summed = pd.DataFrame(out.groupby(["ric", "ex_date"], as_index=False)["amount"].sum())
    if "currency" not in out.columns:
        return summed
    return summed.merge(out.groupby(["ric", "ex_date"], as_index=False)["currency"].first(), on=["ric", "ex_date"])


def capital_changes(
    raw: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    columns: Mapping[str, str] = f.CAPITAL_CHANGE_COLUMNS,
    days_after: int = 1,
) -> pd.DataFrame:
    """``[ric, action_type, ex_date, ratio, verified]``: splits (``verified``: the price confirms them), spin-offs.

    ``ratio`` is new shares per old share (4.0 for a 4-for-1 split, 0.125 for a 1-for-8 reverse split). A factor
    near 1 (a spin-off, a rights issue) is kept as ``spinoff`` - it must not rescale prices like a split.
    ``closes``: ``[ric, trade_date, close]`` unadjusted, around the events. ``columns`` names the date LSEG was asked
    (London's ex-date, :data:`~.fields.UK_CAPITAL_CHANGE_COLUMNS`); the price may confirm a split on that day or on
    one of the ``days_after`` trading days after it.
    """
    out = _take(raw, columns, "capital changes")
    out = out.assign(ex_date=_dates(out["ex_date"]), factor=pd.to_numeric(out["factor"], errors="coerce"))
    out = out.dropna(subset=["ex_date", "factor"])
    out = out[(out["factor"] > 0) & ((out["factor"] - 1).abs() > 1e-9)].drop_duplicates(["ric", "ex_date", "factor"])
    # capital changes on one day act together (Motorola, 2011-01-04: a 1-for-7 reverse split and a spin-off): their
    # factors multiply - keeping one of them would leave the price's jump unexplained
    out = out.groupby(["ric", "ex_date"])["factor"].prod().reset_index()
    is_split = (out["factor"] <= SPLIT_FACTOR_BAND[0]) | (out["factor"] >= 1 / SPLIT_FACTOR_BAND[1])
    rows = []
    by_ric = {str(r): g.set_index("trade_date")["close"].sort_index() for r, g in closes.groupby("ric")}
    for ric, ex_date, factor, split in zip(out["ric"], out["ex_date"], out["factor"], is_split, strict=True):
        if not split:
            rows.append((ric, "spinoff", ex_date, 1.0 / factor, True))
            continue
        day, ok = _split_day(by_ric.get(ric), pd.Timestamp(ex_date), float(factor), days_after)
        rows.append((ric, "split", day, 1.0 / factor, ok))
    # boolean even when empty (a refresh's few days often hold no change): an empty object column used as a mask
    # selects columns, not rows
    return pd.DataFrame(rows, columns=["ric", "action_type", "ex_date", "ratio", "verified"]).astype({"verified": bool})


def _split_day(
    px: pd.Series | None, date: pd.Timestamp, factor: float, days_after: int = 1
) -> tuple[pd.Timestamp, bool]:
    """The vendor's date or one of the ``days_after`` trading days after it that the price moves by ``factor`` on.

    Flagged when none does.
    """
    if px is None or px.empty:
        return date, False
    moves = (px / px.shift(1)).dropna()
    candidates = [date, *px.index[px.index > date][:days_after]]
    scores = {c: abs(float(np.log(moves[c] / factor))) for c in candidates if c in moves.index and moves[c] > 0}
    if not scores:
        return date, False
    best = min(scores, key=scores.__getitem__)
    return (best, True) if scores[best] <= SPLIT_TOLERANCE else (date, False)


def confirm_by_share_count(
    changes: pd.DataFrame,
    closes: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    days_after: int = 1,
    held: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``changes`` (:func:`capital_changes`) with the splits the shares outstanding confirm where the price could not.

    Two kinds the price alone misses, both kept only when the count moves by the split's inverse
    (:data:`SHARE_COUNT_TOLERANCE`) between the last count before the day and the first after it:

    1. a split of LSEG's the price moved its way on (by :data:`JUMP` at least) but not by its ratio - one that came
       with a return of cash (Pinewood's 1-for-20 of 2024 moved the price x8);
    2. a jump no action explains (:data:`JUMP`) - a split LSEG lacks (Grainger's 5-for-1 of 2005) or dates weeks
       off (Capital & Regional's 1-for-10: 2019-11-15 at LSEG, the price on 2020-01-15) - when the count's ratio is a
       whole exchange (:data:`WHOLE_EXCHANGES`), which it is then stored as.

    A collapse or a bad price leaves the count as it was, so it stays unexplained. ``closes``: ``[ric, trade_date,
    close]`` unadjusted, the whole history; ``counts``: ``[ric, as_of_date, shares]`` outstanding; ``held``:
    ``[ric, ex_date, ratio]``, the splits and spin-offs already stored (under another of the line's codes, say) -
    they explain their days too.
    """
    if counts.empty or closes.empty:
        return changes
    out = changes.copy()
    by_ric = {str(r): g.set_index("trade_date")["close"].sort_index() for r, g in closes.groupby("ric")}
    shares = {str(r): g.set_index("as_of_date")["shares"].sort_index() for r, g in counts.groupby("ric")}

    def count_ratio(ric: str, day: pd.Timestamp) -> float:
        """Shares outstanding after ``day`` per share before it; NaN without a count on both sides."""
        s = shares.get(ric)
        if s is None:
            return float("nan")
        near = pd.Timedelta(days=SHARE_COUNT_DAYS)
        before, after = s[(s.index < day) & (s.index >= day - near)], s[(s.index >= day) & (s.index <= day + near)]
        if before.empty or after.empty or before.iloc[-1] <= 0:
            return float("nan")
        return float(after.iloc[0] / before.iloc[-1])

    def whole(ratio: float) -> float | None:
        """The whole exchange ``ratio`` is (:data:`WHOLE_EXCHANGES`), if any."""
        n = max(round(ratio if ratio >= 1 else 1 / ratio), 1)
        for exact in (float(n) if ratio >= 1 else 1 / n, *WHOLE_EXCHANGES):
            if abs(np.log(ratio / exact)) <= WHOLE_TOLERANCE:
                return exact
        return None

    def counted(ratio: float, ric: str, day: pd.Timestamp) -> bool:
        c = count_ratio(ric, day)
        return bool(np.isfinite(c) and c > 0 and abs(np.log(c / ratio)) <= SHARE_COUNT_TOLERANCE)

    # 1. LSEG's splits the price moved towards but not far enough
    open_ = out[(out["action_type"] == "split") & ~out["verified"]]
    for i, ric, ex_date, ratio in zip(open_.index, open_["ric"], open_["ex_date"], open_["ratio"], strict=True):
        day, px = pd.Timestamp(ex_date), by_ric.get(str(ric))
        if px is None or not counted(float(ratio), str(ric), day):
            continue
        rel = (px / px.shift(1)).dropna()
        moves = pd.Series(np.log(rel[rel > 0].to_numpy()), index=rel[rel > 0].index)
        days = [d for d in [day, *px.index[px.index > day][:days_after]] if d in moves.index]
        # its way: a 1-for-20 (ratio 0.05) raises the price
        towards = [d for d in days if -moves[d] * np.sign(np.log(float(ratio))) >= JUMP]
        if towards:
            out.loc[i, ["ex_date", "verified"]] = [max(towards, key=lambda d: abs(moves[d])), True]

    # 2. jumps no kept action explains
    def factors(rows: pd.DataFrame) -> pd.Series:
        """``(ric, day) -> ratio``: one day's actions act together."""
        rows = rows.assign(ric=rows["ric"].astype(str), ex_date=pd.to_datetime(rows["ex_date"]))
        return rows.groupby(["ric", "ex_date"])["ratio"].prod()

    # a day LSEG's answer explains takes its factor; another, the stored actions' (under the line's other codes)
    day_factor = factors(out[out["verified"]])
    if held is not None and not held.empty:
        day_factor = day_factor.combine_first(factors(held))
    explained = {str(r): g.droplevel(0) for r, g in day_factor.groupby(level=0)}
    found = []
    for ric, px in by_ric.items():
        if ric not in shares:
            continue
        moves = (px / px.shift(1)).dropna()
        moves = moves[moves > 0]
        adjusted = moves * explained.get(ric, pd.Series(dtype=float)).reindex(moves.index).fillna(1.0)
        for day in adjusted.index[np.abs(np.log(adjusted)) > JUMP]:
            ratio = count_ratio(ric, pd.Timestamp(day))
            if not (np.isfinite(ratio) and ratio > 0) or abs(np.log(adjusted[day] * ratio)) > SHARE_COUNT_TOLERANCE:
                continue
            exact = whole(ratio)
            if exact is not None:
                found.append((ric, "split", pd.Timestamp(day), exact, True))
    extra = pd.DataFrame(found, columns=out.columns).astype({"verified": bool})
    return pd.concat([p for p in (out, extra) if not p.empty] or [out], ignore_index=True)


# ---------------------------------------------------------------- share counts


def share_counts(shares_raw: pd.DataFrame, float_raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, kind, as_of_date, shares]``: ``outstanding`` and ``float`` counts per share class.

    ``float`` is the part that trades: outstanding x the free-float percentage known on that date.
    """
    s = _take(shares_raw, f.SHARES_COLUMNS, "shares outstanding")
    s = s.assign(as_of_date=_dates(s["as_of_date"]), shares=pd.to_numeric(s["shares"], errors="coerce"))
    s = s.dropna().query("shares > 0").drop_duplicates(["ric", "as_of_date"]).sort_values("as_of_date")
    p = _take(float_raw, f.FREE_FLOAT_COLUMNS, "free float")
    p = p.assign(as_of_date=_dates(p["as_of_date"]), float_pct=pd.to_numeric(p["float_pct"], errors="coerce"))
    p = p.dropna().query("float_pct > 0 and float_pct <= 100").drop_duplicates(["ric", "as_of_date"])
    outstanding = s.assign(kind="outstanding")
    if p.empty or s.empty:
        return outstanding[["ric", "kind", "as_of_date", "shares"]].reset_index(drop=True)
    joined = pd.merge_asof(s, p.sort_values("as_of_date"), on="as_of_date", by="ric", direction="backward")
    floated = joined.dropna(subset=["float_pct"]).assign(
        kind="float", shares=lambda d: d["shares"] * d["float_pct"] / 100.0
    )
    cols = ["ric", "kind", "as_of_date", "shares"]
    return pd.concat([outstanding[cols], floated[cols]], ignore_index=True)


# ---------------------------------------------------------------- statements


def announcements(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, period_end, announced_at]``: when each fiscal quarter's results were first announced."""
    out = _take(raw, f.ANNOUNCEMENT_COLUMNS, "announcements")
    out = out.assign(
        period_end=_dates(out["period_end"]),
        announced_at=pd.to_datetime(out["announced_at"], errors="coerce").dt.tz_localize(None),
    )
    return out.dropna().drop_duplicates(["ric", "period_end"]).reset_index(drop=True)


def statements(raw: pd.DataFrame, announced: pd.DataFrame) -> pd.DataFrame:
    """Financial facts ``[ric, item, period_type, period_start, period_end, value, unit, published_at]``.

    Flows are the quarter's amount (``Q``, starting three months before the end), balances the figure on the day
    (``I``). Each is dated by the quarter's announcement; without one, :data:`FALLBACK_REPORTING_LAG_DAYS` after the
    period end - later than the truth, never earlier.
    """
    columns = {"Period End Date": "period_end", **{col: item for col, item, _ in f.STATEMENT_ITEMS.values()}}
    wide = _take(raw, columns, "statements")
    wide = wide.assign(period_end=_dates(wide["period_end"])).dropna(subset=["period_end"])
    kinds = {item: kind for _, item, kind in f.STATEMENT_ITEMS.values()}
    long = wide.melt(id_vars=["ric", "period_end"], var_name="item", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"]).drop_duplicates(["ric", "item", "period_end"])
    flow = long["item"].map(kinds) == "flow"
    long["period_type"] = np.where(flow, "Q", "I")
    long["period_start"] = (long["period_end"] - pd.DateOffset(months=3) + pd.Timedelta(days=1)).where(flow)
    long["unit"] = np.where(long["item"] == "shares_outstanding", "shares", "USD")
    return _dated(long, announced)


def _dated(long: pd.DataFrame, announced: pd.DataFrame) -> pd.DataFrame:
    """``long``'s facts with ``published_at``: their period's announcement, else a conservative date.

    Without an announcement a figure is dated :data:`FALLBACK_REPORTING_LAG_DAYS` after its period end. LSEG answers the
    announcement dates beside a period end of another field family, lined up by row: for some companies a quarter gets
    the previous quarter's announcement. One made before the period ended cannot be its own - such a figure is dated
    conservatively too.
    """
    long = long.merge(announced, on=["ric", "period_end"], how="left")
    own = long["announced_at"].dt.normalize() >= long["period_end"]
    fallback = long["period_end"] + pd.Timedelta(days=FALLBACK_REPORTING_LAG_DAYS)
    long["published_at"] = long["announced_at"].where(own).fillna(fallback)
    cols = ["ric", "item", "period_type", "period_start", "period_end", "value", "unit", "published_at"]
    return long[cols].reset_index(drop=True)


def annual_statement(raw: pd.DataFrame, field: str, announced: pd.DataFrame) -> pd.DataFrame:
    """One yearly item's facts (:data:`~.fields.ANNUAL_ITEMS`), in :func:`statements`' columns.

    A flow is the fiscal year's amount (``FY``, starting a year before its end), a balance the figure at the year's
    end (``I``); payments are stored as the amount paid. The value's column is whichever LSEG named it; an answer
    without one (LSEG had nothing), or one LSEG labels an estimate, gives no facts.
    """
    item, kind, _ = f.ANNUAL_ITEMS[field]
    cols = ["ric", "item", "period_type", "period_start", "period_end", "value", "unit", "published_at"]
    values = [c for c in raw.columns if c not in ("Instrument", "Period End Date")]
    if raw.empty or not values or "Period End Date" not in raw.columns or f.ESTIMATE_MARKER in str(values[0]):
        return pd.DataFrame(columns=cols)
    long = pd.DataFrame(
        {
            "ric": raw["Instrument"],
            "period_end": _dates(raw["Period End Date"]),
            "value": pd.to_numeric(raw[values[0]], errors="coerce"),
        }
    ).dropna()
    long = long.drop_duplicates(["ric", "period_end"]).assign(item=item)
    if item in f.PAID:
        long["value"] = long["value"].abs()
    flow = kind == "flow"
    long["period_type"] = "FY" if flow else "I"
    long["period_start"] = (long["period_end"] - pd.DateOffset(years=1) + pd.Timedelta(days=1)) if flow else pd.NaT
    long["unit"] = "USD"
    return _dated(long, announced)


def half_year_statements(halves: pd.DataFrame, years: pd.DataFrame, *, unit: str) -> pd.DataFrame:
    """London statements as facts, in :func:`statements`' columns: first halves and fiscal years, as first reported.

    ``halves`` and ``years`` answer :data:`~.fields.UK_STATEMENT_FIELDS` half-yearly and yearly. A year is stored as
    LSEG reports it (``FY``), not as its halves added up; a half ending on a fiscal year's end is that year's second
    half and adds nothing; any other half is the year so far (``YTD``, six months), from which research builds the
    trailing year. Balances are the figure on the day (``I``). Each is dated by its own row's announcement; amounts
    are in ``unit`` (the ``Curn`` asked), share counts in shares; payments as the amount paid.
    """
    cols = ["ric", "item", "period_type", "period_start", "period_end", "value", "unit", "published_at"]
    columns = {
        "Period End Date": "period_end",
        **{col: item for col, item, _ in f.UK_STATEMENT_ITEMS.values()},
        "Original Announcement Date Time": "announced_at",
    }
    frames = [_take(raw, columns, "statements").assign(yearly=y) for raw, y in ((years, True), (halves, False))]
    wide = pd.concat([w for w in frames if not w.empty] or frames[:1], ignore_index=True)
    if wide.empty:
        return pd.DataFrame(columns=cols)
    wide = wide.assign(
        period_end=_dates(wide["period_end"]),
        announced_at=pd.to_datetime(wide["announced_at"], errors="coerce").dt.tz_localize(None),
        yearly=wide["yearly"].astype(bool),
    ).dropna(subset=["period_end"])
    year_ends = set(zip(wide.loc[wide["yearly"], "ric"], wide.loc[wide["yearly"], "period_end"], strict=True))
    second_half = [
        not y and (r, e) in year_ends for r, e, y in zip(wide["ric"], wide["period_end"], wide["yearly"], strict=True)
    ]
    wide = wide[~pd.Series(second_half, index=wide.index)]
    long = wide.melt(id_vars=["ric", "period_end", "announced_at", "yearly"], var_name="item", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"]).drop_duplicates(["ric", "item", "yearly", "period_end"])
    flow = long["item"].map({item: kind for _, item, kind in f.UK_STATEMENT_ITEMS.values()}) == "flow"
    long["period_type"] = np.where(flow, np.where(long["yearly"], "FY", "YTD"), "I")
    year_start = long["period_end"] - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    half_start = long["period_end"] - pd.DateOffset(months=6) + pd.Timedelta(days=1)
    long["period_start"] = year_start.where(long["yearly"], half_start).where(flow)
    long = long.drop_duplicates(["ric", "item", "period_type", "period_end"])  # a balance on a year's end: once
    paid = long["item"].isin(f.PAID)
    long.loc[paid, "value"] = long.loc[paid, "value"].abs()
    long["unit"] = np.where(long["item"] == "shares_outstanding", "shares", unit)
    own = long["announced_at"].dt.normalize() >= long["period_end"]
    fallback = long["period_end"] + pd.Timedelta(days=FALLBACK_REPORTING_LAG_DAYS)
    long["published_at"] = long["announced_at"].where(own).fillna(fallback)
    return long[cols].reset_index(drop=True)


# ---------------------------------------------------------------- I/B/E/S estimates


def _by_ending(raw: pd.DataFrame, endings: Mapping[str, str], what: str) -> pd.DataFrame:
    """``raw``'s ``Instrument`` plus the one column ending with each of ``endings``, renamed.

    LSEG names estimate columns after their measure ("Revenue - Mean", "EBIT - Mean"), so one reader serves them all;
    a missing or ambiguous column raises :class:`MissingColumnsError`.
    """
    if raw.empty:
        return pd.DataFrame(columns=["ric", *endings.values()])
    renames = {}
    for ending, name in endings.items():
        hits = [c for c in raw.columns if str(c).endswith(ending)]
        if len(hits) != 1:
            raise MissingColumnsError(
                f"LSEG {what}: {len(hits)} columns ending {ending!r}; it sent {list(raw.columns)}"
            )
        renames[hits[0]] = name
    if "Instrument" not in raw.columns:
        raise MissingColumnsError(f"LSEG {what}: no column 'Instrument'; it sent {list(raw.columns)}")
    return raw[["Instrument", *renames]].rename(columns={"Instrument": "ric", **renames})


def consensus(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, as_of_date, fiscal_period_end, mean, median, std_dev, estimates]``: one row per RIC and month end.

    A month without a mean (no analyst covered the company) is no row.
    """
    out = _by_ending(raw, f.CONSENSUS_COLUMNS, "consensus estimates")
    out = out.assign(
        as_of_date=_dates(out["as_of_date"]),
        fiscal_period_end=_dates(out["fiscal_period_end"]),
        **{c: pd.to_numeric(out[c], errors="coerce") for c in ("mean", "median", "std_dev", "estimates")},
    )
    out = out.dropna(subset=["as_of_date", "mean"]).drop_duplicates(["ric", "as_of_date"])
    return out.astype({"estimates": "Int64"}).reset_index(drop=True)


def actuals(raw: pd.DataFrame) -> pd.DataFrame:
    """``[ric, period_end, value, announced_at]``: each fiscal year's reported actual and when it was reported."""
    out = _by_ending(raw, f.ACTUAL_COLUMNS, "reported actuals")
    out = out.assign(
        period_end=_dates(out["period_end"]),
        value=pd.to_numeric(out["value"], errors="coerce"),
        announced_at=pd.to_datetime(out["announced_at"], errors="coerce").dt.tz_localize(None),
    )
    return out.dropna(subset=["period_end", "value"]).drop_duplicates(["ric", "period_end"]).reset_index(drop=True)


def revisions(raw: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """``[ric, as_of_date, up, down]``: how many analysts raised and lowered their estimate, on one date.

    The answer has no date of its own (asked for one day, ``as_of``); a RIC with neither count is no row.
    """
    out = _take(raw, f.REVISION_COLUMNS, "estimate revisions")
    out = out.assign(**{c: pd.to_numeric(out[c], errors="coerce") for c in ("up", "down")})
    out = out.dropna(subset=["up", "down"], how="all").drop_duplicates("ric")
    out = out.assign(as_of_date=pd.Timestamp(as_of)).astype({"up": "Int64", "down": "Int64"})
    return out[["ric", "as_of_date", "up", "down"]].reset_index(drop=True)


def in_shares_of_their_date(
    rows: pd.DataFrame, columns: list[str], date: str, splits: pd.DataFrame, basis: pd.Timestamp
) -> pd.DataFrame:
    """Per-share ``columns`` in the shares of each row's own ``date``, not LSEG's basis (the day it was fetched).

    LSEG restates per-share estimates for every later split, so a figure fetched again after a split would change.
    Multiplying by each split ratio with an ex-date in ``(date, basis]`` gives the figure as it was quoted then, which
    no later split alters - the way prices are stored, unadjusted. ``rows`` and ``splits`` (``[listing_id, ex_date,
    ratio]``, new shares per old) are keyed by ``listing_id``.
    """
    if rows.empty or splits.empty:
        return rows
    s = splits[(pd.to_datetime(splits["ex_date"]) <= basis) & (splits["ratio"] > 0)]
    hit = rows[["listing_id", date]].reset_index().merge(s, on="listing_id")
    hit = hit[pd.to_datetime(hit["ex_date"]) > hit[date]]
    factor = hit.groupby("index")["ratio"].prod().reindex(rows.index).fillna(1.0)
    return rows.assign(**{c: rows[c] * factor for c in columns})
