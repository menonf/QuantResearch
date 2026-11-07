"""RIC (Reuters Instrument Code) spellings.

``AAPL.OQ``    a listing: ticker ``AAPL`` on Nasdaq (``.OQ``); ``BRKb.N`` is Berkshire's class B on NYSE
``BAC.N^I98``  a retired RIC: the ``^`` suffix marks the event that ended it (``I98`` = September 1998). Tickers
               are reused (old Dow Chemical ``DOW.N^I17`` vs Dow Inc ``DOW.N``), so a retired RIC is its own listing
``.SPX``       an index level (starts with a dot)
"""

from __future__ import annotations

import pandas as pd

#: Exchange suffix -> ISO 10383 market code. ``L`` is the London Stock Exchange (``VOD.L``): its tickers are not the
#: US ones (``AAL.L`` is Anglo American), so a listing's market decides which vendors' tickers can name it.
EXCHANGE_MIC = {
    "N": "XNYS",
    "OQ": "XNAS",
    "O": "XNAS",
    "A": "XASE",
    "P": "ARCX",
    "K": "BATS",
    "Z": "BATS",
    "L": "XLON",
}
#: Futures / volatility codes the S&P constituent feed has returned as members (``ES`` is also Eversource ``ES.N``).
NON_EQUITY_TICKERS = frozenset({"ES", "NQ", "VIX", "YM", "RTY"})


def is_index(ric: str) -> bool:
    """``.SPX`` and the like."""
    return ric.startswith(".")


def is_fx(ric: str) -> bool:
    """An exchange rate (``GBP=``, ``ZAR=``): a currency's RIC ends with ``=``."""
    return ric.endswith("=")


def is_retired(ric: str) -> bool:
    """A RIC with a ``^`` event suffix: the listing it named has ended."""
    return "^" in ric


def retired_on(ric: str) -> pd.Timestamp | None:
    """The month a retired RIC ended, from its suffix (``BAC.N^I98`` -> 1998-09-01); None for a live RIC."""
    if "^" not in ric:
        return None
    code = ric.split("^", 1)[1]
    if len(code) != 3 or not code[0].isalpha() or not code[1:].isdigit() or not "A" <= code[0].upper() <= "L":
        return None
    year = int(code[1:])
    return pd.Timestamp(year=2000 + year if year < 70 else 1900 + year, month=ord(code[0].upper()) - 64, day=1)


def is_ric(code: str) -> bool:
    """Shaped like a RIC (``AAPL.OQ``, ``BAC.N^I98``, ``.SPX``) - LSEG's feeds sometimes put a SEDOL in its place."""
    return "." in code and not code.replace(".", "").isdigit()


def _root(ric: str) -> str:
    base = ric.split("^", 1)[0]
    return base.rsplit(".", 1)[0] if "." in base else base


def ticker(ric: str) -> str:
    """The ticker part: ``AAPL.OQ`` -> ``AAPL``, ``BAC.N^I98`` -> ``BAC``, ``BRKb.N`` -> ``BRK.B``."""
    root = _root(ric)
    if len(root) > 1 and root[-1].islower():  # a share class as a lower-case letter
        return f"{root[:-1]}.{root[-1].upper()}"
    return root


def share_class(ric: str) -> str | None:
    """``BRKb.N`` -> ``B``; None when the RIC names no class (a London RIC's lower-case letter is none: ``VNHq.L``)."""
    root = _root(ric)
    if exchange_mic(ric) == "XLON":
        return None
    return root[-1].upper() if len(root) > 1 and root[-1].islower() else None


def exchange_mic(ric: str) -> str | None:
    """The market a RIC trades on, from its suffix (``.OQ`` -> ``XNAS``)."""
    base = ric.split("^", 1)[0]
    if is_index(ric) or "." not in base:
        return None
    return EXCHANGE_MIC.get(base.rsplit(".", 1)[1])


def is_non_equity(exchange_ticker: object, ric: str) -> bool:
    """A futures / volatility code returned as an index member (``ESc1``, not Eversource ``ES.N``)."""
    return str(exchange_ticker).upper() in NON_EQUITY_TICKERS and "." not in ric.split("^", 1)[0]


def cusip6(cusip: object, isin: object) -> str | None:
    """The 6-character issuer part of a CUSIP (or of a US / Canadian ISIN, which embeds the CUSIP)."""
    if isinstance(cusip, str) and len(cusip.strip()) >= 6:
        return cusip.strip()[:6]
    if isinstance(isin, str) and len(isin) == 12 and isin[:2] in ("US", "CA"):
        return isin[2:8]
    return None


def blank_to_none(s: pd.Series) -> pd.Series:
    """LSEG answers a missing value with ``""`` or ``<NA>``; both become None."""
    return s.astype("object").where(s.notna() & (s.astype(str).str.strip() != ""), None)
