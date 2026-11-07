"""Calls into ``norgatedata`` (and so into Norgate Data Updater), one method per kind of answer.

Every time series is asked for as a pandas DataFrame indexed by date. Prices come split-adjusted (``CAPITAL``), with
the ``Unadjusted Close`` column beside them: the normaliser divides the two to recover the real prices of each day
and the splits between them.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

#: our index code -> (Norgate's "current & past" watchlist, its index name for constituent series, its level symbol)
INDEXES = {
    "SP500": ("S&P 500 Current & Past", "S&P 500", "$SPX"),
    "SP400": ("S&P MidCap 400 Current & Past", "S&P MidCap 400", "$MID"),
    "SP600": ("S&P SmallCap 600 Current & Past", "S&P SmallCap 600", "$SML"),
}
GICS_LEVELS = {1: "sector", 2: "industry_group", 3: "industry", 4: "sub_industry"}


class NorgateUnavailableError(RuntimeError):
    """Norgate Data Updater is not running (or the library cannot reach it)."""


class NorgateClient:
    """Norgate's answers as pandas objects."""

    def __init__(self) -> None:
        import norgatedata  # noqa: PLC0415 - loaded only when Norgate is used

        self._nd: Any = norgatedata

    def check(self) -> None:
        """Raise :class:`NorgateUnavailableError` unless NDU answers."""
        try:
            ok = bool(self._nd.status())
        except Exception as exc:
            raise NorgateUnavailableError(f"Norgate Data Updater does not answer: {exc}") from exc
        if not ok:
            raise NorgateUnavailableError("Norgate Data Updater is not running - start it, sign in, and retry")

    def symbols(self, watchlist: str) -> list[str]:
        """The symbols in a watchlist (``S&P 500 Current & Past`` includes every past member)."""
        return [str(s) for s in self._nd.watchlist_symbols(watchlist)]

    def describe(self, symbol: str) -> dict[str, Any]:
        """``{assetid, name, exchange, currency, domicile, first_quoted, last_quoted, sector... }`` of one symbol."""
        nd = self._nd
        out: dict[str, Any] = {
            "symbol": symbol,
            "assetid": nd.assetid(symbol),
            "name": nd.security_name(symbol),
            "exchange": nd.exchange_name(symbol),
            "currency": nd.currency(symbol),
            "domicile": nd.domicile(symbol),
            "first_quoted": nd.first_quoted_date(symbol),
            "last_quoted": nd.last_quoted_date(symbol),
        }
        for level, name in GICS_LEVELS.items():
            try:
                out[name] = nd.classification_at_level(symbol, "GICS", "Name", level)
            except Exception:
                out[name] = None
        return out

    def prices(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Daily bars, split-adjusted, with ``Unadjusted Close`` and ``Dividend`` columns."""
        frame: pd.DataFrame = self._nd.price_timeseries(
            symbol,
            stock_price_adjustment_setting=self._nd.StockPriceAdjustmentType.CAPITAL,
            padding_setting=self._nd.PaddingType.NONE,
            start_date=start,
            end_date=end,
            timeseriesformat="pandas-dataframe",
        )
        return frame

    def membership(self, symbol: str, index_name: str, start: str, end: str) -> pd.DataFrame:
        """``Index Constituent`` (1 / 0) per trading day for one index."""
        frame: pd.DataFrame = self._nd.index_constituent_timeseries(
            symbol,
            index_name,
            padding_setting=self._nd.PaddingType.NONE,
            start_date=start,
            end_date=end,
            timeseriesformat="pandas-dataframe",
        )
        return frame

    def shares(self, symbol: str) -> dict[str, Any]:
        """Today's shares outstanding and float (Norgate keeps no history of them)."""
        return {"outstanding": self._nd.sharesoutstanding(symbol), "float": self._nd.sharesfloat(symbol)}
