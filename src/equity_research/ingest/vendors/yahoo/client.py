"""HTTP access to Yahoo Finance's chart endpoint (daily bars, dividends and splits of one symbol).

Unofficial and rate-limited: requests are spaced ``1 / max_per_second`` apart and 429 / 5xx responses wait and retry.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import requests

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


class YahooClient:
    """Rate-limited GET of chart responses."""

    def __init__(self, *, max_per_second: float = 1.0, max_retries: int = 5) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (equity_research)"})
        self._min_interval = 1.0 / max_per_second
        self._max_retries = max_retries
        self._last = 0.0

    def chart(self, symbol: str, start: dt.date | None = None) -> tuple[str, bytes]:
        """``(url, JSON body)`` of ``symbol``'s daily history from ``start`` (the whole history when ``None``)."""
        period1 = int(dt.datetime.combine(start, dt.time(), dt.UTC).timestamp()) if start else 0
        params = {
            "period1": str(period1),
            "period2": str(int(time.time()) + 86_400),
            "interval": "1d",
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
        url = CHART_URL.format(symbol=symbol)
        for attempt in range(1, self._max_retries + 1):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._session.get(url, params=params, timeout=60)
            except requests.RequestException as exc:
                log.warning("Yahoo %s: %s - retry %d", symbol, exc, attempt)
                time.sleep(5.0 * attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                log.warning("Yahoo %s: HTTP %d - retry %d", symbol, resp.status_code, attempt)
                time.sleep(10.0 * attempt)
                continue
            resp.raise_for_status()
            return resp.url, resp.content
        raise RuntimeError(f"Yahoo {symbol}: no answer after {self._max_retries} attempts")
