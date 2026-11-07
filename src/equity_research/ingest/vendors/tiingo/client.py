"""HTTP access to Tiingo's end-of-day API: one ticker's daily history, and the list of tickers it supports.

Every request carries the API key in the ``Authorization`` header; a 429 (the hourly or daily allowance used up)
raises :class:`TiingoThrottledError` at once rather than waiting an hour, and 5xx answers are retried.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from equity_research.settings import api_key

log = logging.getLogger(__name__)

PRICES_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
#: Every ticker Tiingo has end-of-day prices for, with its first and last day - a free file, no allowance used.
SUPPORTED_TICKERS_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"


class TiingoThrottledError(RuntimeError):
    """Tiingo refused a request: the account's allowance (per hour or per day) is used up."""


class TiingoClient:
    """Paced GETs of Tiingo answers; ``requests_made`` counts the ones that use the allowance."""

    def __init__(self, key: str | None = None, *, max_per_second: float = 2.0, max_retries: int = 4) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Token {key or api_key('tiingo')}", "Content-Type": "application/json"}
        )
        self._min_interval = 1.0 / max_per_second
        self._max_retries = max_retries
        self._last = 0.0
        self.requests_made = 0

    def _get(self, url: str, params: dict[str, str] | None = None) -> requests.Response | None:
        """The answer, or ``None`` for a 404 (Tiingo does not know the ticker)."""
        for attempt in range(1, self._max_retries + 1):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._session.get(url, params=params, timeout=60)
            except requests.RequestException as exc:
                log.warning("Tiingo %s: %s - retry %d", url, exc, attempt)
                time.sleep(5.0 * attempt)
                continue
            self.requests_made += 1
            if resp.status_code == 429:
                raise TiingoThrottledError(f"Tiingo: allowance used up after {self.requests_made} requests")
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                log.warning("Tiingo %s: HTTP %d - retry %d", url, resp.status_code, attempt)
                time.sleep(10.0 * attempt)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Tiingo {url}: no answer after {self._max_retries} attempts")

    def prices(self, ticker: str, start: str, end: str) -> tuple[str, bytes] | None:
        """``(url, JSON body)`` of one ticker's daily bars from ``start`` to ``end``; ``None`` for an unknown one."""
        resp = self._get(PRICES_URL.format(ticker=ticker), {"startDate": start, "endDate": end})
        return None if resp is None else (resp.url, resp.content)

    def download_supported_tickers(self, path: Path) -> str:
        """Save the supported-tickers file (a zipped CSV) to ``path``; returns its URL."""
        resp = requests.get(SUPPORTED_TICKERS_URL, timeout=300)  # public: no key sent, nothing counted
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return SUPPORTED_TICKERS_URL
