"""HTTP access to SEC EDGAR within its fair-access rules.

The SEC asks automated clients to declare a contact in the User-Agent and to stay under 10 requests a second; clients
that do not are blocked. The User-Agent comes from the settings (``EQR_SEC_USER_AGENT`` or the keyring); requests are
spaced at least ``1 / max_per_second`` apart; 429 / 5xx responses wait (``Retry-After`` when given) and retry.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

from equity_research.settings import get_settings

log = logging.getLogger(__name__)

COMPANY_FACTS_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
#: Public CIK - CUSIP map built from SEC 13D / 13G filings (https://github.com/leoliu0/cik-cusip-mapping).
CIK_CUSIP_MAP_URL = "https://raw.githubusercontent.com/leoliu0/cik-cusip-mapping/master/cik-cusip-maps.csv"
#: The same file through a CDN, for when GitHub's file host is unreachable (it resets connections from some networks).
CIK_CUSIP_MAP_MIRROR = "https://cdn.jsdelivr.net/gh/leoliu0/cik-cusip-mapping@master/cik-cusip-maps.csv"


class SecClient:
    """Rate-limited GET for SEC endpoints."""

    def __init__(self, user_agent: str | None = None, *, max_per_second: float = 8.0, max_retries: int = 6) -> None:
        ua = user_agent or get_settings().sec_user_agent
        if not ua or "@" not in ua:
            raise RuntimeError(
                "the SEC requires a contact in the User-Agent: set EQR_SEC_USER_AGENT or keyring sec_user_agent"
            )
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self._min_interval = 1.0 / max_per_second
        self._max_retries = max_retries
        self._last = 0.0

    def _pace(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str, *, stream: bool = False, timeout: float = 120.0) -> requests.Response:
        """GET ``url``; 429 / 5xx wait and retry; other errors raise."""
        for attempt in range(1, self._max_retries + 1):
            self._pace()
            try:
                resp = self._session.get(url, timeout=timeout, stream=stream)
            except requests.RequestException as exc:
                wait = min(10.0 * attempt, 120.0)
                log.warning("SEC %s: %s - retry in %.0fs", url, exc, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else min(30.0 * attempt, 600.0)
                log.warning(
                    "SEC %s: HTTP %d - waiting %.0fs (%d/%d)", url, resp.status_code, wait, attempt, self._max_retries
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"SEC {url}: still failing after {self._max_retries} attempts")

    def get_json(self, url: str) -> Any:
        """Parsed JSON."""
        return self.get(url).json()

    def download(self, url: str, path: Path, *, chunk_bytes: int = 8 << 20) -> Path:
        """Stream ``url`` to ``path`` (via ``path.part``, renamed when complete)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        resp = self.get(url, stream=True, timeout=600.0)
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_bytes):
                fh.write(chunk)
        part.replace(path)
        return path
