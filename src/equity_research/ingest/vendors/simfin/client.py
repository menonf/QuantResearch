"""HTTP access to SimFin's bulk downloads: one zipped, semicolon-separated CSV per dataset.

The API key goes in the ``Authorization`` header (``api-key <key>``); the answer redirects to a file store, and
``requests`` drops the header when it follows the redirect to another host.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from equity_research.settings import api_key

log = logging.getLogger(__name__)

BULK_URL = "https://prod.simfin.com/api/bulk-download/s3"


@dataclass(frozen=True)
class Dataset:
    """One bulk file: SimFin's dataset name, variant and market."""

    dataset: str
    variant: str | None = None
    market: str | None = "us"

    @property
    def params(self) -> dict[str, str]:
        """The query string SimFin expects (no ``variant`` / ``market`` when the dataset has none)."""
        out = {"dataset": self.dataset}
        if self.variant:
            out["variant"] = self.variant
        if self.market:
            out["market"] = self.market
        return out


#: our dataset name -> SimFin's file. Banks and insurers have statements of their own shape (deposits, premiums).
DATASETS: dict[str, Dataset] = {
    "companies": Dataset("companies"),
    "industries": Dataset("industries", market=None),
    "income_quarterly": Dataset("income", "quarterly"),
    "balance_quarterly": Dataset("balance", "quarterly"),
    "income_banks_quarterly": Dataset("income-banks", "quarterly"),
    "balance_banks_quarterly": Dataset("balance-banks", "quarterly"),
    "income_insurance_quarterly": Dataset("income-insurance", "quarterly"),
    "balance_insurance_quarterly": Dataset("balance-insurance", "quarterly"),
}


class SimFinRefusedError(RuntimeError):
    """SimFin refused a download (a wrong key, or a dataset the account's plan does not include)."""


class SimFinClient:
    """Downloads of SimFin bulk files."""

    def __init__(self, key: str | None = None) -> None:
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"api-key {key or api_key('simfin')}"})

    def download(self, name: str, path: Path) -> str:
        """Save dataset ``name`` (a key of :data:`DATASETS`) to ``path``; returns the request's URL (no key in it)."""
        ds = DATASETS[name]
        request = requests.Request("GET", BULK_URL, params=ds.params).prepare()
        resp = self._session.get(BULK_URL, params=ds.params, timeout=600, stream=True)
        if 400 <= resp.status_code < 500:
            raise SimFinRefusedError(f"SimFin {name}: HTTP {resp.status_code} {resp.text[:300]}")
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8 << 20):
                f.write(chunk)
        return str(request.url)
