"""Notre Dame's Software Repository for Accounting and Finance: the Loughran-McDonald master dictionary.

One file, fetched with a generic client name (it is not the SEC's, which asks for a contact; nothing personal goes
to the file's host). The dictionary's 1993-2025 edition is the one :mod:`equity_research.ingest.vendors.sec.text`
names in its method.
"""

from __future__ import annotations

import requests

VENDOR = "sraf"
#: The master dictionary's CSV (linked from https://sraf.nd.edu/loughranmcdonald-master-dictionary/).
LM_DICTIONARY_URL = (
    "https://drive.usercontent.google.com/download?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t"
)


def fetch_dictionary(url: str = LM_DICTIONARY_URL, *, timeout: float = 120.0) -> bytes:
    """The dictionary's CSV bytes; a page that is not the CSV (a sign-in or warning page) is an error."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "equity-research"})
    resp.raise_for_status()
    if not resp.content.startswith(b"Word,"):
        raise RuntimeError(f"{url} did not return the Loughran-McDonald CSV")
    return resp.content
