"""Which vendor supplies what to the equity research, in priority order (environment overrides, else the defaults).

    vendor_preferences().prices               # ("refinitiv",) unless EQR_PRICE_VENDORS says otherwise

The one place a vendor is chosen: research, notebooks and the refresh read it from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class VendorPreferences:
    """Which vendor supplies what, in priority order: the first with data for a listing or index wins.

    The one place a vendor is chosen - research, notebooks and the refresh read it from here. Each list can be set in
    the environment as comma-separated codes (``EQR_PRICE_VENDORS=norgate,refinitiv``).
    """

    prices: tuple[str, ...] = ("refinitiv",)
    benchmarks: tuple[str, ...] = ("yahoo", "refinitiv")
    members: tuple[str, ...] = ("refinitiv",)
    #: each company's accounts from the first that has them; a later one only for periods before the first's start
    #: (the SEC's machine-readable accounts begin in 2009-2011; LSEG's reach back to 2000)
    fundamentals: tuple[str, ...] = ("sec", "refinitiv")
    classification: tuple[str, ...] = ("refinitiv", "financedatabase")
    #: the loads a build or refresh runs, in this order (the first creates the listings the others attach to; London's
    #: after New York's, each market's estimates after its listings)
    sources: tuple[str, ...] = (
        "refinitiv",
        "estimates",
        "refinitiv_uk",
        "estimates_uk",
        "yahoo",
        "financedatabase",
        "sec",
        "repository",  # last: its corrections read the prices the others loaded
    )


def _codes(env: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(env)
    return tuple(v.strip() for v in value.split(",") if v.strip()) if value else default


@lru_cache(maxsize=1)
def vendor_preferences() -> VendorPreferences:
    """The vendor preferences (environment overrides, else the defaults) - no database settings needed."""
    d = VendorPreferences()
    return VendorPreferences(
        prices=_codes("EQR_PRICE_VENDORS", d.prices),
        benchmarks=_codes("EQR_BENCHMARK_VENDORS", d.benchmarks),
        members=_codes("EQR_MEMBER_VENDORS", d.members),
        fundamentals=_codes("EQR_FUNDAMENTALS_VENDOR", d.fundamentals),
        classification=_codes("EQR_CLASSIFICATION_VENDORS", d.classification),
        sources=_codes("EQR_SOURCES", d.sources),
    )
