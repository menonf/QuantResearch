"""Country names as vendors write them -> ISO 3166 alpha-2 codes (``United States Of America`` -> ``US``)."""

from __future__ import annotations

import pycountry

#: Names pycountry does not resolve as vendors spell them.
OVERRIDES = {"Macau": "MO", "Turkey": "TR"}


def country_code(name: object) -> str | None:
    """The ISO alpha-2 code of a country name or code; None when unknown."""
    if not isinstance(name, str) or not name.strip():
        return None
    text = name.strip()
    if text in OVERRIDES:
        return OVERRIDES[text]
    if len(text) == 2 and text.isalpha():
        return text.upper()
    try:
        return str(pycountry.countries.lookup(text).alpha_2)
    except LookupError:
        return None
