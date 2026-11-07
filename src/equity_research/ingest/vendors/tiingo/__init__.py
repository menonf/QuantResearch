"""Tiingo (tiingo.com): US end-of-day prices with each day's dividend and split, over a REST API.

* :mod:`.client` - the only module calling Tiingo;
* :mod:`.normalize` - its answers -> canonical rows (pure, tested without the network).

Tiingo knows a security only by its ticker: no CUSIP, ISIN or index membership. So it never creates a listing - it
prices the listings other vendors found, each under the ticker we hold for it, and its prices are checked against
another vendor's where both exist (a ticker can pass to another company). The API key is sent in a header, never
in a URL, so no landed batch records it. A free account allows few requests an hour; a load stops when told the
allowance is used up and the next run continues where it stopped.
"""

VENDOR = "tiingo"
