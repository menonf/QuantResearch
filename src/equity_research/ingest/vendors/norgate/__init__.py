"""Norgate Data (norgatedata.com): US equities with delisted companies, historical index membership, daily prices.

Reached through the ``norgatedata`` library, which talks to Norgate Data Updater (NDU) - a Windows app that keeps a
local copy of the subscribed databases. NDU must be installed, signed in and running.

* :mod:`.client` - the only module calling ``norgatedata``;
* :mod:`.normalize` - its answers -> canonical rows (pure, tested without NDU).

Norgate names a delisted security by its ticker and delisting month (``LEH-200809``) and gives every security a
permanent asset id (our ``NORGATE`` identifier). It has no CUSIP or ISIN, so it is matched to listings other vendors
created by its asset id, else by ticker on a date (:func:`~research_core.data.identity.match_listings`).
"""

VENDOR = "norgate"
