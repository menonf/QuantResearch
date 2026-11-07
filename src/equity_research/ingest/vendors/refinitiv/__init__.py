"""LSEG / Refinitiv, live through LSEG Workspace (the ``lseg-data`` library).

Index members, listings, daily prices, dividends and splits, share counts, sector classification, quarterly statements
and their announcement dates.

* :mod:`.client` - the requests: one session, retries, splitting a request one bad RIC spoils, and a clear stop when
  LSEG's daily limit is reached (:class:`~.client.ThrottledError`);
* :mod:`.fields` - every ``TR.*`` field asked for and the column name LSEG answers with;
* :mod:`.rics` - RIC spellings (``AAPL.OQ``, a retired ``BAC.N^I98``, an index ``.SPX``);
* :mod:`.normalize` - LSEG's answers -> canonical rows (pure functions, tested without LSEG).

Nothing here opens a session at import time.
"""

VENDOR = "refinitiv"
