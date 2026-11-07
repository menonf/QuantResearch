"""SimFin (simfin.com): standardised quarterly statements of US companies, as bulk files.

* :mod:`.client` - the only module calling SimFin;
* :mod:`.normalize` - its files -> canonical financial facts and company codes (pure, tested without the network).

SimFin is a second source of company accounts beside the SEC (choose it with ``EQR_FUNDAMENTALS_VENDOR=simfin``):
already mapped to one chart of accounts, banks and insurers included, one download for every company. Each company
has a permanent SimFin id (our ``SIMFIN`` issuer identifier); it is matched to our issuers by CIK where the file
has one, else through a listing by ISIN or ticker. SimFin creates no issuer or listing of its own.

Its files hold the *latest* value of each figure, dated by when SimFin last restated it - so a value is never known
earlier than it was, and history of corrections builds up from the first download onwards.
"""

VENDOR = "simfin"
