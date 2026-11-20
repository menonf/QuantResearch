"""Fixed income research: bond indices from their constituents, month end by month end.

What is specific to bonds - the index provider's files, bonds as listings with their terms, ratings, spreads and
durations, an index rebuilt from its members - on top of ``research_core``'s storage, engine and analytics. It never
imports ``equity_research``: what both need is in ``research_core``.
"""

#: This package's asset class: its database schema, and what its runs and rule sets are filed under
ASSET_CLASS = "fixed_income"
