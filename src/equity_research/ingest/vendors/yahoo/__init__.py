"""Yahoo Finance: daily levels of the S&P indices and the benchmark ETFs (SPY, RSP), with their dividends and splits.

Free and current, used for benchmarks: what the research compares itself against. Yahoo adjusts its history for
later splits; the adapter undoes that so the stored rows are unadjusted like every other vendor's.
"""

VENDOR = "yahoo"
