# fixed_income_research

Bond indices from their constituents, month end by month end, on top of `research_core`. It never imports
`equity_research`: what both need is in `research_core`.

## What is here

```
data/        schema (the bond tables, in the database schema `fixed_income`), cache (the research cache's table
             list, in cache/fixed_income),
             index_data (one index as monthly panels)
ingest/      ice (ICE's constituent files -> canonical rows), fred (official index levels), reference (the indices)
pipelines/   load_ice (a folder of files, new ones only), load_index_levels (FRED)
quality/     the fixed income gates
analytics/   replication (an index rebuilt from its members; what it is made of)
cli.py       the `fir` command: load-ice, load-index-levels, checks, sync-cache
```

## The data

One Excel file per index and month end from ICE's Index Viewer, named `<index>-<MMDDYYYY>.xlsx`
(`H0A0-01312019.xlsx`), in the folder named by `EQR_ICE_DIR`. Files arrive in any order; a file loaded before is
skipped. Three facts about them shape the model (`ingest/ice.py` has the detail):

1. the file of a month end holds the index **after** that day's rebalance - next month's members and weights, with
   each bond's analytics and its return over the month just ended;
2. a bond that left at that rebalance is in no file for its last month, so about 2 % of the index's weight has no
   return each month - measured by a gate and by notebook 2.1, not hidden;
3. an issuer is ICE's ticker (a company and its financing subsidiaries); industry levels and country are stored per
   issuer and month.

## What it reuses from research_core

The database engine and migrations, the raw zone (every file landed unchanged, every row naming its batch), job
runs and the writer lock, change-only upserts, the security master (a bond is a `core.listing` of a `core.issuer`,
with its ISIN and CUSIP in `core.identifier`), the quality-check framework, the research cache mechanism, the
performance and tracking statistics (`periods_per_year=12` for monthly returns) and the charts.

## Not built yet

Strategies, the run registry for bonds, and risk beyond the provider's own durations. See the top-level README,
*Adding an asset class*, and `docs/ARCHITECTURE.md`.
