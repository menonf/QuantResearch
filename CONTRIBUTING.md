# Working on QuantResearch

The design and its reasons are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). These are the working rules that
follow from it.

## A change is done when

```sh
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
uv run pytest -m integration        # when the change touches the database: on a LOCAL SQL Server, e.g.
                                    #   EQR_DB_SERVER=<your machine>\SQLEXPRESS EQR_DB_NAME=quant_research_gate
                                    #   (rolled back, or on throwaway databases; refused on Azure)
```

All four pass, the new behaviour has a test, and docstrings say *why* where the code cannot.

## Packages

* **`research_core` imports no asset-class package.** What it offers must work for any instrument: an id, a
  return per period, a number of periods in a year passed as a keyword.
* **Asset-class packages never import one another** (`equity_research`, `fixed_income_research`). What two of
  them need moves to `research_core`.
* **`equity_research/research/` is for the notebooks only**: nothing in `src` or `tests` imports it.
* `tests/unit/test_dependency_rules.py` checks all three with every unit-test run.

## Data

* **Vendors stay in `ingest/vendors/<vendor>/`.** Nothing outside it knows a vendor's field names, codes or quirks.
* **Land before you normalise.** Every vendor response goes through `research_core.data.raw_store.land_file` / `land_frame` first; a fix
  to a mapping re-normalises from the raw zone, never re-downloads.
* **Store facts, derive the rest.** Unadjusted prices, corporate actions, accounting items as reported. Adjusted
  prices, total returns, TTM figures and ratios are computed when read (`equity_research/data/`), never stored.
* **Two clocks.** `published_at` is when the world could know a fact; `loaded_at` when we knew it. A research read
  uses only facts published strictly before the decision date.
* **Identifiers are insert-only.** A code already held by another entity is reported (`add_identifiers` returns the
  conflicts), never reassigned. Merging two issuers or two listings (one share line under two RICs) is an explicit
  step (`research_core.data.identity.merge_issuers`, `merge_listings`).
* **Writes go through `research_core.data.db.upsert`.** Change-only, idempotent: a re-run writes nothing new. Pipelines run inside
  `research_core.data.ops.job(...)`, which takes the writer lock, and end with their asset class's quality gates
  (`equity_research.quality.checks`, `fixed_income_research.quality.checks`).

## Schema

Only through Alembic: `uv run alembic revision -m "..."`, review the script, test `upgrade` and `downgrade` on the
database, then `uv run alembic check` must report no differences.

## Research

* Read from the local cache (`uv run eqr sync-cache`; `uv run fir sync-cache` for bonds), not from the shared database.
* Run strategies through their runner with an engine - `factor_strategy.run_strategy`, `garp.run_garp`,
  `qgarp.run_qgarp`, fixed income's `factor_strategy.run_factors` - all ending in `research_core.engine.finish.finish_run`:
  every run is registered, so the number of variants tried - and the deflated Sharpe ratio - is known.
* Backtests end before the hold-out (`EQR_HOLDOUT_START`). Open it once, for a final strategy, and say so.
* Notebooks only set parameters and display results; logic lives in the package with tests.
