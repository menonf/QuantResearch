# Working on EquityResearch

The design and its reasons are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). These are the working rules that
follow from it.

## A change is done when

```sh
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
uv run pytest -m integration        # when the change touches the database: on a LOCAL SQL Server, e.g.
                                    #   EQR_DB_SERVER=MENONPC\SQLEXPRESS EQR_DB_NAME=equity_research_local
                                    #   (rolled back, or on throwaway databases; refused on Azure)
```

All four pass, the new behaviour has a test, and docstrings say *why* where the code cannot.

## Data

* **Vendors stay in `ingest/vendors/<vendor>/`.** Nothing outside it knows a vendor's field names, codes or quirks.
* **Land before you normalise.** Every vendor response goes through `raw.store.land_file` / `land_frame` first; a fix
  to a mapping re-normalises from the raw zone, never re-downloads.
* **Store facts, derive the rest.** Unadjusted prices, corporate actions, accounting items as reported. Adjusted
  prices, total returns, TTM figures and ratios are computed in `research/`, never stored.
* **Two clocks.** `published_at` is when the world could know a fact; `loaded_at` when we knew it. A research read
  uses only facts published strictly before the decision date.
* **Identifiers are insert-only.** A code already held by another entity is reported (`add_identifiers` returns the
  conflicts), never reassigned. Merging two issuers or two listings (one share line under two RICs) is an explicit
  step (`ingest.identity.merge_issuers`, `merge_listings`).
* **Writes go through `db.upsert`.** Change-only, idempotent: a re-run writes nothing new. Pipelines run inside
  `ops.job(...)`, which takes the writer lock, and end with quality gates (`quality.checks`).

## Schema

Only through Alembic: `uv run alembic revision -m "..."`, review the script, test `upgrade` and `downgrade` on the
database, then `uv run alembic check` must report no differences.

## Research

* Read from the local cache (`uv run eqr sync-cache`), not from the shared database.
* Run strategies through `research.strategy.run_strategy` with an engine: every run is registered, so the number of
  variants tried - and the deflated Sharpe ratio - is known.
* Backtests end before the hold-out (`EQR_HOLDOUT_START`). Open it once, for a final strategy, and say so.
* Notebooks only set parameters and display results; logic lives in the package with tests.
