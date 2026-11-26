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

## Continuous integration

`.github/workflows/gate.yml` runs the first line of the gate - `ruff format --check`, `ruff check`, `mypy`, the unit
tests - on every push to `main` and every pull request into it, on Python 3.13 and 3.14. It installs every dependency
group exactly as `uv.lock` has it (`uv sync --locked --all-groups`, which fails when the lock is out of step with
`pyproject.toml`) and runs with an empty keyring, so a test that needs a database, a vendor or a stored setting fails
there. The integration tests need a SQL Server and stay on your machine.

**On GitHub, by hand:** *Actions* -> *Gate* -> *Run workflow*, or from a terminal (the workflow must be on `main`
for either; the run can then be on any branch):

```sh
gh workflow run gate.yml --ref <branch>
gh run watch                                     # follow it
```

**On your machine, the workflow itself** - in a Linux container, as GitHub runs it, with [act](https://nektosact.com).
It needs:

* **Docker** (Docker Desktop on Windows or macOS, in Linux-container mode), running;
* **act** 0.2.98 or later (earlier versions cannot run the Node 24 actions the workflow uses):
  `winget install nektos.act` on Windows, `brew install act` on macOS; open a new terminal afterwards so it is on
  the `PATH`.

From the repository's root (one line, for any shell):

```sh
act workflow_dispatch -W .github/workflows/gate.yml -j gate -P ubuntu-latest=catthehacker/ubuntu:act-latest --matrix python-version:3.14
```

* `--matrix python-version:3.14` runs one Python version; leave it out to run both, one after the other.
* `-P ubuntu-latest=...` names the image act runs the job in (its medium Ubuntu image), so it does not ask.
* `act -l -W .github/workflows/gate.yml` lists the jobs without running anything.
* What runs is your working copy, uncommitted changes included: act copies it into the container, leaving out what
  `.gitignore` names (`.venv/`, `data/`).
* Never add `--bind`: it mounts the folder instead of copying it, and the workflow's `uv sync` would replace your
  `.venv` with a Linux one.
* No GitHub token is needed. The image (2.3 GB) is downloaded once; every run installs the dependencies afresh in a
  new container - on Linux, torch brings about 3 GB of CUDA libraries. Measured on 2026-11-26 (Python 3.14): install
  3 min 40 s, mypy 1 min 20 s, unit tests 50 s - about seven minutes in all.

**On your machine, without Docker** - the same steps, on your own operating system (on Windows, torch is its CUDA
build): `uv sync --locked` makes `.venv` exactly the lock, removing anything installed by hand.

```sh
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring   # an empty keyring, as on the runner (optional)
uv sync --locked --all-groups
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy
uv run --no-sync pytest -q
```

In PowerShell, set the variable with `$env:PYTHON_KEYRING_BACKEND = "keyring.backends.null.Keyring"` and run the
commands one per line (Windows PowerShell 5.1 has no `&&`).

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
