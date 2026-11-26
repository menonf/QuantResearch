# QuantResearch - working notes for Claude

Read `docs/ARCHITECTURE.md` (the design and its rules) and `CONTRIBUTING.md` (the working rules) before changing
code. This file adds what a session needs to work here safely. Machine-learning work follows `docs/ML4T_PLAN.md`
(phases, compatibility rules, progress): read it first and keep its boxes and log current.

## Packages

* Three packages under `src/`: `research_core` (shared: storage, the backtest engine, analytics - imports no
  asset-class package), `equity_research` (everything about shares - imports `research_core` only) and
  `fixed_income_research` (bond indices from ICE's constituent files; command `fir`). Asset-class packages
  never import one another;
  `equity_research/research/` is for the notebooks only. `tests/unit/test_dependency_rules.py` enforces it.
* The database is laid out as the packages are: `research_core` owns the schemas `core` (asset classes, vendors,
  instrument and identifier types, issuers, listings, identifiers), `ingest`, `ops` and `research` (runs and rule
  sets, each naming its `asset_class`); each asset class owns one schema named after it (`equity`, `fixed_income`)
  and registers its schema module under the entry point `research_core.asset_classes` (`pyproject.toml`). A new
  table goes in its owner's schema; a new asset class adds a package, a schema, and its rows in `core.asset_class`,
  `core.instrument_type`, `core.id_type` - never a column on a shared table. `tests/unit/test_database_layout.py` and
  `tests/integration/test_migrations.py` enforce it (docs/ARCHITECTURE.md section 3).
* Equity notebooks are under `notebooks/equity/`, fixed income ones under `notebooks/fixed_income/`, numbered
  folder.position; each says what it needs and produces.
* Fixed income: month-end files `<index>-<MMDDYYYY>.xlsx` in the folder `EQR_ICE_DIR`, loaded by `fir load-ice`
  (new files only, any order). The file of a month end is the index after its rebalance: next month's weights,
  the past month's returns; a bond that left has no return for its last month. Each asset class's research cache
  and artifacts have their own folder: `data/<db>/cache/<asset class>/`, `data/<db>/research/<asset class>/`.
* The repository was called EquityResearch until November 2026 (the `EQR_*` settings, the keyring service
  `equity_research` and the `eqr` command keep their names).
* Migrations: one history from `0001` in `migrations/versions` (started 2026-11-15 with the multi-asset layout).
  The first history (0001-0011, schemas `ref` / `market` / `fund`) was removed on 2026-11-20 (git history has it):
  `equity_research_local` and Azure's `equity_research` are still on it, and the code no longer reads them.

## Environment

* Windows, Git Bash and PowerShell. Python 3.13 or later via `uv` (this machine's venv runs 3.14); run tools from `.venv/Scripts/` (`.venv/Scripts/python`,
  `.venv/Scripts/eqr`, `.venv/Scripts/ruff`, `.venv/Scripts/mypy`) - the shell's `VIRTUAL_ENV` may point at another
  project.
* Databases: the working database is the local `quant_research_local` (new history, built from the vendors from
  2026-11-15; `.env` names it), data folder `data/quant_research_local/`. Only VS Code reads `.env`: the settings read
  the environment, then the keyring, so a shell command without `EQR_DB_SERVER='MENONPC\SQLEXPRESS'
  EQR_DB_NAME=quant_research_local` connects to Azure - set both on every command. The shared **Azure** database
  `equity_research` (with no `EQR_DB_SERVER` set) is still on the old layout: it is to be replaced by a copy of
  `quant_research_local` (`eqr copy-database` into an empty database on the new history) - confirm with the user
  first. It is serverless: the first connection after a pause waits up to a minute while it resumes.
* A local SQL Server Express `MENONPC\SQLEXPRESS` (Windows authentication), chosen by
  `EQR_DB_SERVER='MENONPC\SQLEXPRESS' EQR_DB_NAME=<name>`, is for what must not touch Azure first: integration tests
  (they create and drop scratch databases - never on Azure), new loaders and trial builds, verified there before
  they reach Azure. `quant_research_gate` is an empty, migrated database the integration tests run in (their
  own scratch databases have fixed names - `equity_research_test*` - and are dropped after: never give a database
  one of those names). `equity_research_local` (old layout, migration 0011, rebuilt 2026-11-12) is kept unchanged
  at the user's request: never migrate, empty or drop it. Confirm before anything destructive on Azure (emptying a
  table, a migration that rewrites data, a rebuild).
* Secrets live in the OS keyring (service `equity_research`): never in files, never printed, never in a URL. Vendor
  keys: `eqr set-key tiingo|simfin`.
* Each database has its own data folder `data/<db name>/` (raw zone + research cache); `data/` is not in git.

## Gate - a change is done when all pass

```sh
.venv/Scripts/ruff format . && .venv/Scripts/ruff check . && .venv/Scripts/mypy        # mypy covers src and tests
.venv/Scripts/python -m pytest -q                                                          # unit: no database, no network
EQR_DB_SERVER='MENONPC\SQLEXPRESS' EQR_DB_NAME=quant_research_gate EQR_SEC_USER_AGENT="t t@example.com" \
  .venv/Scripts/python -m pytest -q -m integration                                         # scratch databases, dropped after
```

Data checks: `eqr checks` (the gates) and `eqr audit` (every architecture rule and fitness for research, year by year).

## Conventions

* Match the surrounding code: comments and docstrings say *why* in plain words; module docstrings describe the steps.
* Pure functions for normalisation, matching, panels and research; I/O only in clients, the raw store and pipelines.
* Every vendor answer is landed in the raw zone before use; every canonical row names its `batch_id`; batches read
  are marked `normalized`.
* A rule found while fixing data becomes code (with a test) and, where it can, a gate or an audit rule.
* Do not commit unless asked. Confirm before anything destructive or outward-facing (dropping a database, Azure).

## Vendor facts worth knowing

* **LSEG** (Workspace on this PC): 10,000 requests and 5 GB a day, reset at midnight local time; check with
  `GET http://localhost:9000/api/throttling/limits`. `get_history` costs one request **per RIC** and returns up to
  ~3,000 days, so prices go in 11-year windows; `get_data` costs one per call. lseg-data cannot build a history mixing
  index levels (no volume) with stocks - levels are asked apart. Builds start on 2000-01-01 (prices from 1998, a year
  before for momentum): three price windows per RIC; a full build of both markets with their estimates took ~14,200
  requests, two days of the allowance (2026-11-15/16: US 7,933, its estimates 1,674, London 3,552, its estimates
  1,069 - of which 1,908 were past revision counts, no longer asked); loads resume. Everything answers from 2000, but LSEG's S&P 400 history begins 2001-07-20 and its S&P 600's 2001-01-16
  (every member "joins" that day): before then the S&P 1500 is the S&P 500 alone - start S&P 1500 research in 2002.
  A refresh asks its prices by `get_data` (`TR.PriceClose`, unadjusted: one request per 50 RICs, equal to the
  histories bar for bar) and starts on the first of a month, so a rerun the same day reads back what it landed;
  a daily refresh costs ~1,050 requests, most of them the fixed actions / shares / statements / yearly-items
  buckets.
  Its statement share counts (`TR.F.ComShrOutsTot`) are restated to today's share basis; research values listings
  from their own counts (`equity.share_count`, never restated).
  I/B/E/S estimates (`eqr load-estimates`): per-share history is restated to today's shares too (the loader stores it
  in the shares of its own date). Never ask the revision counts (`TR.NumEstRevisingUp`/`Down`) as a history
  (`EDate`/`Frq`): no answer comes and lseg-data re-sends the request at a request each (~1,000 lost on 2026-10-30);
  the client's runaway guard stops such a call. And asked for a past date (`SDate`) they answer **today's** counts:
  99.6 % of those the 2026-11-17 build asked for 2000-2026 equal each listing's latest. So the loader asks only month
  ends a month old at most, research reads only counts loaded within `REVISION_LIVE_DAYS` (31) of their date
  (`qgarp_estimates.live_revisions`), and the audit lists the rest; backtests use the consensus revisions instead.
  Yearly statement items (cash flows, depreciation, bank lines: `fields.ANNUAL_ITEMS`) answer one field per request
  only (two or more: "Backend error"), banks only beside total assets' period end; LSEG's capital ratios are broker
  estimates (never stored). Share issues, buybacks, borrowing and the capital ratios come from the SEC alone.
* **LSEG London** (`--market UK`; facts in `fields.py`'s London section, `eqr probe-lseg-uk` repeats them): prices in
  pence (`GBX`), official closes `OFF_CLOSE` (not `TRDPRC_1`); dividends in pounds; tickers from `TR.ExchangeTicker`;
  companies by PermID; investment trusts by TRBC industry; statements `TR.F.*` half-yearly with `ReportingState=Orig`
  (as first reported); an unknown `TR.F.*` name is silently left out of the answer - check the columns.
* **SEC**: XBRL accounts begin 2009-2011, so research reads accounts SEC-first, LSEG answering on any date the SEC
  cannot yet (before XBRL, during its first four quarters, a company it lacks) - one vendor per company and date.
  Companies are matched to CIKs by ticker, CUSIP and registrant name; a name-matched CIK only fills the periods
  before the company's main CIK begins (a re-registration's earlier years, never a subsidiary's same years).
* **Norgate**: membership is a daily 1/0 series - a delisted share's spell is closed at its last trade.
* GitHub's raw file host resets connections from this PC; files fetched from GitHub have a jsDelivr fallback.
