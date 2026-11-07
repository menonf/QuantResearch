# EquityResearch - working notes for Claude

Read `docs/ARCHITECTURE.md` (the design and its rules) and `CONTRIBUTING.md` (the working rules) before changing
code. This file adds what a session needs to work here safely. Machine-learning work follows `docs/ML4T_PLAN.md`
(phases, compatibility rules, progress): read it first and keep its boxes and log current.

## Environment

* Windows, Git Bash and PowerShell. Python 3.13 via `uv`; run tools from `.venv/Scripts/` (`.venv/Scripts/python`,
  `.venv/Scripts/eqr`, `.venv/Scripts/ruff`, `.venv/Scripts/mypy`) - the shell's `VIRTUAL_ENV` may point at another
  project.
* Databases: the shared **Azure** database `equity_research` is the working database (since 2026-10-30, a verified
  copy of `equity_research_local`): with no `EQR_DB_SERVER` set - the default, `.env` included - research,
  refreshes and the run registry use it, with its data folder `data/equity_research/`. It is serverless: the first
  connection after a pause waits up to a minute while it resumes.
* A local SQL Server Express `MENONPC\SQLEXPRESS` (Windows authentication), chosen by
  `EQR_DB_SERVER='MENONPC\SQLEXPRESS' EQR_DB_NAME=<name>`, is for what must not touch Azure first: integration tests
  (they create and drop scratch databases - never on Azure), new loaders and trial builds, verified there before
  they reach Azure. `equity_research_gate` is an empty, migrated database the integration tests run in (their
  own scratch databases have fixed names - `equity_research_test*` - and are dropped after: never give a database
  one of those names). `equity_research_local` and `equity_research_norgate` were no longer on the server on
  2026-10-30; Azure holds the verified copy. Confirm before anything destructive on Azure (emptying a table, a
  migration that rewrites data, a rebuild).
* Secrets live in the OS keyring (service `equity_research`): never in files, never printed, never in a URL. Vendor
  keys: `eqr set-key tiingo|simfin`.
* Each database has its own data folder `data/<db name>/` (raw zone + research cache); `data/` is not in git.

## Gate - a change is done when all pass

```sh
.venv/Scripts/ruff format . && .venv/Scripts/ruff check . && .venv/Scripts/mypy        # mypy covers src and tests
.venv/Scripts/python -m pytest -q                                                          # unit: no database, no network
EQR_DB_SERVER='MENONPC\SQLEXPRESS' EQR_DB_NAME=equity_research_gate EQR_SEC_USER_AGENT="t t@example.com" \
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
  index levels (no volume) with stocks - levels are asked apart. A full build is ~7,550 requests; loads resume.
  A refresh asks its prices by `get_data` (`TR.PriceClose`, unadjusted: one request per 50 RICs, equal to the
  histories bar for bar) and starts on the first of a month, so a rerun the same day reads back what it landed;
  a daily refresh costs ~1,050 requests, most of them the fixed actions / shares / statements / yearly-items
  buckets.
  Its statement share counts (`TR.F.ComShrOutsTot`) are restated to today's share basis; research values listings
  from their own counts (`market.share_count`, never restated).
  I/B/E/S estimates (`eqr load-estimates`): per-share history is restated to today's shares too (the loader stores it
  in the shares of its own date). Never ask the revision counts (`TR.NumEstRevisingUp`/`Down`) as a history
  (`EDate`/`Frq`): no answer comes and lseg-data re-sends the request at a request each (~1,000 lost on 2026-10-30);
  the client's runaway guard stops such a call.
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
