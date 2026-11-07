# EquityResearch

A point-in-time equity research and backtesting platform for a small team, built so that **no backtest can see data
it could not have had on the day it trades**, and so that **no data vendor is wired into the research**.

It covers US equities in the S&P 500, 400 and 600 (the S&P 1500) since 2005, companies that later failed or were
bought included. The design and the reasons for it are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the
working rules are in [CONTRIBUTING.md](CONTRIBUTING.md).

## How it differs from other open-source projects

Most open-source quant tools are either **backtesting engines** (Zipline, Backtrader, vectorbt, backtesting.py, bt)
or **data access layers** (OpenBB, yfinance, vendor SDKs). An engine simulates trading on the data you give it; an
access layer fetches today's view of a dataset. Neither decides whether that data is fit to backtest on. Delisted
companies dropped from a universe, index members counted before they joined, accounts restated after the fact, a
reused ticker joined to the wrong company - those are left to the user, and each one makes a backtest look better
than it would have been. The platforms that do solve some of this do it inside their own data: QuantConnect LEAN
with the data sold through its cloud, Qlib with a point-in-time store for revised financial reports in its own format.

EquityResearch is the other half: **a data platform first, with the research on top** - and the data is yours, from
the vendors you choose.

| | Typical open-source tools | EquityResearch |
|---|---|---|
| **Who the companies were** | today's ticker list, or the vendor's own ids | a security master: a company (*issuer*) and each share line (*listing*) kept apart, every code (ticker, RIC, CUSIP, ISIN, CIK...) with the dates it was valid - a reused ticker is two companies, never one |
| **Index membership** | today's members, or a bundle you buy | every S&P 500 / 400 / 600 member since 2005 with the day it joined and left, rebuilt from the vendor's event history; gates fail on a share counted twice or a member dated before it traded |
| **Company accounts** | the latest restated figures, dated by period end | every figure as first reported and every later correction, each dated by the day it was filed (SEC XBRL, free), a company kept whole across re-registrations; earlier years (before XBRL, 2009-2011) from a second vendor, one vendor per company and date, never interleaved. SimFin's figures are dated by first publication only where the SEC confirms them |
| **Prices** | split-adjusted series, sometimes adjusted again by the tool | real (unadjusted) prices plus every split, spin-off and dividend; adjusted prices and total returns are computed when read, one formula for every vendor |
| **Vendors** | one per project, its field names everywhere | seven adapters behind one canonical model (LSEG/Refinitiv, Norgate, SEC, Yahoo, FinanceDatabase, Tiingo, SimFin); which vendor supplies what is a setting, and two vendors' prices for one listing are checked against each other |
| **Where a number came from** | rarely recorded | every row names the raw file it came from; each vendor answer is kept unchanged with its request, so a mapping fixed later is re-applied without downloading again |
| **Is the data complete?** | not checked | quality gates after every load: price coverage of members, benchmark coverage, unexplained price jumps, members after their last price, overlapping or premature memberships, facts about periods not yet ended, rows without provenance |
| **Is the backtest honest?** | any number of runs, no record | every run registered (configuration hash, code version, data snapshot, metrics), the number of trials counted into a deflated Sharpe ratio, and a hold-out period that must be opened on purpose |
| **For more than one person** | files on one laptop | one SQL database as the system of record (built and verified locally, then copied to Azure SQL table by table with checksums) and a Parquet / DuckDB copy on each researcher's machine |

**A first check that it works.** Rebuilding the S&P 500 from its members' prices, splits, dividends and share
counts on Norgate's data (2024-10 to 2026-10) tracks the real index closely: by market value 14.48 % a year against
14.72 % for ^GSPC (tracking error 0.86 %, daily correlation 0.9991); equally weighted 10.15 % against 10.34 % for
^SPXEW (0.69 %, 0.9989). Notebook 2.1 runs the same comparison on the full history.

### What it is not

* **Not an execution simulator.** Portfolios rebalance on daily closes with a cost per trade in basis points; there
  is no order book, no intraday data and no fill model. Where execution detail matters, feed these point-in-time
  panels into an engine such as Zipline or LEAN.
* **Not a trading system** - there is no broker connection.
* **US equities, US dollars.** Research uses one currency: statements in another are converted by the vendor or left
  out, and the research says how many.
* **Not subscription-free for the full history.** The free sources give company accounts (SEC), benchmarks (Yahoo)
  and codes (FinanceDatabase). Index membership with its history and the prices of delisted companies need a vendor
  that has them: LSEG Workspace or Norgate Data.

## How it fits together

```
vendors ──> raw zone ──────────────────> canonical model ──────────> local cache ──────> research
            data/<db>/raw, immutable,     SQL Server / Azure SQL       Parquet + DuckDB     panels, factors,
            one ingest.batch per answer   ref / market / fund          per researcher       backtests, registry
                                          facts as reported
```

* **Canonical model** - issuers and listings with an identifier history; unadjusted prices and corporate actions;
  accounting facts as reported, every version dated by publication. No vendor field reaches research code.
* **Research** - adjusted prices, total returns, trailing-twelve-month figures and ratios are computed at each
  decision date from facts known strictly before it.

## Data sources

| Source | Supplies | Needs |
|---|---|---|
| `refinitiv` (LSEG) | index members with history, listings and codes, sectors, daily prices, dividends and splits, share counts and float, statements | LSEG Workspace signed in; a full build uses ~7,550 of its 10,000 daily requests and resumes if stopped |
| `norgate` | index members with history, listings incl. delisted, unadjusted prices, splits and dividends, GICS | Norgate Data Updater with a subscription |
| `sec` | company accounts as first reported, with every correction | a contact name and e-mail (the SEC asks for one) |
| `yahoo` | benchmark index levels and funds (^GSPC, SPY, RSP, ^SPXEW, SPTM, ^SP400, ^SP600) | nothing |
| `financedatabase` | codes (ISIN, FIGI, Yahoo symbol), where a company is run from, its own sector labels | nothing |
| `tiingo` | unadjusted prices, dividends and splits for listings other sources found, cross-checked | an API key |
| `simfin` | standardised statements, banks and insurers included; its own sector labels | an API key |

The default build runs `refinitiv, yahoo, financedatabase, sec`. Choosing others is a setting - a build without
LSEG, for example:

```sh
EQR_SOURCES=norgate,yahoo,financedatabase,sec  EQR_PRICE_VENDORS=norgate  EQR_MEMBER_VENDORS=norgate  uv run eqr build
```

## Setup

```sh
uv sync --all-groups
uv run pytest                                    # unit tests: no database, no network
uv run pytest -m integration                     # database tests on a local SQL Server (never Azure)
uv run eqr set-key simfin                        # a vendor's API key, kept in the OS keyring (typed without echo)
```

Settings come from environment variables or the OS keyring, never from the repository (see
`src/equity_research/settings.py`). For a local database set `EQR_DB_SERVER` (e.g. `MENONPC\SQLEXPRESS`) and
`EQR_DB_NAME`; without them the shared Azure database's settings are read from the keyring.

## Pipelines

```sh
uv run eqr sources           # every data source: what it supplies, what it needs, whether this build runs it
uv run eqr probe             # each source of this build answers questions with known answers (before a long load)
uv run eqr build             # every source in the preferences (EQR_SOURCES), from nothing or continuing a build
uv run eqr refresh           # every source, incrementally, then the gates, the cache and the clean-up (daily)
uv run eqr checks            # the data quality gates
uv run eqr audit             # every architecture rule and the fitness for research, year by year (read-only)
uv run eqr sync-cache        # refresh your local research cache
uv run eqr prune-raw         # remove superseded copies of public files (the SEC's bulk file) from the raw zone
uv run eqr load-lseg         # one source alone - also load-norgate, load-sec, load-benchmarks, load-financedatabase,
                             #   load-tiingo, load-simfin, load-estimates; and probe-lseg, probe-norgate, probe-tiingo,
                             #   probe-simfin, probe-estimates
uv run eqr copy-database "MENONPC\SQLEXPRESS" equity_research_local
                             # copy a verified local build into the configured (Azure) database, checked per table
```

## Notebooks

Written for an undergraduate in finance, economics or a quantitative subject: standard terms (look-ahead bias,
tracking error, information ratio...) are used and defined, with the method behind each result.

Grouped by topic in five folders; each notebook's number is *folder.position* (3.2 is the second strategy), and the
notebooks refer to one another by it.

**`notebooks/1_data_load/`** - building and maintaining the database

* 1.1 `01_initial_load` - builds the database from nothing: every dataset, where it comes from and why it is needed.
* 1.2 `02_refresh` - keeps it current.

**`notebooks/2_index_replication/`** - rebuilding real indices from our own data, the first check that it is right

* 2.1 `01_sp500` - rebuilds the S&P 500 by market value and equally, and compares it with ^GSPC, SPY, ^SPXEW and RSP.
* 2.2 `02_ftse_all_share` - rebuilds the FTSE All-Share and its FTSE 100, 250 and SmallCap segments in pounds,
  compares them with LSEG's index levels, and shows what equal weights and leaving out investment trusts do.

**`notebooks/3_investment_strategies/`** - strategies, backtested and registered

* 3.1 `01_multifactor_long_short` - a multi-factor long-short strategy on the S&P 1500.
* 3.2 `02_multifactor_long_only` - a long-only strategy on the S&P 1500, against SPTM, an equal-weight S&P 1500 and
  SPY.
* 3.3 `03_multifactor_long_only_extra_signals` - four candidate signals (earnings surprise, residual momentum,
  free-cash-flow yield, accruals), each added on its own to 3.2's long-only portfolio.
* 3.4 `04_multifactor_long_only_ml_value` - a machine-learned value score (after *Build Your Own AI Investor*),
  trained walk-forward, against the plain one in 3.2's long-only portfolio.
* 3.5 `05_sp500_garp` - the S&P 500 GARP index (growth at a reasonable price) rebuilt from its published rules and
  tested against SPY.
* 3.6 `06_barings_garp` - Barings' GEMS factors and QGARP score rebuilt from their production code on our data, and a
  book of the best QGARP names tested against SPTM (without I/B/E/S estimates for now: quality and value only).
* 3.7 `07_arima_garch` - a daily long/short timing of the S&P 500 by an ARIMA model chosen by AIC and a GJR-GARCH
  model fitted each day to the last 24 returns (after `menonf/PythonProjects`), held in SPY against SPY.

**`notebooks/4_portfolio_analytics/`** - tools for looking at stocks and portfolios rather than strategies

* 4.1 `01_stock_screen` - any registered strategy (chosen by its run id) ranking today's companies by its own
  score, with the numbers behind each rank and the book it would hold if it rebalanced today.
* 4.2 `02_factor_dashboard` - S&P's monthly factor dashboard for any registered runs of one universe, backtested
  again without registering them (by default our S&P 500 factor books, as of the last month end before the hold-out).
* 4.3 `03_risk_model` - a fundamental factor risk model after BlackRock's BFRE: a portfolio's risk split into
  market, industry, style and stock-specific parts, with its VaR and the model's own tests.

**`notebooks/5_analysis/`** - questions about method rather than strategies

* 5.1 `01_choosing_ml_algorithm` - the eight learning algorithms of *Build Your Own AI Investor* explained, how
  the book chose among them, and the choice made again without hindsight: candidates fixed in advance, each
  trained walk-forward, chosen each January by its out-of-sample rank IC, every portfolio registered.
* 5.2 `02_choosing_ml_algorithm_v2` - the same choice after four changes diagnosed from 5.1: learn each company's
  percentile rather than its return, percentile features, settings for a weak signal, and price, size, issuance,
  surprise and growth features beside the book's ratios; a new registry family, deflated by both studies' trials.
* 5.3 `03_choosing_ml_algorithm_v3` - why 5.2's better ranking earned nothing (a percentile target learns the typical
  year, a portfolio is paid the average one), and the choice made a third time: a clipped excess-return target, the
  models judged by what their top 30 % went on to earn, a forecast that ranks nobody holding the largest companies.

## A backtest

```python
from equity_research.db.engine import get_engine
from equity_research.research.cache import ResearchCache
from equity_research.research.strategy import StrategyConfig, run_strategy

cfg = StrategyConfig(family="value_quality_ls", universe="SP1500", factors=(("value", 1.0), ("quality", 1.0)))
result = run_strategy(ResearchCache(), cfg, engine=get_engine())  # fenced, backtested, registered
result.metrics  # cagr, vol, sharpe, max_drawdown, turnover, trials, deflated_sharpe
```

## Layout

```
src/equity_research/
  settings.py   cli.py   ops.py (job runs, writer lock)
  db/           engine, schema (SQLAlchemy Core), change-only upserts, inventory
  raw/          the raw zone: land a vendor answer, register its batch, read it back, prune public files
  ingest/       identity (issuers, listings, identifiers, matching, merges), membership (spells made consistent),
                reference (indices, universes), classification (sector history), countries;
                vendors/<vendor>/ adapters: refinitiv (LSEG), norgate, sec, yahoo, financedatabase, tiingo, simfin
  pipelines/    sources (the registry a build runs), load_<vendor> for each adapter, refresh, copy_database
  quality/      checks and gates; audit (the architecture rules and fitness for research)
  research/     cache, panels, accounting, ratios, compare, factors, portfolio, backtest, registry, data,
                strategy, indices, screen
  charts.py     the notebooks' charts
migrations/     Alembic
tests/          unit/ (pure) and integration/ (a scratch database on a local SQL Server)
```
