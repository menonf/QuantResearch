# QuantResearch

A point-in-time research and backtesting platform for a small team, built so that **no backtest can see data it
could not have had on the day it trades**, and so that **no data vendor is wired into the research**. It is one
shared core (`research_core`: storage, the backtest engine, analytics) and one package per asset class: equities
are built (`equity_research`); fixed income has begun (`fixed_income_research`: ICE's bond index
constituents, loaded and rebuilt into the index). The
repository was called EquityResearch until November 2026.

The equity research covers two markets since 2000, companies that later failed or were bought included: US equities in the S&P 500,
400 and 600 (the S&P 1500), in dollars, and London's FTSE All-Share (FTSE 100, 250 and SmallCap), in pounds. The
design and the reasons for it are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the working rules are in
[CONTRIBUTING.md](CONTRIBUTING.md); the machine-learning research follows [docs/ML4T_PLAN.md](docs/ML4T_PLAN.md).

## How it differs from other open-source projects

Most open-source quant tools are either **backtesting engines** (Zipline, Backtrader, vectorbt, backtesting.py, bt)
or **data access layers** (OpenBB, yfinance, vendor SDKs). An engine simulates trading on the data you give it; an
access layer fetches today's view of a dataset. Neither decides whether that data is fit to backtest on. Delisted
companies dropped from a universe, index members counted before they joined, accounts restated after the fact, a
reused ticker joined to the wrong company - those are left to the user, and each one makes a backtest look better
than it would have been. The platforms that do solve some of this do it inside their own data: QuantConnect LEAN
with the data sold through its cloud, Qlib with a point-in-time store for revised financial reports in its own format.

QuantResearch is the other half: **a data platform first, with the research on top** - and the data is yours, from
the vendors you choose.

| | Typical open-source tools | QuantResearch |
|---|---|---|
| **Who the companies were** | today's ticker list, or the vendor's own ids | a security master: a company (*issuer*) and each share line (*listing*) kept apart, every code (ticker, RIC, CUSIP, ISIN, CIK...) with the dates it was valid - a reused ticker is two companies, never one |
| **Index membership** | today's members, or a bundle you buy | every S&P 500 / 400 / 600 and FTSE 100 / 250 / SmallCap member since 2000 (LSEG's S&P 400 and 600 histories begin in 2001) with the day it joined and left, rebuilt from the vendor's event history; gates fail on a share counted twice or a member dated before it traded |
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
^SPXEW (0.69 %, 0.9989). [Notebook 2.1](notebooks/equity/2_index_replication/01_sp500.ipynb) runs the same comparison on
the full history.

### What it is not

* **Not an execution simulator.** Portfolios rebalance on daily closes with a cost per trade in basis points; there
  is no order book, no intraday data and no fill model. Where execution detail matters, feed these point-in-time
  panels into an engine such as Zipline or LEAN.
* **Not a trading system** - there is no broker connection.
* **Two markets, one currency at a time.** A study runs on one universe in its own currency (the S&P 1500 in
  dollars, the FTSE All-Share in pounds): statements in another currency are converted by the vendor or left out,
  and the research says how many.
* **Not subscription-free for the full history.** The free sources give company accounts (SEC), benchmarks (Yahoo)
  and codes (FinanceDatabase). Index membership with its history and the prices of delisted companies need a vendor
  that has them: LSEG Workspace or Norgate Data.

## How it fits together

```
vendors ──> raw zone ──────────────────> canonical model ──────────> local cache ──────> research
            data/<db>/raw, immutable,     SQL Server / Azure SQL       Parquet + DuckDB     panels, factors,
            one ingest.batch per answer   core / equity / fixed_income per researcher       backtests, registry
                                          facts as reported
```

* **Canonical model** - issuers and listings with an identifier history; unadjusted prices and corporate actions;
  accounting facts as reported, every version dated by publication. No vendor field reaches research code.
* **Research** - adjusted prices, total returns, trailing-twelve-month figures and ratios are computed at each
  decision date from facts known strictly before it.

## Data sources

| Source | Supplies | Needs |
|---|---|---|
| `refinitiv` (LSEG) | US: index members with history, listings and codes, sectors, daily prices, dividends and splits, share counts and float, statements | LSEG Workspace signed in; a full build uses ~7,900 of its 10,000 daily requests and resumes if stopped |
| `estimates` (LSEG) | analysts' estimates (I/B/E/S): monthly consensus history, reported actuals, revision counts | LSEG Workspace; ~400 requests from nothing (revision counts only for the latest month end) |
| `refinitiv_uk`, `estimates_uk` (LSEG) | London: FTSE 100 / 250 / SmallCap members, official closes in pence, dividends, splits, share counts, half-yearly statements as first reported, estimates in pounds | LSEG Workspace; ~4,000 requests from nothing |
| `norgate` | index members with history, listings incl. delisted, unadjusted prices, splits and dividends, GICS | Norgate Data Updater with a subscription |
| `sec` | company accounts as first reported, with every correction | a contact name and e-mail (the SEC asks for one) |
| `yahoo` | benchmark index levels and funds (^GSPC, SPY, RSP, ^SPXEW, SPTM, ^SP400, ^SP600) | nothing |
| `financedatabase` | codes (ISIN, FIGI, Yahoo symbol), where a company is run from, its own sector labels | nothing |
| `tiingo` | unadjusted prices, dividends and splits for listings other sources found, cross-checked | an API key |
| `simfin` | standardised statements, banks and insurers included; its own sector labels | an API key |

The default build runs `refinitiv, estimates, refinitiv_uk, estimates_uk, yahoo, financedatabase, sec, repository`
(the repository's own corrections, last), in that order - about 12,500 LSEG requests, so two days of its
allowance; a build stopped by the allowance continues the next day. Choosing others is a setting - a build without
LSEG, for example:

```sh
EQR_SOURCES=norgate,yahoo,financedatabase,sec  EQR_PRICE_VENDORS=norgate  EQR_MEMBER_VENDORS=norgate  uv run eqr build
```

Filing text is loaded apart (`eqr load-sec-text`, notebook 1.3): the SEC's filing index, every annual report since
2003 (about 10 GB of documents, kept in the raw zone) and their text measures, with the Loughran-McDonald dictionary
from Notre Dame's SRAF; all free.

## Setup

```sh
uv sync --all-groups                             # or: pip install -r requirements.txt (the same versions, pinned)
uv run pytest                                    # unit tests: no database, no network
uv run pytest -m integration                     # database tests on a local SQL Server (never Azure)
uv run eqr set-key simfin                        # a vendor's API key, kept in the OS keyring (typed without echo)
```

On Windows torch is its CUDA build (`[tool.uv.sources]` in `pyproject.toml`): the neural models (FinBERT, the LSTM,
the MLP ensemble) run on a GPU when there is one and on the processor otherwise.

Settings come from environment variables or the OS keyring, never from the repository (see
`src/research_core/settings.py`). For a local database set `EQR_DB_SERVER` (e.g. `<your machine>\SQLEXPRESS`) and
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
uv run eqr load-lseg         # one source alone (--market UK for London) - also load-norgate, load-sec,
                             #   load-benchmarks, load-financedatabase, load-tiingo, load-simfin, load-estimates; and
                             #   probe-lseg, probe-lseg-uk, probe-norgate, probe-tiingo, probe-simfin, probe-estimates
uv run eqr load-sec-text --tone --tone-device cuda
                             # filing index, annual reports (10-K) and their text measures; --tone adds FinBERT
uv run eqr load-rules        # the repository's strategy rule sets (QGARP's cleaning limits...) into the shared
                             #   tables research.rule_set / research.strategy_rule; a loaded set is never changed
uv run eqr load-corrections  # corporate actions no vendor recorded (ingest/corrections/corporate_actions.csv:
                             #   Alphabet's class C shares of 2014...), written under vendor `repository`
uv run eqr copy-database "<your machine>\SQLEXPRESS" quant_research_local
                             # copy a verified local build into the configured (Azure) database - every asset
                             #   class's tables, on the same migration history - checked per table
```

## Running the notebooks

Written for an undergraduate in finance, economics or a quantitative subject: standard terms (look-ahead bias,
tracking error, information ratio...) are used and defined, with the method behind each result. Each notebook says
under its title what it **needs** and what it **produces**.

### 1. Environment

```sh
uv sync --all-groups     # Python 3.13 or later: the three packages, the Jupyter kernel and the optional groups
```

Open the repository in VS Code (or start Jupyter from it) and choose the `.venv` interpreter as the kernel. Three
optional groups are needed by a few notebooks only: `ml` (LightGBM, Optuna) by 5.4, `deep` (torch) by 5.7 and 5.8,
`text` (transformers, for FinBERT) by 1.3; a fourth, `vendors` (norgatedata), by a build from Norgate. One notebook
from the command line:

```sh
uv run jupyter execute --inplace --timeout=-1 notebooks/equity/2_index_replication/01_sp500.ipynb
```

### 2. Data: what must exist before the research notebooks run

Nothing ships with the repository: `data/` is not in git, and the vendors' data may not be passed on. You need

* **a SQL Server database** - SQL Server Express on your own machine is enough (with ODBC Driver 18), or an Azure
  SQL database. Name it with `EQR_DB_SERVER` (e.g. `<your machine>\SQLEXPRESS`) and `EQR_DB_NAME`; a `.env` file at the
  repository's root holding those two lines is read by VS Code's kernels and terminals;
* **a source of index members and prices with their history** - LSEG Workspace signed in on the same machine (the
  default sources), or Norgate Data (`EQR_SOURCES=norgate,yahoo,financedatabase,sec`; see *Data sources*). The SEC,
  Yahoo and FinanceDatabase are free; the SEC asks for a contact (`EQR_SEC_USER_AGENT="name e-mail"`);
* **notebook 1.1 run once** (or `uv run eqr build`): it creates the tables, loads every source, runs the quality
  gates and syncs the local research cache. With LSEG that is two days of its daily allowance; a build
  stopped by the allowance continues where it stopped.

After that the research notebooks read the local cache (`data/<database>/cache/<asset class>`) and, where they register a run,
the database. Nothing they do needs a vendor.

### 3. Run order

Run them in the order of their numbers - *folder.position* (3.2 is the second strategy), which is also how they
refer to one another. The equity notebooks are under `notebooks/equity/`; the fixed income ones, numbered the
same way, under `notebooks/fixed_income/` (listed after them).

**`1_data_ingestion/`** - building and maintaining the database

| | Notebook | What it does |
|---|---|---|
| 1.1 | [Initial load](notebooks/equity/1_data_ingestion/01_initial_load.ipynb) | builds the database from nothing: every dataset, where it comes from and why it is needed |
| 1.2 | [Refresh](notebooks/equity/1_data_ingestion/02_refresh.ipynb) | keeps it current (optional before the research notebooks) |
| 1.3 | [Filing text](notebooks/equity/1_data_ingestion/03_filing_text.ipynb) | the SEC's filing index, every annual report since 2003 and their text measures; only 5.10 needs it |

**`2_index_replication/`** - rebuilding real indices from our own data, the first check that it is right

| | Notebook | What it does |
|---|---|---|
| 2.1 | [S&P 500](notebooks/equity/2_index_replication/01_sp500.ipynb) | rebuilds the S&P 500 by market value and equally, against ^GSPC, SPY, ^SPXEW and RSP |
| 2.2 | [FTSE All-Share](notebooks/equity/2_index_replication/02_ftse_all_share.ipynb) | rebuilds the FTSE All-Share and its three segments in pounds, against LSEG's index levels |

**`3_investment_strategies/`** - strategies from simple to advanced, each backtested and registered

| | Notebook | What it does |
|---|---|---|
| 3.1 | [Multi-factor long-only](notebooks/equity/3_investment_strategies/01_multifactor_long_only.ipynb) | a long-only strategy on the S&P 1500, against SPTM, an equal-weight S&P 1500 and SPY |
| 3.2 | [Long-only with extra signals](notebooks/equity/3_investment_strategies/02_long_only_extra_signals.ipynb) | four candidate signals (earnings surprise, residual momentum, free-cash-flow yield, accruals), each added to 3.1's portfolio |
| 3.3 | [Learned value score](notebooks/equity/3_investment_strategies/03_long_only_learned_value.ipynb) | a machine-learned value score (after *Build Your Own AI Investor*), trained walk-forward, against the plain one |
| 3.4 | [Multi-factor long-short](notebooks/equity/3_investment_strategies/04_multifactor_long_short.ipynb) | a beta-neutral long-short strategy; explains the run registry, the deflated Sharpe ratio and the hold-out |
| 3.5 | [S&P 500 GARP and QGARP](notebooks/equity/3_investment_strategies/05_sp500_garp_and_qgarp.ipynb) | the S&P 500 GARP index rebuilt from its published rules; Part 2 runs 3.6's QGARP on the same universe |
| 3.6 | [QGARP](notebooks/equity/3_investment_strategies/06_qgarp.ipynb) | the QGARP strategy reconstructed from its rules, and a book of its best names against SPTM |
| 3.7 | [Revised long-only](notebooks/equity/3_investment_strategies/07_revised_long_only.ipynb) | the review's revised specification ([docs/STRATEGY_REVIEW.md](docs/STRATEGY_REVIEW.md)), written down before it ran: once on the S&P 1500, then unchanged on the FTSE All-Share |

**`4_portfolio_analytics/`** - performance, risk and attribution of what the strategies hold

| | Notebook | What it does |
|---|---|---|
| 4.1 | [Factor dashboard](notebooks/equity/4_portfolio_analytics/01_factor_dashboard.ipynb) | S&P's monthly factor dashboard for any registered strategies of one universe: returns, risk, overlap, exposures |
| 4.2 | [Risk model](notebooks/equity/4_portfolio_analytics/02_risk_model.ipynb) | a fundamental factor risk model after BlackRock's BFRE: a portfolio's risk by source, its VaR, stress tests |
| 4.3 | [Stock screen](notebooks/equity/4_portfolio_analytics/03_stock_screen.ipynb) | any registered strategy ranking today's companies by its own score, and the book it would hold today |

**`5_machine_learning/`** - the machine-learning plan ([docs/ML4T_PLAN.md](docs/ML4T_PLAN.md)), after *Machine
Learning for Trading* (3rd edition); each notebook reads what the ones before it kept in the artifact store

| | Notebook | What it does |
|---|---|---|
| 5.1 | [Research protocol](notebooks/equity/5_machine_learning/01_research_protocol.ipynb) | the rules every later notebook follows, the labels, and whether the data can support the plan |
| 5.2 | [Features](notebooks/equity/5_machine_learning/02_features.ipynb) | about fifty company features, each triaged alone against next month's return |
| 5.3 | [Linear models](notebooks/equity/5_machine_learning/03_linear_models.ipynb) | ridge, lasso, elastic net and logistic regression, tuned inside each walk-forward fold |
| 5.4 | [Gradient boosting](notebooks/equity/5_machine_learning/04_gradient_boosting.ipynb) | LightGBM with three losses, a grid and an Optuna search; SHAP importance |
| 5.5 | [ML strategy](notebooks/equity/5_machine_learning/05_ml_strategy.ipynb) | four out-of-sample signals as 44 registered portfolios |
| 5.6 | [Strategy assessment](notebooks/equity/5_machine_learning/06_strategy_assessment.ipynb) | break-even costs, tail risk, signal drift, the Fundamental Law, White's Reality Check |
| 5.7 | [Latent factors](notebooks/equity/5_machine_learning/07_latent_factors.ipynb) | PCA, RP-PCA, IPCA and a conditional autoencoder against the risk model's factors |
| 5.8 | [Deep learning](notebooks/equity/5_machine_learning/08_deep_learning.ipynb) | NLinear, LSTM and TSMixer on weekly returns; an MLP ensemble on the features |
| 5.9 | [Causal ML](notebooks/equity/5_machine_learning/09_causal_ml.ipynb) | double machine learning of momentum's effect; a factor-zoo check |
| 5.10 | [Text features](notebooks/equity/5_machine_learning/10_text_features.ipynb) | the words of 10-Ks and 8-Ks: dictionary tone, year-on-year change, FinBERT, events |
| 5.11 | [Synthetic data](notebooks/equity/5_machine_learning/11_synthetic_data.ipynb) | three generators of synthetic markets, and 5.5's books run on them |

**`notebooks/fixed_income/`** - needs a folder of ICE's month-end constituent files (`EQR_ICE_DIR`) and the
database migrated to the latest schema; nothing from the equity notebooks

| | Notebook | What it does |
|---|---|---|
| 1.1 | [Index constituents](notebooks/fixed_income/1_data_ingestion/01_index_constituents.ipynb) | loads the files (bonds, issuers, weights, analytics), the index's official level from FRED, runs the gates, syncs the cache |
| 2.1 | [US High Yield](notebooks/fixed_income/2_index_replication/01_us_high_yield.ipynb) | rebuilds the ICE BofA US High Yield Index from its bonds, against the official index; its spread, yield, duration and ratings |
| 3.1 | [US High Yield factors](notebooks/fixed_income/3_investment_strategies/01_us_high_yield_factors.ipynb) | value, momentum, size and low risk in the index: every fifth of every signal backtested, each signal's best fifth registered |

Three things to know before running everything:

* the data-ingestion notebooks call the vendors and write to the database - run them when you mean to load data, not
  as a test;
* a strategy notebook registers its runs again each time it runs. A configuration registered before is not another
  trial, so the trial counts and deflated Sharpe ratios do not move; notebooks 3.5 and 3.6 run with the hold-out open;
* the machine-learning notebooks take about five hours the first time; afterwards they read what they kept.

## A backtest

```python
from research_core.data.db.engine import get_engine
from equity_research.data.cache import ResearchCache
from equity_research.strategies.factor_strategy import StrategyConfig, run_strategy

cfg = StrategyConfig(family="value_quality_ls", universe="SP1500", factors=(("value", 1.0), ("quality", 1.0)))
result = run_strategy(ResearchCache(), cfg, engine=get_engine())  # fenced, backtested, registered
result.metrics  # cagr, vol, sharpe, max_drawdown, turnover, trials, deflated_sharpe
```

## Project layout

```
src/
  research_core/            shared by every asset class - it imports none of them
    settings.py             database, data folder, keyring, hold-out start (environment variables, then the keyring)
    charts.py               the notebooks' charts
    data/                   db/ (engine, the shared tables, change-only upserts, migrations, database copy),
                            raw_store (the raw zone), ops (job runs, writer lock), quality (checks as gates),
                            identity (issuers, listings, identifiers), cache (the research cache), artifact_store,
                            calendars
    engine/                 backtest (books to returns), portfolio (scores to weights), tilt (a benchmark plus tilts
                            on a risk model), allocation, registry (runs, trials, deflated Sharpe, hold-out), finish
                            (a run measured and registered)
    analytics/              performance/ (metrics, tracking, periods, holdings), risk/ (statistics),
                            signals/ (labels, evaluation, validation)
  equity_research/          everything that knows about shares - it imports research_core only
    cli.py                  the `eqr` command
    preferences.py          which vendor supplies what
    data/                   the equity tables, the research cache's table list, the artifact store bound to the
                            returns method, reads, panels, accounting, ratios, compare, sectors, universe (a
                            backtest's data)
    ingest/                 vendors/<vendor>/ adapters, membership, reference, classification, countries, corrections
    pipelines/              sources, load_<vendor>, load_estimates, load_sec_text, load_rules, load_corrections, refresh
    quality/                the equity gates; the audit
    strategies/             factors, factor_strategy, construction, learned_value, garp, qgarp (with
                            qgarp_estimates, qgarp_math), rules, runs
    analytics/              indices, dashboard, screen, risk/ (the BFRE factor risk model, a portfolio's risk)
    ml/                     the machine-learning plan: features, models, strategies, their assessment
    research/               helpers for the notebooks only
    regression.py           the regression check: strategies and the risk model measured again
  fixed_income_research/    bond indices from their constituents - it imports research_core only
    cli.py                  the `fir` command
    data/ ingest/ pipelines/ quality/ analytics/ strategies/    see its README
migrations/                 Alembic: one database, one history for every asset class
notebooks/equity/           the five folders above;  notebooks/fixed_income/  1.1, 2.1 and 3.1
docs/                       ARCHITECTURE.md (the design), ML4T_PLAN.md (the machine-learning plan and its log),
                            STRATEGY_REVIEW.md (the review of the strategy notebooks and what was done)
tests/                      unit/ (pure) and integration/ (a scratch database on a local SQL Server)
```

**Dependency rules** (checked by `tests/unit/test_dependency_rules.py`, part of the unit tests):

1. `research_core` imports no asset-class package;
2. the asset-class packages (`equity_research`, `fixed_income_research`) never import one another - what two of
   them need moves to `research_core`;
3. `equity_research/research/` holds helpers for the notebooks only: nothing in `src` or `tests` imports it.

## Adding an asset class

1. **A package** `src/<asset>_research/` (`__init__.py` with `ASSET_CLASS = "<asset>"`, `py.typed`); name it in
   `pyproject.toml` (`[tool.hatch.build.targets.wheel] packages`), in `ASSET_CLASSES` of
   `tests/unit/test_dependency_rules.py` and in `PACKAGES` of `tests/unit/test_database_layout.py`.
2. **Tables** in its own `data/schema.py`, all in the database schema named after the asset class, defined on
   `research_core.data.db.schema.metadata` and reusing the shared ones (`core.issuer`, `core.listing`,
   `core.identifier`: a bond is a listing of an issuer; a crypto asset or an exchange rate a listing without one).
   Register the module in `pyproject.toml` under `[project.entry-points."research_core.asset_classes"]` - that is
   how the migrations, `copy-database` and an issuer merge find its tables - and run `uv sync`.
3. **One migration** under `migrations/versions/` creating its schema and tables and inserting its rows in
   `core.asset_class`, `core.instrument_type` (its kinds of listing), `core.id_type` (its kinds of code) and
   `core.vendor` (its vendors, also in `VENDORS`). The shared tables do not change (docs/ARCHITECTURE.md section 3).
4. **Ingestion**: vendor adapters that land every answer through `research_core.data.raw_store`, write through
   `research_core.data.db.upsert` inside `research_core.data.ops.job`, and end with their own
   `research_core.data.quality.Check`s.
5. **Reads**: a `ResearchCache` with its table list, in `cache/<asset class>` (as `equity_research/data/cache.py`),
   and an `ArtifactStore` of its asset class bound to the versions of its own methods (as
   `equity_research/data/artifacts.py`).
6. **Returns and backtests**: build a `date x instrument` total-return panel - for bonds, carry and roll-down
   belong there - and hand it to `research_core.engine.backtest.run_backtest` (`periods_per_year` when it is not
   daily); finish with `research_core.engine.finish.finish_run(..., asset_class=ASSET_CLASS)`, so every asset class's
   runs are registered and deflated alike, and filed under their asset class.
7. **Notebooks** under `notebooks/<asset>/`, numbered folder.position, each saying what it needs and produces.
