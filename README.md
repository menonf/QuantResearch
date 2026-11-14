# EquityResearch

A point-in-time equity research and backtesting platform for a small team, built so that **no backtest can see data
it could not have had on the day it trades**, and so that **no data vendor is wired into the research**.

It covers two markets since 2005, companies that later failed or were bought included: US equities in the S&P 500,
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

EquityResearch is the other half: **a data platform first, with the research on top** - and the data is yours, from
the vendors you choose.

| | Typical open-source tools | EquityResearch |
|---|---|---|
| **Who the companies were** | today's ticker list, or the vendor's own ids | a security master: a company (*issuer*) and each share line (*listing*) kept apart, every code (ticker, RIC, CUSIP, ISIN, CIK...) with the dates it was valid - a reused ticker is two companies, never one |
| **Index membership** | today's members, or a bundle you buy | every S&P 500 / 400 / 600 and FTSE 100 / 250 / SmallCap member since 2005 with the day it joined and left, rebuilt from the vendor's event history; gates fail on a share counted twice or a member dated before it traded |
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
^SPXEW (0.69 %, 0.9989). [Notebook 2.1](notebooks/2_index_replication/01_sp500.ipynb) runs the same comparison on
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
| `refinitiv` (LSEG) | US: index members with history, listings and codes, sectors, daily prices, dividends and splits, share counts and float, statements | LSEG Workspace signed in; a full build uses ~7,500 of its 10,000 daily requests and resumes if stopped |
| `estimates` (LSEG) | analysts' estimates (I/B/E/S): monthly consensus history, reported actuals, revision counts | LSEG Workspace; ~3,200 requests from nothing |
| `refinitiv_uk`, `estimates_uk` (LSEG) | London: FTSE 100 / 250 / SmallCap members, official closes in pence, dividends, splits, share counts, half-yearly statements as first reported, estimates in pounds | LSEG Workspace; ~4,000 requests from nothing |
| `norgate` | index members with history, listings incl. delisted, unadjusted prices, splits and dividends, GICS | Norgate Data Updater with a subscription |
| `sec` | company accounts as first reported, with every correction | a contact name and e-mail (the SEC asks for one) |
| `yahoo` | benchmark index levels and funds (^GSPC, SPY, RSP, ^SPXEW, SPTM, ^SP400, ^SP600) | nothing |
| `financedatabase` | codes (ISIN, FIGI, Yahoo symbol), where a company is run from, its own sector labels | nothing |
| `tiingo` | unadjusted prices, dividends and splits for listings other sources found, cross-checked | an API key |
| `simfin` | standardised statements, banks and insurers included; its own sector labels | an API key |

The default build runs `refinitiv, estimates, refinitiv_uk, estimates_uk, yahoo, financedatabase, sec`, in that
order - about 15,000 LSEG requests, so two or three days of its allowance; a build stopped by the allowance continues
the next day. Choosing others is a setting - a build without LSEG, for example:

```sh
EQR_SOURCES=norgate,yahoo,financedatabase,sec  EQR_PRICE_VENDORS=norgate  EQR_MEMBER_VENDORS=norgate  uv run eqr build
```

Filing text is loaded apart (`eqr load-sec-text`, notebook 1.3): the SEC's filing index, every annual report since
2003 (about 10 GB of documents, kept in the raw zone) and their text measures, with the Loughran-McDonald dictionary
from Notre Dame's SRAF; all free.

## Setup

```sh
uv sync --all-groups
uv run pytest                                    # unit tests: no database, no network
uv run pytest -m integration                     # database tests on a local SQL Server (never Azure)
uv run eqr set-key simfin                        # a vendor's API key, kept in the OS keyring (typed without echo)
```

On Windows torch is its CUDA build (`[tool.uv.sources]` in `pyproject.toml`): the neural models (FinBERT, the LSTM,
the MLP ensemble) run on a GPU when there is one and on the processor otherwise.

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
uv run eqr load-lseg         # one source alone (--market UK for London) - also load-norgate, load-sec,
                             #   load-benchmarks, load-financedatabase, load-tiingo, load-simfin, load-estimates; and
                             #   probe-lseg, probe-lseg-uk, probe-norgate, probe-tiingo, probe-simfin, probe-estimates
uv run eqr load-sec-text --tone --tone-device cuda
                             # filing index, annual reports (10-K) and their text measures; --tone adds FinBERT
uv run eqr load-rules        # the repository's strategy rule sets (Barings' cleaning limits...) into the shared
                             #   tables research.rule_set / research.strategy_rule; a loaded set is never changed
uv run eqr load-corrections  # corporate actions no vendor recorded (ingest/corrections/corporate_actions.csv:
                             #   Alphabet's class C shares of 2014...), written under vendor `repository`
uv run eqr copy-database "MENONPC\SQLEXPRESS" equity_research_local
                             # copy a verified local build into the configured (Azure) database, checked per table
```

## Notebooks

Written for an undergraduate in finance, economics or a quantitative subject: standard terms (look-ahead bias,
tracking error, information ratio...) are used and defined, with the method behind each result.

Grouped by topic in six folders; each notebook's number is *folder.position* (3.2 is the second strategy), and the
notebooks refer to one another by it.

**`notebooks/1_data_load/`** - building and maintaining the database

* [1.1 Initial load](notebooks/1_data_load/01_initial_load.ipynb) - builds the database from nothing: every dataset,
  where it comes from and why it is needed.
* [1.2 Refresh](notebooks/1_data_load/02_refresh.ipynb) - keeps it current.
* [1.3 Filing text](notebooks/1_data_load/03_filing_text.ipynb) - the words: the SEC's filing index, every annual
  report (10-K) since 2003 and their text measures (Loughran-McDonald counts, similarity to the year before, FinBERT's
  tone - on the GPU when there is one).

**`notebooks/2_index_replication/`** - rebuilding real indices from our own data, the first check that it is right

* [2.1 S&P 500](notebooks/2_index_replication/01_sp500.ipynb) - rebuilds the S&P 500 by market value and equally, and
  compares it with ^GSPC, SPY, ^SPXEW and RSP.
* [2.2 FTSE All-Share](notebooks/2_index_replication/02_ftse_all_share.ipynb) - rebuilds the FTSE All-Share and its
  FTSE 100, 250 and SmallCap segments in pounds, compares them with LSEG's index levels, and shows what equal weights
  and leaving out investment trusts do.

**`notebooks/3_investment_strategies/`** - strategies, backtested and registered

* [3.1 Multi-factor long-short](notebooks/3_investment_strategies/01_multifactor_long_short.ipynb) - a multi-factor
  long-short strategy on the S&P 1500.
* [3.2 Multi-factor long-only](notebooks/3_investment_strategies/02_multifactor_long_only.ipynb) - a long-only
  strategy on the S&P 1500, against SPTM, an equal-weight S&P 1500 and SPY.
* [3.3 Long-only with extra signals](notebooks/3_investment_strategies/03_multifactor_long_only_extra_signals.ipynb) -
  four candidate signals (earnings surprise, residual momentum, free-cash-flow yield, accruals), each added on its own
  to 3.2's long-only portfolio.
* [3.4 Learned value score](notebooks/3_investment_strategies/04_multifactor_long_only_ml_value.ipynb) - a
  machine-learned value score (after *Build Your Own AI Investor*), trained walk-forward, against the plain one in
  3.2's long-only portfolio.
* [3.5 S&P 500 GARP](notebooks/3_investment_strategies/05_sp500_garp.ipynb) - the S&P 500 GARP index (growth at a
  reasonable price) rebuilt from its published rules and tested against SPY; Part 2 runs notebook 3.6's Barings
  QGARP strategy on the same S&P 500 universe.
* [3.6 Barings GARP](notebooks/3_investment_strategies/06_barings_garp.ipynb) - Barings' GEMS factors and QGARP score
  rebuilt from their production code on our data, and a book of the best QGARP names tested against SPTM (without
  I/B/E/S estimates for now: quality and value only).
* [3.7 ARIMA-GARCH](notebooks/3_investment_strategies/07_arima_garch.ipynb) - a daily long/short timing of the S&P 500
  by an ARIMA model chosen by AIC and a GJR-GARCH model fitted each day to the last 24 returns (after
  `menonf/PythonProjects`), held in SPY against SPY.

**`notebooks/4_portfolio_analytics/`** - tools for looking at stocks and portfolios rather than strategies

* [4.1 Stock screen](notebooks/4_portfolio_analytics/01_stock_screen.ipynb) - any registered strategy (chosen by its
  run id) ranking today's companies by its own score, with the numbers behind each rank and the book it would hold if
  it rebalanced today.
* [4.2 Factor dashboard](notebooks/4_portfolio_analytics/02_factor_dashboard.ipynb) - S&P's monthly factor dashboard
  for any registered runs of one universe, backtested again without registering them (by default our S&P 500 factor
  books, as of the last month end before the hold-out).
* [4.3 Risk model](notebooks/4_portfolio_analytics/03_risk_model.ipynb) - a fundamental factor risk model after
  BlackRock's BFRE: a portfolio's risk split into market, industry, style and stock-specific parts, with its VaR and
  the model's own tests.

**`notebooks/5_analysis/`** - questions about method rather than strategies

* [5.1 Choosing a learning algorithm](notebooks/5_analysis/01_choosing_ml_algorithm.ipynb) - the eight learning
  algorithms of *Build Your Own AI Investor* explained, how the book chose among them, and the choice made again
  without hindsight: candidates fixed in advance, each trained walk-forward, chosen each January by its out-of-sample
  rank IC, every portfolio registered.
* [5.2 Choosing a learning algorithm, second version](notebooks/5_analysis/02_choosing_ml_algorithm_v2.ipynb) - the
  same choice after four changes diagnosed from 5.1: learn each company's percentile rather than its return,
  percentile features, settings for a weak signal, and price, size, issuance, surprise and growth features beside the
  book's ratios; a new registry family, deflated by both studies' trials.
* [5.3 Choosing a learning algorithm, third version](notebooks/5_analysis/03_choosing_ml_algorithm_v3.ipynb) - why
  5.2's better ranking earned nothing (a percentile target learns the typical year, a portfolio is paid the average
  one), and the choice made a third time: a clipped excess-return target, the models judged by what their top 30 %
  went on to earn, a forecast that ranks nobody holding the largest companies.

**`notebooks/6_machine_learning/`** - the machine-learning plan (`docs/ML4T_PLAN.md`), after *Machine Learning for
Trading* (3rd edition)

* [6.1 Research protocol](notebooks/6_machine_learning/01_research_protocol.ipynb) - the rules every later notebook
  follows (hold-out, walk-forward folds purged of overlapping labels, searches registered whole, signals triaged
  first), the labels, and feasibility: breadth, moves against trading costs, and the evaluation toolkit on nine known
  factors.
* [6.2 Features](notebooks/6_machine_learning/02_features.ipynb) - about fifty company features (momentum, risk,
  liquidity, accounts, model-based volatility and trend) and two date features (market regime, calendar), each triaged
  alone against next month's return with false-discovery control, their decay, redundancy and overlap with the
  existing factors and the risk model's styles.
* [6.3 Linear models](notebooks/6_machine_learning/03_linear_models.ipynb) - ridge, lasso, elastic net and logistic
  regression on all the features, tuned inside each walk-forward fold, against a plain average of 6.2's two survivors;
  coefficient stability and conformal intervals.
* [6.4 Gradient boosting](notebooks/6_machine_learning/04_gradient_boosting.ipynb) - LightGBM with squared, Huber and
  ranking losses, a grid tuned in each fold and an Optuna search before 2015, against 6.3's models, the composite and
  notebook 3.4's learned value; SHAP importance.
* [6.5 ML strategy](notebooks/6_machine_learning/05_ml_strategy.ipynb) - four out-of-sample signals as 44 portfolios:
  long-only and long-short deciles, equal, inverse-volatility, HRP and mean-variance weights, and volatility, drawdown
  and Kelly overlays, all registered.
* [6.6 Strategy assessment](notebooks/6_machine_learning/06_strategy_assessment.ipynb) - those strategies' break-even
  costs, tail risk, signal drift, transfer coefficients and the Fundamental Law, White's Reality Check and Rademacher
  bounds; what the hold-out step needs.
* [6.7 Latent factors](notebooks/6_machine_learning/07_latent_factors.ipynb) - PCA, RP-PCA, IPCA and a conditional
  autoencoder on characteristic-managed portfolios, out of sample, against the risk model's factors: explained
  variance, pricing errors, the SDF's Sharpe ratio.
* [6.8 Deep learning](notebooks/6_machine_learning/08_deep_learning.ipynb) - NLinear, LSTM and TSMixer on weekly
  returns against last week's reversal, and an MLP ensemble on the features against the linear models and gradient
  boosting.
* [6.9 Causal ML](notebooks/6_machine_learning/09_causal_ml.ipynb) - double machine learning of momentum's effect with
  the other characteristics as controls, and a factor-zoo check of the features that survived the triage.
* [6.10 Text features](notebooks/6_machine_learning/10_text_features.ipynb) - the words of 10-Ks and 8-Ks (loaded by
  1.3): Loughran-McDonald tone, year-on-year change of the report ("Lazy Prices"), FinBERT's tone of the MD&A, 8-K
  events; triaged and checked against the price and accounts features.
* [6.11 Synthetic data](notebooks/6_machine_learning/11_synthetic_data.ipynb) - three generators of synthetic markets
  (block bootstrap, factor GARCH-t, regime switching), their realism against the stylized facts, and 6.5's books run
  on 200 synthetic decades and two stress scenarios.

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
                reference (indices, universes), classification (sector history), countries, corrections
                (corporate actions no vendor recorded);
                vendors/<vendor>/ adapters: refinitiv (LSEG), norgate, sec (accounts; filing index and text), yahoo,
                financedatabase, tiingo, simfin, sraf (the Loughran-McDonald dictionary)
  pipelines/    sources (the registry a build runs), load_<vendor> for each adapter, load_estimates, load_sec_text,
                load_rules, load_corrections, refresh, copy_database
  quality/      checks and gates; audit (the architecture rules and fitness for research)
  research/     the data: cache, store (results kept by data version), panels, accounting, ratios, compare, data
                strategies: factors, portfolio, backtest, strategy, registry, runs, rules (shared rule sets),
                indices, screen, dashboard,
                risk_model, garp and barings_* (GARP, QGARP), arima_garch, ml and model_choice* (notebooks 5.x)
                the ML plan: labels, validation, evaluation, features, text_features, linear, boosting, deep,
                latent, autoencoder, causal, allocation, assessment, ml_strategy, synthetic, regression
  charts.py     the notebooks' charts
migrations/     Alembic
notebooks/      the six folders above
docs/           ARCHITECTURE.md (the design), ML4T_PLAN.md (the machine-learning plan and its log)
tests/          unit/ (pure) and integration/ (a scratch database on a local SQL Server)
```
