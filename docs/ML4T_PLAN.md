# Plan: the ideas of *Machine Learning for Trading* (3rd edition) on this repository's data

A working plan, kept up to date as phases land: tick the boxes, note what changed and why. The design rules in
`ARCHITECTURE.md` and the working rules in `CONTRIBUTING.md` apply to all of it; the compatibility rules below are
what keeps the existing research and notebooks unchanged while it grows.

## Source

* Repository: <https://github.com/stefan-jansen/machine-learning-for-trading>, third edition (27 chapters, nine case
  studies), read at commit `a4e1d50` (2026-10-29). Code under the MIT licence; the book text is not needed - each
  chapter's README maps sections to notebooks, and the methods come from the papers they cite.
* The second edition is on the repository's `second-edition` branch; this plan follows the third.
* Methods are **reimplemented** here as pure, tested functions on our panels, not ported: the book's notebooks run on
  Polars, its own `ml4t-*` libraries (0.1.x) and Docker images, on data we do not have (NASDAQ Data Link).
* Two case studies are the templates, because they are the ones our data can run:

| Case study | Cadence | Book's data | Ours |
|---|---|---|---|
| `case_studies/us_equities_panel` | daily, label `fwd_ret_1d` (also 5d, 21d) | ~3,200 US stocks 1990-2018Q1, OHLCV only, 62 features, 16 folds (10y train, 1y validation) | S&P 1500 point in time, 2005-2024 |
| `case_studies/us_firm_characteristics` | monthly, label `fwd_ret_1m` | anonymised characteristics 1996-2016, 6-month accounting lag, 57 characteristics | the same characteristics from our accounts, dated by publication - no lag guess needed |

Their pipeline is the plan's spine: feasibility -> labels -> features -> evaluation -> linear -> GBM -> deep learning ->
latent factors -> causal -> model analysis -> backtest -> portfolio -> risk -> costs -> hold-out.

## Our data, as it bears on the plan

* Prices: LSEG from 1998-11 (2003-11 before the rebuild of 2026-11-17), daily, unadjusted with corporate actions;
  total-return and split-adjusted panels in `panels.py`. Membership: the S&P 500 from 2000, the S&P 400 and 600 from
  2001; the study starts in 2005.
* Accounts: SEC (XBRL from 2009-2011) and LSEG (published from 1999), point in time by `published_at`.
* Walk-forward: 10 years' training then one year's validation gives 10 folds, validating 2015-2024. A shorter or
  expanding training window gives more; the choice is a registered config, not a tweak.
* Hold-out: 2025-01-01 onward (22 months by now) - the book's "evidence boundary". Only the final step opens it.
* Breadth: ~1,500 names, not ~3,200 - by the Fundamental Law the same IC buys less; the feasibility step shows how much.
* Compute: CPU only (no usable GPU). Deep models are sized for it: smaller windows, weekly cadence where the book
  offers it. *Since 2026-11-08 an RTX 4060 Laptop GPU (8 GB, CUDA) works and is used wherever it helps - see the log.*

## Chapters: in scope, later, out

| Chapter | Verdict | Why |
|---|---|---|
| 1 Process is edge (regimes) | in (P1) | factor and macro regimes; macro needs P8 |
| 2 Data universe | already done | point-in-time, survivorship, provider comparison, quality gates are this repository |
| 3 Market microstructure | partly (A6) | 15-minute bars and quote features from a free SIP source or LSEG; ITCH order books out |
| 4 Fundamental and alternative data | partly (P8) | SEC XBRL done; FRED macro, Form 4 insiders, 13F holdings are new data; crypto, Kalshi, Polymarket out |
| 5 Synthetic data | later (P10) | low value for this project |
| 6 Strategy research framework | in (P0) | universe, schedule, costs, walk-forward folds, run logging |
| 7 Defining the learning task | in (P0) | labels, IC with HAC errors, multiple testing, quantile analysis |
| 8 Financial features | in (P1) | price-volume families, liquidity and spread estimates from daily bars, fundamentals, calendar |
| 9 Model-based features | in (P1) | GARCH, HAR, fractional differencing, Kalman, HMM regimes, structural breaks |
| 10 Text features | later (P9) | needs a filing-text store |
| 11 ML pipeline (linear, SHAP, conformal) | in (P2) | the baseline every later model must beat |
| 12 Gradient boosting, tabular DL | in (P3; TabM in P6) | LightGBM, Optuna, TreeSHAP, conformal |
| 13 DL for time series | in (P6) | NLinear, LSTM, TSMixer; transformers / Mamba optional |
| 14 Latent factors | in (P5) | PCA, IPCA, RP-PCA, conditional and supervised autoencoders, SDF - set against the BFRE risk model |
| 15 Causal ML | in (P7) | Double ML on momentum, factor-zoo validation |
| 16 Strategy simulation | in (P4) | our engine; DSR exists; add Rademacher Anti-Serum, White's Reality Check |
| 17 Portfolio construction | in (P4) | mean-variance on the risk model, HRP, Kelly; deep allocation optional |
| 18 Transaction costs | partly (P4, A6) | daily spread estimates and cost grids in P4; measured spreads and impact with A6's intraday data |
| 19 Risk management | partly done (P4) | VaR, stress, factor exposure exist (notebook 4.2); add CVaR, overlays, drift |
| 20 Strategy synthesis | in (P4) | IC-to-Sharpe, Fundamental Law diagnostics, the hold-out step |
| 21 RL execution, hedging | **out** | needs order books and options |
| 22-23 RAG, knowledge graphs | later (P9) | on SEC filings; an LLM's cost and keys are a decision |
| 24 Autonomous agents | **out** | not a data-research method |
| 25 Live trading | **out** | this repository is research only; no broker layer |
| 26 MLOps | partly (P4) | drift monitoring of a live signal; feature stores and MLflow are what the artifact store and the registry already are |
| 27 Systematic edge | read | conclusions |

The other seven case studies (ETFs, FX, futures, crypto perpetuals, NASDAQ-100 intraday, two on options) need data we
do not hold yet; where it can be had free or from LSEG they are scoped in "Other asset classes" below.

## Design within the existing architecture

* **No new architecture for P0-P7.** Pure functions in `equity_research/ml/` (what any asset class can use in
  `research_core`: labels, evaluation and validation in `analytics/signals/`, allocation in `engine/`), a notebook per
  phase (in `notebooks/equity/5_machine_learning/`, numbered 5.1 on) that only sets parameters and shows results, unit
  tests on synthetic panels with known answers.
* **Artifact store** (`research_core/data/artifact_store.py`, bound to the returns method in
  `equity_research/data/artifacts.py`): labels, features and predictions as Parquet under
  `data/<db name>/research/<asset class>/<kind>/<name>-<hash>.parquet`, each with a sidecar recording its config, the data snapshot
  (max `loaded_at`) and the code version - computed once per data version, like the book's `labels/*.parquet` plus
  `.digest`. `data/` is not in git.
* **Training runs are registered runs.** A fitted model configuration is a row in `research.run` (no schema change):
  `family` names the idea (`ml4t_daily_gbm`), the config carries the stage (`training` / `backtest` / `holdout`),
  the fold scheme and hyperparameters; the metrics carry the fold ICs. Every Optuna trial is a registered config, so
  `trial_count` - and the deflated Sharpe - see the whole search.
* **New data (P8-P9) follows "Choosing vendors, adding one"** in `ARCHITECTURE.md`: a client, the raw zone, batches,
  canonical tables by Alembic migration, gates and audit rules.
* **Dependencies** in optional groups, added in the phase that first needs them:
  `ml` = lightgbm, optuna (P3; statsmodels and arch are base dependencies); `deep` = torch (P5-P6); `text` =
  transformers (P9's FinBERT). Checked 2026-10-30: adding both leaves
  numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1, duckdb 2.0.0 and pyarrow 26.0.0 where they are, and
  every one has a Windows wheel for Python 3.14 (lightgbm 4.7.0, statsmodels 0.15.0, optuna 5.0.0, torch 2.15.0).
  The book's `ml4t-*` libraries are not dependencies; their results may be used in a test as a cross-check.

## Compatibility rules (so nothing existing breaks)

1. **Existing functions keep their signatures and defaults.** What the notebooks of sections 2-4 import stays as it
   is: `universe.load_universe_data`, `factor_strategy.StrategyConfig` / `run_strategy` / `factor_scores` /
   `TRAIN_FROM`, `metrics.performance` / `active_performance` / `decile_returns`, `registry.deflated_sharpe` /
   `HoldoutClosedError`, `indices.*`, `sectors.listing_sectors`, `screen.*`, `runs`, `garp.*`, `qgarp.*`,
   `dashboard`, `learned_value`, `factors`, `risk_model`, `portfolio_risk`, `risk_data`, `risk_styles`. New behaviour
   is a new function or a new keyword whose default is today's behaviour.
2. **New registry families only.** Never reuse an existing family name, so existing trial counts and deflated Sharpes
   do not move.
3. **A new table is migrated in every local database before it joins the cache.** `ResearchCache.sync` and
   `copy-database` walk every table in the schema; a database without the table would fail. Verify the migration on
   a local database first (the integration tests' scratch databases, `quant_research_gate`), then migrate the working
   database (`quant_research_local`) in the same change, and Azure only with confirmation.
4. **Lock changes are read before they are kept.** A dependency group that moves numpy, pandas, scikit-learn, scipy,
   duckdb or pyarrow is a decision, not a side effect (`learned_value.py`'s model and every notebook number depend on
   them).
5. **A regression check after each phase.** Before P0, record the key results of the strategy and analytics notebooks
   (registered run metrics, the risk model's summary numbers: today 3.1-3.7, 4.1 and 4.2, `regression.FAMILIES`);
   after each phase, rerun them on the same data snapshot and compare.
6. **Optional dependencies stay optional.** Modules needing torch import it lazily; their tests skip without it, so
   the base gate runs without the `deep` group.
7. **The hold-out stays shut** until a phase's final step, once per final strategy, recorded by the registry.
8. **Other asset classes stay out of the equity research.** Their instruments get their own indices and universes;
   `load_universe_data`, the S&P universes and every existing panel read only what they read today. New instrument
   types and tables arrive by migration (rule 3); code that annualises or builds calendars takes the periods per year
   and the calendar as keywords defaulting to today's (252 NYSE days).

## Phases

Each phase is done when the gate passes (`ruff format`, `ruff check`, `mypy`, unit tests; integration tests when the
database is touched), `ARCHITECTURE.md` lists the new modules and notebook, the notebook runs from start to finish on
the working database (`quant_research_local`; Azure `equity_research` until 2026-11-13), and the regression check
(rule 5) shows no change.

### P0 - Groundwork: protocol, labels, folds, evaluation (chapters 6, 7)

Book: `06_strategy_definition/02_cv_foundations`, `07_defining_the_learning_task/03_label_methods`, `05_signal_evaluation`,
`06_ic_inference`, `07_multiple_testing`; case study `01_feasibility_analysis`, `02_labels`, `05_evaluation`.

- [x] Baseline for the regression check (rule 5); a git commit so runs record a code version (today every run's
      `code_version` is empty - the repository has no commits)
- [x] `research_core/data/artifact_store.py` - the artifact store
- [x] `research_core/analytics/signals/labels.py` - forward returns (1, 5, 21 days; 1 month), triple-barrier, trend-scanning
- [x] `research_core/analytics/signals/validation.py` - walk-forward folds with purging (label horizon) and an embargo; nested CV for tuning
- [x] `research_core/analytics/signals/evaluation.py` - IC series (rank and Pearson), Newey-West t-statistics, IC decay by horizon, quantile
      spreads, turnover and rank autocorrelation, Benjamini-Hochberg false-discovery control, a triage ledger
- [x] registry helper: register each point of a search as a trial (`registry.record_search`)
- [x] Notebook 5.1 *ML research protocol*: feasibility (breadth by date, move against cost, the folds)

### P1 - Features (chapters 8, 9, 1)

Book: `08_financial_features/01-07`, `09_model_based_features/02-14`; case studies `03_financial_features`,
`04_model_based_features`.

- [x] `equity_research/ml/features.py` - price-volume families (momentum, reversal, volatility, liquidity: Amihud, Corwin-Schultz
      and Roll spreads from daily bars), the fundamentals from `ratios.py` / `factors.py`, calendar. Corwin-Schultz
      is not possible: our LSEG bars carry close and volume only (no high, low or open); Roll's spread needs closes
- [x] `equity_research/ml/model_features.py` - GARCH(1,1), HAR volatility, fractional differencing, Kalman trend, HMM regimes,
      structural breaks. No `ml` group needed: `arch` and `statsmodels` are base dependencies already
- [x] Notebook 5.2 *features*: every feature through `evaluation.py`; the triage ledger; overlap with existing factors
      and the risk model's styles

### P2 - Linear baseline (chapter 11)

Book: `11_ml_pipeline/01-08`; case studies `06_linear` / `05_linear`.

- [x] ridge, lasso, elastic net, logistic, fitted per fold; nested tuning; coefficient stability
- [x] conformal prediction intervals
- [x] Notebook 5.3 *linear models*, against an equal-weighted composite of the surviving features

### P3 - Gradient boosting (chapter 12)

Book: `12_gradient_boosting/02, 04, 07, 08, 11`; case studies `07_gbm` / `06_gbm`.

- [x] LightGBM per fold; a grid over leaf profiles and losses, then Optuna (every trial registered)
- [x] TreeSHAP through LightGBM's `pred_contrib`; conformal intervals
- [x] Notebook 5.4 *gradient boosting*, against P2 and `learned_value.py`'s learned value

### P4 - From predictions to a strategy (chapters 16-20, 26)

Book: `16_strategy_simulation/11-14`, `17_portfolio_construction/02, 03, 04, 06`, `18_transaction_costs/02, 09-11`,
`19_risk_management/01, 02, 04, 07`, `20_strategy_synthesis/*`; case studies `16_backtest` to `22_strategy_analysis`.

- [x] entry schemes: daily top-K long-short, monthly deciles - through `backtest.py` (new functions, rule 1). Top-K
      is built (`allocation.top_k_book`) but not run: every signal is monthly, and 5.1 found a day's move covers 2.5
      round trips
- [x] allocators: equal, inverse volatility, mean-variance on the risk model's covariance, HRP, Kelly scaling
- [x] cost grids and break-even costs; spread estimates from P1
- [x] risk overlays (volatility target, drawdown control), CVaR and historical VaR, signal drift
- [x] multiple testing beyond the DSR: Rademacher Anti-Serum, White's Reality Check (raw and studentised)
- [x] Fundamental Law diagnostics (IC, breadth, transfer coefficient -> Sharpe)
- [ ] the hold-out step: refit the chosen configuration on history before 2025, trade 2025 onward once - **left
      for the owner's decision** (which strategy, if any; see 5.6's findings)
- [x] Notebooks 5.5 *ML strategy* and 5.6 *strategy assessment*

### P5 - Latent factors (chapter 14)

Book: `14_latent_factors/01-08`; case studies `13a_pca`, `13b_ipca`, `08a-08d`.

- [x] PCA and eigenportfolios, RP-PCA, IPCA
- [x] conditional autoencoder (Gu, Kelly, Xiu), supervised autoencoder, SDF (dependency group `deep`). The supervised
      autoencoder is left out: a second network for little extra here; the SDF is each model's tangency portfolio
- [x] Notebook 5.7 *latent factors*: explained variance and pricing errors set against the BFRE risk model

### P6 - Deep learning (chapters 13, 12)

Book: `13_dl_time_series/01, 03, 06, 10`, `12_gradient_boosting/03`; case studies `08_tabular_dl` to `12_dl_weekly`.

- [x] NLinear, LSTM, TSMixer (weekly cadence first, for the CPU); a TabM-style MLP ensemble on the features
- [x] fixed seeds, deterministic CPU training, every architecture registered
- [x] Notebook 5.8 *deep learning*, against P2 and P3

### P7 - Causal ML (chapter 15)

Book: `15_causal_estimation/03, 05, 11`; case studies `14_causal_dml` / `09_causal_dml`.

- [x] Double ML with cross-fitting (scikit-learn): the effect of 12-1 momentum with factor confounders
- [x] factor-zoo validation of the features that survived P1
- [x] Notebook 5.9 *causal ML*

### P8 - New data: macro, insiders, institutional holdings (chapter 4) - needs a decision

Book: `04_fundamental_alternative_data/03, 06, 07, 10`.

- [ ] FRED / ALFRED macro series with vintages (point in time: GDP and payrolls are revised)
- [ ] SEC Form 4 insider transactions
- [ ] SEC 13F institutional holdings
- [ ] each: client, raw zone, tables by migration (rule 3), gates, audit rules; features fed back into P1

### P9 - Text and language models (chapters 10, 22, 23) - needs a decision

Book: `10_text_feature_engineering/03, 06, 08, 09`, `22_rag_financial_research/01`, `23_knowledge_graphs/08`.

- [x] filing text from EDGAR in the raw zone; a document table with publication times - every 10-K, 10-Q and 8-K
      listed in `equity.document` (with the acceptance instant and an 8-K's items); the text of the 10-Ks only (the
      owner's choice), 47,540 of them since 2003
- [x] Loughran-McDonald sentiment, year-on-year 10-K change ("Lazy Prices"), 8-K event extraction - notebook 5.10
- [x] FinBERT - 45,547 reports scored on the GPU, in the rebuilt local database; notebook 5.10 rerun with it
- [ ] (not wanted, the owner's decision of 2026-11-08) RAG and knowledge graphs

### P10 - Optional

- [x] synthetic data for stress-testing strategies (chapter 5): `synthetic.py`, notebook 5.11
- [ ] (not wanted, the owner's decision of 2026-11-08) minute bars for the whole S&P 1500 (beyond A6's 115 names at
      15 minutes): ~150 million rows a year, past SQL
      Server Express's 10 GB a database, so they would live in Parquet only - an exception to "every canonical row
      is in SQL" that `ARCHITECTURE.md` would have to state

## Other asset classes: the seven remaining case studies

Each runs the same pipeline as the equity work (P0-P4 must exist first): what changes is the data, the instruments and
a few pieces of the backtest. Sources checked from this PC on 2026-10-30 (LSEG Workspace was not running: its column
is from what the platform offers and is confirmed by the probe below before any build).

| Case study (book's setup) | Free source | LSEG | Verdict | Effort |
|---|---|---|---|---|
| **A1 ETFs** - 100 ETFs, 9 asset classes, daily, monthly decisions, 2006-2025 (`case_studies/etfs/config/setup.yaml`) | Tiingo and Yahoo (clients exist; prices with dividends and splits) | yes - one history request per RIC | **in**, both | S |
| **A2 FX pairs** - 20 G10 pairs, daily, 2005-2025, momentum and carry | FRED H.10 daily rates (API host answers - needs a free key; the web host times out from this PC); short rates for carry are monthly there | yes - spot (`EUR=`), deposit rates and forward points for daily carry | **in**: LSEG first, FRED as the second vendor | M |
| **A3 CME futures** - 30 products, 7 sectors, daily, weekly decisions, 2011-2025, carry and term structure | none adequate (Yahoo has the front month only; the book's Databento is paid) | yes - continuation RICs by tenor (`ESc1`, `ESc2`...) with open interest; expired contracts by `^` RIC | **in**, LSEG only | M-L |
| **A4 Crypto perpetuals** - 19 Binance perps, 8-hourly, funding settlements, 2020-2025 | Binance futures API and `data.binance.vision` bulk files (both answer here, no key), Bybit as a second vendor | no Binance funding | **in**, free | M |
| **A5 S&P 500 equity + option analytics** - IV level, skew, term structure, VRP for 633 stocks, weekly, 2017-2021 | none for single-stock implied-volatility history; Cboe index series (VIX, VIX3M, SKEW) for a market-level version | per-stock implied-volatility history: **to probe** | **in if the probe finds it**; otherwise a reduced version on index-level volatility features | M |
| **A6 NASDAQ-100 microstructure** - 115 stocks, 15-minute bars, order flow and spreads, 2020-2021 | Alpaca free account: consolidated (SIP) bars, trades and quotes since 2016 - **to verify**; IEX HIST pcaps since 2016 (answer here, free) but IEX is one venue's few per cent of volume | intraday summaries with bid/ask, but only a recent window (depth **to probe**) | **in with Alpaca** (2020-2021 as the book); LSEG alone gives only recent months, inside our hold-out | L |
| **A7 S&P 500 option straddles** - weekly short ATM straddles held to expiry, daily delta hedge, 2017-2021 | none (Cboe DataShop and OptionMetrics are paid) | expired option RICs, one request each: ~52,000 contracts a year for 500 names, ~2,400 for S&P 100 monthly | **reduced only**: S&P 100, monthly entries, LSEG (~2 days of allowance for 5 years) | XL |

### X0 - What the new asset classes need from the architecture (before A2-A7)

ETFs (A1) fit what exists: `etf` is already an instrument type and their prices are daily bars. The rest need, each
by migration with gates and audit rules, migrated in every local database (rule 3):

- [ ] instrument types `future`, `perp`, `option` (listings without an issuer, as index levels are today; `fx` exists
      since migration 0001 - the equity research's exchange rates)
- [ ] `<asset class>.contract` (each in the schema of the package that loads it, docs/ARCHITECTURE.md section 3) -
      a future's root, expiry, multiplier and sector; an option's underlying, expiry, strike and right
- [ ] `<asset class>.intraday_bar` - timestamped bars (UTC start, interval) with bid/ask where the source gives them: crypto's
      8-hour bars and A6's 15-minute bars (115 names x 26 bars x 252 days ~ 0.75 million rows a year: SQL is fine)
- [ ] `<asset class>.rate` - per-instrument rates: perpetual funding, FX deposit rates; shares its design with P8's macro
      series
- [ ] calendars: 24/7 crypto, the FX week, exchange sessions for futures - research code takes the calendar and the
      periods per year as keywords (rule 8)
- [ ] the LSEG probe (below), then a probe pipeline per new source, as `load_tiingo` / `load_simfin` have

### A1 - ETFs (chapters 16-17; the book's portfolio-construction setting)

- [ ] the book's 100 ETFs as a listing set; an index `ETF100` with a spell per ETF from its first trade - the list is
      today's survivors, a bias the notebook states
- [ ] prices from LSEG and Tiingo (two vendors, compared as `compare.py` does for accounts); distributions as corporate
      actions so total returns are right
- [ ] the pipeline at monthly decisions, 21-day label, 5-15 bps a leg; the IC-against-Sharpe comparison across model
      families; allocators from P4
- [ ] Notebook *ETF rotation*

### A2 - FX pairs

- [ ] 20 pairs' daily spot (LSEG, FRED second); daily carry from deposit rates or forward points
- [ ] returns that include carry; costs of 1-8 bps; the pipeline at daily decisions, 5-year training windows
- [ ] Notebook *FX momentum and carry*

### A3 - CME futures

- [ ] the 30 products' continuation series to the 2nd-3rd tenor with open interest; expired contracts where the roll
      needs them
- [ ] continuous series by ratio back-adjustment, roll yield and carry - pure functions with tests
- [ ] the backtest in notional exposure (leverage), costs of commission + spread + roll slippage; weekly decisions
- [ ] Notebook *futures carry and term structure*

### A4 - Crypto perpetuals

- [ ] 8-hourly bars, mark prices and funding settlements for the 19 pairs from Binance; Bybit as the second vendor
- [ ] funding cash flows in the return; 2 bps maker / 4 bps taker; 8-hour label; two folds as the book
- [ ] Notebook *crypto funding*

### A5 - S&P 500 equities with option information

- [ ] probe first; with per-stock implied volatility: IV level, skew, term structure, variance risk premium against
      realised volatility, lagged one day
- [ ] equity features from P1, weekly Friday decisions, Monday-open execution (needs open-price execution in the
      backtest - a new keyword, rule 1)
- [ ] without it: market-level volatility features (VIX term structure, SKEW) as regime inputs to P3's model
- [ ] Notebook *option information in equity returns*

### A6 - NASDAQ-100 microstructure

- [ ] a free Alpaca account (key in the keyring) and a probe of its history; 15-minute bars plus quote-derived
      features (relative spread, order-flow imbalance, quote staleness), computed from trades and quotes landed in
      the raw zone - only the aggregates become rows
- [ ] measured spreads replace P4's estimates for these names; cost model per share plus half-spread
- [ ] a hold-out inside the study's own window (as the book: 6-month folds) - stated, not the 2025 date
- [ ] Notebook *intraday microstructure*

### A7 - S&P 100 option straddles (reduced)

- [ ] option contracts by expired RIC: monthly ATM straddles on the S&P 100, daily bid/ask, implied volatility, delta
- [ ] a cohort backtest - overlapping positions, an option leg and its daily delta hedge, premium-denominated costs;
      a new engine beside `backtest.py`, which holds one weight per listing
- [ ] Notebook *option straddles*; only after A5 shows the option data is good

### The LSEG probe (~20 requests, when Workspace is running)

- [ ] ETFs: `SPY`, `TLT.O`, `GLD` daily from 2006, with dividends
- [ ] FX: `EUR=`, `JPY=` from 2005; a deposit rate (`EUR1MD=`, `USD1MD=`) and 1-month forward points (`EUR1M=`)
- [ ] futures: `ESc1`, `ESc2`, `CLc1`, `CLc2`, `TYc1`, `GCc1` from 2011 with open interest; one expired contract by `^` RIC
- [ ] intraday: `AAPL.O` 15-minute bars with bid/ask - how far back
- [ ] options: per-stock implied-volatility history fields for `AAPL.O`; one expired monthly option's daily history

## Open decisions

1. ~~A baseline git commit before P0~~ - taken 2026-11-07 (`4e16099`).
2. ~~Dependency groups `ml` and `deep`~~ - added 2026-11-08.
3. Whether P8 is wanted (new tables and vendors). P9 was chosen on 2026-11-08 (10-K text only, no RAG).
4. ~~Whether P10 is in scope~~ - synthetic data chosen 2026-11-08, minute bars not.
5. Which of A1-A7 to build, and in what order (suggested: A1, A2, A3, A4, then A5 after the probe; A6 and A7 last).
6. Free accounts: a FRED API key (A2, P8), an Alpaca account (A6) - both keys go in the keyring.
7. Running the LSEG probe (start Workspace; ~20 requests of the 10,000 a day).

## Log

* 2026-10-30 - plan written; book repository read at `a4e1d50`; dependency resolution checked.
* 2026-10-30 - the seven other case studies scoped (A1-A7, X0). Free sources answering from this PC: Binance futures
  API, data.binance.vision, Bybit, IEX HIST, FRED's API host (key needed); FRED's web host times out. LSEG not
  probed: port 9000 was held by another program speaking a binary protocol, so Workspace was not reachable.
* 2026-10-30 - Azure is now the working database: `equity_research_local` copied with `eqr copy-database` (18 tables,
  18.4 million rows, every table's rows and checksum equal; raw zone 1,065 files equal); its cache seeded from the
  local one and synced. Phases run on `equity_research` from here; the regression baseline (rule 5) is taken there.
* 2026-11-06 - notebooks grouped into folders (`1_data_load`, `2_index_replication`, `3_investment_strategies`,
  `4_portfolio_analytics`) and numbered folder.position; this plan's notebooks become 5.1-5.9 in
  `5_machine_learning/` (they were 12-20, which clashed with the QGARP and FTSE notebooks).
* 2026-11-07 - notebook 5.1 `5_analysis/01_choosing_ml_algorithm` and `research/model_choice.py`: the book's
  eight learners (*Build Your Own AI Investor*) compared by out-of-sample rank IC and chosen walk-forward, registry
  family `ml_model_choice_lo`. Its IC and Newey-West t-statistic are the first pieces of P0's `evaluation.py`.
  Result (runs 130-139, 10 trials): no learner's mean rank IC over 2009-2023 differs from zero (best +0.003, t +0.5);
  portfolios 12.6-14.7 % a year against SPTM's 13.7 %, best deflated active Sharpe 28 % (KNN, never chosen). Tree
  fits follow row order and last-bit float noise between loads, so inputs are now sorted and rounded to float32
  before fitting (runs 120-129 predate both, 130-139 the rounding).
* 2026-11-07 - notebook 5.2 `5_analysis/02_choosing_ml_algorithm_v2` and `research/model_choice_v2.py` (family
  `ml_model_choice_v2_lo`, runs 150-159; 140-149 are its first execution, deflated by 1-10 trials): percentile target
  and features, settings for a weak signal (depth-2 histogram boosting, leaves of 500, 500 neighbours, elastic net
  tuned by blocked CV), 8 price / size / issuance / surprise / growth features. Rank IC now +0.04 to +0.06 (five with
  t 2.0-2.6) but portfolios unchanged (13.3-15.2 % a year against SPTM's 13.7 %, best deflated 46 % / 33 % over both
  families): a percentile target learns medians (avoid volatile shares) while a portfolio earns means. Open for a
  version 3: an excess-return target with trimmed extremes; a flat forecast should score IC 0 and hold the index
  (the elastic net's flat 2011-2014 held cash and froze the choice 2012-2016). `strategy.deflated_active` and
  `runs.score_metrics` added for studies spanning families.
* 2026-11-07 - notebook 5.3 `5_analysis/03_choosing_ml_algorithm_v3` and `research/model_choice_v3.py` (family
  `ml_model_choice_v3_lo`, runs 160-169): version 2 with a clipped (1 %/99 %) excess-return target, the choice by the
  top-30 % spread, a flat forecast scoring 0 and holding the largest companies. Worse: the elastic net was flat on
  all 192 month-ends, the other models' top spreads -1.4 % to +1.3 % (rank IC ~0), and the models learned to buy past
  losers - 2008-2009 decision dates carry 70 % of the squared target in the 2011 window (20 % in 2024), 2020 another
  19 %. The choice held the largest companies 2011-2021, then switched to the decision tree and lost 13.2 and 10.0
  points in 2023-2024. Thirty trials over three families, none deflated above 25 %: the line stops here; what is
  left (shorter horizons, purged CV, feature triage) is P0-P1.
  The folder `5_analysis` now holds notebook 5.1, so this plan's `5_machine_learning` numbering (5.1-5.9) clashes:
  to settle before P0's notebook is written.
* 2026-11-07 - P0 started. Numbering settled: `5_analysis` keeps 5.1-5.3 and this plan's notebooks become 6.1-6.9 in
  `notebooks/6_machine_learning/`. Open decision 1 taken: the baseline commit `4e16099` (the reference PDFs in `docs/`
  stay out of git), so runs from here record a code version.
* 2026-11-07 - P0 done. `store.py`, `labels.py`, `validation.py`, `evaluation.py`, `regression.py` and
  `registry.record_search`, with tests; `model_choice` now takes its IC and Newey-West t from `evaluation`. Regression
  baseline taken on commit `4e16099`, data snapshot 2026-11-02T09:38:38 (7 strategy families rerun, the risk model
  rebuilt: 188 numbers); measured again with P0's code: 0 differ. Notebook 6.1: three of nine known factors survive
  Benjamini-Hochberg at 10 % against next month's return (issuance t 3.85, cash-flow yield 2.98, quality 2.42);
  momentum and the earnings surprise are fast signals, low volatility, quality and issuance slow; a day's move covers
  2.5 round trips, 63 days' 20 - P1 starts from monthly labels and slow signals.
* 2026-11-07 - P1 built: `features.py` (16 price-volume features, the book's 18 ratios and 5 account extras, calendar),
  `model_features.py` (GARCH, HAR, fractional differencing, Kalman trend, break statistic, market regime) and
  notebook 6.2; features kept in the artifact store (kind `features`, 45 company features, 360,361 member-months).
  Triage against next month's return, 2005-2024, Benjamini-Hochberg at 10 % over 45 features: two discoveries, both
  existing factors - share-count change (t -3.85; the issuance factor, rank correlation -1.00) and ROCE (t +2.99;
  0.83 with quality). Nine near-misses at |t| 2.0-2.5 (profitability, volatility in five forms, sales growth, zero
  days). Valuation ratios and every momentum measure about zero at this horizon. P2's yardstick: an equal-weighted
  composite of issuance and ROCE.
  Regression check after P1: 188 numbers, 0 differ (`regression.py` now reads every registered run before the
  first rerun, as the serverless database paused mid-check once).
* 2026-11-08 - P2 built: `linear.py` (ridge, lasso, elastic net, logistic; nested purged tuning; coefficient
  stability; split conformal; the composite) and notebook 6.3, 27 trials in `ml4t_linear_monthly`. Out of sample
  2015-2024 against next month's return: logistic IC +0.0236 (t 2.67), ridge +0.0139, elastic net +0.0087, lasso
  +0.0053; the composite of issuance and ROCE +0.0249 (t 3.18; flattered: chosen by a triage that saw the test years).
  No model beats it (logistic -0.0013, t -0.18), at four times its turnover. Conformal 90 % intervals cover 89 %, 30
  times wider than the predictions vary. A flat forecast now scores IC 0 in every tuning loop (`ic_flat_zero`).
  Dependency group `ml` added (lightgbm 4.7.0, optuna 5.0.0; the core libraries unmoved).
* 2026-11-08 - P7 built: `causal.py` (DML with cross-fitting on purged date blocks, date-clustered errors; OLS
  comparisons; the factor zoo with Benjamini-Hochberg) and notebook 6.9. Momentum 12-1, 2005-2024: no effect, alone
  (-0.09 %, t -0.22), with linear controls or by DML (-0.05 %, t -0.15), in either half. Factor zoo of the 11
  triage survivors and near-misses, each against all the others: only zero-return days (t -3.21), the GARCH forecast
  (-2.84) and idiosyncratic volatility (-2.60) keep an effect; share-count change (t -1.54) and ROCE (-0.03) do not -
  the low-volatility / illiquidity effect is the common thread of P1-P7.
* 2026-11-08 - P5 built: `latent.py` (managed portfolios; PCA, RP-PCA, IPCA by ALS; observed factors; total R2,
  pricing error, SDF Sharpe, walk-forward 2015-2024), `autoencoder.py` (CA1 on torch) and notebook 6.7. Dependency
  group `deep` added (torch 2.15.0 CPU; the core libraries unmoved). IPCA with 5 factors: total R2 98.2 %, pricing
  error 0.011 % a month, SDF Sharpe 1.29 - better on all three than BFRE's 24 factors (94.5 %, 0.020 %, 0.30) or its
  market and styles (94.5 %, 0.021 %, 0.89). RP-PCA prices better than PCA. The autoencoder (one seed, CPU-sized)
  stalled near one factor's worth (92.0 %). The market carries 92 % of the managed portfolios' second moment.
* 2026-11-08 - P3 built: `boosting.py` (LightGBM grid of 3 losses x 3 leaf profiles tuned per fold; Optuna, 40 trials
  on 2005-2014; TreeSHAP; conformal) and notebook 6.4, 51 trials in `ml4t_gbm_monthly`. Huber with large trees won 9
  folds of 10; LambdaRank's validation IC was negative. Out of sample 2015-2024: GBM grid IC +0.0286 (t 2.74), Optuna
  +0.0265 (t 2.56) - the best so far, but not significantly above the composite (+0.0037, t 0.29) or logistic
  (+0.0050, t 0.40); only above 3.4's learned value (+0.0041; difference t 2.15). Top-minus-bottom fifth +0.62-0.73 %
  a month (twice the composite's) at 53-56 % turnover a month. SHAP: zero-return days, the fractionally differenced
  price and the GARCH forecast lead in every fold. Validation +0.044 against +0.029 out of sample.
* 2026-11-08 - P6 built: `deep.py` (weekly NLinear, LSTM, TSMixer on 52-week rank sequences; an 8-network MLP
  ensemble on the features; seeded, deterministic, targets scaled for training) and notebook 6.8, 4 architectures in
  `ml4t_deep`. Weekly ICs: NLinear +0.0071, LSTM +0.0060, TSMixer +0.0014 - all below last week's reversal (+0.0135,
  t 2.04). MLP ensemble monthly IC +0.0219 (t 2.36): above ridge, lasso, elastic net (t 2.05-2.91), level with
  logistic, GBM and the composite. Across P2, P3 and P6 (82 registered configurations) no model's IC is
  distinguishable from the composite's.
* 2026-11-08 - P4 built: `allocation.py` (decile and top-K books; equal, inverse volatility, HRP, mean-variance on
  the risk model; volatility target, drawdown control, half Kelly), `ml_strategy.py` (backtests, registry family
  `ml4t_strategy_monthly`, 44 strategies, runs 275-318) and `assessment.py` (cost grids, break-even, VaR/CVaR, IC
  drift, transfer coefficient, White's Reality Check raw and studentised, Rademacher bounds); notebooks 6.5 and 6.6.
  Every long-only equal-weighted book trails SPTM (-0.8 to -3.6 % a year); the best long-only is logistic
  mean-variance (+1.4 %, IR 0.28, deflated 9 %); overlays all cost return. Best overall: the composite long-short
  mean-variance (Sharpe 0.70, deflated 49 %, break-even at 3.0x costs). Boosting ranks best before costs (Sharpe
  0.82 equal-weighted long-short) and loses it to turnover (0.20). Effective breadth 8-4,838 bets a year against
  18,048 nominal. Reality Check: p 0.049 on raw means (dominated by the most volatile book), 0.129 studentised. The
  Rademacher bound is vacuous at 119 months. The hold-out step is left for a decision.
* 2026-11-08 - regression check after P2-P7: 188 numbers measured on the baseline's data snapshot, 0 differ. Gate:
  ruff and mypy clean, 383 unit and 31 integration tests pass. Nothing since `4e16099` is committed (the owner's
  choice); open for the owner: the hold-out step, P8-P10, A1-A7.
* 2026-11-08 - the owner chose P9 and P10 with: 10-K text only (no 10-Q or 8-K text; 8-K item codes come from the
  filing index), no RAG or knowledge graphs, no minute bars.
* 2026-11-08 - P10 built: `synthetic.py` (stationary block bootstrap of whole days; a factor model of 5 principal
  components with GARCH(1,1)-t factors and Student-t residuals; a two-state regime-switching factor model fitted by
  Baum-Welch; stylized facts; a discriminator; books held on synthetic paths) and notebook 6.11. Realism: a
  classifier tells bootstrap months from real ones 49 % of the time (chance), factor GARCH 78 %, regime 68 %; all
  match the stocks' co-movement (0.28-0.29 against 0.30), the factor models miss single-stock volatility clustering.
  Four 6.5 books on 200 paths per generator, which keep their exposures but break forecast-to-month: the real decade
  beat 94-100 % of the factor models' paths for all four (96-100 % of bootstrap paths for the mean-variance long-short
  books; 72 % for the composite's equal-weighted one, 69 % for the long-only logistic book). Without selection the
  long-short books lose money on 89-100 % of factor paths (costs and borrowing 3.4-11 % a year). A year of crisis or
  50 % more volatility deepens the long-only book's worst-twentieth drawdown to 64 % / 84 % (the index: 52 % / 73 %).
* 2026-11-08 - the GPU (RTX 4060 Laptop, 8 GB) works after the owner repaired its driver; the owner asked for it to be
  used wherever it helps. torch is now its CUDA build on Windows (2.15.0+cu132 from PyTorch's index, `[tool.uv.sources]`;
  other platforms and every other package unchanged). `deep.train`/`predict`/`walk_forward_*`, `autoencoder.fit` and
  FinBERT (`eqr load-sec-text --tone --tone-device cuda`) take a `device` keyword, default `cpu`, so cached CPU results
  stand; `deep.best_device()` picks the GPU. Measured: FinBERT 0.08 s a report against 0.84 s, LSTM training
  hundreds of times faster, the MLP ensemble twice, TSMixer the same, the autoencoder slower (a month of 1,500 stocks
  is too small for a GPU) - it stays on the CPU. GPU and CPU predictions agree (correlation 1.000000; FinBERT within
  5e-7).
* 2026-11-08 - P9 loaded and assessed (notebook 6.10). `eqr load-sec-text`: 829,161 filings listed, 47,540 annual
  reports landed and measured (3,013 companies); loads hit two EDGAR facts made code - a combined report is one
  filing under two CIKs (fetched and landed once, measured for both companies), and Azure drops hour-long sessions
  (a landing chunk is retried). Results: 11 of 33 text features are discoveries at a 10 % false-discovery rate, but
  with the 45 features of 6.2 controlled for only the count of acquisition-or-disposal 8-Ks (item 2.01) keeps an
  effect (double ML t -3.58): such companies trailed the rest by 0.36 % a month (t -3.91), in both halves. Lazy
  Prices does not repeat (quintile spreads |t| < 0.9); the whole report's tone does nothing (IC t +0.07); most text
  discoveries repeat balance-sheet strength, profitability or size. Caution: no asset-growth feature in 6.2. The
  strategy candidates are unchanged; adding the event count would be a new trial, tested before the hold-out.
* 2026-11-08 - Azure's free monthly compute ran out during the loads (error 42119): the database is paused until
  2026-12-01, the owner's choice (no hurry). Pending then: the FinBERT write, `eqr checks` / `eqr audit`, the data
  refresh, the hold-out. Research continues from the local cache meanwhile.
* 2026-11-08 - the owner asked how to run a chosen strategy as a hybrid: a systematic monthly book that a person
  reviews (vetoes only, each with a reason), tracked against the unvetoed shadow book. Designed in ARCHITECTURE.md
  section 7 (schema `book`, `eqr rebalance` / `veto` / `approve`, paper before live); not built - B0-B3 wait for the
  owner's review and for Azure (the migration), paper starts after the hold-out is opened.
* 2026-11-13 - the owner chose not to wait for Azure: `equity_research_local` was rebuilt from the vendors (US and
  London, two days of LSEG allowance), the 10-K text loaded with FinBERT (notebook 1.3, new), and notebooks 6.2 and
  6.10 rerun on it. 6.2: the same two discoveries (share-count change t -3.87, ROCE t +2.89), numbers moved by
  thousandths. 6.10: twelve text discoveries, still only the acquisition-or-disposal 8-K count survives the controls
  (double ML t -3.51); FinBERT's tone points the expected way but is no discovery (IC +0.0057, t +1.60). The local
  registry and artifact store start afresh: 6.1 and 6.3-6.9, 6.11 still to rerun on the rebuilt data.
* 2026-11-14 - the repository restructured into `research_core` (shared), `equity_research` and an empty
  `fixed_income_research`; notebooks moved to `notebooks/equity/`. Nothing in a strategy or a result changed.
  **Numbers in this log above this entry are the old ones**: this plan's notebooks were 6.1-6.11 and are now
  5.1-5.11 (`notebooks/equity/5_machine_learning/`); the model-choice study's notebooks 5.1-5.3 (`5_analysis`)
  and its modules `model_choice*.py` were removed (their runs stay in the registry, families
  `ml_model_choice*`); the strategies are now 3.1 long-only, 3.2 extra signals, 3.3 learned value (was 3.4),
  3.4 long-short (was 3.1); section 4 is 4.1 dashboard, 4.2 risk model, 4.3 screen. This plan's modules are in
  `equity_research/ml/`, except `labels`, `evaluation` and `validation` (`research_core/analytics/signals/`),
  `allocation` (`research_core/engine/`) and the artifact store (`research_core/data/artifact_store.py`, bound
  to the returns method in `equity_research/data/artifacts.py`). Compatibility rule 1 holds for signatures and
  defaults; only the import paths moved. Rule 8's keyword exists now: `periods_per_year` on `performance`,
  `run_backtest`, `BacktestResult.metrics`, `tracking` and `yearly_tracking_error`, defaulting to 252.
* 2026-11-17 - the multi-asset database layout (one Alembic history from `0001`; schemas `core`, `ingest`, `ops`,
  `research`, `equity`, `fixed_income`) built as `quant_research_local` from the vendors, **with history from
  2000** (prices from 1998 for momentum) instead of 2005; the run registry and artifact store start afresh again
  (registry from run 1, so the deflated ratios count only the trials since). Every equity notebook, 2.1-5.11,
  rerun on it without an error. The two discoveries of 5.2 are unchanged (share-count change t -3.53, ROCE
  t +2.94); seven near-misses where there were nine (EBIT over assets and the GARCH forecast fell just past
  p 0.05), so 5.9's factor zoo tests nine and two keep an effect (zero-return days, idiosyncratic volatility).
  5.4: grid and Optuna now rank alike out of sample (+0.0276, +0.0274); the lead over learned value is t 1.86.
  5.5: the best deflated ratio 43 % (composite long-short mean-variance). 5.6: White's Reality Check p 0.116 (was
  0.049), boosting's IC decline significant (t -2.35). 5.8: NLinear +0.0121 (t 2.12), still below reversal.
  5.10: eleven text discoveries (restatements dropped); acquisition-or-disposal count still the only survivor
  (t -3.50). 3.3: the learned score now trails SPTM (-0.5 % a year; it led by 0.4 % on 14 November). Each
  notebook's findings were rewritten to its new outputs.
* 2026-11-20 - a review of the strategy notebooks 3.1-3.6, implemented on branch `strategy_improvements`
  (`docs/STRATEGY_REVIEW.md`). Two data errors fixed outright: LSEG's revision counts are a snapshot whatever
  `SDate` asks (99.6 % equal each listing's latest), so research reads only counts loaded within 31 days of their
  date and the loader asks no older month end; EBIT / EV is missing where EV <= 0. Everything else is a new setting
  whose default is the old behaviour and which is registered only when set (compatibility rule 1: old hashes kept):
  `mix`, `within_sectors`, `construction="tilted"` (`research_core/engine/tilt.py`: benchmark plus Grinold-Kahn
  tilts on 4.2's risk model, beta-, sector- and size-neutral), `max_weight`, `benchmark` (own cap-weighted index),
  `drift_band`, `cost_multiplier`, `train_from`; factors `earnings_value`, `consensus_revisions` (4 months),
  `ml_value_rank`. Notebooks 3.1-3.6 rerun; 3.7 (family `revised_lo`) specified and run once: **does not pass**
  (S&P 1500 +0.41 % a year against its own index, t 0.73, negative at twice the costs; FTSE All-Share unchanged
  +0.99 %, t 2.24). 3.4's rebuilt long-short book (`revised_ls`) does not pass (Sharpe 0.27 against 0.35). Without
  the counts' look-ahead QGARP returns 0.6 points a year less on both universes. The regression check now covers
  `revised_lo` and `revised_ls`; its baseline is to be retaken once the branch is accepted. A third data fix
  came from rerunning 3.6: the same configuration gave slightly different results (up to 0.007 points a year)
  because LSEG back-filled some early actuals with one announcement time and `Estimates.actual` took whichever year
  the cache returned last; the latest year now wins (stable sort, tested), and 3.5 and 3.6 were run again.
* 2026-11-24 - QGARP's registry family renamed `qgarp` (modules `strategies/qgarp.py`, `qgarp_estimates.py`; rule
  set `gems-2026-11` filed under it). Its 46 runs in `quant_research_local` were moved with their config hashes
  recomputed: the same 14 configurations, so its trial count and deflated ratios are unchanged (rule 2 is about
  reusing a name, which this is not). The regression baseline is keyed by the family list, so it is to be retaken -
  as it already was for `revised_lo` and `revised_ls`.
