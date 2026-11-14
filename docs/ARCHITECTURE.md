# EquityResearch - architecture

A vendor-neutral platform for point-in-time equity research and backtesting, built for one to a few researchers.
This document records the decisions and the reasons for them; the code follows it.

## 1. Principles

1. **Vendor neutrality.** Research code never sees a vendor. Vendors are adapters that land raw data and normalise it
   into one canonical model; swapping or adding a vendor touches only `ingest/vendors/<vendor>/`.
2. **Never fetch twice.** Every vendor response is landed unchanged in the raw zone (immutable, with a batch record)
   before it is normalised. Fixing a mapping or a formula re-normalises from the raw zone - it never re-downloads.
3. **Store facts, derive the rest.** The database holds what vendors report: unadjusted prices, corporate actions,
   accounting items as reported, index membership events. Adjusted prices, total returns, TTM figures and ratios are
   computed when read, at the decision date - one formula, applied to any vendor.
4. **Two clocks on every fact.** `published_at` - when the world could know it (filing / announcement / trade date);
   `loaded_at` - when we knew it. A backtest decision at time *t* sees only facts published before *t*; a replay
   "as known on day *k*" also requires `loaded_at <= k`.
5. **Company and listing are different things.** Accounts belong to an **issuer** (one CIK, one set of statements);
   prices belong to a **listing** (GOOG and GOOGL, BRK.A and BRK.B). Identifiers (ticker, RIC, CUSIP, ISIN, CIK)
   live in one history table with validity dates - a reused ticker is two rows, never a merged company.
6. **Functional core, imperative shell.** Normalisation, identity matching, panel building, factors, weighting and
   backtesting are pure functions over DataFrames, unit-tested without a database or network. I/O lives at the edges
   (vendor clients, raw store, repositories, cache).
7. **Gates between steps.** Each pipeline step ends with data-quality checks (coverage, freshness, plausibility); a
   failed gate stops everything downstream instead of letting it run on partial data.
8. **Research discipline is enforced, not advised.** Every backtest run is registered (config hash, code version,
   data snapshot, results), so the number of trials is known; a hold-out date fence must be opened explicitly.
9. **Typed, linted, tested.** `mypy --strict`, `ruff`, `pytest`; a change is done when all three pass.

## 2. Storage

| Layer | Technology | Holds | Why |
|---|---|---|---|
| **System of record** | Azure SQL Database (serverless, free offer, 32 GB) | reference data, canonical facts, ingest batches, ops, research run registry, later live trading | shared by several people from anywhere, transactional, roles per person, no server to run |
| **Raw zone** | files: Parquet (tabular) or the original JSON / zip, under `data/<database>/raw/<vendor>/<dataset>/` | every vendor response, unchanged, with the request that produced it | immutable source for re-normalisation; a load resumes by reading back requests already answered |
| **Research cache** | Parquet + DuckDB per researcher, under `data/<database>/cache/` | a local, incrementally synced copy of the canonical tables | backtests read millions of rows; local columnar reads are fast and cost no cloud compute |

The raw zone and the cache belong to one database (the folder is named after it), so two databases never mix
files. A database is built and verified locally (SQL Server Express), then copied to Azure table by table with
every id unchanged (`eqr copy-database`, which checks rows and a checksum per table on both sides).

Free-tier compute (100k vCore-seconds a month) is spent on loads and syncs, not on research reads. A load that
downloads for hours (the SEC's 10-Ks) stages its files and lands them in short write phases, each under the writer
lock, so the serverless database can pause in between.

Research results kept in the artifact store are keyed by the cache's **data version** - the latest `loaded_at` it has
synced. Filing text (`fund.document`, `fund.text_measure`) has a version of its own (`store.SEPARATE_TABLES`):
loading it does not make every price- and accounts-based result stale, and research that reads it keys its results
by both versions.

## 3. Data model (Azure SQL)

Schemas separate concerns and permissions (a researcher role reads `ref`, `market`, `fund`; writes `research`).

### `ref` - who is who
| Table | Key | Notes |
|---|---|---|
| `vendor` | `vendor_id` | the data sellers with fixed small ids (1 refinitiv / LSEG, 2 sec, 3 yahoo, 4 financedatabase, 5 norgate, 6 tiingo, 7 simfin, 8 sraf - Notre Dame's Loughran-McDonald dictionary, 9 repository - corrections this repository states itself), used by every other table and the same in every database |
| `issuer` | `issuer_id` | a company: name, `domicile_country` (where it is run - not where its shares trade, which is the listing's exchange), `incorporation_country` (where it is registered: UK stamp duty follows it) |
| `classification` | (`issuer_id`, `vendor_id`, `scheme`, `valid_from`) | a company's sector / industry group / industry / sub-industry per vendor and scheme (GICS), with the dates each held; a change closes the old row and opens a new one, so history builds from the day it was first recorded |
| `listing` | `listing_id` | one share line - the same security whatever it was called (every RIC it had is an identifier: a ticker change does not split its history): `issuer_id`, share class, exchange (MIC), currency, `instrument_type` (equity / etf / index / fund - a closed-end fund, a UK investment trust: an index member, but research leaves it out / fx - one unit of its currency, priced in US dollars: an exchange rate), `listed_on`, `delisted_on` |
| `identifier` | `identifier_id`; unique (`id_type`, `value`, `valid_from`, `market`) | (`entity_type` issuer / listing, `entity_id`, `id_type` TICKER / RIC / CUSIP / ISIN / CIK / SEDOL / FIGI / PERMID / YAHOO / NORGATE / SIMFIN (an issuer's), `value`, `valid_from`, `valid_to`, `market`, `vendor_id` that told us); insert-only - a code held by another entity is reported, never moved. A ticker names a share only among one country's exchanges (London's AAL is Anglo American, Nasdaq's American Airlines), so its `market` (`US`, `GB`, from its listing's exchange) is part of the key; other codes are the same everywhere (`''`). A vendor's tickers match only listings on that vendor's markets |
| `index_definition` | `index_id` | an index (our codes SP500 / SP400 / SP600, FTSE100 / FTSE250 / FTSESC): name, provider, currency, the listing carrying its level |
| `index_membership` | (`index_id`, `listing_id`, `start_date`) | point-in-time intervals; `end_date` exclusive; `vendor_id` |
| `universe`, `universe_component` | | a research universe = indices x segment (cost, borrow) and a benchmark; defined in `ingest/reference.py` |

### `market` - what traded
| Table | Key | Notes |
|---|---|---|
| `price_bar` | (`listing_id`, `vendor_id`, `trade_date`) | **unadjusted** open / high / low / close, volume, currency |
| `corporate_action` | (`listing_id`, `vendor_id`, `action_type`, `ex_date`) | cash dividend (amount), split (ratio), spin-off; `published_at` when known |
| `share_count` | (`listing_id`, `vendor_id`, `kind`, `as_of_date`) | shares per share class: `outstanding`, and `float` (the part that trades - what index weights use) |

### `fund` - what companies reported
| Table | Key | Notes |
|---|---|---|
| `filing` | (`issuer_id`, `vendor_id`, `filing_ref`) | each report: form, period end, fiscal year / period, `published_at` |
| `financial_fact` | (`issuer_id`, `vendor_id`, `item`, `period_type`, `period_end`, `published_at`) | canonical accounting items: `period_type` Q (3-month flow), YTD, FY (fiscal-year flow), I (balance at an instant). SEC facts are as first reported, each restatement a new row with a later `published_at`; LSEG facts are its quarterly figures dated by their first announcement; SimFin facts are its latest figures, dated by their first publication where the SEC's first report has the same value, else by when SimFin last re-dated them (usually a year later). Every vendor goes through the same ratio code, which is how they are compared |
| `estimate` | (`listing_id`, `vendor_id`, `measure`, `period`, `as_of_date`) | analysts' consensus (I/B/E/S, through LSEG) as it stood at each month end: mean, median, standard deviation, number of estimates and the fiscal year's end, for EPS, revenue, EBIT, dividends and cash flow per share and ROE, FY1-FY3. Per listing - a per-share forecast belongs to a share class; per-share figures in the shares of their own date (LSEG restates them for every later split; the loader undoes that with LSEG's splits), so a later split never rewrites them |
| `estimate_revision` | (`listing_id`, `vendor_id`, `measure`, `period`, `as_of_date`) | how many analysts raised and lowered their EPS estimate (FY1, FY2), at each month end, for that day's index members |
| `document` | (`issuer_id`, `vendor_id`, `filing_ref`) | every 10-K, 10-Q and 8-K (and amendment) a company filed since 2003, from its EDGAR index: form, filing date, the period or event date, `published_at` - the instant EDGAR accepted it, when it became public - and an 8-K's item codes (2.02 results, 5.02 an officer leaving, 4.02 a restatement...) |
| `text_measure` | (`issuer_id`, `vendor_id`, `filing_ref`, `section`, `measure`) | what research reads of an annual report's text, which stays in the raw zone (gzip-compressed EDGAR HTML: forty thousand reports would fill the database): word counts by Loughran-McDonald category for the whole report, its MD&A and its risk factors; cosine and Jaccard similarity to the company's previous annual report; FinBERT's tone of sampled MD&A sentences. Each row names its `method` (dictionary edition, model), so a new one re-measures from the raw zone |
| `estimate_actual` | (`listing_id`, `vendor_id`, `measure`, `period_end`) | each fiscal year's reported figure as the analysts' service counts it (comparable to the estimates) and when it was reported |

### `ingest`, `ops`, `research`
| Table | Notes |
|---|---|
| `ingest.batch` | one row per landed raw file: `vendor_id`, dataset, request (`source`), path, checksum, rows, `fetched_at`, status |
| `ops.job_run` | one row per pipeline run / step with status, rows, detail; an application lock prevents overlapping writers |
| `ops.check_result` | the outcome of every quality gate |
| `research.run` | every backtest: config, config hash, code version, data snapshot (max `loaded_at`), metrics, whether it touched the hold-out |
| `research.rule_set` | a named set of rules one strategy applies together (`gems-2026-11`: Barings' cleaning limits): its strategy, description, source and the checksum of its rules |
| `research.strategy_rule` | one rule of a set: its subject (a factor, a composite, `*` for the strategy), its type (`lower_limit`, `upper_limit`, `exclude_financials`, `invert`, `weight`, `threshold`, `setting`) and its value |

Every canonical row carries `batch_id` (provenance back to the raw file) and `loaded_at`.

**Strategy rules** are shared through the database and owned by the repository. Each rule set is a file beside
`research/rules.py`, versioned and tested with the code; `eqr load-rules` writes it to the two tables, and research
reads it from there through the cache - so everyone, and any other tool, applies the same rules. A set is **never
changed once loaded**: a run's registered settings name the set it used, and the loader refuses a file whose rules
differ from the loaded set of the same code (by checksum). A changed rule is a new file under a new code. The tables
are copied whole at each cache sync and do not move the data version.

### Choosing vendors, adding one

* **One place chooses.** `settings.vendor_preferences()` says which vendor supplies prices, benchmarks, memberships,
  accounts and sectors, in priority order, and which sources a build or refresh runs (each overridable in the
  environment, e.g. `EQR_PRICE_VENDORS`, `EQR_SOURCES`). Research and notebooks read it; none names a vendor.
* **Two vendors never mix silently.** Prices take the first preferred vendor with data per listing and day; each
  index's membership comes from the first preferred vendor that has that index; sectors from the first that classifies
  the company; a company's accounts, on each decision date, from the first preferred vendor that knows its trailing
  net income then (default SEC, then LSEG). Each vendor's figures are built from its own facts, and all of a company's
  figures on one date come from one vendor - so LSEG answers before the SEC's XBRL accounts begin (2009-2011) and
  through the year the SEC's first filings need to build four quarters, and one company's quarters never interleave
  two vendors' figures. (Cutting LSEG at the SEC's first *period end* left most companies without accounts in
  2008-2011: the SEC's first filings report the year before as comparatives, published only years later.) LSEG's
  figures are its latest corrected ones dated by first announcement: for those years a later correction can show
  early.
* **A new vendor** is an adapter (`ingest/vendors/<vendor>/`), a row in `ref.vendor` (a migration), an entry in
  `pipelines/sources.py`, and a word in the preferences. Its securities attach to our listings through
  `identity.match_listings` (ISIN, then CUSIP, then ticker on the day; a code two listings hold is flagged, never
  guessed); unmatched ones become new listings.
* **Replacing LSEG with Norgate** is configuration only: `EQR_SOURCES=norgate,yahoo,financedatabase,sec`,
  `EQR_PRICE_VENDORS=norgate`, `EQR_MEMBER_VENDORS=norgate`, `EQR_CLASSIFICATION_VENDORS=norgate,financedatabase` (or
  the same words as the defaults in `settings.py`). Index levels are found by any vendor's code (`.SPX`, `$SPX`,
  `^GSPC`), so the index definitions and Yahoo's benchmarks attach to whichever listing exists. Norgate has no
  statements: accounts stay with the SEC, or come from SimFin (`EQR_FUNDAMENTALS_VENDOR=simfin`, `simfin` in
  `EQR_SOURCES`).
* **Prices from Tiingo** are configuration too: `tiingo` in `EQR_SOURCES` after the source that creates the listings,
  and first in `EQR_PRICE_VENDORS` to prefer it. Tiingo knows only tickers, so it creates nothing and each listing's
  prices are checked against another vendor's before they are written.
* **API keys** (Tiingo, SimFin) live in the keyring (`equity_research` / `tiingo_api_key`, `simfin_api_key`) or
  `EQR_TIINGO_API_KEY` / `EQR_SIMFIN_API_KEY`; they travel in a request header, never in a URL, so no landed
  batch's `source` holds one.
* **Market values from each share line's own count.** A listing's value is its price times its own outstanding
  count (``market.share_count``, dated, never restated); a company's value is the sum over its listings. A count from
  the accounts is used only where a listing has none: LSEG's company counts carry today's share basis back in time
  (Alphabet's 2015 count in post-2022 shares, twenty times too many), so a value built from them is wrong for any
  company that has split since.
* **One currency per universe.** A universe is in its indices' currency: the S&P universes in US dollars (LSEG statements
  are requested converted, `Curn=USD`), the FTSE ones in pounds (`Curn=GBP`). London prices are stored as quoted, in
  pence (`GBX` - never `GBp`, which a case-insensitive database reads as pounds), and read as pounds;
  research refuses prices in another currency and leaves out accounting facts in one, saying how many.

### Rules every load keeps (and the gates that check them)

* **One spell per membership.** Spells of one listing in one index never overlap: two codes of one share line (a RIC
  before and after a rename) each bring a spell, and once both name the same listing they are merged
  (`ingest/membership.py`). Gate `overlapping_membership_spells` fails on any.
* **One index of a market at a time.** A market's indices never share a member (S&P 500 / 400 / 600, FTSE 100 / 250 /
  SmallCap), but a replay can leave a listing in two: a leaver with no join event is dated a member since the window
  opened, a joiner with no leave event a member until today - LSEG's London events miss many moves before 2010 (3i in
  both the FTSE 100 and the SmallCap from 2005 to 2009). `ingest/membership.disjoint` keeps what the events and today's
  member lists say and moves the dates the replay inferred to fit them; a move neither event dates (Computacenter, from
  the FTSE 250 to the 100 some time after 2009) is put halfway, and two contradicting events go to the larger index.
  Gate `members_of_two_indices` lists any left; research's `indices.segments` puts each listing-day in one index too.
* **No member before it traded.** A vendor replaying today's members backwards dates a member with no join event to
  the start of its window (LSEG misses some joins - Arcosa's 2018 spin-off). A spell never starts before the
  listing's first price; gate `members_before_first_trade` lists any left (a listing priced after the replay).
* **No fact about the future.** A financial fact whose period ends after it was published is a filer's tagging slip
  (debt "at 2041-03-15" in a 2011 10-K) and is dropped; gate `facts_published_before_period_end` fails on any.
* **A bar is consistent or partial.** A close outside its own high / low (LSEG's bars of 2006-10-24 mix two days)
  keeps the close and loses the range; gate `bars_outside_their_range`.
* **Every row names its batch** - the raw file it came from, each row its own when a load read several (one
  instruments batch per bucket of RICs). Lists we keep ourselves (the Yahoo benchmarks, Norgate's index levels) are
  landed like a vendor answer first. Gate `rows_without_provenance`; every batch read is marked `normalized`.
* **Every equity has a company** - by CUSIP, else by its share's ISIN, else (LSEG sends neither) by its own RIC until
  the SEC's CIK joins it to the right one; gate `equity_listings_without_issuer`.
* **A split is what the price says.** Capital changes on one day act together (their factors multiply: Motorola's
  reverse split and spin-off of 2011-01-04); a split the unadjusted price does not confirm within a factor of 2 on its
  day or the next is left out (a phantom "100-for-1" would multiply a day's return a hundredfold) - unless the shares
  outstanding confirm it: they move by the split's inverse (within 10%) from the last monthly count before its day to
  the first after. The count also finds a split LSEG lacks or dates weeks off, at a jump of 2.5x no action explains,
  when its ratio is a whole exchange (n-for-1, 1-for-n, 3-for-2...): Capital & Regional's 1-for-10 of 2020, Grainger's
  5-for-1 of 2005. A crash that met an issue of shares matches the count too but no whole exchange (National City,
  2008), so it stays a crash. A build replays the actions after the share counts (no requests). Gate
  `unexplained_price_jumps` lists the moves no action explains.
* **A spin-off is reinvested, not lost.** A capital change near 1 that lowers the price (a spin-off, a rights issue:
  stored as `spinoff` with ratio > 1) adjusts prices and returns like a split - the spun-off value is treated as
  reinvested in the parent - but never share counts. Like a split, it is what the price says: placed on the vendor's
  date or one of the two trading days after, whichever the price drops on, and left out when no drop within half its
  size confirms it (LSEG dates many US spin-offs a day early, some London ones days). Ratios below 1 (consolidations
  that come with a special payout) are not applied: the price does not rise by them. The method has a version
  (`panels.RETURNS_VERSION`) in every research artifact's key, so a change to it rebuilds cached results. Gate
  `suspect_halvings` lists the halvings left: a new share class handed out or a missed split.
* **What no vendor recorded, the repository states.** A corporate action every vendor lacks - a new share class
  handed to holders (Alphabet's class C of 2014, Under Armour's of 2016), a split LSEG never had (C.H. Robinson's
  2-for-1 of 2005) - is a row of `ingest/corrections/corporate_actions.csv`, named by RIC and saying what happened.
  `eqr load-corrections` (source `repository`, last in a build or refresh) lands the file in the raw zone and writes it
  to `corporate_action` under vendor 9 `repository`; a distribution's ratio is the holder's value on its ex-date over
  the parent's price, from both lines' closes. Research takes a correction only where the prices' vendor has no split
  or spin-off within 5 days of it, so a vendor that records the event later does not count it twice. A correction
  comes from evidence (`suspect_halvings`, the share counts), never from a rule guessed from prices alone: a new share
  line or a halving is also what a collapse with new shares looks like (Guaranty Financial, 2008).
* **A figure is not known before its period ends** - for any vendor: an LSEG announcement dated before its quarter
  ended belongs to another quarter (LSEG lines announcement dates up by row), so the figure is dated conservatively.
* **A leaver the vendor missed still leaves.** An open membership of a listing that stopped trading ends the day after
  its last trade (Norgate's daily series stop at 1; LSEG's event list misses some S&P 600 leavers).
* **The research cache only gains rows.** A repair that deletes rows in the database (re-deriving facts from the raw
  zone) must be followed by `eqr sync-cache --rebuild <table>`, or the cache keeps the deleted rows.

## 4. Code layout

```
src/equity_research/
  settings.py          configuration (environment variables + keyring), no secrets in code
  db/                  engine, schema (SQLAlchemy Core), Alembic migrations, repositories
  raw/                 the raw zone: land a vendor response, register its batch, read it back
  ingest/
    vendors/<vendor>/  fetch + normalise for one vendor: refinitiv (LSEG, live), norgate, sec (XBRL facts; filing
                       indexes and the text of annual reports: documents.py, text.py), yahoo,
                       financedatabase (open source: codes, domicile, sector labels), tiingo (prices),
                       simfin (statements), sraf (the Loughran-McDonald dictionary)
    countries.py       country names as vendors write them -> ISO codes
    identity.py        vendor identifiers -> issuer / listing (matching rules, conflicts, merges)
    membership.py      membership spells made consistent: overlaps merged, none before the first trade
    reference.py       our indices and universes (codes, costs, benchmarks)
    classification.py  sector history: record what a vendor says today, close what changed
    corrections/       corporate actions no vendor recorded (corporate_actions.csv), stated by the repository
  quality/             data-quality checks used as gates; the audit (architecture rules, fitness for research)
  pipelines/           sources (the list a build / refresh runs), load_lseg (+ probe), load_norgate (+ probe),
                       load_benchmarks, load_financedatabase, load_sec, load_tiingo (+ probe),
                       load_simfin (+ probe), load_estimates (+ probe), load_sec_text (filing indexes, 10-K
                       text measures, FinBERT tone: staged downloads, short write phases), load_rules (the
                       repository's strategy rule sets into the shared tables, each loaded once),
                       load_corrections (the repository's corporate-action corrections),
                       refresh (with the gates), copy_database; the CLI
  research/
    cache.py           sync canonical tables to the local Parquet / DuckDB cache
    reads.py           the cache reads research loaders share (companies, accounts, actions, share counts) and
                       the first preferred vendor's row - each written once, so a fix reaches every loader
    panels.py          point-in-time panels: prices (raw, adjusted for splits and spin-offs, total return), membership, accounts as of t
    accounting.py      TTM figures, earnings surprises and items as known on a date (vendor-neutral; implausible
                       counts dropped)
    ratios.py          ratios at decision dates; compare.py checks two vendors' ratios against each other
    factors.py         momentum (plain and residual), value, quality, low volatility, net share issuance, cash-flow
                       yield, accruals, magic formula
    ml.py              a learned value score: the book's ratios -> next year's return, fitted walk-forward
    model_choice.py    the book's eight learners, fixed in advance, compared by out-of-sample rank IC and chosen
                       walk-forward each January (predictions stored under data/<db>/research/ml_candidates/)
    model_choice_v2.py the same choice with the learning task repaired: percentile target and features, settings
                       for a weak signal, price, size, issuance, surprise and growth features
    model_choice_v3.py version 2 learning and judged by what a portfolio earns: clipped excess-return target, top-30 %
                       spread yardstick, a flat forecast scoring 0 and holding the largest companies
    garp.py            the S&P 500 GARP strategy rebuilt from its published rules (growth, then quality & value)
    barings_garp.py    Barings' GEMS factors and QGARP composite rebuilt from their production code, and a QGARP book
    barings_estimates.py  its I/B/E/S factors (NTM / LTM, forward yields, expected growth, revisions); qgarp_math.py
    arima_garch.py     a daily S&P 500 timing signal: ARIMA chosen by AIC, then GJR-GARCH on its residuals, refitted each
                       day (each day's fit stored under data/<db>/research/arima_garch/)
    dashboard.py       a factor dashboard's numbers: period returns, risk, overlap, active share, exposures
    portfolio.py       weights: quantile, buffered, by market value, beta-neutral
    backtest.py        engine, costs and borrow, metrics; strategy.py: one config -> a registered run
    indices.py         rebuild an index (by market value or equally) and compare it with the real one
    screen.py          today's ranking of a factor mix or any registered strategy, with the numbers behind it
    classification.py  the sector a company had on a day
    registry.py        research-run registry, deflated Sharpe and the hold-out fence; record_search registers every
                       point of a search as a trial (stage: training / backtest / holdout)
    rules.py           strategies' rule sets: the files they come from, their checks and checksum, and reading
                       them from the shared tables through the cache
    runs.py            a registered run rebuilt from its config: its scores and book on a day (no backtest, no
                       trial), or its backtest again without registering it; each score's newest run in a family
    store.py           the artifact store: Parquet under data/<db>/research/<kind>/<name>-<hash>.parquet, keyed by
                       config and data snapshot, with a sidecar of config, snapshot and code version
    labels.py          forward returns (1, 5, 21 days; to the next decision date), triple barrier, trend scanning
    validation.py      walk-forward folds purged of overlapping labels; purged, embargoed k-fold for tuning
    evaluation.py      IC (rank, Pearson) and Newey-West t, IC decay, quantile returns, turnover, rank
                       autocorrelation, Benjamini-Hochberg, the triage ledger, moves against costs
    regression.py      the regression check: existing strategies rerun and the risk model rebuilt, against a baseline
    features.py        company features on decision dates: momentum and reversal, risk, liquidity (Amihud, Roll
                       spread, zero days), the book's ratios and account extras, calendar; daily traded value
    model_features.py  features from fitted models, walk-forward: GARCH(1,1), HAR, fractional differencing, Kalman
                       trend, a break statistic, a two-state market regime
    linear.py          the linear baseline: ridge, lasso, elastic net, logistic, tuned by purged k-fold inside each
                       walk-forward fold; coefficient stability, split-conformal intervals, the composite yardstick
    boosting.py        LightGBM (optional group `ml`): a loss x leaf-profile grid tuned in each fold, an Optuna
                       search on the years before the first test, TreeSHAP importance, conformal intervals
    allocation.py      predictions to books: decile and top-K entry; equal, inverse-volatility, HRP and mean-variance
                       (risk model) weights; volatility-target, drawdown and Kelly overlays
    ml_strategy.py     a P4 strategy backtested and measured (long-only against the index, long-short against cash)
    assessment.py      cost grids and break-even, historical VaR / CVaR, IC drift, transfer coefficient and the
                       Fundamental Law, White's Reality Check, Rademacher bounds
    latent.py          latent factors on characteristic-managed portfolios: PCA, RP-PCA, IPCA, observed (BFRE)
                       factors; out-of-sample total R2, pricing errors, SDF Sharpe
    autoencoder.py     a conditional autoencoder (Gu, Kelly and Xiu) on torch (optional group `deep`)
    deep.py            weekly sequence models (NLinear, LSTM, TSMixer) and an MLP ensemble, seeded and deterministic
    causal.py          double machine learning with cross-fitting: a characteristic's effect, the factor-zoo check
    text_features.py   features from filing text on decision dates: dictionary tone, change since last year's 10-K
                       ("Lazy Prices"), FinBERT tone, 8-K events in the last quarter
    synthetic.py       synthetic markets (block bootstrap, factor GARCH-t, regime-switching factors), their
                       stylized facts and a discriminator score; real books run on synthetic paths
    risk_styles.py     a factor risk model's substyles (BlackRock's BFRE): size, volatility, momentum, liquidity,
                       value, yields, profitability, growth, leverage, mid-cap; risk_data.py reads its inputs
    risk_model.py      exposures, daily factor-return regressions, factor covariance and specific risk (BFRE 1.0
                       and 2.0: gamma scaling, structural specific risk)
    portfolio_risk.py  a portfolio's risk and its decomposition, analytical VaR, bias / Q / Kupiec tests
  charts.py            the notebooks' charts, one consistent style
tests/                 unit tests (pure, no I/O) and integration tests (database, marked)
notebooks/             parameters, explanations and displays - no logic; numbered folder.position
  1_data_load/         1.1 initial load, 1.2 refresh, 1.3 filing text
  2_index_replication/ 2.1 S&P 500, 2.2 FTSE All-Share
  3_investment_strategies/  3.1 long-short, 3.2 long-only, 3.3 long-only extra signals, 3.4 long-only ML value,
                       3.5 S&P 500 GARP, 3.6 Barings GARP, 3.7 ARIMA-GARCH timing
  4_portfolio_analytics/  4.1 screen, 4.2 factor dashboard, 4.3 risk model
  5_analysis/          5.1 choosing a learning algorithm without hindsight, 5.2 the same with the learning task
                       repaired, 5.3 learning and judging what a portfolio earns
  6_machine_learning/  docs/ML4T_PLAN.md: 6.1 research protocol, 6.2 features, 6.3 linear models, 6.4 gradient
                       boosting, 6.5 ML strategies, 6.6 strategy assessment, 6.7 latent factors, 6.8 deep learning,
                       6.9 causal ML, 6.10 text features, 6.11 synthetic markets
```

## 5. Building a database

From an empty database (notebook 1, or `eqr load-lseg`, `eqr load-benchmarks`, `eqr load-sec`):

* `load_lseg --market UK` - London: the FTSE 100 / 250 / SmallCap the same way, with what differs there (checked live,
  `eqr probe-lseg-uk`): official closes (`OFF_CLOSE` - London's last trade is often after the closing auction), in pence;
  tickers from LSEG (London RICs do not spell them: `SJP.L` is STJ); companies by PermID (no CUSIPs), with where they
  are incorporated; investment trusts as `fund` listings; dividends in pounds; statements half-yearly and yearly **as
  first reported** (`TR.F.*`, `ReportingState=Orig`), the first half stored as a six-month year to date and the year as
  reported (not its halves added up) - research builds a half-year's trailing year as last year + this half - last
  year's half. Capital changes are dated by their ex-date (a rights issue's effective date is weeks later). Some
  London lines change the currency they are quoted in (IHG: pence to dollars, 2026) and LSEG's histories do not say:
  each bar's currency is today's quote and, walking back, changes where the price jumps by an exchange rate. Dividends
  are kept as declared (Anglo American's in dollars); daily exchange rates are `fx` listings, and research converts
  prices and dividends into the universe's currency on their day. Each market's requests name only its own RICs, so
  loading London never changes a US request.
  `load_estimates --market UK` asks I/B/E/S in pounds. Sources `refinitiv_uk` and `estimates_uk`, in the default
  `EQR_SOURCES` after New York's (a London refresh: a few hundred requests a day, its I/B/E/S after each month end). LSEG
  lacks some London consolidations or dates them weeks off: the share count confirms most (Capital & Regional's of
  2020); Wolseley's 1-for-10 of 2009 (no counts that year) and Shell's unification of 2005 (0.287 new shares per old,
  no whole exchange) stay among `unexplained_price_jumps`.
* `load_lseg` - members of the S&P 500 / 400 / 600 since 2005 (today's lists replayed backwards through every joiner
  and leaver), listings, codes and sectors, unadjusted daily prices, dividends, splits (confirmed by the price) and
  spin-offs, monthly share counts and float, quarterly statements with their announcement dates, and yearly items
  LSEG answers only one field per request (cash flows, depreciation, inventories, unusual items, bank lines). LSEG allows 10,000
  requests (and 5 GB) a day, reset at midnight; a price history costs one request per RIC (lseg-data asks per
  instrument) and returns up to ~3,000 days, so prices are asked in 11-year windows - about 6,500 requests, ~7,550 for
  a whole build. The client reads the allowance left from Workspace before each call and stops before it runs out;
  every request is landed with its description, so the next run continues.
* `load_norgate` (instead of LSEG, or beside it) - every symbol ever in the S&P 500 / 400 / 600 (Norgate's "current &
  past" watchlists, delisted ones as `TICKER-YYYYMM`), daily membership turned into intervals, unadjusted prices
  recovered from Norgate's split-adjusted ones (`Unadjusted Close` / `Close` is each day's factor; a change in it is a
  split, or a spin-off when small), dividends, the index levels ($SPX, $MID, $SML), today's GICS and domicile. NDU runs
  on this computer, so there is no daily limit. A trial reaches back about two years; `eqr probe-norgate` shows how
  far.
* `load_benchmarks` - index levels and the SPY / RSP funds from Yahoo, filed under the same listing where LSEG or
  Norgate supplies the same index.
* `load_financedatabase` - the open-source FinanceDatabase equities file: ISIN / FIGI / Yahoo codes, the country a
  company is run from (filled only where no other source has one), its own sector labels (scheme `FD`).
* `load_sec` - XBRL company facts, as first reported and as restated: income, balance sheet and cash flows, with
  depreciation, financing flows (dividends, share issues and buybacks, borrowing), unusual items and the bank lines
  (capital ratios as pure numbers). Companies are matched to SEC registrants by
  ticker, by CUSIP, and - where a CUSIP names several registrants - by the name each files under (the bulk file's
  `entityName`). A company's main CIK is the one its ticker or CUSIP names; a CIK matched by name only fills the
  periods before the main one begins - a company that re-registered (Disney 2019, Cigna 2018: a new holding company,
  a new CIK) keeps its earlier accounts, while a subsidiary filing its own accounts for the same years (Duke Energy
  Carolinas, a REIT's operating partnership) adds nothing. Companies sharing a main CIK are merged; a name never
  merges two.
* `load_tiingo` (after the source that creates the listings) - unadjusted daily prices, dividends and splits for every
  listing ever in one of our indices, under its last ticker and only within its life; tickers Tiingo does not list
  (its free supported-tickers file) cost no request. Closes must agree with another vendor's where both exist (median
  within 2%), else the ticker is another company's at Tiingo and nothing is written. A free account allows few
  requests an hour: the load stops at Tiingo's 429 (or `--max-requests`) and the next run continues.
* `load_simfin` (after `load_sec`, whose CIKs it uses) - SimFin's bulk quarterly statements (general, banks,
  insurers) as financial facts, its companies matched by SimFin id, CIK, ISIN or ticker on the day they last
  published; its sector labels under scheme `SF`. US-dollar statements only. SimFin re-dates a quarter whenever a
  later report repeats it, so a figure is dated by its first publication only when the SEC's first report confirms
  it (within 0.5%), else by that later date - run it after `load_sec`. The free plan holds about five years ending
  about a year ago (2020-11 to 2025-09 on 2026-10-24): useful history and a cross-check of the SEC, not current
  accounts. `eqr probe-simfin` shows what the account's plan includes and which codes the companies file carries.

* `load_estimates` (after `load_lseg`, whose listings and splits it uses) - analysts' consensus history, reported
  actuals and revision counts from LSEG (`eqr load-estimates`, `eqr probe-estimates`); ~3,200 requests for 2005 to
  today, resumable. The consensus answers as a monthly history (one call per group of ~165 RICs, measure and
  period); the revision counts only one date at a time - asked as a history they never answer, and lseg-data re-sends
  the request at one request each until the client's runaway guard stops it. It is a source of its own
  (`estimates`, after `refinitiv` in `EQR_SOURCES`), so `eqr build` and `eqr refresh` run it; the load ends at the
  last month end (fixed request descriptions all month), and a refresh asks only once a new month end has passed
  (~400 requests).

Afterwards `eqr refresh` keeps everything current (the daily job) and ends with the quality gates.

## 6. Environments and people

One database; separation by **role**, not by copies: `loader` writes `ref` / `market` / `fund` / `ingest` / `ops`,
`researcher` reads them and writes `research`. Each person has their own login and their own local cache.
Schema changes go through Alembic migrations only.

## 7. Running a strategy month by month (design - not built yet)

*Proposed 2026-11-08 for review; nothing below exists in code yet.* A tested strategy can be run as a **book**: each
month the system builds the portfolio the strategy would hold, a person reviews it and may only remove names, and
both books - the systematic one and the one actually held - are tracked from then on. The person's judgement is
then measured, not assumed: after a year the gap between the two books says whether the vetoes earned or cost money.

### Why this shape

* **The strategy picks; the person checks.** The edge measured in research belongs to the whole book - every name at
  its weight. A person who chose favourites from a screen would run a different, untested strategy; a person who
  removes a name for a stated reason (a pending deal, a restatement, bad data) applies information the model cannot
  see - 6.10 found that 8-K events carry news prices and accounts do not.
* **Every decision is recorded.** An unrecorded override is an untested trial (principle 8); a veto names its reason
  from a fixed list, and the book without vetoes - the **shadow book** - is always computable.
* **Decisions are facts.** What was targeted, removed and held on a date happened; it is stored once and never
  recomputed (principle 3: store facts, derive the rest - returns and the shadow comparison are derived).
* **Paper first.** A book can run in `paper` mode (no money) from any registered run: from its start it trades on
  data nobody has seen, the cleanest forward test there is. `live` mode needs the strategy's hold-out look recorded
  (a run with `holdout_opened`), so money follows evidence.

### Tables (schema `book`, one migration)

| Table | One row per | Holds |
|---|---|---|
| `book.strategy` | book | name, the registered `run_id` it runs, mode (`paper` / `live`), start date, status, the veto rules' version |
| `book.target` | book x rebalance date x listing | the systematic weight, score and rank; the code version and data snapshot that built it. Rebuilt freely until approved, then frozen |
| `book.veto` | removal | book, date, listing, reason (`pending_deal`, `restatement`, `data_error`, `news_event`, `cannot_trade`), a note, who, when. A veto lasts one rebalance: next month the name is reviewed afresh |
| `book.holding` | book x date x listing | the weights held after the vetoes, as approved; later the shares and fills a broker reports |

A `manager` role writes `book`; `researcher` reads it.

### The monthly cycle

1. **Refresh** (`eqr refresh`) early in the month, while the free compute allowance has room; the gates must pass.
2. **`eqr rebalance --book <name>`** - refuses if the last refresh failed a gate or the data is stale; refits the
   strategy by its registered rule (the yearly refit of the walk-forward), builds today's book through the same code
   as the backtest (`runs.snapshot`) and writes `book.target`; writes a **review sheet**: names entering and leaving,
   weights, each name's scores and ratios (`screen`), and its 8-Ks of the last quarter (`fund.document`).
3. **Review** - `eqr veto --book <name> <ticker> --reason <code> --note "..."`; nothing can be added or reweighted.
4. **`eqr approve --book <name>`** - the strategy's own allocator runs again without the vetoed names (so the
   constraints - net, gross, risk - still hold), the result is written to `book.holding`, and a trade list (CSV, from
   the previous holding) is written for the broker. Orders are placed by hand until a broker is connected.
5. **Report** - a notebook: the held book against the shadow book, both valued from prices and charged the same cost
   model; each veto's outcome (the removed name against its replacement over the month); exposures (`dashboard`).

### Code (when built)

* `research/book.py` - pure functions: today's target from a snapshot, applying vetoes, the trade list, the two
  books' returns, each veto's outcome. Unit-tested without a database.
* `research/runs.py` - a snapshot for the ML4T strategies (notebook 6.5's books: the composite or a model's scores,
  the allocators of `allocation.py`), beside the factor, GARP and QGARP snapshots it has.
* `pipelines/rebalance.py` - I/O: the gate check, writing targets and holdings under the writer lock.
* CLI `rebalance`, `veto`, `approve`, `books`; migration `0009_book`; notebook *book review*; integration tests on a
  scratch database, as every pipeline has.

### Order of work

B0 the ML4T snapshot -> B1 the `book` tables -> B2 rebalance, veto, approve -> B3 the report -> **paper** from the
first rebalance after the 2025 hold-out is opened -> a year of paper results -> `live`, if the evidence holds. A
broker connection, and whether to run long-short (borrowing) or long-only, are decided then, on that evidence.
