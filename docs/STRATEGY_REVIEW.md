# The review of the investment strategies (notebooks 3.1-3.6), and what was done

A review of notebooks 3.1-3.6 and the code behind them (20 November 2026) found that the scores carry some real
information but the portfolios lose most of it to sector, size and beta bets, and that several measurements were
wrong. Its suggested order: fix the measurements, change how the books are built, prune the signals, then test one
written specification once and replicate it on another market. This records, item by item, what the data showed
when checked, what changed in the code, and what was left. Branch `strategy_improvements`.

How changes were made, following `CONTRIBUTING.md` and `docs/ML4T_PLAN.md`'s compatibility rules:

* **Errors in the data are fixed outright** (1.1, the EBIT / EV guard of 1.3): they move registered results, and the
  notebooks that use them were rerun.
* **Everything else is a new setting** of `StrategyConfig` (or a QGARP input) whose default is how runs were made
  before. A setting left at its default is not registered (`factor_strategy.registered_settings`, as QGARP's
  `registered_settings` already did), so every earlier run keeps its hash and repeating it adds no trial.
* New strategies are new registry families (`revised_lo`, `revised_ls`), specified before they ran.

## 1. Measurements

| # | The review | Checked on the data | Done |
|---|---|---|---|
| 1.1 | revision counts may be a snapshot (look-ahead) | **Confirmed.** In `quant_research_local`, 99.97 % of month-to-month revision counts are unchanged and 99.6 % equal the listing's latest: GE 21 raised / 0 lowered, Exxon 20 / 2, JPMorgan 0 / 6 in every June from 2000 to 2026. LSEG answers `TR.NumEstRevisingUp` / `Down` with the day's counts whatever `SDate` names; a delisted company answers 0 / 0 for the same reason (not survivorship, as the notebooks said). | Research reads only counts loaded within 31 days of their date (`qgarp_estimates.live_revisions`, `schema.REVISION_LIVE_DAYS`); the loader asks no older month end (`load_estimates.revision_days`, ~1,900 requests saved per build); the audit lists the snapshot rows (`estimates` area). Backtests have no breadth; the consensus revisions stand in. CLAUDE.md and ARCHITECTURE.md say so. |
| 1.2 | learned value compared over different dates | Confirmed by the code: a model needs eight training dates with a complete year-ahead outcome, so with `TRAIN_FROM` 2006 the first score is 2007 and the book held cash 2004-2006. | `run_strategy` refuses a book empty on any rebalance date (`EmptyBookError`, naming its first holding date). Setting `train_from`; notebook 3.3 trains from 2000 so every score exists for all of 2004-2024. |
| 1.3 | the factor mix is not equal-weighted; missing factors favoured | The mechanism is in `factors.combine` (a mean of z-scores, a missing one skipped). | Setting `mix="standardised"`: each factor z-scored again, a missing factor counted as neutral (`combine(missing="neutral")`). **EBIT / EV is missing where EV <= 0** (`ratios.py`), as FCF / EV was: 0.24 % of S&P 1500 member-months (0.5-0.7 % in 2020-2024), 1.7 % of the FTSE All-Share's (up to 3 %). |
| 1.4 | benchmark against our own cap-weighted index | SPTM tracked other indices until 2020 (SSGA). | Setting `benchmark`: `cap_weighted` / `equal_weighted` (the universe rebuilt from its members) or a fund symbol. `StrategyResult.benchmark` carries it. |
| 1.5 | a one-day spike in mid-2020 | **Not a price error.** The long-short book lost 8.5 % on 5 June and 15.5 % on 8 June 2020, then gained 10.6 % and 8.3 %: distressed energy shares in its short leg squeezed in the "junk rally" (Valaris $0.35 to $2.19, Noble +161 %, Oasis, Denbury, Callon), all real moves. What made it hurt was the book's size: 179 % long and 94 % short that day (see 4.1). | No data fix; a check for one-day reversals would flag genuine moves (March 2020 has dozens). The rebuilt long-short book is dollar- and beta-neutral. |
| 1.6 | write-ups out of step with outputs | Confirmed (3.1's labels, 3.2, 3.3, 3.5, 3.6: 3.5's Part 1 still described a rebuild from 2009 that starts in 2004, 3.6's findings a database since replaced). | 3.1's run labels fixed; alpha shown beside active return in 3.1-3.3; 3.3's and 3.7's findings printed from the results; 3.2's, 3.5's and 3.6's findings rewritten after the reruns; chart titles and texts name the years each run covers. |

## 2. Portfolio construction

`research_core/engine/tilt.py` builds a book as **the benchmark plus active tilts on a factor risk model** (Grinold and
Kahn): each name's tilt is proportional to its score over its specific risk, subject to no active beta, no exposure
to chosen factors (sectors, size), a cap per name, never short (long-only), and scaled to a budget of predicted
active risk. It is solved exactly (a quadratic programme with a diagonal Hessian, by damped Newton steps on its dual)
and tested on a synthetic market. `equity_research/strategies/construction.py` feeds it notebook 4.2's risk model
and the universe by market value; the factor strategy uses it with `construction="tilted"`.

| # | The review | Done |
|---|---|---|
| 2.1 | hold beta at 1, judge on alpha | the tilted construction has no active beta by constraint; alpha shown in every summary |
| 2.2 | remove the sector bet | sector-neutral tilts; setting `within_sectors` ranks each factor within GICS sectors (`factors.within_groups`) |
| 2.3 | remove the size bet | size-neutral tilts (the risk model's Size style) |
| 2.4 | build the books with the risk model | `construction="tilted"`, settings `risk`, `max_active`, `neutral`, `gross` |
| 2.5 | use the bottom of the ranking | a tilted book underweights the worst names down to zero; a quantile book can already hold "the index less its worst fifth" (`enter=0.8`, `weighting="market_cap"`) |
| 2.6 | spend active risk in mid and small caps | not a setting: the tilts follow score over specific risk within the per-name cap. Measured: at 2 % predicted active risk the S&P 1500 book holds 199 names on average, active share 55 % (notebook 3.7). |
| 2.7 | cap single-name weights | `max_weight` for quantile books (`portfolio.capped`); `max_active` for tilted ones |

## 3. Signals

| # | The review | Done |
|---|---|---|
| 3.1 | reweight QGARP towards quality and consensus revisions | one variant, written down in 3.6 before it ran: quality 45 % (alpha 30, defensive 15), value 15, growth 15, 4-month consensus revisions 25; to 2024, hold-out shut |
| 3.2 | narrow the growth composite to sustainable growth | not built |
| 3.3 | value on earnings and cash flow, within sectors | factor `earnings_value` (EBIT / EV, earnings / price, FCF / EV) |
| 3.4 | momentum and low volatility out of the long-only mix | the revised specification leaves them out |
| 3.5 | test the 4-month consensus revision | factor `consensus_revisions` (4 months, read from the cache: `factor_strategy.cache_factors`); QGARP input `zscore_consensus_revisions_4_mth`; tested in 3.6 and in the deciles and triage of 3.5 and 3.6. S&P 1500, 2004-2026: t +2.23 (not a discovery) against the 1-month version's +2.70, with 28 % of its top fifth replaced a month instead of 70 %; best tenth +2.9 % a year (t 1.6). S&P 500, 2010-2026: t +1.18, best tenth +3.7 % (t 1.8). As QGARP's revisions input, active return +0.1 % a year to 2024 |
| 3.6 | stabilise the learned value score | factor `ml_value_rank`: a rank target, depth-3 regularised trees, five seeds averaged (`learned_value.learned_value_rank`); tested in 3.3 |

## 4. Long-short

| # | The review | Done |
|---|---|---|
| 4.1 | replace the beta-neutral leg scaling | `construction="tilted"` with `long_only=False`: dollar-, beta-, sector- and size-neutral, gross capped (`gross`); notebook 3.4's rebuilt book |
| 4.2 | combine factors as separate sleeves | not built: the standardised mix on a neutral construction is the alternative tested |
| 4.3 | adopt notebook 5.5's construction | the tilted construction is that idea (the risk model's covariance) with neutrality constraints |
| 4.4 | cut the turnover | see 5 |

## 5. Turnover and costs

* Setting `drift_band`: a holding within that fraction of its target is not traded back at a rebalance
  (`backtest.banded`); only entries, exits and names that drifted further trade.
* Setting `cost_multiplier` stresses every trading cost; notebook 3.7 reports its books at twice the costs.
* Staggered quarterly rebalancing: not built.

## 6. Testing discipline

Notebook 3.7 states one specification, its pass mark and the trials it is counted against before it runs; runs it
once on the S&P 1500 (2004-2024, the hold-out shut) and unchanged on the FTSE All-Share; and prints its findings from
the results. Notebook 3.4's rebuilt long-short book is specified the same way.

## Results

*Runs of 20 November 2026 on `quant_research_local`, each registered; the hold-out (2025 on) shut unless said. Every
notebook of section 3 was rerun after the fixes.*

### The revised long-only specification (notebook 3.7): does not pass

One of the three conditions written before the run holds:

| | S&P 1500, 2004-2024 | FTSE All-Share, unchanged |
|---|---|---|
| return a year (its own index) | 10.7 % (10.3 %) | 8.4 % (7.3 %) |
| active return after costs | +0.41 % | +0.99 % |
| tracking error, information ratio, t | 2.61 %, +0.16, **t 0.73** (the pass mark asked 1.0) | 2.06 %, +0.49, t 2.24 |
| beta to its index, alpha | 0.99, +0.56 % | 1.00, +1.01 % |
| costs a year; active return at twice the costs | 0.67 %; **-0.27 %** | 1.48 % (stamp duty); -0.49 % |
| deflated active Sharpe: this family / all 46 long-only trials | 58 % / 6 % | 96 % / 50 % |

* **The construction did what it promised:** a beta of 0.99 and 1.00 to its index (the books of 3.1-3.6: 0.8 to 1.2),
  199 names of 1,500 and 83 of 509 on average, active share 55 % and 40 %, no name more than 1 % from its index weight.
* **The edge is small and costs take it.** On the S&P 1500 the active return is a third of a point a year with a
  t of 0.7, and doubling the costs turns it negative. Its two worst years are the mega-cap years 2019 (-5.9 points)
  and 2020 (-8.0), although the book has no size bet by the risk model's measure.
* **The FTSE All-Share is the encouraging part:** a market none of these strategies had been tested on, the
  settings unchanged, +1.0 % a year with an information ratio of 0.49 and ahead of its index in 15 of 21 years. It
  is one market and one run; costs at twice the assumed level would also take it.
* Against SPTM and SPY (daily fund prices) the S&P 1500 book is +0.38 % and +0.42 % a year, information ratios
  +0.12 and +0.16.
* **Measured twice.** The first execution (runs 195-196) compared the books with an index that had one return a
  month - the benchmark read month-end market values as daily ones. Fixed (`construction.index_returns`, and
  `indices.cap_weighted` refuses such values), tested, and the same two configurations measured again (runs
  231-232, and once more as 244-245 with the same numbers); the books did not change.

### The rebuilt long-short book (notebook 3.4, run 219): does not pass

Market-neutral as built - a beta to SPY of -0.00 (the old book +0.19), volatility 3.8 % (16.4 %), maximum drawdown
-23 % (-50 %) - but a Sharpe ratio after costs and borrowing of 0.27 against the old book's 0.35: 1.0 % a year. With
no beta, sector or size bet it earns only what its signals earn, and they lost five years running, 2016-2020. Part
of the old book's Sharpe ratio was the market's: it was on average 1.40 long and 0.74 short.

### QGARP reweighted (notebook 3.6, S&P 1500, to 2024, hold-out shut)

| QGARP weights | return a year | active return vs SPTM | information ratio | t |
|---|---|---|---|---|
| the original's (run 246) | 10.1 % | -0.1 % | +0.07 | +0.34 |
| 4-month consensus revisions in the revisions' 10 % (run 250) | 10.4 % | +0.1 % | +0.10 | +0.48 |
| weight moved to quality (45 %) and the 4-month consensus revisions (25 %) (run 251) | 11.2 % | +0.8 % | +0.19 | +0.88 |

The direction the review expected, not evidence: a t below 1, weights chosen after reading the triage of the same
years, and the family's fourteenth trial.

### What removing the revision counts' look-ahead changed (1.1)

Read as a history, the counts were the strongest signals in the book: the 4-month breadth's best tenth beat the
S&P 1500 by 7.5 % a year (t 6.7) and topped 3.6's triage (t +5.36), with a month-to-month rank autocorrelation of
1.00 - the mark of a snapshot. Without them:

| QGARP | with the counts (17 November) | without (20 November) |
|---|---|---|
| S&P 1500, 2004 - November 2026 (3.6) | 11.0 % a year, active +0.2 % | 10.4 %, active -0.3 % |
| S&P 1500, the hold-out alone | active +0.4 % a year, t +0.06 | active -2.0 % a year, t -0.19 |
| S&P 500, 2010 - November 2026 (3.5) | 13.5 % a year, active -0.8 % | 12.9 %, active -1.3 % |
| S&P 500, the hold-out alone | active +0.1 % a year, t +0.02 | active -2.3 % a year, t -0.23 |

Both hold-out looks remain inconclusive by their pass marks (a t below -1 would count against the screen), now
leaning the other way. They were read again after a data fix, not after a research choice.

### Learned value over the same dates (1.2, 3.6; notebook 3.3)

Every score now exists from 2003 (training from 2000). Alone, 2004-2024: formula value 10.0 % a year (-0.3 % against
SPTM, information ratio +0.03), learned value 8.8 % (-1.4 %, -0.21), the steadier learned value 9.0 % (-1.2 %,
-0.32; turnover 1.7 instead of 2.9). The learned score's lead in the old findings came from comparing different
dates. Inside the five-factor portfolio the learned score instead of formula value gives 10.7 % against 10.5 % a
year (active +0.3 % against +0.2 %).

### Other findings of the reruns

* **3.2:** free-cash-flow yield's top tenth beat the S&P 1500 by 3.0 % a year (t 2.0), the strongest of the
  signals tested there, yet added nothing to the cap-weighted portfolio; 3.7's `earnings_value` uses it inside a
  value score instead.
* **3.1:** from 2000 the first version is +1.9 % a year ahead of SPTM, from 2006 -1.5 %: its lead is 2000-2002.
* **QGARP on the S&P 1500 was not reproducible to the last digit.** The same six configurations of 3.6, run twice on
  the same code and data (runs 202-207, 233-238), differed by up to 0.007 points of return a year. The cause: LSEG
  back-filled some companies' early actuals with one announcement time (2001, 2002 and 2003 all "announced" on
  1 September 2004 - 130 company-measure pairs, mostly cash flow per share), and `Estimates.actual` chose among years
  known at the same moment by the order the cache returned them. Fixed - the latest year wins, by a stable sort -
  with a test in every row order; 3.5 and 3.6 run again. What remains is floating-point noise in two listings'
  daily returns (2e-16), far inside the regression check's tolerance.

### Not built, or left open

* 3.2 (a sustainable-growth composite), 4.2 (factor sleeves), staggered quarterly rebalancing, a setting for 2.6.
* The revision counts stored for past dates stay in the database (the audit's `estimates` area lists them);
  deleting them is the owner's decision.
* The regression check's baseline (`equity_research/regression.py`, now covering `revised_lo` and `revised_ls`)
  should be retaken once this branch is accepted: its strategies' numbers moved on purpose.
