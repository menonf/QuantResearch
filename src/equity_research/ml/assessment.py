"""Is a strategy real, affordable and safe? Costs, tail risk, drift, the Fundamental Law and selection-aware tests.

    grid = cost_grid(book, data, benchmark)                    # the strategy at 0x .. 4x its trading costs
    even = break_even(result, benchmark)                       # the cost at which it earns nothing over the benchmark
    tails = tail_risk(result.net)                              # historical VaR and CVaR
    drift = ic_drift(ic)                                       # is the signal's IC fading through time
    tc = transfer_coefficient(book, score, vol, members)       # how much of the signal the portfolio carries
    lower = rademacher_bounds(ics)                             # each signal's IC, after allowing for the search
    rc = reality_check(active)                                 # could the best strategy's edge be luck

*Machine Learning for Trading* (chapters 16-20) assesses a strategy beyond its backtest. Each function here answers
one question:

1. **Costs** - the backtest rerun with every trading cost scaled (:func:`cost_grid`), and the **break-even** cost:
   the multiple of the assumed costs at which the active return falls to zero, from the active return before and after
   costs (it falls in proportion to the costs, since turnover does not change).
2. **Tail risk** - historical value at risk (the loss exceeded on 1 % of days) and conditional value at risk, or
   expected shortfall (the average loss on those days).
3. **Drift** - the signal's IC over a rolling year, and the slope of its IC through time with a Newey-West
   t-statistic: a significantly negative slope says the signal is decaying (crowding, or a regime that has passed).
4. **Fundamental Law diagnostics** (Grinold, 1989; Clarke, de Silva and Thorley, 2002) -
   ``IR ~ TC x IC x sqrt(breadth)``. The **transfer coefficient** TC is the correlation, across companies on a date,
   between the portfolio's active weights and the weights the forecasts alone would ask for (``score / volatility``):
   how much of the signal survives the portfolio's constraints. Comparing the law's prediction with the realised
   information ratio shows how many independent bets there really were.
5. **Selection-aware tests** - choosing the best of many strategies picks luck along with skill.

   * **White's Reality Check** (White, 2000) - the best strategy's mean return over the benchmark, compared with
     the best of the same strategies in bootstrap samples centred on zero (a stationary bootstrap, Politis and
     Romano 1994, keeps months' dependence). The p-value is the chance that luck alone gives a best this good.
   * **The Rademacher bound** behind Paleologo's "Rademacher anti-serum" (2024) - for values in an interval of width
     ``w``, with probability at least ``1 - delta`` every one of ``N`` signals' true mean exceeds its sample mean less
     ``2 R + 3 w sqrt(log(2 / delta) / (2 T))`` (Mohri, Rostamizadeh and Talwalkar, 2018, theorem 3.3).
     ``R`` is the empirical Rademacher complexity of the set of signals: how well the best of them correlates with
     random signs. It is applied to monthly ICs, which lie in [-1, 1].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from equity_research.data.universe import UniverseData
from research_core.analytics.performance.metrics import active_performance
from research_core.analytics.signals.evaluation import newey_west_t
from research_core.engine.backtest import BacktestResult, run_backtest

SEED = 42


def cost_grid(
    book: pd.DataFrame,
    data: UniverseData,
    benchmark: pd.Series,
    *,
    multiples: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per multiple of the universe's trading and borrowing costs: the book's return, active return and IR."""
    returns = data.returns if end is None else data.returns.loc[:end]
    rows = {}
    for m in multiples:
        result = run_backtest(
            book, returns, cost_bps=data.cost_bps * m, borrow_bps=data.borrow_bps * m, buy_tax_bps=data.buy_tax_bps * m
        )
        active = active_performance(result.net, benchmark)
        rows[m] = {"cagr": result.metrics()["cagr"], **{k: active[k] for k in ("active_return", "information_ratio")}}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("cost multiple")


def break_even(result: BacktestResult, benchmark: pd.Series) -> dict[str, float]:
    """The multiple of the assumed costs, and the one-way cost in basis points, at which the active return is zero.

    Costs reduce the active return in proportion (the trades do not change), so with ``g`` the active return before
    costs and ``n`` after, the break-even multiple is ``g / (g - n)``; the cost in basis points is that times the
    average cost the backtest charged per unit traded. Negative before costs: no cost breaks even (NaN).
    """
    gross = active_performance(result.gross, benchmark)["active_return"]
    net = active_performance(result.net, benchmark)["active_return"]
    traded = float(result.turnover.sum())
    charged = float(result.costs.sum()) / traded * 1e4 if traded > 0 else float("nan")
    multiple = gross / (gross - net) if gross > 0 and gross > net else float("nan")
    return {"gross active": gross, "net active": net, "cost charged (bp)": charged, "break-even multiple": multiple,
            "break-even cost (bp)": multiple * charged}  # fmt: skip


def ic_drift(ic: pd.Series, *, window: int = 12, lags: int = 0) -> dict[str, Any]:
    """The rolling ``window`` mean IC and the IC's linear trend a year, with its Newey-West t-statistic.

    The trend's t-statistic is that of the slope in a regression of the IC on time, computed as the t of the mean of
    ``(t - mean t) (IC - mean IC) / var(t)`` with Newey-West errors over ``lags``.
    """
    x = ic.dropna()
    years = (pd.DatetimeIndex(x.index) - pd.DatetimeIndex(x.index)[0]).days.to_numpy() / 365.25
    centred = years - years.mean()
    contributions = pd.Series(centred * (x.to_numpy() - x.mean()) / (centred @ centred / len(x)), index=x.index)
    return {
        "rolling": ic.rolling(window, min_periods=window // 2).mean().rename("rolling IC"),
        "trend a year": float(contributions.mean()),
        "t_nw": newey_west_t(contributions, lags),
    }


def transfer_coefficient(
    book: pd.DataFrame, score: pd.DataFrame, vol: pd.DataFrame, members: pd.DataFrame, benchmark_weights: pd.DataFrame
) -> pd.Series:
    """Per date: the correlation across members of the book's active weights with ``score`` z-scores over volatility.

    The active weight is the book's weight less the benchmark's (``benchmark_weights``, e.g. market-value weights);
    ``score / volatility`` is what an unconstrained mean-variance portfolio would hold for Grinold's alpha rule
    (alpha proportional to IC x volatility x score, divided by variance).
    """
    out = {}
    for d in book.index:
        on = members.reindex(columns=book.columns).loc[d].fillna(False).astype(bool) if d in members.index else None
        if on is None or d not in score.index:
            continue
        s = score.loc[d].reindex(book.columns).where(on)
        v = vol.loc[d].reindex(book.columns).where(on) if d in vol.index else None
        if v is None:
            continue
        z = (s - s.mean()) / s.std()
        optimal = z / v
        active = book.loc[d] - benchmark_weights.reindex(columns=book.columns).loc[d].fillna(0.0)
        both = optimal.notna() & np.isfinite(optimal) & on
        if both.sum() > 30:
            out[d] = float(np.corrcoef(active[both].to_numpy(), optimal[both].to_numpy())[0, 1])
    return pd.Series(out, name="transfer coefficient")


def fundamental_law(ic: float, breadth: float, tc: float) -> float:
    """The information ratio the law predicts: ``tc x ic x sqrt(breadth)``."""
    return tc * ic * math.sqrt(breadth)


def stationary_bootstrap(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of one stationary-bootstrap sample of length ``n`` (blocks of geometric length, mean ``mean_block``)."""
    out = np.empty(n, dtype=int)
    i = int(rng.integers(n))
    for t in range(n):
        out[t] = i
        i = int(rng.integers(n)) if rng.random() < 1 / mean_block else (i + 1) % n
    return out


def reality_check(
    active: pd.DataFrame, *, n_boot: int = 2000, mean_block: float = 6.0, seed: int = SEED, studentise: bool = False
) -> dict[str, Any]:
    """White's Reality Check: is the best column's mean (a return over a benchmark) more than luck across columns?

    ``active`` is ``period x strategy``. Returns the best strategy, its mean and the bootstrap p-value of
    ``max sqrt(T) mean`` against the same statistic of the re-centred bootstrap samples.

    White's statistic compares raw means, so the most volatile strategy dominates the bootstrap maximum and the
    others barely count as trials. ``studentise`` divides each column by its standard deviation first, comparing
    t-statistics instead (the step Hansen's SPA test adds), so every strategy counts as a trial on equal terms.
    """
    x = active.dropna().to_numpy(dtype=float)
    if studentise:
        x = x / x.std(axis=0, ddof=1)
    t = len(x)
    means = x.mean(axis=0)
    stat = math.sqrt(t) * means.max()
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap(t, mean_block, rng)
        boot[b] = math.sqrt(t) * (x[idx].mean(axis=0) - means).max()
    best = int(np.argmax(means))
    raw = active.dropna().to_numpy(dtype=float)[:, best].mean()
    return {"best": str(active.columns[best]), "mean": float(raw), "p_value": float((boot >= stat).mean()),
            "strategies": x.shape[1], "periods": t}  # fmt: skip


def rademacher_bounds(
    values: pd.DataFrame, *, width: float = 2.0, delta: float = 0.05, draws: int = 2000, seed: int = SEED
) -> pd.DataFrame:
    """Per column (a signal's per-period values in an interval of ``width``): its mean and a lower bound on the truth.

    The bound holds for every column at once with probability at least ``1 - delta``: the mean less twice the
    empirical Rademacher complexity (estimated from ``draws`` random sign vectors) less
    ``3 width sqrt(log(2 / delta) / (2 T))``. Periods where any column is missing are dropped.
    """
    x = values.dropna().to_numpy(dtype=float)
    t = len(x)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, t))
    complexity = float(np.mean(np.max(signs @ x / t, axis=1)))
    slack = 3 * width * math.sqrt(math.log(2 / delta) / (2 * t))
    means = x.mean(axis=0)
    return pd.DataFrame(
        {"mean": means, "rademacher": complexity, "slack": slack, "lower bound": means - 2 * complexity - slack},
        index=values.columns,
    )
