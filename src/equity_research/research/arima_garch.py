"""An ARIMA-GARCH timing strategy on the S&P 500, after ``menonf/PythonProjects`` (``strategy_single_core_CPU``).

    cfg = ArimaGarchConfig()                                   # the script's settings, 2022-2024
    result = run_arima_garch(cache, cfg, engine=engine)        # fenced, fitted day by day, backtested, registered
    result.forecasts                                           # each day's chosen ARIMA order, forecast and position

Steps, for each trading day t of the period:

1. the last ``window`` daily log returns of the signal index (^GSPC) up to and including t - 24, as the script's
   ``LogReturns[(1 + i):(RollingWindowLength + i)]`` with ``RollingWindowLength = 25`` holds;
2. every ARIMA(p, d, q) of the grid (p and q 1-5, d 1) fitted to them by maximum likelihood, the lowest AIC kept;
3. a GARCH(1, o, 1) with a constant mean fitted to that ARIMA's residuals, ``o`` (the GJR asymmetric term) set to the
   chosen ``d`` as in the script - so with d = 1 it is a GJR-GARCH(1, 1, 1);
4. the sign of its one-step mean forecast is the position held over day t + 1: +1 long, -1 short (0 if exactly zero).

The positions are a one-name book in the traded fund (SPY, with dividends), which :func:`run_backtest` trades at each
close with costs; the run is fenced and registered like every strategy. The script itself scored ``signal x log
return`` of the index, summed (:func:`script_view` repeats that); a fund held long or short earns simple returns.

**What the signal is.** arch's constant-mean model forecasts the mean as its constant mu at every horizon, so step 4
is the sign of mu - in effect of the average ARIMA residual over the window. The ARIMA's own forecast never reaches the
position; each day's record keeps it (``arima_forecast``) and the residuals' mean (``resid_mean``) to show this.

**Cost.** 25 ARIMA fits a day take about 8 seconds on one core. Each day's result is kept in
``data/<db>/research/arima_garch/<hash>.parquet``, keyed by the model's settings, with a digest of the returns it was
fitted to: a rerun reuses the days whose inputs are unchanged and fits only the rest, so a stopped run resumes.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sqlalchemy.engine import Engine

from equity_research.research.backtest import TRADING_DAYS, BacktestResult, run_backtest
from equity_research.research.cache import ResearchCache
from equity_research.research.indices import vendor_returns
from equity_research.research.registry import config_hash, fence
from equity_research.research.strategy import finish_run
from equity_research.settings import get_settings

#: The settings that change a day's fitted models - the key of the stored forecasts. Dates, the traded fund and costs
#: are left out: they change which days are needed and what is earned, not what is fitted on a day.
MODEL_SETTINGS = ("signal_symbol", "window", "p_values", "d_values", "q_values", "garch_p", "garch_q")
#: The columns of a day's record (:func:`forecast_day`), after ``date``.
RECORD = ("digest", "p", "d", "q", "aic", "mean_forecast", "signal", "arima_forecast", "resid_mean")
#: How many newly fitted days between saves, so a stopped run loses little.
SAVE_EVERY = 20


@dataclass(frozen=True)
class ArimaGarchConfig:
    """Every setting of a run (hashed into its registry entry); the defaults are the script's."""

    family: str = "arima_garch"
    signal_symbol: str = "^GSPC"  # the series the models are fitted to: the script's S&P 500 closes
    traded_symbol: str = "SPY"  # the fund the position is held in, dividends included
    start: str = "2022-01-01"  # the script ran on its last three years; ours are the three before the hold-out
    end: str = "2024-12-31"
    window: int = 24
    p_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    d_values: tuple[int, ...] = (1,)
    q_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    garch_p: int = 1
    garch_q: int = 1
    cost_bps: float = 1.0  # one way, per unit traded: SPY's spread is a fraction of a basis point
    open_holdout: bool = False


@dataclass(frozen=True)
class ArimaGarchResult:
    """A finished run: each day's models and position, the book, its backtest, and the fund held throughout."""

    config: ArimaGarchConfig
    forecasts: pd.DataFrame
    book: pd.DataFrame
    backtest: BacktestResult
    benchmark: pd.Series
    metrics: dict[str, Any]
    run_id: int | None


def best_arima(
    window: np.ndarray, p_values: Sequence[int], d_values: Sequence[int], q_values: Sequence[int]
) -> tuple[float, tuple[int, int, int] | None, Any]:
    """``(aic, order, fitted)`` of the grid's ARIMA with the lowest AIC on ``window`` (``None``s if none fits).

    The script's ``evaluate_models``: an order whose fit fails is passed over. statsmodels' defaults apply - no
    constant once d >= 1. Its warnings (non-stationary starts, slow convergence) are routine on 24 points and silenced.
    """
    best: tuple[float, tuple[int, int, int] | None, Any] = (float("inf"), None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        fitted = sm.tsa.arima.ARIMA(window, order=(p, d, q)).fit()
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if fitted.aic < best[0]:
                        best = (float(fitted.aic), (p, d, q), fitted)
    return best


def forecast_day(window: np.ndarray, config: ArimaGarchConfig) -> dict[str, Any]:
    """One day's models on ``window`` (log returns up to the day): the chosen order, the forecasts and the position.

    The GARCH model is the script's ``arch_model(resid, p=1, o=d, q=1)`` - constant mean, normal errors, data not
    rescaled. Its one-step mean forecast is the constant mu, whose sign is the position. A day on which no ARIMA or the
    GARCH model fits is flat (signal 0).
    """
    nan = float("nan")
    record: dict[str, Any] = {
        "p": -1,
        "d": -1,
        "q": -1,
        "aic": nan,
        "mean_forecast": nan,
        "signal": 0.0,
        "arima_forecast": nan,
        "resid_mean": nan,
    }
    aic, order, fitted = best_arima(window, config.p_values, config.d_values, config.q_values)
    if order is None:
        return record
    resid = np.asarray(fitted.resid, dtype=float)
    record |= {
        "p": order[0],
        "d": order[1],
        "q": order[2],
        "aic": aic,
        "arima_forecast": float(np.asarray(fitted.forecast(1))[0]),
        "resid_mean": float(resid.mean()),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "y is poorly scaled": the script fits raw log returns, and so do we
        try:  # arch re-enables its own convergence warning inside fit; show_warning turns it off
            model = arch_model(resid, p=config.garch_p, o=order[1], q=config.garch_q)
            garch = model.fit(disp="off", show_warning=False)
        except (ValueError, np.linalg.LinAlgError):
            return record
        mean = float(garch.forecast(horizon=1, reindex=False).mean.to_numpy()[-1, 0])
    record |= {"mean_forecast": mean, "signal": float(np.sign(mean))}
    return record


def window_digest(window: np.ndarray) -> str:
    """A digest of a day's inputs: a stored day is reused only while the returns it was fitted to are unchanged."""
    return hashlib.sha256(np.round(np.asarray(window, dtype=float), 12).tobytes()).hexdigest()[:16]


def rolling_forecasts(
    log_returns: pd.Series,
    config: ArimaGarchConfig,
    days: pd.DatetimeIndex,
    *,
    known: pd.DataFrame | None = None,
    save: Callable[[pd.DataFrame], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Each of ``days``' record (:func:`forecast_day`) from the ``window`` log returns ending on it, indexed by date.

    ``known`` (records from an earlier run) are reused where their digest matches the day's inputs; the rest are
    fitted, in date order, with ``save`` called on all records so far every :data:`SAVE_EVERY` new fits and at the end,
    and ``progress(fitted, to_fit)`` after each. A day without ``window`` returns up to it cannot be forecast.
    """
    r = log_returns.dropna().sort_index()
    position = r.index.get_indexer(days)
    if (position < 0).any():
        raise ValueError(f"{int((position < 0).sum())} days have no return of {log_returns.name}")
    if (position < config.window - 1).any():
        raise ValueError(f"the first day needs {config.window} returns up to it: start later or load earlier data")
    values = r.to_numpy(dtype=float)
    windows = {day: values[i - config.window + 1 : i + 1] for day, i in zip(days, position, strict=True)}
    digests = {day: window_digest(w) for day, w in windows.items()}
    kept: dict[Any, dict[Any, Any]] = {} if known is None else known.to_dict("index")
    records: dict[Any, dict[Any, Any]] = {
        day: kept[day] for day in days if day in kept and kept[day].get("digest") == digests[day]
    }
    to_fit = [day for day in days if day not in records]

    def frame() -> pd.DataFrame:
        out = pd.DataFrame.from_dict(records, orient="index", columns=list(RECORD)).sort_index()
        out.index.name = "date"
        return out

    for n, day in enumerate(to_fit, start=1):
        records[day] = {"digest": digests[day], **forecast_day(windows[day], config)}
        if progress is not None:
            progress(n, len(to_fit))
        if save is not None and (n % SAVE_EVERY == 0 or n == len(to_fit)):
            save(frame())
    return frame().loc[days]


def signal_book(forecasts: pd.DataFrame, traded: str) -> pd.DataFrame:
    """The book: each day's position in the ``traded`` fund, set at that close (earning from the next day)."""
    return forecasts[["signal"]].rename(columns={"signal": traded}).astype(float)


def script_view(signal: pd.Series, log_returns: pd.Series) -> pd.DataFrame:
    """The script's own scoring: ``1 + cumulative sum`` of the index's log returns and of ``signal x log return``.

    ``signal`` is dated by the day it is set; it multiplies the next day's return, as the script's did.
    """
    held = signal.shift(1).dropna()
    r = log_returns.reindex(held.index)
    return pd.DataFrame({"Buy and Hold": r.cumsum() + 1, "ARIMA-GARCH Strategy": (held * r).cumsum() + 1})


def signal_diagnostics(forecasts: pd.DataFrame, traded_returns: pd.Series) -> dict[str, float]:
    """How the positions behave: time long and short, changes a year, and the hit rate against the up-day rate.

    The hit rate is the share of days the position had the next return's sign. A coin flip that is long as often as
    the signal is would hit ``long x up + short x down`` of the time; ``hit_z`` is the binomial z-statistic of the
    hit rate against that rate - near 0 means the timing adds nothing.
    """
    held = forecasts["signal"].shift(1)
    both = pd.concat({"s": held, "r": traded_returns}, axis=1).dropna()
    both = both[(both["s"] != 0) & (both["r"] != 0)]
    n = len(both)
    hits = float((np.sign(both["s"]) == np.sign(both["r"])).mean()) if n else float("nan")
    long = float((both["s"] > 0).mean()) if n else float("nan")
    up = float((both["r"] > 0).mean()) if n else float("nan")
    chance = long * up + (1 - long) * (1 - up)
    signal = forecasts["signal"]
    years = len(signal) / TRADING_DAYS
    return {
        "days": float(len(signal)),
        "long": float((signal > 0).mean()),
        "short": float((signal < 0).mean()),
        "flat": float((signal == 0).mean()),
        "changes_per_year": float((signal.diff().fillna(0) != 0).sum() / years) if years else float("nan"),
        "hit_rate": hits,
        "up_days": up,
        "chance_hit_rate": chance,
        "hit_z": float((hits - chance) / np.sqrt(chance * (1 - chance) / n)) if n else float("nan"),
        "mu_sign_matches_resid_mean": float(
            (np.sign(forecasts["mean_forecast"]) == np.sign(forecasts["resid_mean"])).mean()
        ),
        "mu_sign_matches_arima_forecast": float(
            (np.sign(forecasts["mean_forecast"]) == np.sign(forecasts["arima_forecast"])).mean()
        ),
    }


def forecasts_path(config: ArimaGarchConfig, root: Path | None = None) -> Path:
    """Where the day records of ``config``'s model settings are kept (:data:`MODEL_SETTINGS`)."""
    settings = {k: v for k, v in asdict(config).items() if k in MODEL_SETTINGS}
    folder = (root or get_settings().data_dir / "research") / "arima_garch"
    return folder / f"forecasts-{config_hash(settings)}.parquet"


def run_arima_garch(
    cache: ResearchCache,
    config: ArimaGarchConfig,
    *,
    engine: Engine | None = None,
    root: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ArimaGarchResult:
    """Fence, load, fit each day (reusing stored days), build the book, backtest, and (with ``engine``) register."""
    holdout = fence(config.end, open_holdout=config.open_holdout)
    start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
    log_returns = np.log1p(vendor_returns(cache, config.signal_symbol, total=False)).rename(config.signal_symbol)
    log_returns = log_returns[log_returns.index <= end]
    days = pd.DatetimeIndex(log_returns.index[log_returns.index >= start])

    path = forecasts_path(config, root)
    known = pd.read_parquet(path) if path.exists() else None

    def save(frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = frame if known is None else pd.concat([known.drop(frame.index, errors="ignore"), frame]).sort_index()
        merged.to_parquet(path)

    forecasts = rolling_forecasts(log_returns, config, days, known=known, save=save, progress=progress)
    traded = vendor_returns(cache, config.traded_symbol, total=True)
    traded = traded[traded.index <= end]
    book = signal_book(forecasts, config.traded_symbol)
    result = run_backtest(book, traded.to_frame(config.traded_symbol), cost_bps=config.cost_bps)
    benchmark = traded.reindex(result.net.index)
    metrics, run_id = finish_run(
        result, benchmark, asdict(config), family=config.family, engine=engine, holdout=holdout
    )
    return ArimaGarchResult(config, forecasts, book, result, benchmark, metrics, run_id)
