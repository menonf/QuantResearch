from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from equity_research.research import arima_garch as ag
from equity_research.research.backtest import run_backtest

DAYS = pd.bdate_range("2020-01-01", periods=60)
#: One ARIMA order instead of 25, so a day fits in a fraction of a second.
SMALL = ag.ArimaGarchConfig(window=24, p_values=(1,), d_values=(1,), q_values=(1,))


def _returns(seed: int = 0, n: int = 60) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n), index=DAYS[:n], name="^GSPC")


def _fake(calls: list[np.ndarray]) -> Any:
    """A stand-in for a day's fit: records the window it was given, long when the window's sum is positive."""

    def fake(window: np.ndarray, config: ag.ArimaGarchConfig) -> dict[str, Any]:
        calls.append(window.copy())
        s = float(np.sign(window.sum()))
        return {
            "p": 1,
            "d": 1,
            "q": 1,
            "aic": 0.0,
            "mean_forecast": s,
            "signal": s,
            "arima_forecast": s,
            "resid_mean": s,
        }

    return fake


def test_the_lowest_aic_order_of_the_grid_is_chosen() -> None:
    y = _returns().to_numpy()[:24]
    aic, order, _ = ag.best_arima(y, (1, 2), (1,), (1, 2))
    fits = {(p, 1, q): sm.tsa.arima.ARIMA(y, order=(p, 1, q)).fit().aic for p in (1, 2) for q in (1, 2)}
    assert order == min(fits, key=lambda k: fits[k])
    assert aic == pytest.approx(min(fits.values()))


def test_the_position_is_the_sign_of_the_garch_constant_mean() -> None:
    record = ag.forecast_day(_returns().to_numpy()[:24], SMALL)
    assert (record["p"], record["d"], record["q"]) == (1, 1, 1)
    assert record["signal"] == np.sign(record["mean_forecast"]) and record["signal"] in (-1.0, 1.0)
    # the constant-mean forecast is mu, an estimate of the residuals' mean: same order of size, not the ARIMA forecast
    assert abs(record["mean_forecast"]) < 10 * abs(record["resid_mean"]) + 0.01


def test_a_day_with_no_fitting_arima_is_flat() -> None:
    record = ag.forecast_day(_returns().to_numpy()[:24], ag.ArimaGarchConfig(p_values=(), q_values=()))
    assert record["signal"] == 0.0 and record["p"] == -1


def test_each_day_is_fitted_to_the_window_ending_on_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[np.ndarray] = []
    monkeypatch.setattr(ag, "forecast_day", _fake(calls))
    r = _returns()
    days = pd.DatetimeIndex(DAYS[30:35])
    out = ag.rolling_forecasts(r, SMALL, days)
    assert list(out.index) == list(days)
    for day, window in zip(days, calls, strict=True):
        i = int(r.index.searchsorted(day))
        np.testing.assert_array_equal(window, r.to_numpy()[i - 23 : i + 1])  # 24 returns, the day's last


def test_a_later_return_never_changes_an_earlier_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ag, "forecast_day", _fake([]))
    r = _returns()
    days = pd.DatetimeIndex(DAYS[30:40])
    before = ag.rolling_forecasts(r, SMALL, days)
    shocked = r.copy()
    shocked.iloc[35] = 0.5
    after = ag.rolling_forecasts(shocked, SMALL, days)
    pd.testing.assert_frame_equal(before.iloc[:5], after.iloc[:5])  # days 30-34 are before the shock


def test_stored_days_are_reused_only_while_their_inputs_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[np.ndarray] = []
    monkeypatch.setattr(ag, "forecast_day", _fake(calls))
    r = _returns()
    days = pd.DatetimeIndex(DAYS[30:40])
    saved: list[pd.DataFrame] = []
    first = ag.rolling_forecasts(r, SMALL, days, save=saved.append)
    assert len(calls) == 10 and len(saved[-1]) == 10
    calls.clear()
    again = ag.rolling_forecasts(r, SMALL, days, known=first)
    assert calls == [] and again.equals(first)
    revised = r.copy()
    revised.iloc[38] += 0.001  # a revised price: the days whose window holds it are fitted again
    ag.rolling_forecasts(revised, SMALL, days, known=first)
    assert len(calls) == 2  # days 38 and 39


def test_a_day_without_a_full_window_is_refused() -> None:
    with pytest.raises(ValueError, match="24 returns"):
        ag.rolling_forecasts(_returns(), SMALL, pd.DatetimeIndex(DAYS[10:12]))


def test_a_position_set_at_a_close_earns_the_next_days_return() -> None:
    forecasts = pd.DataFrame({"signal": [1.0, -1.0, 1.0, 1.0]}, index=DAYS[:4])
    returns = pd.DataFrame({"SPY": [0.0, 0.01, 0.02, -0.03]}, index=DAYS[:4])
    result = run_backtest(ag.signal_book(forecasts, "SPY"), returns, cost_bps=0.0)
    np.testing.assert_allclose(result.gross.to_numpy(), [0.01, -0.02, -0.03])


def test_the_script_view_sums_signal_times_the_next_log_return() -> None:
    signal = pd.Series([1.0, -1.0, 1.0], index=DAYS[:3])
    logs = pd.Series([0.0, 0.01, 0.02], index=DAYS[:3])
    view = ag.script_view(signal, logs)
    np.testing.assert_allclose(view["Buy and Hold"].to_numpy(), [1.01, 1.03])
    np.testing.assert_allclose(view["ARIMA-GARCH Strategy"].to_numpy(), [1.01, 0.99])


def test_diagnostics_compare_the_hit_rate_with_chance() -> None:
    forecasts = pd.DataFrame(
        {"signal": [1.0, 1.0, -1.0, 1.0], "mean_forecast": [1.0, 1.0, -1.0, 1.0], "resid_mean": [1.0, -1.0, -1.0, 1.0]},
        index=DAYS[:4],
    )
    forecasts["arima_forecast"] = forecasts["mean_forecast"]
    traded = pd.Series([0.0, 0.01, -0.01, -0.02], index=DAYS[:4])
    d = ag.signal_diagnostics(forecasts, traded)
    assert d["hit_rate"] == pytest.approx(2 / 3)  # long-up, long-down (miss), short-down
    assert d["long"] == pytest.approx(0.75) and d["changes_per_year"] > 0
    assert d["mu_sign_matches_resid_mean"] == pytest.approx(0.75)


def test_stored_forecasts_are_keyed_by_the_model_settings_only(tmp_path: Any) -> None:
    base = ag.forecasts_path(ag.ArimaGarchConfig(), tmp_path)
    assert ag.forecasts_path(ag.ArimaGarchConfig(start="2015-01-01", cost_bps=5.0), tmp_path) == base
    assert ag.forecasts_path(ag.ArimaGarchConfig(window=50), tmp_path) != base
