from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from equity_research.research import ml

MONTH_ENDS = pd.date_range("2006-01-31", "2011-12-31", freq="ME")
NAMES = list(range(1, 41))


def _features(values: np.ndarray) -> pd.DataFrame:
    """``(date, listing_id)`` x one feature ``f``, from a ``date x listing`` array."""
    wide = pd.DataFrame(values, index=MONTH_ENDS, columns=NAMES)
    long = wide.rename_axis(index="date", columns="listing_id").reset_index()
    long = long.melt(id_vars="date", var_name="listing_id", value_name="f")
    return long.set_index(["date", "listing_id"]).sort_index()


class _Recorder:
    """A model that remembers what it was trained on and predicts the training rows' count."""

    fits: ClassVar[list[tuple[int, int]]] = []

    def __init__(self) -> None:
        self.n = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_Recorder":
        self.n = len(x)
        # the training target encodes its own decision date (as months since 2000), so the test can read it back
        _Recorder.fits.append((len(x), int(np.nanmax(y))))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), float(self.n))


def test_each_year_is_predicted_only_from_returns_already_known() -> None:
    _Recorder.fits = []
    months_since_2000 = np.array([(d.year - 2000) * 12 + d.month for d in MONTH_ENDS], dtype=float)
    target = pd.DataFrame(np.repeat(months_since_2000[:, None], len(NAMES), axis=1), index=MONTH_ENDS, columns=NAMES)
    members = pd.DataFrame(True, index=MONTH_ENDS, columns=NAMES)
    out = ml.walk_forward(_features(np.ones((len(MONTH_ENDS), len(NAMES)))), target, members, MONTH_ENDS, _Recorder)
    first_year = out.dropna(how="all").index.min().year
    # training dates are quarter ends whose year-ahead return ended before January: 2006-03 .. 2007-12 is the first
    # set of eight, used for 2009 - nothing for 2006-2008
    assert first_year == 2009 and out.loc[:"2008-12-31"].isna().all().all()
    for (rows, latest), year in zip(_Recorder.fits, range(2009, 2012), strict=True):
        latest_date = pd.Timestamp(year=2000 + (latest - 1) // 12, month=(latest - 1) % 12 + 1, day=1)
        assert latest_date + pd.DateOffset(years=1) < pd.Timestamp(f"{year}-01-01")  # its year was already over
        assert rows % len(NAMES) == 0  # whole dates of members, quarter ends only
    assert _Recorder.fits[1][0] > _Recorder.fits[0][0]  # the window expands


def test_the_model_learns_a_relation_that_holds_through_time() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(len(MONTH_ENDS), len(NAMES)))
    target = pd.DataFrame(0.1 * x + rng.normal(0, 0.02, x.shape), index=MONTH_ENDS, columns=NAMES)
    members = pd.DataFrame(True, index=MONTH_ENDS, columns=NAMES)
    out = ml.walk_forward(_features(x), target, members, MONTH_ENDS)
    last = pd.Series(out.iloc[-1].to_numpy(dtype=float), index=NAMES)
    truth = pd.Series(x[-1], index=NAMES)
    assert last.corr(truth, method="spearman") > 0.8


def test_non_members_are_neither_learned_from_nor_predicted() -> None:
    _Recorder.fits = []
    members = pd.DataFrame(True, index=MONTH_ENDS, columns=NAMES)
    members[40] = False
    target = pd.DataFrame(1.0, index=MONTH_ENDS, columns=NAMES)
    out = ml.walk_forward(_features(np.ones((len(MONTH_ENDS), len(NAMES)))), target, members, MONTH_ENDS, _Recorder)
    assert out[40].isna().all()
    assert all(rows % 39 == 0 for rows, _ in _Recorder.fits)


def test_year_ahead_returns_end_on_the_last_trading_day_a_year_later() -> None:
    days = pd.bdate_range("2020-01-01", "2021-06-30")
    returns = pd.DataFrame({1: [0.001] * len(days)}, index=days)
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-31"), pd.Timestamp("2020-12-31")])
    out = ml.forward_returns(returns, dates)[1]
    held = ((days > dates[0]) & (days <= dates[0] + pd.DateOffset(years=1))).sum()
    assert out.iloc[0] == pytest.approx(1.001**held - 1)
    assert np.isnan(out.iloc[1])  # its year has not ended within the data


def test_the_books_ratios_are_clipped_and_altman_z_flags_distress() -> None:
    index = pd.MultiIndex.from_product([MONTH_ENDS[-1:], [1, 2]], names=["date", "listing_id"])
    ratios = pd.DataFrame(
        {
            "P/E": [5000.0, 12.0],
            "(CA-CL)/TA": [-0.5, 0.2],
            "RE/TA": [-1.0, 0.3],
            "EBIT/TA": [-0.2, 0.1],
            "Book Equity/TL": [0.1, 1.0],
        },
        index=index,
    )
    features = ml.book_features(ratios)
    assert list(features.columns) == list(ml.BOOK_CLIPS) and features["P/E"].max() == 1000.0
    z = ml.altman_z(ratios)
    assert z.iloc[0] < ml.DISTRESS_Z < z.iloc[1]
