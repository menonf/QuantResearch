import numpy as np
import pandas as pd
import pytest

from research_core.analytics.signals import validation as v

CALENDAR = pd.bdate_range("2010-01-01", "2016-12-31")


def test_label_ends_count_trading_days() -> None:
    dates = CALENDAR[[0, 10, len(CALENDAR) - 3]]
    ends = v.label_ends(dates, CALENDAR, 5)
    assert ends.iloc[0] == CALENDAR[5] and ends.iloc[1] == CALENDAR[15]
    assert ends.iloc[2] == pd.Timestamp.max  # past the data: never known
    months = pd.DatetimeIndex(["2010-01-29", "2010-02-26", "2010-03-31"])
    assert v.next_date_ends(months).tolist() == [months[1], months[2], pd.Timestamp.max]


def test_walk_forward_trains_only_on_labels_known_before_the_test_year() -> None:
    ends = v.label_ends(CALENDAR, CALENDAR, 21)
    folds = v.walk_forward(CALENDAR, ends, train_years=3)
    assert [f.test.min().year for f in folds] == [2013, 2014, 2015, 2016]
    for fold in folds:
        start = fold.test.min()
        assert (ends.reindex(fold.train) < start).all()  # every training label was complete
        assert fold.train.min().year == start.year - 3  # a three-year window
        assert len(fold.purged) == 21  # the last 21 days' labels ran into the test year
        assert (fold.purged < start).all() and fold.train.max() < fold.purged.min()
    expanding = v.walk_forward(CALENDAR, ends, train_years=None, first_test=2012)
    assert expanding[0].train.min() == CALENDAR[0] and expanding[-1].train.min() == CALENDAR[0]
    assert len(expanding[-1].train) > len(expanding[0].train)


def test_walk_forward_with_period_labels_purges_the_last_date() -> None:
    months = pd.date_range("2010-01-31", "2014-12-31", freq="ME")
    folds = v.walk_forward(months, v.next_date_ends(months), train_years=None, first_test=2012)
    assert folds[0].purged.tolist() == [pd.Timestamp("2011-12-31")]  # its label ends in January 2012
    assert folds[0].train.max() == pd.Timestamp("2011-11-30")


def test_purged_kfold_keeps_label_windows_apart_and_embargoes() -> None:
    dates = CALENDAR[:500]
    ends = v.label_ends(dates, CALENDAR, 10)
    folds = v.purged_kfold(dates, ends, k=5, embargo=5)
    assert len(folds) == 5
    assert pd.DatetimeIndex(np.concatenate([f.test for f in folds])).equals(dates)  # each date tested once
    for fold in folds:
        lo, hi = fold.test.min(), ends.reindex(fold.test).max()
        train_ends = ends.reindex(fold.train)
        overlap = (fold.train <= hi) & (train_ends.to_numpy() >= lo)
        assert not overlap.any()  # no training label window touches the block's
        after = dates[dates > fold.test.max()][:5]
        assert not set(after) & set(fold.train)  # the embargo
    # the 10 days before the block have labels reaching into it; the 10 after start inside its label window
    assert len(folds[2].purged) == 10 + 10
    longer = v.purged_kfold(dates, ends, k=5, embargo=15)  # an embargo beyond the label window drops 5 more
    assert len(longer[2].purged) == 10 + 15
    with pytest.raises(ValueError, match="k must be"):
        v.purged_kfold(dates[:3], ends, k=5)


def test_describe_lists_each_fold() -> None:
    ends = v.label_ends(CALENDAR, CALENDAR, 21)
    table = v.describe(v.walk_forward(CALENDAR, ends, train_years=3))
    assert list(table.index) == [1, 2, 3, 4]
    assert (table["purged"] == 21).all() and table["test from"].iloc[0].year == 2013
