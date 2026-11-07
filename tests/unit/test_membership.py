import pandas as pd

from equity_research.ingest.membership import clip_to_first_trade, close_at_delisting, coalesce, disjoint


def missing(value: object) -> bool:
    """A missing value (NaN, NaT, None) - typed ``object`` so the checker does not narrow what a frame holds."""
    return bool(pd.isna(value))  # type: ignore[call-overload]


def _spells(*rows: tuple[int, str, str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_id": 1,
            "listing_id": [r[0] for r in rows],
            "start_date": pd.to_datetime([r[1] for r in rows]),
            "end_date": pd.to_datetime(pd.Series([r[2] for r in rows], dtype="object")),
            "vendor_id": 1,
        }
    )


def test_overlapping_spells_of_one_listing_become_one() -> None:
    # Anteon: one RIC's spell 2005-01-01..2006-06-09, the other's 2005-05-02..2006-03-13 - one share line
    out = coalesce(_spells((7, "2005-01-01", "2006-06-09"), (7, "2005-05-02", "2006-03-13")))
    assert len(out) == 1
    assert (out.loc[0, "start_date"], out.loc[0, "end_date"]) == (
        pd.Timestamp("2005-01-01"),
        pd.Timestamp("2006-06-09"),
    )


def test_an_open_spell_keeps_the_merged_one_open() -> None:
    out = coalesce(_spells((7, "2005-01-01", None), (7, "2015-10-02", None)))
    assert len(out) == 1 and missing(out.loc[0, "end_date"])


def test_spells_apart_and_other_listings_stay_apart() -> None:
    out = coalesce(_spells((7, "2005-01-01", "2006-01-01"), (7, "2008-01-01", None), (8, "2005-06-01", None)))
    assert len(out) == 3
    touching = coalesce(_spells((7, "2005-01-01", "2006-01-01"), (7, "2006-01-01", None)))  # end is exclusive
    assert len(touching) == 1


def test_no_spell_starts_before_the_listing_traded() -> None:
    spells = _spells(
        (1, "2005-01-01", None), (2, "2005-01-01", None), (3, "2005-01-01", "2006-01-01"), (4, "2005-01-01", None)
    )
    first = pd.Series({1: pd.Timestamp("2005-01-03"), 2: pd.Timestamp("2018-11-01"), 3: pd.Timestamp("2010-01-04")})
    out, clipped = clip_to_first_trade(spells, first)
    starts = dict(zip(out["listing_id"], out["start_date"], strict=True))
    assert starts[1] == pd.Timestamp("2005-01-01")  # within a week of its first trade: left alone
    assert starts[2] == pd.Timestamp("2018-11-01")  # Arcosa: a member from its first day, not from 2005
    assert 3 not in starts  # a spell that ended before the listing traded at all
    assert starts[4] == pd.Timestamp("2005-01-01")  # no price yet: nothing to clip against
    assert clipped == 2


def test_a_delisted_shares_open_spell_ends_the_day_after_its_last_trade() -> None:
    spells = _spells((1, "2024-10-24", None), (2, "2024-10-24", None), (3, "2024-10-24", "2025-03-01"))
    out, closed = close_at_delisting(spells, pd.Series({1: pd.Timestamp("2024-11-21"), 3: pd.Timestamp("2025-06-30")}))
    ends = dict(zip(out["listing_id"], out["end_date"], strict=True))
    assert ends[1] == pd.Timestamp("2024-11-22")  # Marathon Oil: bought, its last trade on 2024-11-21
    assert missing(ends[2])  # still trading: still a member
    assert ends[3] == pd.Timestamp("2025-03-01")  # a spell that had ended keeps its end
    assert closed == 1


FTSE100, FTSE250, SMALLCAP = 4, 5, 6
PRIORITY = {FTSE100: 0, FTSE250: 1, SMALLCAP: 2}


def _in(*rows: tuple[int, str, str | None]) -> pd.DataFrame:
    """Spells of listing 7: ``(index_id, start, end)``."""
    return pd.DataFrame(
        {
            "index_id": [r[0] for r in rows],
            "listing_id": 7,
            "start_date": pd.to_datetime([r[1] for r in rows]),
            "end_date": pd.to_datetime(pd.Series([r[2] for r in rows], dtype="object")),
            "vendor_id": 1,
        }
    )


def _dates(out: pd.DataFrame) -> list[tuple[int, str, str | None]]:
    return [
        (int(i), pd.Timestamp(s).date().isoformat(), None if missing(e) else pd.Timestamp(e).date().isoformat())
        for i, s, e in zip(out["index_id"], out["start_date"], out["end_date"], strict=True)
    ]


def test_an_inferred_start_follows_the_other_indices_the_listing_left() -> None:
    # Enterprise Inns: three leavers and no joiners, so the replay has it in all three since 2005
    spells = _in(
        (FTSE100, "2005-01-01", "2008-09-22"),
        (FTSE250, "2005-01-01", "2011-09-19"),
        (SMALLCAP, "2005-01-01", "2012-12-24"),
        (FTSE250, "2012-12-24", None),
    )
    out, counts = disjoint(spells, "2005-01-01", {(FTSE250, 7)}, PRIORITY)
    assert _dates(out) == [
        (FTSE100, "2005-01-01", "2008-09-22"),
        (FTSE250, "2008-09-22", "2011-09-19"),
        (FTSE250, "2012-12-24", None),
        (SMALLCAP, "2011-09-19", "2012-12-24"),
    ]
    assert counts == {"moved": 2, "halved": 0, "by_priority": 0}


def test_an_inferred_open_end_closes_when_the_listing_joins_another_index() -> None:
    # joined the SmallCap in 2010 with no leave event since, then joined the FTSE 250 (today's member) in 2015
    out, counts = disjoint(
        _in((SMALLCAP, "2010-03-22", None), (FTSE250, "2015-06-22", None)), "2005-01-01", {(FTSE250, 7)}, PRIORITY
    )
    assert _dates(out) == [(FTSE250, "2015-06-22", None), (SMALLCAP, "2010-03-22", "2015-06-22")]
    assert counts["by_priority"] == 0


def test_a_move_neither_event_dates_is_put_halfway() -> None:
    # Computacenter: joined the FTSE 250 in 2009 with no leave event, in the FTSE 100 today with no join event
    out, counts = disjoint(
        _in((FTSE100, "2005-01-01", None), (FTSE250, "2009-03-23", None), (SMALLCAP, "2007-06-18", "2009-03-23")),
        "2005-01-01",
        {(FTSE100, 7)},
        PRIORITY,
    )
    halfway = pd.Timestamp("2009-03-23") + pd.Timedelta(
        days=(pd.Timestamp.today().normalize() - pd.Timestamp("2009-03-23")).days // 2
    )
    assert _dates(out) == [
        (FTSE100, halfway.date().isoformat(), None),
        (FTSE250, "2009-03-23", halfway.date().isoformat()),
        (SMALLCAP, "2007-06-18", "2009-03-23"),
    ]
    assert counts["by_priority"] == 0 and counts["halved"] > 6000


def test_what_the_events_cannot_settle_goes_to_the_larger_index_and_is_counted() -> None:
    # 3i: a SmallCap leaver in 2011 with no joiner; the FTSE 250 spell beside it has both its events
    out, counts = disjoint(
        _in(
            (FTSE100, "2005-01-01", "2009-03-23"),
            (FTSE100, "2009-06-22", "2011-09-19"),
            (FTSE250, "2009-03-23", "2009-06-22"),
            (FTSE250, "2011-09-19", "2014-06-23"),
            (SMALLCAP, "2005-01-01", "2011-09-29"),
        ),
        "2005-01-01",
        set(),
        PRIORITY,
    )
    assert _dates(out) == [
        (FTSE100, "2005-01-01", "2009-03-23"),
        (FTSE100, "2009-06-22", "2011-09-19"),
        (FTSE250, "2009-03-23", "2009-06-22"),
        (FTSE250, "2011-09-19", "2014-06-23"),
    ]
    assert counts["by_priority"] == 10  # the SmallCap's last ten days, inside the FTSE 250 spell


def test_a_listing_in_one_index_and_a_member_today_are_left_alone() -> None:
    spells = _in((FTSE250, "2005-01-01", None))
    out, counts = disjoint(spells, "2005-01-01", {(FTSE250, 7)}, PRIORITY)
    assert _dates(out) == [(FTSE250, "2005-01-01", None)] and counts == {"moved": 0, "halved": 0, "by_priority": 0}
