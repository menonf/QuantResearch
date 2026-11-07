"""Index membership intervals made consistent, whichever vendor replayed them. Pure functions.

* :func:`coalesce` - spells of one listing that overlap or touch become one. Two codes of one share line (a RIC
  before and after a rename) each bring their own spell; once both name the same listing they are one membership.
* :func:`disjoint` - a listing is in one of a market's indices at a time (the S&P 500, 400 and 600 never share a
  member, nor the FTSE 100, 250 and SmallCap). A replay leaves some in two: a leaver with no join event is dated a
  member since the window's start, and a joiner with no leave event a member until today. What a vendor's events and
  today's member list say is kept; a date the replay inferred gives way to it.
* :func:`clip_to_first_trade` - no spell starts before the listing first traded. A vendor that replays today's
  members backwards dates a member with no join event to the start of its window: for a company that did not exist
  yet, the first day it traded is the earliest it can have joined.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

KEY = ["index_id", "listing_id"]
#: a spell is clipped only when it starts this much before the first trade (a listing's first bar can lag a day or two)
CLIP_TOLERANCE_DAYS = 7


def coalesce(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` (``index_id, listing_id, start_date, end_date`` + other columns) with overlapping spells merged.

    ``end_date`` is exclusive and NaT means still a member; the merged spell keeps the other columns of its first row.
    """
    if rows.empty:
        return rows
    df = rows.assign(
        start_date=pd.to_datetime(rows["start_date"]),
        end_date=pd.to_datetime(rows["end_date"]),
    ).sort_values([*KEY, "start_date"], kind="stable")
    far = pd.Timestamp.max.normalize()
    end = df["end_date"].fillna(far)
    # the latest end of the earlier spells of the same listing: a spell starting after it opens a new membership
    reach = end.groupby([df[k] for k in KEY]).transform(lambda s: s.cummax().shift(1))
    new = reach.isna() | (df["start_date"] > reach)
    group = new.cumsum()
    out = df.groupby(group, sort=False).agg({c: "first" for c in df.columns if c != "end_date"})
    out["end_date"] = df.groupby(group, sort=False)["end_date"].agg(_last_end)
    return out.reset_index(drop=True)[rows.columns]


def _last_end(ends: pd.Series) -> pd.Timestamp | NaTType:
    """The merged spell's end: the latest, or NaT (still a member) when any spell is still open."""
    return pd.Timestamp(ends.max()) if ends.notna().all() else pd.NaT


@dataclass
class _Spell:
    row: int
    index_id: int
    start: pd.Timestamp
    end: pd.Timestamp | None
    start_inferred: bool  # no join event: dated to the replay window's start
    end_inferred: bool  # no leave event, and not a member today
    pieces: list[tuple[pd.Timestamp, pd.Timestamp | None]] = field(default_factory=list)


def disjoint(
    rows: pd.DataFrame,
    origin: str | pd.Timestamp,
    members_today: set[tuple[int, int]],
    priority: Mapping[int, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """``(rows, counts)``: coalesced spells of one market's indices, a listing in at most one of them on any day.

    ``origin`` is the replay window's start (a spell starting there had no join event); ``members_today`` the
    ``(index_id, listing_id)`` in today's member lists (an open spell of one ended with no leave event only if it is
    not). Where a listing's spells in two indices overlap:

    1. a start the replay inferred moves to the latest end of the listing's other spells before its own end - it
       joined after it left them (Enterprise Inns: FTSE 100 until 2008-09, then 250 until 2011-09, then SmallCap);
    2. an inferred open end moves to the earliest start of its other spells after its own start - it left when it
       joined them;
    3. an inferred start against an inferred end - one move neither event dates (Computacenter, from the FTSE 250
       in 2009 to the 100 it is in today) - meet halfway through their overlap, and the days are counted;
    4. what is left (two events contradicting) goes to the index first in ``priority`` (0 first), and is counted.

    ``counts``: ``moved`` spells whose dates the events changed, ``halved`` and ``by_priority`` listing-days settled
    by rules 3 and 4.
    """
    if rows.empty:
        return rows, {"moved": 0, "halved": 0, "by_priority": 0}
    df = rows.assign(start_date=pd.to_datetime(rows["start_date"]), end_date=pd.to_datetime(rows["end_date"]))
    df = df.reset_index(drop=True)
    first = pd.Timestamp(origin)
    moved = halved = by_priority = 0
    out: list[pd.DataFrame] = []
    for _, group in df.groupby("listing_id", sort=False):
        listing_id = int(group["listing_id"].iloc[0])
        if group["index_id"].nunique() < 2:
            out.append(group)
            continue
        spells = [
            _Spell(
                row=int(i),
                index_id=int(ix),
                start=pd.Timestamp(s),
                end=None if pd.isna(e) else pd.Timestamp(e),
                start_inferred=pd.Timestamp(s) <= first,
                end_inferred=pd.isna(e) and (int(ix), listing_id) not in members_today,
            )
            for i, ix, s, e in zip(
                group.index.tolist(), group["index_id"].tolist(), group["start_date"], group["end_date"], strict=True
            )
        ]
        before = [(sp.start, sp.end) for sp in spells]
        _follow_events(spells)
        moved += sum(1 for sp, was in zip(spells, before, strict=True) if (sp.start, sp.end) != was)
        halved += _halve(spells)
        by_priority += _by_priority(spells, priority)
        out += [_dated(group.loc[[sp.row]], s, e) for sp in spells for s, e in sp.pieces]
    result = pd.concat(out, ignore_index=True) if out else df.iloc[0:0]
    return result.sort_values([*KEY, "start_date"], kind="stable").reset_index(drop=True)[rows.columns], {
        "moved": moved,
        "halved": halved,
        "by_priority": by_priority,
    }


#: an open spell's days are counted to this day
_TODAY = pd.Timestamp.today().normalize()


def _follow_events(spells: list[_Spell]) -> None:
    """Rules 1 and 2: inferred starts and open ends moved to what the listing's other spells' dates say."""
    for _ in range(len(spells) + 1):  # each pass moves dates towards the events; a few settle them
        changed = False
        for a in spells:
            others = [b for b in spells if b.index_id != a.index_id]
            if a.start_inferred:
                ends = [b.end for b in others if b.end is not None and b.end > a.start and _before(b.end, a.end)]
                if ends:
                    a.start, changed = max(ends), True
            if a.end_inferred:
                starts = [b.start for b in others if b.start > a.start]
                if starts and (a.end is None or min(starts) < a.end):
                    a.end, changed = min(starts), True
        if not changed:
            return


def _halve(spells: list[_Spell]) -> int:
    """Rule 3: an inferred start and another index's inferred end meet halfway; the days they shared."""
    days = 0
    for a in (sp for sp in spells if sp.start_inferred):
        for b in (sp for sp in spells if sp.end_inferred and sp.index_id != a.index_id):
            begin = max(a.start, b.start)
            stop = min(_TODAY if a.end is None else a.end, _TODAY if b.end is None else b.end)
            if begin < stop:
                days += (stop - begin).days
                a.start = b.end = begin + pd.Timedelta(days=(stop - begin).days // 2)
    return days


def _by_priority(spells: list[_Spell], priority: Mapping[int, int]) -> int:
    """Rule 4: each spell's ``pieces`` - its days no higher-priority spell has; the days given away."""
    days = 0
    taken: list[tuple[pd.Timestamp, pd.Timestamp | None]] = []
    for sp in sorted(spells, key=lambda sp: priority.get(sp.index_id, len(priority))):
        whole = [(sp.start, sp.end)] if sp.end is None or sp.end > sp.start else []
        pieces = whole
        for t in taken:
            pieces = [q for piece in pieces for q in _minus(piece, t)]
        days += sum(_days(*q) for q in whole) - sum(_days(*q) for q in pieces)
        sp.pieces = pieces
        taken += pieces
    return days


def _dated(row: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    out = row.copy()
    out["start_date"] = start
    out["end_date"] = pd.NaT if end is None else end
    return out


def _before(end: pd.Timestamp, other_end: pd.Timestamp | None) -> bool:
    return other_end is None or end <= other_end


def _days(start: pd.Timestamp, end: pd.Timestamp | None) -> int:
    return max(0, ((end if end is not None else _TODAY) - start).days)


def _minus(
    piece: tuple[pd.Timestamp, pd.Timestamp | None], taken: tuple[pd.Timestamp, pd.Timestamp | None]
) -> list[tuple[pd.Timestamp, pd.Timestamp | None]]:
    """``piece`` without the days of ``taken`` (ends exclusive, None open): nothing, one piece or two."""
    (s, e), (ts, te) = piece, taken
    if (te is not None and te <= s) or (e is not None and e <= ts):
        return [piece]  # no overlap
    out: list[tuple[pd.Timestamp, pd.Timestamp | None]] = []
    if s < ts:
        out.append((s, ts))
    if te is not None and (e is None or te < e):
        out.append((te, e))
    return out


def clip_to_first_trade(rows: pd.DataFrame, first_trade: pd.Series) -> tuple[pd.DataFrame, int]:
    """``(rows, spells clipped)``: each spell starting no earlier than its listing's first trade.

    ``first_trade``: ``listing_id`` -> day. A listing with no price yet is left as it is; a spell that would end
    before it starts is dropped.
    """
    first = pd.to_datetime(rows["listing_id"].map(first_trade))
    start = pd.to_datetime(rows["start_date"])
    early = first.notna() & (start < first - pd.Timedelta(days=CLIP_TOLERANCE_DAYS))
    out = rows.assign(start_date=start.where(~early, first))
    end = pd.to_datetime(out["end_date"])
    keep = end.isna() | (end > out["start_date"])
    return out[keep].reset_index(drop=True), int(early.sum())


def close_at_delisting(rows: pd.DataFrame, last_day: pd.Series) -> tuple[pd.DataFrame, int]:
    """``(rows, spells closed)``: an open spell of a listing that stopped trading ends the day after its last trade.

    ``last_day``: ``listing_id`` -> its last trading day, for delisted listings only. A vendor whose membership is a
    daily 1 / 0 series (Norgate) stops the series on a delisted share's last day, still at 1 - the spell would stay
    open for ever and count a company bought years ago as a member today.
    """
    end = pd.to_datetime(rows["end_date"])
    last = pd.to_datetime(rows["listing_id"].map(last_day))
    shut = end.isna() & last.notna()
    out = rows.assign(end_date=end.where(~shut, last + pd.Timedelta(days=1)))
    return out, int(shut.sum())
