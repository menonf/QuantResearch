"""Who is who: resolve or create issuers and listings, and record their identifiers.

    ids = ensure_listings(conn, frame, anchor_type="RIC", issuer_anchor_type="CUSIP", vendor="refinitiv", batch_id=b)

``frame`` has one row per listing: ``anchor`` (the listing's code in ``anchor_type``, e.g. the vendor's RIC),
``issuer_anchor`` (the issuer's code in ``issuer_anchor_type``, e.g. the 6-character CUSIP issuer code; NULL for
an instrument without an issuer such as an index level), ``issuer_name`` and the ``core.listing`` attributes.

* a listing whose anchor is known is updated in place (name, dates...); an unknown one is created;
* an issuer is found by its anchor or created - listings sharing an issuer anchor share one issuer
  (GOOG and GOOGL, BRK.A and BRK.B);
* identifiers are **insert-only**: a code already held by another entity is reported, never reassigned;
* provenance: ``batch_id`` names the raw batch of every row written - or, when ``frame`` has a ``batch_id`` column,
  each row's own (a load that read its listings from several batches).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection

from research_core.data.db import schema
from research_core.data.db.schema import vendor_id
from research_core.data.db.upsert import insert_returning, upsert

LISTING_ATTRIBUTES = (
    "name",
    "instrument_type",
    "share_class",
    "exchange_mic",
    "currency",
    "listed_on",
    "delisted_on",
)
IDENTIFIER_KEY = ["id_type", "value", "valid_from", "entity_type", "market"]
OPEN_START = pd.Timestamp("1900-01-01")
#: The US exchanges (ISO 10383): where the tickers of the US-only vendors (SEC, Norgate, SimFin, FinanceDatabase,
#: Tiingo) trade.
US_MARKETS = frozenset({"XNYS", "XNAS", "XASE", "ARCX", "BATS"})
#: Exchange -> the market its tickers are unique in: a country's exchanges share one set of tickers. A ticker of a
#: listing on another exchange (or none recorded) is unique only with ``''``, as codes the same everywhere are.
TICKER_MARKETS = {**dict.fromkeys(US_MARKETS, "US"), "XLON": "GB"}


@dataclass
class IdentityResult:
    """Anchor -> listing id, and what was created."""

    listing_ids: dict[str, int]
    listings_created: int = 0
    issuers_created: int = 0


def read_identifiers(conn: Connection, entity_type: str, id_type: str) -> pd.DataFrame:
    """``[entity_id, value, valid_from, valid_to]`` for one identifier type."""
    t = schema.identifier
    rows = conn.execute(
        sa.select(t.c.entity_id, t.c.value, t.c.valid_from, t.c.valid_to).where(
            t.c.entity_type == entity_type, t.c.id_type == id_type
        )
    ).all()
    return pd.DataFrame(rows, columns=["entity_id", "value", "valid_from", "valid_to"])


def anchor_map(conn: Connection, entity_type: str, id_type: str) -> dict[str, int]:
    """``{code: entity id}`` for an anchor identifier type (codes valid from the open start)."""
    df = read_identifiers(conn, entity_type, id_type)
    return {str(v): int(e) for v, e in zip(df["value"], df["entity_id"], strict=True)}


def add_identifiers(conn: Connection, rows: pd.DataFrame, *, batch_id: int | None) -> pd.DataFrame:
    """Insert ``[entity_type, entity_id, id_type, value, valid_from, valid_to, vendor_id]`` rows that are new.

    Returns the conflicts: rows whose (type, value, valid_from, market) already belongs to a *different* entity. A
    ticker's market is its listing's (:data:`TICKER_MARKETS`), so London's AAL and Nasdaq's AAL are two codes.
    """
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["valid_from"] = pd.to_datetime(rows["valid_from"]).fillna(OPEN_START)
    rows["market"] = _markets(conn, rows)
    rows = rows.dropna(subset=["value"]).drop_duplicates(IDENTIFIER_KEY, keep="first")
    t = schema.identifier
    existing = pd.DataFrame(
        conn.execute(
            sa.select(t.c.entity_type, t.c.entity_id, t.c.id_type, t.c.value, t.c.valid_from, t.c.market).where(
                t.c.id_type.in_(rows["id_type"].unique().tolist())
            )
        ).all(),
        columns=["entity_type", "held_by", "id_type", "value", "valid_from", "market"],
    )
    if not existing.empty:
        existing["valid_from"] = pd.to_datetime(existing["valid_from"])
        merged = rows.merge(existing, on=IDENTIFIER_KEY, how="left")
        conflicts = merged[merged["held_by"].notna() & (merged["held_by"] != merged["entity_id"])]
    else:
        conflicts = rows.iloc[0:0].assign(held_by=pd.Series(dtype="Int64"))
    upsert(conn, schema.identifier, rows, key=IDENTIFIER_KEY, batch_id=batch_id, insert_only=True)
    return conflicts.reset_index(drop=True)


def _markets(conn: Connection, rows: pd.DataFrame) -> pd.Series:
    """Each row's market: a listing's ticker takes its listing's (:data:`TICKER_MARKETS`), every other code ``''``."""
    ticker = (rows["id_type"] == "TICKER") & (rows["entity_type"] == "listing")
    if not ticker.any():
        return pd.Series("", index=rows.index)
    t = schema.listing
    mics = dict(conn.execute(sa.select(t.c.listing_id, t.c.exchange_mic)).all())  # all: a few thousand, one query
    market = rows["entity_id"].map(lambda i: TICKER_MARKETS.get(str(mics.get(int(i)))) if pd.notna(i) else None)
    return market.where(ticker, None).fillna("").astype(str)


def ensure_issuers(
    conn: Connection, frame: pd.DataFrame, *, anchor_type: str, vendor: str, batch_id: int | None
) -> tuple[dict[str, int], int]:
    """``{issuer anchor: issuer_id}`` for ``frame[issuer_anchor, issuer_name, issuer_domicile]``; creates new ones."""
    known = anchor_map(conn, "issuer", anchor_type)
    wanted = frame.dropna(subset=["issuer_anchor"]).drop_duplicates("issuer_anchor")
    new = wanted[~wanted["issuer_anchor"].astype(str).isin(known)]
    batches = _batches(new, batch_id)
    created = insert_returning(
        conn,
        schema.issuer,
        pd.DataFrame(
            {
                "anchor": new["issuer_anchor"].astype(str),
                "name": new["issuer_name"].fillna(new["issuer_anchor"]).astype(str).str[:256],
                "domicile_country": new.get("issuer_domicile"),
                "batch_id": batches,
            }
        ),
        anchor="anchor",
        id_col="issuer_id",
    )
    if created:
        batch_of = dict(zip(new["issuer_anchor"].astype(str), batches, strict=True))
        add_identifiers(
            conn,
            pd.DataFrame(
                {
                    "entity_type": "issuer",
                    "entity_id": list(created.values()),
                    "id_type": anchor_type,
                    "value": list(created.keys()),
                    "valid_from": OPEN_START,
                    "valid_to": None,
                    "vendor_id": vendor_id(vendor),
                    "batch_id": [batch_of[a] for a in created],
                }
            ),
            batch_id=None,
        )
    return {**known, **created}, len(created)


def _batches(frame: pd.DataFrame, batch_id: int | None) -> pd.Series:
    """Each row's raw batch: the frame's own ``batch_id`` column where it has one, else ``batch_id``."""
    if "batch_id" in frame.columns:
        return frame["batch_id"].where(frame["batch_id"].notna(), batch_id).astype("Int64")
    return pd.Series(batch_id, index=frame.index, dtype="Int64")


def _children(parent: sa.Table) -> list[tuple[sa.Table, str]]:
    """Every ``(table, column)`` with a foreign key to ``parent``, in every asset class's tables (new ones covered).

    An issuer may have shares and bonds: a merge run from either asset class's command must move both kinds of rows.
    """
    schema.load_asset_classes()
    out = []
    for table in schema.metadata.sorted_tables:
        for fk in table.foreign_keys:
            if fk.column.table is parent:
                out.append((table, fk.parent.name))
    return out


def _merge(conn: Connection, parent: sa.Table, entity_type: str, survivor: int, others: list[int]) -> None:
    """Move every row pointing at ``others`` to ``survivor``, then delete ``others`` from ``parent``.

    A row that would duplicate one the survivor already has (the same filing, the same day's price) is dropped first;
    the entities' identifiers move with them.
    """
    if not others:
        return
    key = next(iter(parent.primary_key.columns)).name
    expanding: sa.BindParameter[list[int]] = sa.bindparam("o", expanding=True)
    params = {"s": survivor, "o": others}
    for table, col in _children(parent):
        target = f"[{table.schema}].[{table.name}]"
        rest = [c.name for c in table.primary_key.columns if c.name != col]
        if col in table.primary_key.columns and rest:
            same = " AND ".join(f"s.[{c}] = o.[{c}]" for c in rest)
            conn.execute(
                sa.text(
                    f"DELETE o FROM {target} o WHERE o.[{col}] IN :o "
                    f"AND EXISTS (SELECT 1 FROM {target} s WHERE s.[{col}] = :s AND {same})"
                ).bindparams(expanding),
                params,
            )
        # a moved row is new to readers that sync by loaded_at (the research cache)
        stamp = ", [loaded_at] = SYSUTCDATETIME()" if "loaded_at" in table.c else ""
        conn.execute(
            sa.text(f"UPDATE {target} SET [{col}] = :s{stamp} WHERE [{col}] IN :o").bindparams(expanding), params
        )
    conn.execute(
        sa.text(
            "UPDATE core.identifier SET entity_id = :s, loaded_at = SYSUTCDATETIME() "
            "WHERE entity_type = :t AND entity_id IN :o"
        ).bindparams(expanding),
        {**params, "t": entity_type},
    )
    conn.execute(
        sa.text(f"DELETE FROM [{parent.schema}].[{parent.name}] WHERE [{key}] IN :o").bindparams(expanding), params
    )


def merge_issuers(conn: Connection, survivor: int, others: list[int]) -> None:
    """Fold ``others`` into ``survivor``: one company recorded twice (two CIK-sharing issuers, say)."""
    _merge(conn, schema.issuer, "issuer", survivor, others)


def merge_listings(conn: Connection, survivor: int, others: list[int]) -> None:
    """Fold ``others`` into ``survivor``: one share line recorded under two codes (a RIC that changed).

    Prices, actions, share counts, memberships and codes move; on a day both had a price, the survivor's stays.
    """
    _merge(conn, schema.listing, "listing", survivor, others)


def ensure_listings(
    conn: Connection,
    frame: pd.DataFrame,
    *,
    anchor_type: str,
    issuer_anchor_type: str,
    vendor: str,
    batch_id: int | None,
) -> IdentityResult:
    """Resolve or create the listings of ``frame`` (see the module docstring)."""
    frame = frame.assign(anchor=frame["anchor"].astype(str))
    issuers, issuers_created = ensure_issuers(
        conn, frame, anchor_type=issuer_anchor_type, vendor=vendor, batch_id=batch_id
    )
    frame = frame.assign(issuer_id=frame["issuer_anchor"].map(lambda a: issuers.get(str(a)) if pd.notna(a) else None))
    attrs = [c for c in LISTING_ATTRIBUTES if c in frame.columns]
    known = anchor_map(conn, "listing", anchor_type)
    frame = frame.assign(batch_id=_batches(frame, batch_id))
    new = frame[~frame["anchor"].isin(known)]
    created = insert_returning(
        conn,
        schema.listing,
        new[["anchor", "issuer_id", *attrs, "batch_id"]],
        anchor="anchor",
        id_col="listing_id",
    )
    if created:
        batch_of = dict(zip(new["anchor"], new["batch_id"], strict=True))
        add_identifiers(
            conn,
            pd.DataFrame(
                {
                    "entity_type": "listing",
                    "entity_id": list(created.values()),
                    "id_type": anchor_type,
                    "value": list(created.keys()),
                    "valid_from": OPEN_START,
                    "valid_to": None,
                    "vendor_id": vendor_id(vendor),
                    "batch_id": [batch_of[a] for a in created],
                }
            ),
            batch_id=None,
        )
    existing = frame[frame["anchor"].isin(known)]
    if not existing.empty:  # attributes may have changed (a name, a delisting date)
        rows = existing.assign(listing_id=existing["anchor"].map(known))[
            ["listing_id", "issuer_id", *attrs, "batch_id"]
        ]
        # a listing whose company this call cannot name (no code of that kind) keeps the company it has: writing NULL
        # would only be undone by the call that names it by another code - two writes a run, every run
        named = rows["issuer_id"].notna()
        if named.any():
            upsert(conn, schema.listing, rows[named], key=["listing_id"])
        if not named.all():
            upsert(conn, schema.listing, rows[~named].drop(columns=["issuer_id"]), key=["listing_id"])
    return IdentityResult({**known, **created}, listings_created=len(created), issuers_created=issuers_created)


#: The codes a new vendor's securities are matched on, most reliable first: an ISIN or CUSIP names one security; a
#: ticker only on a given day (tickers are reused).
MATCH_ORDER = ("ISIN", "CUSIP", "TICKER")


def listings_on(markets: Collection[str]) -> sa.ColumnElement[bool]:
    """``core.listing`` rows a ticker of ``markets`` can name: on one of them, or on a market not recorded.

    A ticker names a share only within its market - London's ``AAL`` is Anglo American, Nasdaq's American Airlines.
    A listing whose market is unknown (LSEG's ``.TXS`` suffix has no code yet) stays matchable, as before markets
    were checked; one known to trade elsewhere never is.
    """
    t = schema.listing
    return sa.or_(t.c.exchange_mic.in_(sorted(markets)), t.c.exchange_mic.is_(None))


def match_listings(
    conn: Connection,
    codes: pd.DataFrame,
    *,
    markets: Collection[str] | None,
    on: dt.date | None = None,
    order: tuple[str, ...] = MATCH_ORDER,
) -> pd.DataFrame:
    """``[key, listing_id, matched_by, ambiguous]``: which of our listings each of a vendor's securities is.

    ``codes`` has a ``key`` (the vendor's own symbol) and any of the ``order`` columns. A code counts when it was valid
    on ``on`` (default today) - or on the row's own day when ``codes`` has an ``on`` column (a delisted company's
    ticker on its last day: tickers are reused). The first code that names exactly one listing wins. A code naming
    several listings that day matches nothing and is flagged ``ambiguous`` - for a person to look at, never guessed.
    ``markets`` are the exchanges the vendor's tickers trade on (:func:`listings_on`); an ISIN or CUSIP is the same
    everywhere. ``None`` lets a ticker match any market - only for a vendor whose tickers already say theirs.
    """
    default = pd.Timestamp(on or dt.date.today())
    days = (
        pd.to_datetime(codes["on"]).fillna(default) if "on" in codes.columns else pd.Series(default, index=codes.index)
    )
    out = pd.DataFrame({"key": codes["key"], "listing_id": pd.NA, "matched_by": None, "ambiguous": False})
    out = out.astype({"listing_id": "Int64"})
    allowed = (
        None
        if markets is None
        else set(conn.execute(sa.select(schema.listing.c.listing_id).where(listings_on(markets))).scalars())
    )
    for id_type in order:
        if id_type not in codes.columns:
            continue
        todo = out["listing_id"].isna() & ~out["ambiguous"]
        want = pd.DataFrame({"row": codes.index[todo], "value": codes.loc[todo, id_type], "day": days[todo]})
        want = want.dropna(subset=["value"]).astype({"value": str})
        held = read_identifiers(conn, "listing", id_type)
        if id_type == "TICKER" and allowed is not None:
            held = held[held["entity_id"].isin(allowed)]
        hits = want.merge(held, on="value")
        hits = hits[
            (pd.to_datetime(hits["valid_from"]) <= hits["day"])
            & (hits["valid_to"].isna() | (pd.to_datetime(hits["valid_to"]) > hits["day"]))
        ]
        owners = hits.groupby("row")["entity_id"].agg(["nunique", "first"])
        one = owners[owners["nunique"] == 1]
        out.loc[one.index, "listing_id"] = one["first"].astype(int).to_numpy()
        out.loc[one.index, "matched_by"] = id_type
        out.loc[owners.index[owners["nunique"] > 1], "ambiguous"] = True
    return out
