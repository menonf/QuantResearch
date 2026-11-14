"""The artifact store: research results kept as Parquet, each beside a sidecar saying what made it.

    snapshot = data_snapshot(cache)                                   # the data version: the latest row synced
    frame = cached("labels", "fwd_ret_21d", config, build, snapshot=snapshot)    # read back, or built and saved
    meta = sidecar(artifact_path("labels", "fwd_ret_21d", config, snapshot))     # config, snapshot, code, when

Labels, features, predictions and baselines take minutes to build and are used many times, so each is computed once
per configuration and data version and kept under ``data/<db name>/research/<kind>/<name>-<hash>.parquet``:

1. **Key** - the hash is of the configuration, the data snapshot (the latest ``loaded_at`` the research cache has
   synced) and the returns method (:data:`~equity_research.research.panels.RETURNS_VERSION`), so a new
   configuration, new data or a new way of deriving returns is a new file: an old file is never read for them.
2. **Sidecar** - ``<name>-<hash>.json`` records the kind, name, configuration, data snapshot, code version (the git
   commit) and when it was written: what the book's ``.digest`` files record, readable.
3. **Columns** - Parquet needs text column names; a frame whose columns are listing ids (integers) is written with
   them as text and read back as integers, as the sidecar says.

``data/`` is not in git: artifacts are rebuilt, never shared.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from equity_research.raw.store import code_version
from equity_research.research.cache import ResearchCache
from equity_research.research.panels import RETURNS_VERSION
from equity_research.research.registry import config_hash
from equity_research.settings import get_settings

#: Tables with a data version of their own: loading filing text does not make every price- and accounts-based
#: artifact stale. Research that reads them keys its artifacts by both versions.
SEPARATE_TABLES = frozenset({"fund.document", "fund.text_measure"})


def data_snapshot(cache: ResearchCache, tables: Collection[str] | None = None) -> str | None:
    """The data version of ``cache``: the latest ``loaded_at`` it has synced of its tables (ISO), ``None`` if empty.

    By default of every table but :data:`SEPARATE_TABLES`; ``tables`` names the tables to version instead.
    """
    manifest = cache.manifest()
    names = [t for t in manifest if t not in SEPARATE_TABLES] if tables is None else list(tables)
    stamps = [manifest[t] for t in names if manifest.get(t)]
    return max(stamps) if stamps else None


def artifact_path(
    kind: str, name: str, config: Mapping[str, Any], snapshot: str | None = None, root: Path | None = None
) -> Path:
    """Where the artifact ``kind``/``name`` of ``config`` on data ``snapshot`` is kept (whether or not it exists)."""
    key = config_hash(
        {"kind": kind, "name": name, "config": dict(config), "snapshot": snapshot, "returns": RETURNS_VERSION}
    )
    return (root or get_settings().data_dir / "research") / kind / f"{name}-{key}.parquet"


def sidecar(path: Path) -> dict[str, Any]:
    """The sidecar of the artifact at ``path``: what made it."""
    result: dict[str, Any] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return result


def save(
    frame: pd.DataFrame,
    kind: str,
    name: str,
    config: Mapping[str, Any],
    *,
    snapshot: str | None = None,
    root: Path | None = None,
) -> Path:
    """Write ``frame`` and its sidecar; returns the Parquet path."""
    path = artifact_path(kind, name, config, snapshot, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    integer_columns = len(frame.columns) > 0 and pd.api.types.is_integer_dtype(frame.columns)
    out = frame.copy()
    out.columns = [str(c) for c in frame.columns]
    out.to_parquet(path)
    meta = {
        "kind": kind,
        "name": name,
        "config": dict(config),
        "data_snapshot": snapshot,
        "returns_version": RETURNS_VERSION,
        "code_version": code_version(),
        "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": len(frame),
        "integer_columns": integer_columns,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load(
    kind: str, name: str, config: Mapping[str, Any], *, snapshot: str | None = None, root: Path | None = None
) -> pd.DataFrame | None:
    """The artifact saved for exactly this ``config`` and ``snapshot``, or ``None``."""
    path = artifact_path(kind, name, config, snapshot, root)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if sidecar(path).get("integer_columns"):
        frame.columns = [int(c) for c in frame.columns]
    return frame


def cached(
    kind: str,
    name: str,
    config: Mapping[str, Any],
    build: Callable[[], pd.DataFrame],
    *,
    snapshot: str | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """The stored artifact, or ``build()``'s result saved for next time."""
    stored = load(kind, name, config, snapshot=snapshot, root=root)
    if stored is not None:
        return stored
    frame = build()
    save(frame, kind, name, config, snapshot=snapshot, root=root)
    return frame
