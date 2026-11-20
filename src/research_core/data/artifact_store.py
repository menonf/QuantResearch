"""The artifact store: research results kept as Parquet, each beside a sidecar saying what made it.

    store = ArtifactStore("equity", method={"returns": 2})            # one per asset class, bound to its methods
    snapshot = store.data_snapshot(cache)                             # the data version: the latest row synced
    frame = store.cached("labels", "fwd_ret_21d", config, build, snapshot=snapshot)    # read back, or built and saved
    meta = sidecar(store.artifact_path("labels", "fwd_ret_21d", config, snapshot))     # config, snapshot, code, when

Labels, features, predictions and baselines take minutes to build and are used many times, so each is computed once
per configuration and data version and kept under
``data/<db name>/research/<asset class>/<kind>/<name>-<hash>.parquet`` - each asset class in a folder of its own:

1. **Key** - the hash is of the configuration, the data snapshot (the latest ``loaded_at`` the research cache has
   synced) and the store's ``method`` - the versions of how an asset class derives its numbers (the equity
   research's returns method) - so a new configuration, new data or a new way of deriving the numbers is a new file:
   an old file is never read for them.
2. **Sidecar** - ``<name>-<hash>.json`` records the kind, name, configuration, data snapshot, method versions, code
   version (the git commit) and when it was written: what the book's ``.digest`` files record, readable.
3. **Columns** - Parquet needs text column names; a frame whose columns are instrument ids (integers) is written
   with them as text and read back as integers, as the sidecar says.

``data/`` is not in git: artifacts are rebuilt, never shared.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from research_core.data.cache import ResearchCache
from research_core.data.raw_store import code_version
from research_core.settings import get_settings


def config_hash(config: Mapping[str, Any]) -> str:
    """16-character hash of a config (key order does not matter)."""
    canonical = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def sidecar(path: Path) -> dict[str, Any]:
    """The sidecar of the artifact at ``path``: what made it."""
    result: dict[str, Any] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return result


@dataclass(frozen=True)
class ArtifactStore:
    """An asset class's store, whose keys carry ``method`` and whose data version leaves out ``separate_tables``.

    ``method`` names the versions of how an asset class derives its numbers (``{"returns": 2}``): each is part of
    every key and is written to the sidecar as ``<name>_version``. ``separate_tables`` have a data version of their
    own: loading them does not make every other artifact stale, and research that reads them keys its artifacts by
    both versions.
    """

    asset_class: str
    method: Mapping[str, Any] = field(default_factory=dict)
    separate_tables: frozenset[str] = frozenset()

    def data_snapshot(self, cache: ResearchCache, tables: Collection[str] | None = None) -> str | None:
        """The data version of ``cache``: the latest ``loaded_at`` it has synced of its tables (ISO), ``None`` if empty.

        By default of every table but ``separate_tables``; ``tables`` names the tables to version instead.
        """
        manifest = cache.manifest()
        names = [t for t in manifest if t not in self.separate_tables] if tables is None else list(tables)
        stamps = [manifest[t] for t in names if manifest.get(t)]
        return max(stamps) if stamps else None

    def artifact_path(
        self, kind: str, name: str, config: Mapping[str, Any], snapshot: str | None = None, root: Path | None = None
    ) -> Path:
        """Where the artifact ``kind``/``name`` of ``config`` on data ``snapshot`` is kept (existing or not)."""
        key = config_hash({"kind": kind, "name": name, "config": dict(config), "snapshot": snapshot, **self.method})
        return (root or get_settings().data_dir / "research" / self.asset_class) / kind / f"{name}-{key}.parquet"

    def save(
        self,
        frame: pd.DataFrame,
        kind: str,
        name: str,
        config: Mapping[str, Any],
        *,
        snapshot: str | None = None,
        root: Path | None = None,
    ) -> Path:
        """Write ``frame`` and its sidecar; returns the Parquet path."""
        path = self.artifact_path(kind, name, config, snapshot, root)
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
            **{f"{key}_version": version for key, version in self.method.items()},
            "code_version": code_version(),
            "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "rows": len(frame),
            "integer_columns": integer_columns,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return path

    def load(
        self, kind: str, name: str, config: Mapping[str, Any], *, snapshot: str | None = None, root: Path | None = None
    ) -> pd.DataFrame | None:
        """The artifact saved for exactly this ``config`` and ``snapshot``, or ``None``."""
        path = self.artifact_path(kind, name, config, snapshot, root)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        if sidecar(path).get("integer_columns"):
            frame.columns = [int(c) for c in frame.columns]
        return frame

    def cached(
        self,
        kind: str,
        name: str,
        config: Mapping[str, Any],
        build: Callable[[], pd.DataFrame],
        *,
        snapshot: str | None = None,
        root: Path | None = None,
    ) -> pd.DataFrame:
        """The stored artifact, or ``build()``'s result saved for next time."""
        stored = self.load(kind, name, config, snapshot=snapshot, root=root)
        if stored is not None:
            return stored
        frame = build()
        self.save(frame, kind, name, config, snapshot=snapshot, root=root)
        return frame
