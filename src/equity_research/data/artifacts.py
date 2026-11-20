"""The equity research's artifact store: :mod:`research_core.data.artifact_store` bound to how equities derive numbers.

    snapshot = data_snapshot(cache)                                   # the data version: the latest row synced
    frame = cached("labels", "fwd_ret_21d", config, build, snapshot=snapshot)    # read back, or built and saved
    meta = sidecar(artifact_path("labels", "fwd_ret_21d", config, snapshot))     # config, snapshot, code, when

Every key carries the returns method (:data:`~equity_research.data.panels.RETURNS_VERSION`), so a change to how
returns are derived rebuilds every artifact. Filing text has a data version of its own (:data:`SEPARATE_TABLES`).
"""

from __future__ import annotations

from equity_research import ASSET_CLASS
from equity_research.data.panels import RETURNS_VERSION
from research_core.data.artifact_store import ArtifactStore, sidecar

#: Tables with a data version of their own: loading filing text does not make every price- and accounts-based
#: artifact stale. Research that reads them keys its artifacts by both versions.
SEPARATE_TABLES = frozenset({"equity.document", "equity.text_measure"})

#: The store every equity artifact goes through: its keys name the returns method.
STORE = ArtifactStore(ASSET_CLASS, method={"returns": RETURNS_VERSION}, separate_tables=SEPARATE_TABLES)

data_snapshot = STORE.data_snapshot
artifact_path = STORE.artifact_path
save = STORE.save
load = STORE.load
cached = STORE.cached

__all__ = ["SEPARATE_TABLES", "STORE", "artifact_path", "cached", "data_snapshot", "load", "save", "sidecar"]
