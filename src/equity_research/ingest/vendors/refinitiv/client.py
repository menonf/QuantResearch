"""Requests to LSEG through a running LSEG Workspace (``lseg-data``), with the habits LSEG needs.

* **One session**, opened on the first request, pinned to the port Workspace records in ``.portInUse`` (the library
  otherwise probes ports 9000 / 9060, which another program may hold).
* **Transient errors** (gateway timeouts) are retried with a growing pause.
* **One bad RIC fails a whole request** ("Unable to resolve all requested identifiers"): the request is split in
  halves until the bad RIC is alone, and that one is skipped and named in the log.
* **LSEG's daily limit** is 10,000 requests (and 5 GB) a day, reset at midnight local time; past it LSEG answers
  "Too many requests" (HTTP 429) and waiting does not help for hours. The client stops with :class:`ThrottledError`;
  the load records what it finished and the next run continues from there.
* **A call that keeps costing without answering is stopped.** lseg-data re-sends some requests on its own until an
  answer comes - for estimate revision counts asked as a history, none ever does, and one such call cost ~1,000
  requests (2026-10-30). While a call is open the client watches Workspace's count; past the call's own cost plus
  :data:`RUNAWAY_MARGIN` it closes the session and raises :class:`RunawayRequestError`.
* **What LSEG counts**: ``get_history`` asks once *per RIC* (lseg-data calls one URL per instrument), so a history of
  50 RICs costs 50 of the day's requests; ``get_data`` asks once for all its RICs. ``lseg_requests`` counts the
  day's cost that way, and before each call the client reads what is left of the allowance from Workspace
  (:meth:`LsegClient.allowance`): a call that would need more raises :class:`ThrottledError` before it is sent.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

#: The proxy's address. Not "localhost": Windows tries IPv6 (::1) first, where Workspace does not listen, and falls back
#: to IPv4 only after ~2 s - which every call, the library's own included, paid (measured 2026-10-30: 2.06 s vs 0.02 s).
WORKSPACE_HOST = "127.0.0.1"
#: Where Workspace records the port its Data API proxy listens on.
PORT_FILES = [
    Path(os.environ.get("APPDATA", "")) / vendor / app / ".portInUse"
    for vendor in ("Refinitiv", "LSEG", "Thomson Reuters")
    for app in ("Data API Proxy", "Eikon API proxy", "Eikon Scripting Proxy")
]
#: Messages no retry changes: an unknown instrument, field or parameter.
#: Messages about some of the RICs (one cannot be resolved, or has no data for a field): splitting the request lets
#: the others through. The last is lseg-data's own table builder failing on one RIC's answer in a multi-RIC history
#: (an IndexError, seen 2026-10-25 for a 2008 batch): every retry fails the same, the other RICs read fine apart.
PERMANENT = (
    "Unable to resolve all requested identifiers",
    "Unable to collect data for the field",
    "list assignment index out of range",
)
#: LSEG's backend refusing a request it accepts in smaller parts: usually gone on a retry, but a large estimates
#: history (177 RICs x 20 years of EBIT, 2026-10-31) failed every retry - after the retries the request is split.
BACKEND = ("Backend error",)
#: Messages about the request itself (a field or parameter LSEG does not know): splitting would only repeat the
#: same error for every RIC, burning the daily allowance - it stops at once.
SHAPE = (
    "Unable to resolve all requested fields",
    "is unrecognized",
    "does not support the following fields",  # a history field an instrument lacks (TRDPRC_1 of a currency)
)


class ThrottledError(RuntimeError):
    """LSEG refuses requests for now (its daily limit): run the load again later; it resumes."""


class PermanentRequestError(RuntimeError):
    """LSEG cannot answer for some of the RICs asked about."""


class BackendRequestError(RuntimeError):
    """LSEG's backend failed a request through every retry: asked in smaller parts it may answer."""


class RequestShapeError(RuntimeError):
    """LSEG does not know a field or parameter of the request: no RIC would be answered."""


class RunawayRequestError(RuntimeError):
    """A call went on costing requests without an answer (lseg-data re-sending it): it was stopped."""


#: Requests a call may cost beyond its own before it counts as runaway, and how often the count is read while it is
#: open. lseg-data asks for a large answer and then polls for it every ~15 s, each poll a request: ~170 RICs x 20 years
#: of a monthly history usually answer after ~5, but at night one took over 12 (2026-10-31) - stopped by a margin of
#: 10. A runaway's polls never end, so 25 costs a true one little more and gives a slow answer ~6 minutes.
RUNAWAY_MARGIN = 25
WATCH_SECONDS = 5.0


class _Worker:
    """One long-lived thread that runs the calls, so the caller can watch a call and walk away from one that hangs.

    Long-lived because lseg-data sets itself up again in every new thread: a call in a fresh thread took ~6 s against
    ~1.8 s (measured 2026-10-30). A daemon, so a call abandoned as runaway never keeps the process from exiting.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], dict[str, Any], threading.Event]] = queue.Queue()
        threading.Thread(target=self._run, name="lseg calls", daemon=True).start()

    def _run(self) -> None:
        while True:
            request, box, done = self._jobs.get()
            try:
                box["answer"] = request()
            except Exception as exc:  # handed to the caller's thread
                box["error"] = exc
            finally:
                done.set()

    def submit(self, request: Callable[[], Any]) -> tuple[dict[str, Any], threading.Event]:
        """Queue ``request``; ``(box, done)``: the box holds ``answer`` or ``error`` once ``done`` is set."""
        box: dict[str, Any] = {}
        done = threading.Event()
        self._jobs.put((request, box, done))
        return box, done


def is_throttled(exc: BaseException) -> bool:
    """LSEG's limit answer (``429`` / "Too many requests")."""
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def is_permanent(exc: BaseException) -> bool:
    """An error a retry cannot fix."""
    return any(marker in str(exc) for marker in PERMANENT)


def workspace_port() -> str | None:
    """The proxy port Workspace wrote to ``.portInUse``, if it is running."""
    for path in PORT_FILES:
        try:
            port = path.read_text().strip()
        except OSError:
            continue
        if port.isdigit():
            return port
    return None


def _listening(port: str) -> bool:
    try:
        with socket.create_connection((WORKSPACE_HOST, int(port)), timeout=1.0):
            return True
    except OSError:
        return False


class LsegClient:
    """``get_data`` / ``get_history`` with the habits of the module docstring."""

    def __init__(
        self, *, max_retries: int = 4, pause_seconds: float = 1.0, watch_seconds: float = WATCH_SECONDS
    ) -> None:
        self._max_retries = max_retries
        self._pause = pause_seconds
        self._watch = watch_seconds
        self._worker: _Worker | None = None  # started with the first watched call
        self._open = False
        self.requests = 0  # calls made, retries included
        self.lseg_requests = 0  # what they cost of LSEG's daily allowance (a history: one per RIC)
        self.skipped: list[str] = []  # RICs LSEG could not resolve

    def allowance(self) -> dict[str, int] | None:
        """Workspace's own count of today's allowance (``daily_request_remaining``, ``daily_volume_remaining``...).

        ``None`` when it cannot be asked (no Workspace on this computer, or a version without the endpoint).
        """
        base = os.environ.get("DP_PROXY_BASE_URL") or (
            f"http://{WORKSPACE_HOST}:{port}" if (port := workspace_port()) else None
        )
        if base is None:
            return None
        try:
            resp = requests.get(f"{base}/api/throttling/limits", timeout=5)
            resp.raise_for_status()
            limits = resp.json()
        except (requests.RequestException, ValueError):
            return None
        return {str(k): int(v) for k, v in limits.items()} if isinstance(limits, dict) else None

    def _check_allowance(self, cost: int, what: str) -> None:
        limits = self.allowance()
        left = None if limits is None else limits.get("daily_request_remaining")
        if left is not None and left < cost:
            raise ThrottledError(
                f"LSEG allowance: {what} needs {cost} requests, {left} left today (resets at midnight)"
            )

    def _ensure_session(self) -> Any:
        import lseg.data as ld  # noqa: PLC0415 - optional dependency, loaded only when LSEG is used

        if self._open:
            return ld
        port = workspace_port()
        if port and not os.environ.get("DP_PROXY_BASE_URL"):
            if not _listening(port):
                raise RuntimeError(f"LSEG Workspace is not running (nothing on port {port}): start it, sign in, retry")
            os.environ["DP_PROXY_BASE_URL"] = f"http://{WORKSPACE_HOST}:{port}"
        session = ld.open_session()
        if session.open_state != ld.OpenState.Opened:
            raise RuntimeError("could not open the LSEG session - is LSEG Workspace running and signed in?")
        self._open = True
        return ld

    def close(self) -> None:
        """Close the session (the next request opens it again)."""
        if self._open:
            import lseg.data as ld  # noqa: PLC0415

            ld.close_session()
            self._open = False

    def _call(self, request: Any, what: str, cost: int = 1) -> pd.DataFrame:
        """One answer, retried on transient errors; ``cost``: LSEG requests each attempt uses."""
        for attempt in range(1, self._max_retries + 1):
            time.sleep(self._pause)
            self._check_allowance(cost, what)
            self.requests += 1
            self.lseg_requests += cost
            try:
                frame = self._watched(request, what, cost)
                return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
            except RunawayRequestError:
                raise  # asking again would run away again
            except Exception as exc:
                if is_throttled(exc):
                    raise ThrottledError(f"LSEG limit reached ({what}): {str(exc)[:160]}") from exc
                if any(marker in str(exc) for marker in SHAPE):
                    raise RequestShapeError(f"{what}: {str(exc)[:300]}") from exc
                if is_permanent(exc):
                    raise PermanentRequestError(str(exc)[:300]) from exc
                if attempt == self._max_retries:
                    if any(marker in str(exc) for marker in BACKEND):
                        raise BackendRequestError(f"{what}: {str(exc)[:300]}") from exc
                    raise
                wait = 15.0 * attempt
                log.warning("%s: %s - retry %d in %.0fs", what, str(exc)[:120], attempt, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def _watched(self, request: Any, what: str, cost: int) -> Any:
        """``request()``, stopped when Workspace's count falls by more than ``cost`` + :data:`RUNAWAY_MARGIN` meanwhile.

        The request runs in a worker thread (:class:`_Worker`) so a call that never answers cannot hold the load: on a
        runaway the session is closed - which ends the library's re-sending (seen live: the count stops falling) - and
        the worker is abandoned for a new one. Another program
        using LSEG at the same time also lowers the count - the error says so.
        """
        before = (self.allowance() or {}).get("daily_request_remaining")
        if before is None:
            return request()  # no count to watch (no Workspace endpoint): as before
        if self._worker is None:
            self._worker = _Worker()
        box, done = self._worker.submit(request)
        while not done.wait(self._watch):
            now = (self.allowance() or {}).get("daily_request_remaining")
            if now is not None and before - now > cost + RUNAWAY_MARGIN:
                self.close()
                self._worker = None  # it may still be inside the call: the next call gets a new one
                raise RunawayRequestError(
                    f"{what}: {before - now} requests and no answer - lseg-data kept re-sending it (or another "
                    "program used LSEG meanwhile); stopped"
                )
        if "error" in box:
            raise box["error"]
        return box["answer"]

    def get_data(
        self, universe: Sequence[str], fields: Sequence[str], parameters: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """One ``ld.get_data`` answer for ``universe``; RICs LSEG cannot answer for are dropped (see ``skipped``).

        A request LSEG refuses for some of its RICs, or whose backend fails it through every retry, is split in halves
        until the parts answer; a single RIC still refused is skipped and named.
        """
        ld = self._ensure_session()
        rics = list(universe)
        try:
            return self._call(
                lambda: ld.get_data(universe=rics, fields=list(fields), parameters=parameters or {}),
                f"get_data {len(rics)} RICs",
            )
        except (PermanentRequestError, BackendRequestError) as exc:
            if len(rics) == 1:
                log.warning("LSEG cannot answer for %s - skipped (%s)", rics[0], str(exc)[:120])
                self.skipped.append(rics[0])
                return pd.DataFrame()
            half = len(rics) // 2
            parts = [self.get_data(rics[:half], fields, parameters), self.get_data(rics[half:], fields, parameters)]
            parts = [p for p in parts if not p.empty]
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def get_history(self, universe: Sequence[str], fields: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Daily unadjusted history (``ld.get_history``) as one long table ``[Instrument, Date, <field>...]``.

        Unlike ``get_data``, a RIC without data in the window comes back blank instead of failing the request, so
        prices are asked for this way.
        """
        ld = self._ensure_session()
        rics = list(universe)
        # get_history's bounds behave as exclusive: widen by a day each side, then trim
        lo = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hi = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            frame = self._call(
                lambda: ld.get_history(
                    universe=rics, fields=list(fields), start=lo, end=hi, interval="daily", adjustments=["unadjusted"]
                ),
                f"get_history {len(rics)} RICs",
                cost=len(rics),
            )
            long = _long_history(frame, rics, fields)
        except PermanentRequestError:
            if len(rics) == 1:
                log.warning("LSEG has no history for %s - skipped", rics[0])
                self.skipped.append(rics[0])
                return pd.DataFrame()
            half = len(rics) // 2
            parts = [
                self.get_history(rics[:half], fields, start, end),
                self.get_history(rics[half:], fields, start, end),
            ]
            parts = [p for p in parts if not p.empty]
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if long.empty:
            return long
        inside = (long["Date"] >= pd.Timestamp(start)) & (long["Date"] <= pd.Timestamp(end))
        return long[inside].reset_index(drop=True)


def _long_history(frame: pd.DataFrame, rics: list[str], fields: Sequence[str]) -> pd.DataFrame:
    """A ``get_history`` answer as ``[Instrument, Date, <field>...]``, whichever of its shapes lseg-data built.

    Several RICs and fields: ``(RIC, field)`` columns. Several RICs and one field: a column per RIC. One RIC: a column
    per field. Any other shape (RICs answering with different numbers of fields come back garbled) raises
    :class:`PermanentRequestError`, so the request is split like one with a bad RIC.
    """
    if frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels != 2:
            raise PermanentRequestError(f"lseg-data built {frame.columns.nlevels} column levels for {len(rics)} RICs")
        frame.columns = frame.columns.set_names(["Instrument", "field"])
        long = frame.stack(level="Instrument", future_stack=True).reset_index()  # noqa: PD013 - one row per RIC and day
    elif len(rics) > 1:
        if len(fields) != 1 or not set(map(str, frame.columns)) <= set(rics):
            raise PermanentRequestError(f"lseg-data built columns {list(frame.columns)[:6]} for {len(rics)} RICs")
        long = frame.melt(ignore_index=False, var_name="Instrument", value_name=fields[0]).reset_index()
    else:
        long = frame.reset_index().assign(Instrument=rics[0])
    long = long.rename(columns={long.columns[0]: "Date"})
    long.columns = [str(c) for c in long.columns]
    long["Date"] = pd.to_datetime(long["Date"])
    return long
