from typing import Any, cast

import pytest
from sqlalchemy.engine import Engine

from equity_research.pipelines import refresh as r
from equity_research.quality.checks import CheckResult, GateFailedError
from equity_research.research.cache import ResearchCache


class _Cache:
    synced = False

    def sync(self, engine: object) -> dict[str, int]:
        self.synced = True
        return {}


def _steps(monkeypatch: pytest.MonkeyPatch, status: str) -> list[str]:
    ran: list[str] = []

    def sources(engine: object, mode: str) -> dict[str, object]:
        ran.extend(["lseg", "benchmarks", "sec"])
        return {}

    def prune(engine: object) -> dict[str, object]:
        ran.append("prune")
        return {"files": 0, "megabytes": 0.0}

    monkeypatch.setattr(r, "run_sources", sources)
    monkeypatch.setattr(r, "prune_raw", prune)
    monkeypatch.setattr(r, "run_checks", lambda engine, checks: [CheckResult("x", status, 0.5, "fail <= 0.9")])
    return ran


def test_every_source_then_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    ran = _steps(monkeypatch, "pass")
    cache = _Cache()
    report: dict[str, Any] = r.refresh(cast(Engine, None), cache=cast(ResearchCache, cache))
    assert ran == ["lseg", "benchmarks", "sec", "prune"] and cache.synced
    assert report["checks"] == {"x": {"status": "pass", "value": 0.5}}


def test_a_failed_check_stops_before_the_cache_sees_the_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _steps(monkeypatch, "fail")
    cache = _Cache()
    with pytest.raises(GateFailedError, match=r"x = 0.5"):
        r.refresh(cast(Engine, None), cache=cast(ResearchCache, cache))
    assert not cache.synced


def test_written_counts_every_write_in_a_report() -> None:
    report = {
        "lseg": {"prices": {"price_bars": {"inserted": 5, "updated": 1}, "requests": 3}},
        "benchmarks": {"SPY": {"listing_id": 9, "bars": {"inserted": 2, "updated": 0}}},
        "checks": {"x": {"status": "pass"}},
    }
    table = r.written(report)
    assert table.values.tolist() == [["lseg", "prices / price_bars", 5, 1], ["benchmarks", "SPY / bars", 2, 0]]


def test_sources_run_in_the_preferred_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_research.pipelines import sources as src  # noqa: PLC0415

    ran: list[tuple[str, str]] = []

    def record(name: str, mode: str) -> dict[str, object]:
        ran.append((name, mode))
        return {}

    def source(name: str) -> object:
        return src.Source(
            name,
            "",
            build=lambda e: record(name, "build"),
            refresh=lambda e: record(name, "refresh"),
        )

    fake = {name: source(name) for name in ("refinitiv", "yahoo", "sec", "norgate")}
    monkeypatch.setattr(src, "SOURCES", fake)
    src.run_sources(cast(Engine, None), "refresh", ("norgate", "sec"))
    assert ran == [("norgate", "refresh"), ("sec", "refresh")]
    with pytest.raises(KeyError, match="no source 'nobody'"):
        src.run_sources(cast(Engine, None), "build", ("nobody",))
