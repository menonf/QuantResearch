import datetime as dt

import pandas as pd
import pytest

from equity_research.pipelines import sources
from equity_research.pipelines.sources import SOURCES, Source, describe_sources, probe_sources


def _fake(probe: object = None) -> Source:
    return Source("x", "things", build=lambda engine: {}, refresh=lambda engine: {}, needs="a key", probe=probe)  # type: ignore[arg-type]


def test_every_source_says_what_it_supplies_and_needs() -> None:
    table = describe_sources(("refinitiv", "yahoo"))
    assert table["source"].tolist()[:2] == ["refinitiv", "yahoo"]  # this build's first, in its order
    assert set(table["source"]) == set(SOURCES)
    assert table.set_index("source").loc["tiingo", "in this build"] == False  # noqa: E712 - a numpy bool
    assert table["needs"].str.len().gt(0).all()


def test_each_sources_probe_lands_in_one_table(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> pd.DataFrame:
        raise RuntimeError("Norgate Data Updater is not running")

    fakes = {
        "answers": _fake(lambda: pd.DataFrame([{"dataset": "prices", "ok": True, "rows": 5, "detail": ""}])),
        "public": _fake(),
        "broken": _fake(broken),
    }
    monkeypatch.setattr(sources, "SOURCES", fakes)
    out = probe_sources(("answers", "public", "broken")).set_index("source")
    assert out.loc["answers", "check"] == "prices"  # an LSEG-style "dataset" column reads as a check
    assert bool(out.loc["public", "ok"]) and out.loc["public", "detail"] == "a key"
    assert not bool(out.loc["broken", "ok"]) and "not running" in str(out.loc["broken", "detail"])


def test_a_refresh_starts_on_the_first_of_the_month_a_few_days_back() -> None:
    assert sources.refresh_start(dt.date(2026, 10, 23)) == dt.date(2026, 10, 1)
    # a refresh that wrote prices before failing moves the newest price on: the rerun asks the same questions
    assert sources.refresh_start(dt.date(2026, 10, 30)) == sources.refresh_start(dt.date(2026, 10, 23))
    assert sources.refresh_start(dt.date(2026, 11, 4)) == dt.date(2026, 10, 1)  # never fewer than the overlap days
