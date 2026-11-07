from pathlib import Path

import pytest

from equity_research import settings


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        settings,
        "_get",
        lambda env, key=None, default=None: {
            "EQR_DB_SERVER": "srv.database.windows.net",
            "EQR_DB_USER": "u",
            "EQR_DB_PASSWORD": "p",
            "EQR_DATA_DIR": str(tmp_path),
        }.get(env, default),
    )
    settings.get_settings.cache_clear()
    s = settings.get_settings()
    assert s.database.server == "srv.database.windows.net"
    assert s.database.name == "equity_research"
    assert s.database.uses_sql_auth
    assert s.raw_dir == tmp_path / "raw"
    settings.get_settings.cache_clear()


def test_a_missing_server_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "_get", lambda env, key=None, default=None: default)
    settings.get_settings.cache_clear()
    with pytest.raises(settings.SettingMissingError, match="EQR_DB_SERVER"):
        settings.get_settings()
    settings.get_settings.cache_clear()


def test_a_server_from_the_environment_never_takes_the_keyring_login(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EQR_DB_USER", "EQR_DB_PASSWORD", "EQR_DB_NAME", "EQR_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EQR_DB_SERVER", r"PC\SQLEXPRESS")
    monkeypatch.setenv("EQR_DB_NAME", "fresh")
    monkeypatch.setattr("equity_research.settings.keyring.get_password", lambda service, key: "from-keyring")
    settings.get_settings.cache_clear()
    db = settings.get_settings().database
    assert (db.server, db.name, db.user, db.password) == (r"PC\SQLEXPRESS", "fresh", None, None)
    assert not db.uses_sql_auth  # Windows authentication
    assert settings.get_settings().data_dir == settings.REPO_ROOT / "data" / "fresh"  # its own raw zone and cache
    settings.get_settings.cache_clear()


def test_vendor_preferences_default_and_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    settings.vendor_preferences.cache_clear()
    prefs = settings.vendor_preferences()
    assert prefs.prices == ("refinitiv",)
    assert prefs.fundamentals == ("sec", "refinitiv")  # LSEG's accounts only before the SEC's begin
    monkeypatch.setenv("EQR_PRICE_VENDORS", "norgate, refinitiv")
    settings.vendor_preferences.cache_clear()
    assert settings.vendor_preferences().prices == ("norgate", "refinitiv")
    settings.vendor_preferences.cache_clear()
