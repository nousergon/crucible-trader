"""Where the trader is pointed, and every way that resolution is refused.

The negative cases ARE the test. A settings module whose only test is the happy
path grades nothing about the property it exists for: that a trader nobody
pointed at a store stops, rather than reading one it was not pointed at.
"""

from __future__ import annotations

import pytest

from crucible_trader import __version__
from crucible_trader.settings import (
    SERVED_SLOT,
    SLOT_VAR,
    STORE_URI_VAR,
    Settings,
    SettingsError,
)


def test_resolves_an_s3_store_and_defaults_the_slot() -> None:
    settings = Settings.from_env({STORE_URI_VAR: "s3://a-bucket/a-prefix"})

    assert settings.store_uri == "s3://a-bucket/a-prefix"
    assert settings.slot == SERVED_SLOT


def test_resolves_a_file_store_for_a_local_exercise() -> None:
    settings = Settings.from_env({STORE_URI_VAR: "file:///tmp/store", SLOT_VAR: "M"})

    assert settings.store_uri == "file:///tmp/store"
    # Case-folded: an operator typing the slot in caps is pointed at the same
    # contract, not refused for a shift key.
    assert settings.slot == "m"


def test_settings_are_frozen() -> None:
    settings = Settings.from_env({STORE_URI_VAR: "s3://a-bucket/a-prefix"})

    with pytest.raises(AttributeError):
        settings.store_uri = "s3://somewhere-else"  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "   "])
def test_an_unset_or_blank_store_is_refused_not_defaulted(value: str) -> None:
    with pytest.raises(SettingsError) as raised:
        Settings.from_env({STORE_URI_VAR: value})

    assert STORE_URI_VAR in str(raised.value)
    assert "no default" in str(raised.value)


def test_a_missing_store_variable_is_refused() -> None:
    with pytest.raises(SettingsError):
        Settings.from_env({})


def test_a_bare_path_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(SettingsError) as raised:
        Settings.from_env({STORE_URI_VAR: "/var/lib/crucible"})

    assert "not a store URI" in str(raised.value)


def test_another_slot_is_refused_because_the_feed_is_the_m_contract() -> None:
    with pytest.raises(SettingsError) as raised:
        Settings.from_env({STORE_URI_VAR: "s3://a-bucket/a-prefix", SLOT_VAR: "r"})

    assert SLOT_VAR in str(raised.value)
    assert SERVED_SLOT in str(raised.value)


def test_the_process_environment_is_the_default_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STORE_URI_VAR, "s3://from-the-process/env")
    monkeypatch.delenv(SLOT_VAR, raising=False)

    assert Settings.from_env().store_uri == "s3://from-the-process/env"


def test_version_is_a_literal_not_package_metadata() -> None:
    # A checkout that has not been installed has no package metadata, and
    # `importlib.metadata.version` raises there — the shape that produced eight
    # false test failures in a `crucible` worktree on 2026-09-11.
    assert __version__ == "0.1.0"
