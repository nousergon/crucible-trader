"""Branches of `crucible_trader.box.release_alert` the ported suite leaves unread."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from crucible_trader.box import release_alert as r
from crucible_trader.box.notify import send_trading_notice


def test_hm_et_reads_an_iso_instant_and_refuses_anything_else() -> None:
    assert r._hm_et("2026-11-10T15:30:00Z") == "10:30"  # pyright: ignore[reportPrivateUsage]
    assert r._hm_et("2026-11-10T15:30:00") == "10:30", "a naive instant is read as UTC"  # pyright: ignore[reportPrivateUsage]
    for bad in ("", None, 5, "not a time"):
        assert r._hm_et(bad) == ""  # pyright: ignore[reportPrivateUsage]


def test_float_wants_a_positive_number() -> None:
    assert r._float(True) is None  # pyright: ignore[reportPrivateUsage]
    assert r._float("3") is None  # pyright: ignore[reportPrivateUsage]
    assert r._float(0) is None  # pyright: ignore[reportPrivateUsage]
    assert r._float(2) == 2.0  # pyright: ignore[reportPrivateUsage]


def test_a_watch_without_an_entries_list_has_no_entries() -> None:
    assert r.entries(None) == []
    assert r.entries({"entries": "x"}) == []
    assert r.entries({"entries": [{"a": 1}, "junk"]}) == [{"a": 1}]


def test_shares_is_zero_when_nothing_positive_is_recorded() -> None:
    assert r.shares({}) == 0
    assert r.shares({"released_quantity": True, "quantity": -1}) == 0
    assert r.shares({"released_quantity": 0, "quantity": 7}) == 7


def test_price_line_says_what_is_missing() -> None:
    assert r.price_line({}) == "price not recorded"
    assert (
        r.price_line({"reference_price": 100.0}) == "release price not recorded (planned $100.00)"
    )
    assert r.price_line({"release_price": 99.5}) == "$99.50 (no planned price recorded)"
    assert r.price_line({"release_price": 99.0, "reference_price": 100.0}) == (
        "$99.00 vs planned $100.00 (-1.0%)"
    )


def test_an_alerted_file_without_a_list_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "day.json"
    path.write_text(json.dumps({"alerted": "x"}))
    with pytest.raises(r.StateUnreadable, match="no 'alerted' list"):
        r.load_alerted(path)
    path.write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(r.StateUnreadable, match="no 'alerted' list"):
        r.load_alerted(path)


def _tuesday_ten() -> dt.datetime:
    return dt.datetime(2026, 11, 10, 15, 0, tzinfo=dt.UTC)  # 10:00 ET


def test_main_ticks_and_maps_the_outcome_to_an_exit_status(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = object()
    monkeypatch.setattr(r, "read_env_file", lambda _path: {})
    monkeypatch.setattr(r, "open_store", lambda _env: store)
    seen: dict[str, Any] = {}
    outcomes = [
        r.Outcome(sent=1, reason="1 sent"),
        r.Outcome(undelivered=2, reason="2 not delivered"),
        r.Outcome(reason="state unreadable: x"),
    ]

    def fake_tick(passed: object, **kwargs: Any) -> r.Outcome:
        seen.update(kwargs, store=passed)
        return outcomes.pop(0)

    monkeypatch.setattr(r, "tick", fake_tick)
    assert r.main([], now=_tuesday_ten()) == 0
    assert seen["store"] is store and seen["send"] is send_trading_notice
    with caplog.at_level("ERROR"):
        assert r.main([], now=_tuesday_ten()) == 1
    assert "2 release alert(s) not delivered" in caplog.text
    assert r.main([], now=_tuesday_ten()) == 1
