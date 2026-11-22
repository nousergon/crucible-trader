"""Branches of `crucible_trader.box.session_watchdog` the ported suite leaves unread.

The ported suite ran in the v1 executor, which had no coverage gate; this repository
measures every statement, so each defensive branch gets its own case here.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from crucible_trader.box import session_watchdog as w
from crucible_trader.box.market_alert import Unit
from crucible_trader.box.trade_alert import Store

NY = ZoneInfo("America/New_York")


def _at(hour: int, minute: int, day: dt.date = dt.date(2026, 11, 11)) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).astimezone(dt.UTC)


def _done(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_the_default_runner_shells_out_with_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs, argv=argv)
        return _done(0, "out")

    monkeypatch.setattr(w.subprocess, "run", fake)
    assert w._run(["echo", "x"]).stdout == "out"  # pyright: ignore[reportPrivateUsage]
    assert seen["argv"] == ["echo", "x"] and seen["timeout"] == 60 and seen["check"] is False


def test_an_unparseable_timestamp_reads_as_none() -> None:
    assert w._when("not a time") is None  # pyright: ignore[reportPrivateUsage]
    assert w._when(None) is None  # pyright: ignore[reportPrivateUsage]


def test_held_reason_ignores_a_run_without_a_metrics_list_and_non_dict_metrics() -> None:
    assert w.held_reason({"status": "ok", "metrics": "none"}) is None
    held = {"name": w.SERVED_METRIC, "status_reason": w.HELD_PREFIX + " inputs late"}
    assert w.held_reason({"status": "ok", "metrics": ["junk", 3, held]}) == "inputs late"


def test_stop_requested_is_unknown_for_a_session_that_never_started() -> None:
    assert w.stop_requested(Unit({"ActiveState": "inactive"})) is None


def test_the_instance_id_is_read_from_the_cloud_init_file(tmp_path: Path) -> None:
    path = tmp_path / "instance-id"
    path.write_text("placeholder-id\n")
    assert w._instance_id(path) == "placeholder-id"  # pyright: ignore[reportPrivateUsage]
    path.write_text("\n")
    assert w._instance_id(path) == "<trading box instance id>"  # pyright: ignore[reportPrivateUsage]
    assert w._instance_id(tmp_path / "absent") == "<trading box instance id>"  # pyright: ignore[reportPrivateUsage]


def test_a_refused_hard_stop_is_reported_with_the_refusal(tmp_path: Path) -> None:
    sent: list[str] = []

    def send(text: str, **_: Any) -> bool:
        sent.append(text)
        return True

    outcome = w.act(
        w.Decision("hard_stop", "still running at 16:05 ET"),
        now=_at(16, 5),
        state=w.State(),
        state_path=tmp_path / "state.json",
        run=lambda _argv: _done(1, stderr="sudo: a password is required"),
        send=send,
        command="cmd",
    )
    assert outcome.delivered and "the stop was refused (sudo: a password is required)" in (
        outcome.decision.reason
    )
    assert len(sent) == 1


def test_tick_does_nothing_on_a_day_the_exchange_is_closed(tmp_path: Path) -> None:
    saturday = _at(11, 0, dt.date(2026, 11, 14))
    outcome = w.tick(
        Store(object(), "b", ""),  # type: ignore[arg-type]
        now=saturday,
        send=lambda *_a, **_k: pytest.fail("nothing is sent on a closed day"),
        state_dir=tmp_path,
    )
    assert outcome.decision.action == "none" and "not an NYSE session" in outcome.decision.reason


def test_main_ticks_in_the_window_and_fails_on_an_undelivered_notice(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = object()
    monkeypatch.setattr(w, "read_env_file", lambda _path: {"CRUCIBLE_TRADER_STORE_URI": "s3://x"})
    monkeypatch.setattr(w, "open_store", lambda _env: store)
    monkeypatch.setattr(w, "_instance_id", lambda: "placeholder-id")
    monkeypatch.setenv("AWS_REGION", "us-test-1")
    seen: dict[str, Any] = {}
    results = [
        w.Outcome(w.Decision("none", "healthy"), True),
        w.Outcome(w.Decision("start", "dead"), False),
    ]

    def fake_tick(passed: object, **kwargs: Any) -> w.Outcome:
        seen.update(kwargs, store=passed)
        return results.pop(0)

    monkeypatch.setattr(w, "tick", fake_tick)
    assert w.main([], now=_at(10, 0)) == 0
    assert seen["store"] is store
    assert "us-test-1" in seen["command"] and "placeholder-id" in seen["command"]
    with caplog.at_level("WARNING"):
        assert w.main([], now=_at(10, 0)) == 1
    assert "the start notice was not delivered" in caplog.text
