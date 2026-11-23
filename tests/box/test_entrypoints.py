"""Each module runs as ``python -m crucible_trader.box.<name>``, and day_end's runner shells out.

`runpy` executes the module as ``__main__`` with the logging setup stubbed, so the
guarded entry block is read without touching the box's logging or network. Each
module is driven to its own usage refusal (exit 2), which needs no store.
"""

from __future__ import annotations

import contextlib
import runpy
import subprocess
import sys
from typing import Any

import pytest

from crucible_trader.box import day_end

MODULES = [
    ("trade_alert", "v2-trade-alert", ["extra"]),
    ("market_alert", "v2-market-alert", []),
    ("release_alert", "v2-release-alert", ["extra"]),
    ("session_watchdog", "v2-session-watchdog", ["extra"]),
]


@pytest.mark.parametrize(("module", "log_name", "argv"), MODULES)
def test_the_module_runs_as_main_inside_the_logging_guard(
    monkeypatch: pytest.MonkeyPatch, module: str, log_name: str, argv: list[str]
) -> None:
    import krepis.logging as klog

    seen: list[tuple[str, dict[str, Any]]] = []
    guarded: list[bool] = []

    @contextlib.contextmanager
    def guard() -> Any:
        guarded.append(True)
        yield

    monkeypatch.setattr(klog, "setup_logging", lambda name, **kw: seen.append((name, kw)))
    monkeypatch.setattr(klog, "guard_entrypoint", guard)
    monkeypatch.setattr(sys, "argv", [module, *argv])
    monkeypatch.delitem(sys.modules, f"crucible_trader.box.{module}", raising=False)
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module(f"crucible_trader.box.{module}", run_name="__main__")
    assert exit_.value.code == 2
    assert seen == [(log_name, {})], "no flow-doctor yaml: that package is not a dependency"
    assert guarded == [True]


def test_day_end_runs_as_main_and_refuses_an_unknown_argument(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["day_end", "--bogus"])
    monkeypatch.delitem(sys.modules, "crucible_trader.box.day_end", raising=False)
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("crucible_trader.box.day_end", run_name="__main__")
    assert exit_.value.code == 2
    assert "usage: v2_day_end [--dry-run]" in capsys.readouterr().err


def test_day_end_runner_shells_out_with_a_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs, argv=argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(day_end.subprocess, "run", fake)
    assert day_end._run(["true"]).stdout == "ok"  # pyright: ignore[reportPrivateUsage]
    assert seen["argv"] == ["true"] and seen["timeout"] == day_end.STOP_TIMEOUT_SECONDS + 30
    assert seen["capture_output"] is True and seen["check"] is False
