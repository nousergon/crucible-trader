"""The v2 box units this repository carries match their crucible-executor originals.

Each ``infrastructure/systemd/crucible-v2-*`` unit is a copy of the executor unit
it replaces (kept under ``fixtures/executor_units/``). Only the unit name, the
``ExecStart`` and the ``WorkingDirectory`` may differ: schedule, ``User``,
``Environment*``, timeouts and ``OnFailure`` must be identical, so moving a unit
between repositories cannot change when or how it runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "infrastructure" / "systemd"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "executor_units"
TRADER_BOX = ROOT / "scripts" / "trader_box.sh"

OLD_PREFIX = "alpha-engine-v2-"
NEW_PREFIX = "crucible-v2-"
NAMES = ["trade-alert", "market-open-alert", "market-close-alert", "session-watchdog", "day-end"]
PAIRS = [
    f"{name}.{kind}"
    for name in NAMES
    for kind in ("service", "timer")
    if (FIXTURES / f"{OLD_PREFIX}{name}.{kind}").exists()
]
DIFFERS = {"ExecStart", "WorkingDirectory"}

Unit = dict[tuple[str, str], list[str]]


def parse(path: Path) -> Unit:
    """Map (section, key) to every value given, in order; comments and blanks dropped."""
    out: Unit = {}
    section = ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key, _, value = line.partition("=")
        out.setdefault((section, key.strip()), []).append(value.strip())
    return out


def allow_list() -> set[str]:
    text = TRADER_BOX.read_text()
    match = re.search(r"case \"\$module\" in\n\s*([^)]*?)\) ;;", text)
    assert match, "trader_box.sh allow-list not found"
    return {m.strip() for m in match.group(1).split("|")}


def test_every_executor_unit_has_a_counterpart() -> None:
    assert len(PAIRS) == 9
    assert sorted(p.name for p in UNITS.glob("crucible-v2-*")) == sorted(
        f"{NEW_PREFIX}{p}" for p in PAIRS
    )


@pytest.mark.parametrize("suffix", PAIRS)
def test_only_name_exec_and_working_directory_differ(suffix: str) -> None:
    old = parse(FIXTURES / f"{OLD_PREFIX}{suffix}")
    new = parse(UNITS / f"{NEW_PREFIX}{suffix}")
    assert old.keys() == new.keys()
    for key in old:
        if key[1] in DIFFERS:
            continue
        renamed = [v.replace(OLD_PREFIX, NEW_PREFIX) for v in old[key]]
        assert new[key] == renamed, key


@pytest.mark.parametrize("suffix", [p for p in PAIRS if p.endswith(".service")])
def test_exec_lines_run_trader_box_from_this_checkout(suffix: str) -> None:
    old = parse(FIXTURES / f"{OLD_PREFIX}{suffix}")[("Service", "ExecStart")]
    new = parse(UNITS / f"{NEW_PREFIX}{suffix}")
    assert new[("Service", "WorkingDirectory")] == ["/home/ec2-user/crucible-trader"]
    execs = new[("Service", "ExecStart")]
    assert len(execs) == len(old)
    prefix = "/usr/bin/env bash /home/ec2-user/crucible-trader/scripts/trader_box.sh "
    for line, executor_line in zip(execs, old, strict=True):
        # A leading "-" (failure ignored) is part of the contract and must survive.
        assert line.startswith("-") == executor_line.startswith("-")
        assert line.lstrip("-").startswith(prefix), line
        assert line.lstrip("-")[len(prefix) :].split()[0] in allow_list()


def test_the_allow_list_is_what_the_units_need() -> None:
    assert allow_list() == {
        "trade_alert",
        "market_alert",
        "release_alert",
        "session_watchdog",
        "day_end",
    }


@pytest.mark.parametrize("suffix", [p for p in PAIRS if p.endswith(".timer")])
def test_a_timer_triggers_its_own_unit(suffix: str) -> None:
    new = parse(UNITS / f"{NEW_PREFIX}{suffix}")
    assert new[("Timer", "Unit")] == [f"{NEW_PREFIX}{suffix.replace('.timer', '.service')}"]
