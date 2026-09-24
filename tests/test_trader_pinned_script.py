"""The box-side entry script for the daily session and the shadow books parses,
runs only its allowlisted entry points, and refuses before it touches anything.

The install-and-run half runs only on the executor box; what CI can grade is
that the script is valid bash and that its refusals fire before any download.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "trader_pinned.sh"


def _run(*args: str, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    child = {"PATH": os.environ["PATH"], **(extra or {})}
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=child, check=False
    )


def test_the_script_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_only_the_allowlisted_entry_points_run() -> None:
    for args in ((), ("commands",), ("kill_switch",), ("daily_session; rm -rf /",)):
        result = _run(*args)
        assert result.returncode == 2 and "usage" in result.stderr


def test_the_session_refuses_without_its_gateway_by_name() -> None:
    result = _run("daily_session", extra={"CRUCIBLE_TRADER_STORE_URI": "file:///tmp/x"})
    assert result.returncode == 2 and "CRUCIBLE_TRADER_IB_HOST is unset" in result.stderr


def test_the_shadow_books_need_only_a_store() -> None:
    result = _run("shadow_books_daily")
    assert result.returncode == 2 and "CRUCIBLE_TRADER_STORE_URI is unset" in result.stderr


def test_nothing_in_the_script_turns_routing_on() -> None:
    """Routing is an operator decision (I11545 ruling 3): the script may
    mention the switch, never set it."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "CRUCIBLE_TRADER_ORDER_ROUTING=" not in text.replace(
        "CRUCIBLE_TRADER_ORDER_ROUTING=ib_paper is set", ""
    )
