"""The box-side entry script runs only its five allow-listed modules and refuses the rest.

What CI can grade is that the script is valid bash, that its allow-list is the
box package's alert and watchdog modules and nothing else, and that a refusal
fires before anything is executed. The ``exec`` itself runs only on the box.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "trader_box.sh"
ALLOWED = ("trade_alert", "market_alert", "release_alert", "session_watchdog", "day_end")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": "/nonexistent"},
        check=False,
    )


def test_the_script_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_an_unknown_module_is_refused_before_anything_runs() -> None:
    for args in ((), ("daily_session",), ("notify",), ("trade_alert; rm -rf /",), ("../x",)):
        result = _run(*args)
        assert result.returncode == 2 and "usage" in result.stderr


def test_the_allow_list_is_literally_the_five_box_modules() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    pattern = re.search(r"^  ([a-z_ |]+\|[a-z_ |]+\)) ;;$", text, re.M)
    assert pattern is not None
    assert tuple(name.strip() for name in pattern[1].rstrip(")").split("|")) == ALLOWED
    for name in ALLOWED:
        assert (SCRIPT.parents[1] / "crucible_trader" / "box" / f"{name}.py").is_file()


def test_the_script_runs_the_module_in_the_locked_environment() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'exec uv run --frozen --project "$HOME/crucible-trader" python -m ' in text
    assert '"crucible_trader.box.${module}"' in text
