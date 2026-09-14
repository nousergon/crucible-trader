"""The box-side smoke script parses, and refuses before it touches anything.

The install-and-connect half runs only on the trader box against a live paper
gateway; what CI can grade is that the script is valid bash and that its
refusals fire before any download or install.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "trader_paper_smoke.sh"


def _run(*args: str, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    child = {"PATH": os.environ["PATH"], **(extra or {})}
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=child, check=False
    )


def test_the_script_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_a_missing_or_short_sha_is_refused() -> None:
    for args in ((), ("abc",), ("A" * 40,)):
        result = _run(*args)
        assert result.returncode == 2 and "usage" in result.stderr


def test_a_missing_store_or_gateway_is_refused_by_name() -> None:
    result = _run("c" * 40, extra={"CRUCIBLE_TRADER_STORE_URI": "file:///tmp/x"})
    assert result.returncode == 2 and "CRUCIBLE_TRADER_IB_HOST is unset" in result.stderr
