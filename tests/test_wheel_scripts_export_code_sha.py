"""Every box script that runs a release WHEEL exports $CRUCIBLE_CODE_SHA first.

`crucible.runner.run_job` reads `code_sha` from that variable and otherwise
runs `git rev-parse HEAD` in the installed wheel's directory, which is no
checkout. The box's 2026-10-01 pin smoke refused there with `CodeShaError`
before it ran a single step. What CI can grade without a box is that each
script sets the variable to the release sha, and does so before the line
that starts the wheel's interpreter.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
EXPORTS = {
    "trader_paper_smoke.sh": 'export CRUCIBLE_CODE_SHA="$sha"',
    "trader_pinned.sh": 'export CRUCIBLE_CODE_SHA="$pinned_sha"',
    "trader_reconcile.sh": 'export CRUCIBLE_CODE_SHA="$pinned_sha"',
}


@pytest.mark.parametrize("name", sorted(EXPORTS))
def test_the_release_sha_is_exported_before_the_wheel_runs(name: str) -> None:
    text = (SCRIPTS / name).read_text()
    export = text.find(EXPORTS[name] + "\n")
    run = text.rfind('"$work/venv/bin/python" -c')
    assert export != -1, f"{name} does not export {EXPORTS[name]}"
    assert run != -1 and export < run, f"{name} exports the sha after the wheel runs"


@pytest.mark.parametrize("name", ["trader_pinned.sh", "trader_reconcile.sh"])
def test_the_pinned_sha_comes_from_the_pin_read(name: str) -> None:
    text = (SCRIPTS / name).read_text()
    assert "sha = read_trader_pin(store)" in text
    assert "print(name, sha)" in text
    assert 'pinned_sha="${wheel#* }"' in text and 'wheel="${wheel%% *}"' in text


def test_the_split_recovers_both_fields() -> None:
    """The scripts' parameter expansions, run on the heredoc's own output shape."""
    out = subprocess.run(
        [
            "bash",
            "-c",
            'wheel="crucible-0.1-py3-none-any.whl ' + "a" * 40 + '"; '
            'pinned_sha="${wheel#* }"; wheel="${wheel%% *}"; echo "$wheel|$pinned_sha"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "crucible-0.1-py3-none-any.whl|" + "a" * 40
