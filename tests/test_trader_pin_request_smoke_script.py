"""The box-side smoke of the QUEUED trader pin (alpha-engine-config-I11545):
valid bash, refuses before it reads anything, exits 0 with a reason when no
smoke is owed, hands a fresh request's sha to `trader_paper_smoke.sh`, and
fails -- never shrugs -- when the request cannot be read.

Run end to end against a local store through this checkout's own locked
environment, exactly as the box runs it. The broker half is not reachable in
CI: a fresh request is shown reaching `trader_paper_smoke.sh`, whose own first
refusal (no gateway configured) is the proof the sha arrived there validated.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

from crucible.release import TRADER_PIN_KEY
from crucible.store import LocalStore

from crucible_trader.pin_request import REQUEST_KEY, UNREADABLE_EXIT

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "trader_pin_request_smoke.sh"
SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 40


def _run(*args: str, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    passed = {k: os.environ[k] for k in ("PATH", "HOME", "UV_CACHE_DIR") if k in os.environ}
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**passed, **(extra or {})},
        check=False,
    )


def _store(tmp_path, *, pin: str | None, sha: str | None, from_sha: str | None = SHA_A):
    store = LocalStore(tmp_path)
    if pin is not None:
        store.put_bytes(
            TRADER_PIN_KEY,
            json.dumps(
                {
                    "sha": pin,
                    "target": "trader",
                    "pinned_at": "2026-09-08T21:00:00Z",
                    "smoke_run_id": "01JG0000000000000000000000",
                    "smoke_status": "ok",
                    "smoke_manifest_key": f"runs/trader.smoke/2026-09-08/{pin[:12]}/run.json",
                }
            ).encode(),
        )
    if sha is not None:
        store.put_bytes(
            REQUEST_KEY,
            json.dumps(
                {
                    "schema_version": "trader_pin_request.v1",
                    "sha": sha,
                    "from_sha": from_sha,
                    "requested_by": "brian",
                    "requested_at": "2026-09-08T18:00:00Z",
                    "run_url": None,
                }
            ).encode(),
        )
    return {"CRUCIBLE_TRADER_STORE_URI": str(tmp_path)}


def test_the_script_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_it_takes_no_arguments() -> None:
    result = _run(SHA_B)
    assert result.returncode == 2 and "usage" in result.stderr


def test_it_refuses_without_a_store_before_reading_anything() -> None:
    result = _run()
    assert result.returncode == 2 and "CRUCIBLE_TRADER_STORE_URI is unset" in result.stderr


def test_nothing_queued_exits_zero_with_a_reason(tmp_path) -> None:
    result = _run(extra=_store(tmp_path, pin=SHA_A, sha=None))
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("trader-pin-request: none: ")


def test_a_request_already_pinned_exits_zero(tmp_path) -> None:
    result = _run(extra=_store(tmp_path, pin=SHA_B, sha=SHA_B))
    assert result.returncode == 0, result.stderr
    assert "trader-pin-request: noop: " in result.stdout


def test_a_stale_request_exits_zero_and_smokes_nothing(tmp_path) -> None:
    result = _run(extra=_store(tmp_path, pin=SHA_C, sha=SHA_B))
    assert result.returncode == 0, result.stderr
    assert "trader-pin-request: stale: " in result.stdout
    assert "smoking" not in result.stdout


def test_a_fresh_request_hands_its_sha_to_the_paper_smoke(tmp_path) -> None:
    result = _run(extra=_store(tmp_path, pin=SHA_A, sha=SHA_B))
    assert f"smoking {SHA_B}" in result.stdout
    # trader_paper_smoke.sh's own first refusal: the sha passed its 40-hex
    # check and it stopped at the missing gateway, before any download.
    assert result.returncode == 2
    assert "CRUCIBLE_TRADER_IB_HOST is unset; the smoke has no default" in result.stderr


def test_an_unreadable_request_fails_rather_than_reading_as_none(tmp_path) -> None:
    env = _store(tmp_path, pin=SHA_A, sha=None)
    LocalStore(tmp_path).put_bytes(REQUEST_KEY, b'{"sha": "not-a-sha"}')
    result = _run(extra=env)
    assert result.returncode == UNREADABLE_EXIT
    assert "does not conform" in result.stderr
    assert "trader-pin-request:" not in result.stdout


def test_the_fresh_branch_is_the_only_one_that_smokes() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("trader_paper_smoke.sh") == 3  # header x2, one exec
    assert 'exec "$repo/scripts/trader_paper_smoke.sh" "$sha"' in text
