"""The operator commands: one command fires the switch; every action is a run."""

from __future__ import annotations

import json

import pytest
from crucible.keys import parse_manifest_key
from crucible.manifest import read_manifest
from crucible.models import JOB_VALUES
from crucible.runner import run_job
from crucible.store import LocalStore
from ib_fakes import DAY, FakeIB, FakeSdk, environ_for

from crucible_trader import commands
from crucible_trader.commands import main
from crucible_trader.fire_drill import FIRE_DRILL_JOB, fire_drill_key
from crucible_trader.kill_switch import KILL_SWITCH_JOB, kill_switch_event_key, read_state


def _never_loaded():
    raise AssertionError("this command must not connect to the broker")


def _real_run_job_on_day(job, fn, **kwargs):
    """The harness's real runner, with the trading day pinned so a test never
    grades against the wall clock -- the same wrapper `test_paper_smoke.py`
    uses for `trader.smoke`. `trader.kill_switch` and `trader.fire_drill` are
    now admitted too (`crucible.models.JOB_VALUES`, since
    `nousergon/crucible@88f6e24`), so every command test below exercises this
    real runner; nothing stands in for it any more."""
    return run_job(
        job, fn, **{**kwargs, "trading_day": DAY, "run_mode": kwargs.get("run_mode") or "live"}
    )


def _main(tmp_path, argv, ib=None, loader=None, **kw):
    ib = ib or FakeIB(positions={"AAA": 4})
    return main(
        argv,
        environ=environ_for(tmp_path),
        sdk_loader=loader or FakeSdk(ib),
        clock=ib.clock,
        **kw,
    ), ib


def _only_manifest(store, job: str) -> dict:
    """The one manifest ``job`` wrote under ``runs/``, read back through the
    real validator (`crucible.manifest.read_manifest`) -- proof the run went
    through the admitted harness contract, not a stand-in."""
    keys = [k for k in store.list_keys(f"runs/{job}/") if k.endswith("/run.json")]
    assert len(keys) == 1, f"expected exactly one {job} manifest, found {keys}"
    parsed_job, day, discriminator = parse_manifest_key(keys[0])
    assert parsed_job == job
    return read_manifest(store, job, day, discriminator=discriminator)


class TestKillSwitchCommand:
    def test_one_command_flattens_the_paper_book(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(commands, "run_job", _real_run_job_on_day)
        rc, ib = _main(
            tmp_path, ["kill-switch", "fire", "--mode", "flatten", "--cause", "incident"]
        )
        store = LocalStore(tmp_path)
        assert rc == 0 and ib.book == {}
        assert ib.connected_with["readonly"] is False and ib.disconnected == 1
        assert read_state(store)["mode"] == "flatten"
        manifest = _only_manifest(store, KILL_SWITCH_JOB)
        assert manifest["status"] == "ok"
        assert store.exists(kill_switch_event_key(DAY.isoformat(), manifest["run_id"]))

    def test_release_never_connects(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(commands, "run_job", _real_run_job_on_day)
        _main(tmp_path, ["kill-switch", "fire", "--mode", "freeze", "--cause", "incident"])
        rc, _ = _main(
            tmp_path, ["kill-switch", "release", "--cause", "resolved"], loader=_never_loaded
        )
        assert rc == 0 and read_state(LocalStore(tmp_path))["engaged"] is False

    def test_the_fire_writes_a_schema_valid_manifest_through_the_real_runner(
        self, tmp_path, monkeypatch
    ) -> None:
        """`trader.kill_switch` is admitted (`crucible.models.JOB_VALUES`, since
        `nousergon/crucible@88f6e24`): the halt document is still written
        FIRST, before any broker call (the protection never waited on
        registration), but the enclosing run now also files a manifest the
        harness's own writer validates on write and this test validates
        again on read -- through `crucible.runner.run_job` and
        `crucible.manifest.read_manifest`, never a fake."""
        assert KILL_SWITCH_JOB in JOB_VALUES
        monkeypatch.setattr(commands, "run_job", _real_run_job_on_day)
        rc, _ = _main(tmp_path, ["kill-switch", "fire", "--mode", "freeze", "--cause", "incident"])
        store = LocalStore(tmp_path)
        assert rc == 0
        manifest = _only_manifest(store, KILL_SWITCH_JOB)
        assert manifest["status"] == "ok"
        assert read_state(store)["engaged"] is True
        assert store.exists(kill_switch_event_key(DAY.isoformat(), manifest["run_id"]))

    def test_flatten_and_freeze_only_from_the_command_line(self, tmp_path) -> None:
        with pytest.raises(SystemExit):
            _main(tmp_path, ["kill-switch", "fire", "--mode", "hold", "--cause", "x"])


class TestFireDrillCommand:
    def test_an_unannounced_drill_records_itself_as_unannounced(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(commands, "run_job", _real_run_job_on_day)
        rc, ib = _main(
            tmp_path, ["fire-drill", "run", "--kind", "kill_switch_freeze", "--unannounced"]
        )
        store = LocalStore(tmp_path)
        manifest = _only_manifest(store, FIRE_DRILL_JOB)
        artifact = json.loads(store.get_bytes(fire_drill_key(DAY.isoformat(), manifest["run_id"])))
        assert rc == 0 and artifact["announced"] is False and artifact["passed"] is True
        assert ib.disconnected == 1

    def test_the_drill_writes_a_schema_valid_manifest_through_the_real_runner(
        self, tmp_path, monkeypatch
    ) -> None:
        """`trader.fire_drill` is admitted (`crucible.models.JOB_VALUES`, since
        `nousergon/crucible@88f6e24`): the drill's own artifact is still
        written by `drill_cycle` before the run returns, but the enclosing
        run now also files a manifest the harness's own writer validates on
        write and this test validates again on read -- through
        `crucible.runner.run_job` and `crucible.manifest.read_manifest`,
        never a fake."""
        assert FIRE_DRILL_JOB in JOB_VALUES
        monkeypatch.setattr(commands, "run_job", _real_run_job_on_day)
        rc, _ = _main(
            tmp_path, ["fire-drill", "run", "--kind", "kill_switch_freeze", "--announced"]
        )
        store = LocalStore(tmp_path)
        assert rc == 0
        manifest = _only_manifest(store, FIRE_DRILL_JOB)
        assert manifest["status"] == "ok"
        artifact = json.loads(store.get_bytes(fire_drill_key(DAY.isoformat(), manifest["run_id"])))
        assert artifact["passed"] is True

    def test_status_is_not_done_when_undrilled(self, tmp_path) -> None:
        lines = []
        rc, _ = _main(
            tmp_path,
            ["fire-drill", "status", "--from", "2026-09-01", "--to", "2026-09-30"],
            loader=_never_loaded,
            printer=lines.append,
        )
        assert rc == 1 and lines[0].startswith("NOT DONE: no fire drill artifact")

    def test_status_is_done_after_two_passing_drills_one_unannounced(
        self, tmp_path, monkeypatch
    ) -> None:
        store = LocalStore(tmp_path)
        for run_id, announced in (("r1", True), ("r2", False)):
            store.put_bytes(
                fire_drill_key("2026-09-08", run_id),
                json.dumps(
                    {
                        "schema_version": "fire_drill.v1",
                        "kind": "kill_switch_flatten",
                        "announced": announced,
                        "trading_day": "2026-09-08",
                        "fired_at": "a",
                        "settled_at": "b",
                        "settle_seconds": 2.0,
                        "bound_seconds": 300.0,
                        "orders_accepted_after_fire": [],
                        "passed": True,
                    }
                ).encode(),
            )
        lines = []
        rc, _ = _main(
            tmp_path,
            ["fire-drill", "status", "--from", "2026-09-01", "--to", "2026-09-30"],
            loader=_never_loaded,
            printer=lines.append,
        )
        assert rc == 0 and lines[0].startswith("DONE")


def test_the_store_comes_from_the_process_environment_by_default(tmp_path, monkeypatch) -> None:
    for var, value in environ_for(tmp_path).items():
        monkeypatch.setenv(var, value)
    printed = []
    rc = main(
        ["fire-drill", "status", "--from", "2026-09-01", "--to", "2026-09-02"],
        printer=printed.append,
    )
    assert rc == 1 and printed
