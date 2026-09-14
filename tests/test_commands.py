"""The operator commands: one command fires the switch; every action is a run."""

from __future__ import annotations

import json

import pytest
from crucible.manifest import ManifestValidationError
from crucible.models import JOB_VALUES
from crucible.store import LocalStore
from ib_fakes import DAY, FakeIB, FakeSdk, environ_for, fake_run_job

from crucible_trader import commands
from crucible_trader.commands import main
from crucible_trader.fire_drill import FIRE_DRILL_JOB, fire_drill_key
from crucible_trader.kill_switch import KILL_SWITCH_JOB, kill_switch_event_key, read_state

RUN_ID = "01JG0000000000000000000000"


def _never_loaded():
    raise AssertionError("this command must not connect to the broker")


def _main(tmp_path, argv, ib=None, loader=None, **kw):
    ib = ib or FakeIB(positions={"AAA": 4})
    return main(
        argv,
        environ=environ_for(tmp_path),
        sdk_loader=loader or FakeSdk(ib),
        clock=ib.clock,
        **kw,
    ), ib


class TestKillSwitchCommand:
    def test_one_command_flattens_the_paper_book(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(commands, "run_job", fake_run_job)
        rc, ib = _main(
            tmp_path, ["kill-switch", "fire", "--mode", "flatten", "--cause", "incident"]
        )
        store = LocalStore(tmp_path)
        assert rc == 0 and ib.book == {}
        assert ib.connected_with["readonly"] is False and ib.disconnected == 1
        assert read_state(store)["mode"] == "flatten"
        assert store.exists(kill_switch_event_key(DAY.isoformat(), RUN_ID))

    def test_release_never_connects(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(commands, "run_job", fake_run_job)
        _main(tmp_path, ["kill-switch", "fire", "--mode", "freeze", "--cause", "incident"])
        rc, _ = _main(
            tmp_path, ["kill-switch", "release", "--cause", "resolved"], loader=_never_loaded
        )
        assert rc == 0 and read_state(LocalStore(tmp_path))["engaged"] is False

    def test_the_halt_is_written_even_while_the_job_is_unregistered(self, tmp_path) -> None:
        """Contract dependency: the manifest is refused until `trader.kill_switch` is
        admitted, but the protection does not wait on that registration."""
        assert KILL_SWITCH_JOB not in JOB_VALUES, (
            "crucible now admits trader.kill_switch: bump the pin, delete this test, and "
            "assert the command writes a valid runs/trader.kill_switch manifest"
        )
        with pytest.raises(ManifestValidationError, match="job"):
            _main(tmp_path, ["kill-switch", "fire", "--mode", "freeze", "--cause", "incident"])
        assert read_state(LocalStore(tmp_path))["engaged"] is True

    def test_flatten_and_freeze_only_from_the_command_line(self, tmp_path) -> None:
        with pytest.raises(SystemExit):
            _main(tmp_path, ["kill-switch", "fire", "--mode", "hold", "--cause", "x"])


class TestFireDrillCommand:
    def test_an_unannounced_drill_records_itself_as_unannounced(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(commands, "run_job", fake_run_job)
        rc, ib = _main(
            tmp_path, ["fire-drill", "run", "--kind", "kill_switch_freeze", "--unannounced"]
        )
        artifact = json.loads(
            LocalStore(tmp_path).get_bytes(fire_drill_key(DAY.isoformat(), RUN_ID))
        )
        assert rc == 0 and artifact["announced"] is False and artifact["passed"] is True
        assert ib.disconnected == 1

    def test_the_drill_job_is_not_yet_admitted(self) -> None:
        assert FIRE_DRILL_JOB not in JOB_VALUES, (
            "crucible now admits trader.fire_drill: bump the pin and delete this test"
        )

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
