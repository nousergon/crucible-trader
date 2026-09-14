"""The operator commands: one command fires the switch; every action is a run."""

from __future__ import annotations

import json

import pytest
from crucible.keys import parse_manifest_key
from crucible.manifest import read_manifest
from crucible.models import JOB_VALUES
from crucible.runner import run_job
from crucible.store import LocalStore, S3Store
from ib_fakes import DAY, FakeIB, FakeSdk, environ_for
from s3_fake import FakeS3, FakeSsm

from crucible_trader import commands
from crucible_trader.commands import main
from crucible_trader.drill_schedule import NONCE_ENV_VAR
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


def _main(tmp_path, argv, ib=None, loader=None, environ_extra=None, **kw):
    ib = ib or FakeIB(positions={"AAA": 4})
    return main(
        argv,
        environ=environ_for(tmp_path, **(environ_extra or {})),
        sdk_loader=loader or FakeSdk(ib),
        clock=ib.clock,
        **kw,
    ), ib


def _seal_via_command(tmp_path, ib, ssm, fire_at: str) -> tuple[list[str], str]:
    """`fire-drill seal` through `main`; returns the printed run command's argv and
    the nonce the operator would fetch from SSM."""
    lines: list[str] = []
    main(
        ["fire-drill", "seal", "--from", "2026-09-08", "--to", "2026-09-11", "--fire-at", fire_at],
        environ=environ_for(tmp_path),
        sdk_loader=_never_loaded,
        clock=ib.clock,
        printer=lines.append,
        ssm_factory=lambda: ssm,
    )
    (parameter,) = list(ssm.parameters.values())
    run = lines[0].split("fire-drill run ", 1)[1].split()
    run[run.index("KIND")] = "kill_switch_freeze"
    return ["fire-drill", "run", *run], parameter["Value"]


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
    def test_an_unannounced_drill_fires_under_a_seal_and_reveals_it(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(commands, "run_job", _real_run_job_on_day)
        ib = FakeIB(positions={"AAA": 4})
        ssm = FakeSsm()
        run_argv, nonce = _seal_via_command(tmp_path, ib, ssm, "2026-09-08T14:00:30Z")
        ib.clock.advance(30)
        rc, _ = _main(tmp_path, run_argv, ib=ib, environ_extra={NONCE_ENV_VAR: nonce})
        store = LocalStore(tmp_path)
        manifest = _only_manifest(store, FIRE_DRILL_JOB)
        artifact = json.loads(store.get_bytes(fire_drill_key(DAY.isoformat(), manifest["run_id"])))
        assert rc == 0 and artifact["announced"] is False and artifact["passed"] is True
        assert artifact["schedule"]["nonce"] == nonce
        assert ib.disconnected == 1

    @pytest.mark.parametrize(
        ("argv", "environ_extra"),
        [
            pytest.param(["--unannounced"], {}, id="unannounced-without-seal-flags"),
            pytest.param(
                [
                    "--unannounced",
                    "--schedule-id",
                    "a",
                    "--window-from",
                    "2026-09-08",
                    "--window-to",
                    "2026-09-11",
                    "--fire-at",
                    "2026-09-08T14:00:00Z",
                ],
                {},
                id="unannounced-without-the-nonce",
            ),
            pytest.param(["--announced", "--schedule-id", "a"], {}, id="announced-with-a-seal"),
        ],
    )
    def test_a_seal_mismatch_on_the_command_line_is_refused(
        self, tmp_path, argv, environ_extra
    ) -> None:
        with pytest.raises(SystemExit):
            _main(
                tmp_path,
                ["fire-drill", "run", "--kind", "kill_switch_freeze", *argv],
                loader=_never_loaded,
                environ_extra=environ_extra,
            )

    def test_seal_never_connects_and_never_prints_the_nonce(self, tmp_path) -> None:
        ssm = FakeSsm()
        lines: list[str] = []
        rc, _ = _main(
            tmp_path,
            [
                "fire-drill",
                "seal",
                "--from",
                "2026-09-08",
                "--to",
                "2026-09-11",
                "--fire-at",
                "2026-09-08T15:00:00Z",
            ],
            loader=_never_loaded,
            printer=lines.append,
            ssm_factory=lambda: ssm,
        )
        (parameter,) = ssm.parameters.values()
        assert rc == 0 and lines[0].startswith("SEALED ") and parameter["Value"] not in lines[0]

    def test_the_default_ssm_client_is_boto3s(self, monkeypatch) -> None:
        import boto3

        monkeypatch.setattr(boto3, "client", lambda service: f"client:{service}")
        assert commands.default_ssm_client() == "client:ssm"

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
        assert rc == 1 and lines[0].startswith("NOT DONE:") and "undrilled" in lines[0]

    def test_status_is_the_gates_reading_done_only_on_a_sealed_unannounced_drill(
        self, tmp_path, monkeypatch
    ) -> None:
        """End to end through the real runner and the harness's own clause, over an
        S3Store (the seal's write time is the store's `LastModified`)."""
        monkeypatch.setattr(commands, "run_job", _real_run_job_on_day)
        ib = FakeIB(positions={"AAA": 4})
        s3 = S3Store("a-test-store", "crucible", client=FakeS3(now=ib.clock))
        monkeypatch.setattr(commands, "open_configured_store", lambda settings: s3)
        status = ["fire-drill", "status", "--from", "2026-09-01", "--to", "2026-09-30"]

        lines: list[str] = []
        _main(tmp_path, ["fire-drill", "run", "--kind", "kill_switch_freeze", "--announced"], ib=ib)
        rc, _ = _main(tmp_path, status, loader=_never_loaded, printer=lines.append)
        assert rc == 1 and "0 unannounced" in lines[0], lines

        ib.clock.advance(5)
        run_argv, nonce = _seal_via_command(tmp_path, ib, FakeSsm(), "2026-09-08T14:01:00Z")
        ib.clock.advance(55)
        run_argv[run_argv.index("--kind") + 1] = "kill_switch_flatten"
        _main(tmp_path, run_argv, ib=ib, environ_extra={NONCE_ENV_VAR: nonce})

        rc, _ = _main(tmp_path, status, loader=_never_loaded, printer=lines.append)
        assert rc == 0 and lines[-1].startswith("DONE:") and "1 unannounced" in lines[-1], lines

    def test_status_says_unmeasurable_when_the_store_cannot_be_listed(
        self, tmp_path, monkeypatch
    ) -> None:
        class Unlistable(LocalStore):
            def list_keys(self, prefix=""):
                raise PermissionError("AccessDenied")

        monkeypatch.setattr(
            commands, "open_configured_store", lambda settings: Unlistable(tmp_path)
        )
        lines: list[str] = []
        rc, _ = _main(
            tmp_path,
            ["fire-drill", "status", "--from", "2026-09-01", "--to", "2026-09-30"],
            loader=_never_loaded,
            printer=lines.append,
        )
        assert rc == 1 and lines[0].startswith("UNMEASURABLE:")


def test_the_store_comes_from_the_process_environment_by_default(tmp_path, monkeypatch) -> None:
    for var, value in environ_for(tmp_path).items():
        monkeypatch.setenv(var, value)
    printed = []
    rc = main(
        ["fire-drill", "status", "--from", "2026-09-01", "--to", "2026-09-02"],
        printer=printed.append,
    )
    assert rc == 1 and printed


def test_a_drill_reads_its_nonce_from_the_process_environment_by_default(
    tmp_path, monkeypatch
) -> None:
    for var, value in environ_for(tmp_path).items():
        monkeypatch.setenv(var, value)
    monkeypatch.delenv(NONCE_ENV_VAR, raising=False)
    with pytest.raises(SystemExit):
        main(
            [
                "fire-drill",
                "run",
                "--kind",
                "kill_switch_freeze",
                *[
                    "--unannounced",
                    "--schedule-id",
                    "a",
                    "--window-from",
                    "2026-09-08",
                    "--window-to",
                    "2026-09-11",
                    "--fire-at",
                    "2026-09-08T14:00:00Z",
                ],
            ],
            sdk_loader=_never_loaded,
        )
