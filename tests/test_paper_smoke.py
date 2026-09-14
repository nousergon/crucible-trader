"""The trader smoke: right wheel, paper book readable, NO order -- and its manifest
is the one `crucible release.pin --target trader` selects on."""

from __future__ import annotations

import json

import crucible.release
import pytest
from crucible.manifest import read_manifest
from crucible.models import JOB_VALUES, MetricRecordRow
from crucible.release import TRADER_PIN_KEY, passing_trader_smoke
from crucible.runner import run_job
from crucible.store import LocalStore
from ib_fakes import DAY, FakeIB, FakeSdk, ctx_for, environ_for

from crucible_trader import paper_smoke
from crucible_trader.broker_session import BrokerSessionUnavailableError, OrderRefusedError
from crucible_trader.paper_smoke import (
    SMOKE_SCHEMA_VERSION,
    TRADER_SMOKE_JOB,
    RunningWheelMismatchError,
    assert_running_wheel,
    expected_version,
    installed_crucible_version,
    main,
    paper_smoke_key,
    run_smoke,
    running_wheel_disagreement,
    smoke_cycle,
)

SHA = "c" * 40


def _real_run_job_on_day(job, fn, **kwargs):
    """The harness's real runner, with the trading day pinned so a test never
    grades against the wall clock (`main` takes no day), and the run mode
    declared where the caller left it to the environment."""
    return run_job(
        job, fn, **{**kwargs, "trading_day": DAY, "run_mode": kwargs.get("run_mode") or "live"}
    )


VERSION = f"0.1.0+g{'c' * 12}"


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def test_the_job_name_is_the_harness_constant() -> None:
    # crucible-PR294 exports it; `passing_trader_smoke` selects on that object.
    assert TRADER_SMOKE_JOB is crucible.release.TRADER_SMOKE_JOB == "trader.smoke"


class TestRunningWheel:
    def test_the_expected_version_is_the_harness_derivation(self) -> None:
        assert expected_version(SHA) == VERSION

    def test_a_different_installed_wheel_is_refused(self) -> None:
        with pytest.raises(RunningWheelMismatchError, match="certifies nothing"):
            assert_running_wheel(SHA, "0.1.0")

    def test_the_installed_version_is_read_from_metadata(self) -> None:
        assert installed_crucible_version().startswith("0.1.0")


class TestPinDisagreementPageCondition:
    def _pin(self, store, sha):
        store.put_bytes(
            TRADER_PIN_KEY,
            json.dumps(
                {"sha": sha, "target": "trader", "pinned_at": "2026-09-05T22:00:00Z"}
            ).encode(),
        )

    def test_an_unset_pin_is_a_disagreement(self, store) -> None:
        assert "unset" in running_wheel_disagreement(store, VERSION)

    def test_a_pin_naming_another_wheel_is_a_disagreement(self, store) -> None:
        self._pin(store, "d" * 40)
        assert "running crucible" in running_wheel_disagreement(store, VERSION)

    def test_a_pin_naming_the_running_wheel_agrees(self, store) -> None:
        self._pin(store, SHA)
        assert running_wheel_disagreement(store, VERSION) is None


class TestSmokeCycle:
    def test_reads_the_book_places_no_order_and_files_its_output(self, store) -> None:
        ib = FakeIB(positions={"AAA": 10, "BBB": -4}, cash=2_500.0)
        ctx = ctx_for(store, TRADER_SMOKE_JOB)
        document = smoke_cycle(ctx, sha=SHA, client=ib, installed_version=VERSION)
        assert ib.order_calls == [] and ib.cancel_calls == 0
        stored = json.loads(store.get_bytes(paper_smoke_key(DAY.isoformat(), SHA)))
        assert stored == document
        assert document["schema_version"] == SMOKE_SCHEMA_VERSION
        assert (document["n_positions"], document["cash_readable"]) == (2, True)
        (metric,) = ctx.metrics
        MetricRecordRow.model_validate(metric)

    def test_the_session_it_reads_through_cannot_place_an_order(self, store, monkeypatch) -> None:
        """The refusal to place an order is asserted, not conventional: the reader
        is handed a session on which `placeOrder` itself raises."""
        seen = []
        original = paper_smoke.IbPaperStatementSource

        def spy(client):
            seen.append(client)
            return original(client)

        monkeypatch.setattr(paper_smoke, "IbPaperStatementSource", spy)
        ib = FakeIB(positions={"AAA": 1})
        smoke_cycle(ctx_for(store, TRADER_SMOKE_JOB), sha=SHA, client=ib, installed_version=VERSION)
        with pytest.raises(OrderRefusedError):
            seen[0].placeOrder(object(), object())
        assert ib.order_calls == []

    def test_the_wrong_wheel_fails_before_the_book_is_read(self, store) -> None:
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"the broker was touched: {name}")

        with pytest.raises(RunningWheelMismatchError):
            smoke_cycle(
                ctx_for(store, TRADER_SMOKE_JOB),
                sha=SHA,
                client=Untouchable(),
                installed_version="0.1.0",
            )

    def test_unreadable_cash_is_a_failure_not_zero(self, store) -> None:
        ib = FakeIB()
        ib.accountSummary = lambda: []
        with pytest.raises(Exception, match="Cash that cannot be read is not zero"):
            smoke_cycle(
                ctx_for(store, TRADER_SMOKE_JOB), sha=SHA, client=ib, installed_version=VERSION
            )


class TestRunSmoke:
    def test_writes_a_schema_valid_manifest_the_trader_pin_selects(self, store) -> None:
        assert TRADER_SMOKE_JOB in JOB_VALUES
        ib = FakeIB(positions={"AAA": 1})
        run_smoke(
            store,
            SHA,
            connector=lambda: ib,
            installed_version=VERSION,
            trading_day=DAY,
            run_mode="live",
        )
        assert ib.order_calls == [] and ib.disconnected == 1
        manifest = read_manifest(
            store, TRADER_SMOKE_JOB, DAY.isoformat(), discriminator=SHA[:12]
        )  # validated on read
        assert (manifest["status"], manifest["release_sha"]) == ("ok", SHA)
        assert store.exists(paper_smoke_key(DAY.isoformat(), SHA))
        passing_trader_smoke(store, SHA)  # raises TraderPinRefusedError when it does not select

    def test_an_expired_session_propagates_named_and_nothing_was_connected(
        self, store, monkeypatch
    ) -> None:
        monkeypatch.setattr(paper_smoke, "run_job", _real_run_job_on_day)

        def refused():
            raise BrokerSessionUnavailableError("broker_session_unavailable: logged out")

        with pytest.raises(BrokerSessionUnavailableError):
            run_smoke(store, SHA, connector=refused, installed_version=VERSION)

    def test_a_bad_sha_is_refused_before_anything_runs(self, store) -> None:
        with pytest.raises(ValueError, match="40-character"):
            run_smoke(store, "abc", connector=lambda: None, installed_version=VERSION)


def test_main_wires_settings_gateway_and_a_readonly_connect(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paper_smoke, "run_job", _real_run_job_on_day)
    ib = FakeIB(positions={"AAA": 2})
    rc = main(
        ["--release", SHA],
        environ=environ_for(tmp_path),
        sdk_loader=FakeSdk(ib),
        version_reader=lambda: VERSION,
    )
    assert rc == 0
    assert ib.connected_with["readonly"] is True and ib.disconnected == 1
    assert ib.order_calls == []
    assert LocalStore(tmp_path).exists(paper_smoke_key(DAY.isoformat(), SHA))


def test_main_reads_the_process_environment_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paper_smoke, "run_job", _real_run_job_on_day)
    for var, value in environ_for(tmp_path).items():
        monkeypatch.setenv(var, value)
    ib = FakeIB()
    assert main(["--release", SHA], sdk_loader=FakeSdk(ib), version_reader=lambda: VERSION) == 0
