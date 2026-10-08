"""The gateway session: expired-session vs defect, paper only, read-only by construction."""

from __future__ import annotations

import socket
import sys

import pytest
from conftest import LONG_UPTIME_S
from ib_fakes import FakeIB, FakeSdk

from crucible_trader.broker_session import (
    BOOT_GRACE_S,
    CONNECT_TIMEOUT_S,
    GATEWAY_NOT_READY_PREFIX,
    READY_DEADLINE_S,
    READY_POLL_S,
    REAUTH_STEP,
    SESSION_UNAVAILABLE_PREFIX,
    BrokerSessionUnavailableError,
    BrokerSettingsError,
    GatewayAddress,
    GatewayNotReadyError,
    OrderRefusedError,
    ReadOnlyBroker,
    connect,
    load_sdk,
    port_listening,
    read_uptime_s,
    wait_for_port,
)

ADDRESS = GatewayAddress("127.0.0.1", 4002, 7)
ENV = {
    "CRUCIBLE_TRADER_IB_HOST": "h",
    "CRUCIBLE_TRADER_IB_PORT": "4002",
    "CRUCIBLE_TRADER_IB_CLIENT_ID": "7",
}


class FakeMonotonic:
    """A monotonic clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Port:
    """A gateway port that starts listening after ``refusals`` probes."""

    def __init__(self, refusals: float) -> None:
        self.refusals = refusals
        self.probes = 0

    def __call__(self, host: str, port: int) -> bool:
        self.probes += 1
        return self.probes > self.refusals


class TestAddress:
    def test_every_missing_variable_is_named_and_nothing_defaults(self) -> None:
        with pytest.raises(BrokerSettingsError, match="CRUCIBLE_TRADER_IB_CLIENT_ID, .*PORT"):
            GatewayAddress.from_env({"CRUCIBLE_TRADER_IB_HOST": "h"})

    def test_a_non_integer_port_is_refused(self) -> None:
        with pytest.raises(BrokerSettingsError, match="integers"):
            GatewayAddress.from_env(
                {
                    "CRUCIBLE_TRADER_IB_HOST": "h",
                    "CRUCIBLE_TRADER_IB_PORT": "paper",
                    "CRUCIBLE_TRADER_IB_CLIENT_ID": "7",
                }
            )

    def test_a_complete_address_resolves(self) -> None:
        assert GatewayAddress.from_env(
            {
                "CRUCIBLE_TRADER_IB_HOST": " h ",
                "CRUCIBLE_TRADER_IB_PORT": "4002",
                "CRUCIBLE_TRADER_IB_CLIENT_ID": "7",
            }
        ) == GatewayAddress("h", 4002, 7)

    def test_the_readiness_timing_defaults_when_unset(self) -> None:
        address = GatewayAddress.from_env(ENV)
        assert (address.ready_deadline_s, address.boot_grace_s) == (READY_DEADLINE_S, BOOT_GRACE_S)
        # The cold-boot reading only holds if the grace outlasts the whole wait.
        assert BOOT_GRACE_S > READY_DEADLINE_S

    def test_the_readiness_timing_is_overridable(self) -> None:
        address = GatewayAddress.from_env(
            {
                **ENV,
                "CRUCIBLE_TRADER_IB_READY_DEADLINE_S": "60",
                "CRUCIBLE_TRADER_IB_BOOT_GRACE_S": " 0 ",
            }
        )
        assert (address.ready_deadline_s, address.boot_grace_s) == (60.0, 0.0)

    @pytest.mark.parametrize("raw", ["soon", "-1", "nan"])
    def test_a_malformed_timing_override_is_refused_not_defaulted(self, raw) -> None:
        with pytest.raises(BrokerSettingsError, match="CRUCIBLE_TRADER_IB_READY_DEADLINE_S"):
            GatewayAddress.from_env({**ENV, "CRUCIBLE_TRADER_IB_READY_DEADLINE_S": raw})

    def test_the_process_environment_is_the_default_source(self, monkeypatch) -> None:
        for var, value in (
            ("CRUCIBLE_TRADER_IB_HOST", "h"),
            ("CRUCIBLE_TRADER_IB_PORT", "1"),
            ("CRUCIBLE_TRADER_IB_CLIENT_ID", "2"),
        ):
            monkeypatch.setenv(var, value)
        assert GatewayAddress.from_env() == GatewayAddress("h", 1, 2)


def test_the_sdk_is_imported_lazily_by_name(monkeypatch) -> None:
    marker = object()
    monkeypatch.setitem(sys.modules, "ib_async", marker)
    assert load_sdk() is marker


class TestPortProbe:
    def test_a_listening_port_is_listening(self) -> None:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            assert port_listening("127.0.0.1", server.getsockname()[1]) is True

    def test_a_closed_port_is_not_listening(self) -> None:
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        assert port_listening("127.0.0.1", port) is False

    def test_a_timed_out_probe_is_not_listening(self, monkeypatch) -> None:
        def stalls(address, timeout):
            raise TimeoutError("timed out")

        monkeypatch.setattr(socket, "create_connection", stalls)
        assert port_listening("127.0.0.1", 4002) is False

    def test_any_other_socket_error_is_a_defect(self, monkeypatch) -> None:
        def unresolvable(address, timeout):
            raise socket.gaierror("Name or service not known")

        monkeypatch.setattr(socket, "create_connection", unresolvable)
        with pytest.raises(socket.gaierror):
            port_listening("no-such-host", 4002)


class TestUptime:
    def test_the_first_field_is_seconds_since_boot(self, tmp_path) -> None:
        path = tmp_path / "uptime"
        path.write_text("58.21 112.40\n")
        assert read_uptime_s(str(path)) == 58.21

    def test_an_unpublished_uptime_is_none(self, tmp_path) -> None:
        assert read_uptime_s(str(tmp_path / "absent")) is None


class TestWaitForPort:
    def test_a_listening_port_returns_without_sleeping(self) -> None:
        clock = FakeMonotonic()
        assert wait_for_port(ADDRESS, probe=Port(0), sleep=clock.sleep, monotonic=clock)
        assert clock.sleeps == []

    def test_it_polls_until_the_port_listens(self) -> None:
        clock, port = FakeMonotonic(), Port(11)
        assert wait_for_port(ADDRESS, probe=port, sleep=clock.sleep, monotonic=clock)
        assert port.probes == 12
        assert clock.sleeps == [READY_POLL_S] * 11

    def test_it_gives_up_at_the_deadline_and_never_sleeps_past_it(self) -> None:
        clock, port = FakeMonotonic(), Port(float("inf"))
        address = GatewayAddress("127.0.0.1", 4002, 7, ready_deadline_s=12.0)
        assert not wait_for_port(address, probe=port, sleep=clock.sleep, monotonic=clock)
        assert clock.sleeps == [5.0, 5.0, 2.0]
        # A final probe at the deadline itself, after the last sleep.
        assert port.probes == 4

    def test_a_zero_deadline_probes_once(self) -> None:
        clock, port = FakeMonotonic(), Port(float("inf"))
        address = GatewayAddress("127.0.0.1", 4002, 7, ready_deadline_s=0.0)
        assert not wait_for_port(address, probe=port, sleep=clock.sleep, monotonic=clock)
        assert (port.probes, clock.sleeps) == (1, [])


def _connect(ib: FakeIB, *, port: Port, uptime_s: float | None) -> FakeMonotonic:
    clock = FakeMonotonic()
    connect(
        ADDRESS,
        readonly=True,
        sdk_loader=FakeSdk(ib),
        probe=port,
        uptime=lambda: uptime_s,
        sleep=clock.sleep,
        monotonic=clock,
    )
    return clock


class TestBootRace:
    """The 10-07 postclose reconcile: connect ~58s after a cold boot, refused."""

    def test_a_gateway_that_opens_inside_the_wait_connects(self) -> None:
        ib, port = FakeIB(), Port(20)
        clock = _connect(ib, port=port, uptime_s=58.0)
        assert ib.connected_with is not None
        assert sum(clock.sleeps) == 20 * READY_POLL_S

    def test_a_port_still_closed_on_a_fresh_box_is_not_ready_not_reauth(self) -> None:
        ib = FakeIB()
        with pytest.raises(GatewayNotReadyError) as raised:
            _connect(ib, port=Port(float("inf")), uptime_s=58.0 + READY_DEADLINE_S)
        message = str(raised.value)
        assert message.startswith(GATEWAY_NOT_READY_PREFIX)
        assert "did not open its API port within 300s" in message
        assert "HUMAN ACTION" not in message and "re-authenticate" not in message
        # The SDK is never asked to connect to a port that never listened.
        assert ib.connected_with is None

    def test_it_is_still_a_session_unavailable_error_for_existing_callers(self) -> None:
        assert issubclass(GatewayNotReadyError, BrokerSessionUnavailableError)

    @pytest.mark.parametrize("error", [ConnectionRefusedError("refused"), TimeoutError("slow")])
    def test_a_refused_api_connect_on_a_fresh_box_is_not_ready(self, error) -> None:
        with pytest.raises(GatewayNotReadyError, match=GATEWAY_NOT_READY_PREFIX):
            _connect(FakeIB(connect_error=error), port=Port(0), uptime_s=120.0)

    def test_an_account_less_session_on_a_fresh_box_is_not_ready(self) -> None:
        ib = FakeIB(accounts=())
        with pytest.raises(GatewayNotReadyError, match="manages no account"):
            _connect(ib, port=Port(0), uptime_s=120.0)
        assert ib.disconnected == 1

    @pytest.mark.parametrize("uptime_s", [BOOT_GRACE_S, LONG_UPTIME_S, None])
    def test_past_the_boot_grace_or_with_no_uptime_it_is_the_human_action(self, uptime_s) -> None:
        with pytest.raises(BrokerSessionUnavailableError) as raised:
            _connect(FakeIB(), port=Port(float("inf")), uptime_s=uptime_s)
        assert not isinstance(raised.value, GatewayNotReadyError)
        assert str(raised.value).startswith(SESSION_UNAVAILABLE_PREFIX)
        assert REAUTH_STEP in str(raised.value)


class TestConnect:
    @pytest.mark.parametrize("error", [ConnectionRefusedError("refused"), TimeoutError("slow")])
    def test_a_refused_or_timed_out_connect_is_the_human_action(self, error) -> None:
        with pytest.raises(BrokerSessionUnavailableError) as raised:
            connect(ADDRESS, readonly=True, sdk_loader=FakeSdk(FakeIB(connect_error=error)))
        assert str(raised.value).startswith(SESSION_UNAVAILABLE_PREFIX)
        assert REAUTH_STEP in str(raised.value)

    def test_any_other_connect_failure_is_a_defect_and_propagates_as_itself(self) -> None:
        with pytest.raises(ValueError, match="bad client id"):
            connect(
                ADDRESS,
                readonly=True,
                sdk_loader=FakeSdk(FakeIB(connect_error=ValueError("bad client id"))),
            )

    def test_a_session_managing_no_account_is_not_logged_in(self) -> None:
        ib = FakeIB(accounts=())
        with pytest.raises(BrokerSessionUnavailableError, match="not logged in"):
            connect(ADDRESS, readonly=True, sdk_loader=FakeSdk(ib))
        assert ib.disconnected == 1

    def test_a_live_account_is_refused(self) -> None:
        ib = FakeIB(accounts=("U1234",))
        with pytest.raises(OrderRefusedError, match="non-paper"):
            connect(ADDRESS, readonly=False, sdk_loader=FakeSdk(ib))
        assert ib.disconnected == 1

    def test_a_paper_session_connects_with_the_requested_readonly_flag(self) -> None:
        ib = FakeIB()
        assert connect(ADDRESS, readonly=True, sdk_loader=FakeSdk(ib)) is ib
        assert ib.connected_with == {
            "host": "127.0.0.1",
            "port": 4002,
            "clientId": 7,
            "timeout": CONNECT_TIMEOUT_S,
            "readonly": True,
        }


class TestReadOnlyBroker:
    def test_reads_delegate(self) -> None:
        ib = FakeIB(positions={"AAA": 3})
        broker = ReadOnlyBroker(ib)
        assert broker.managedAccounts() == ["DU1"]
        assert [p.contract.symbol for p in broker.positions()] == ["AAA"]
        assert [r.tag for r in broker.accountSummary()] == ["TotalCashValue"]

    @pytest.mark.parametrize("name", ["placeOrder", "cancelOrder", "reqGlobalCancel", "client"])
    def test_every_other_attribute_is_refused(self, name) -> None:
        ib = FakeIB()
        with pytest.raises(OrderRefusedError, match="read-only"):
            getattr(ReadOnlyBroker(ib), name)
        assert ib.order_calls == [] and ib.cancel_calls == 0

    def test_it_cannot_be_modified(self) -> None:
        with pytest.raises(OrderRefusedError, match="cannot be modified"):
            ReadOnlyBroker(FakeIB())._client = object()
