"""The gateway session: expired-session vs defect, paper only, read-only by construction."""

from __future__ import annotations

import sys

import pytest
from ib_fakes import FakeIB, FakeSdk

from crucible_trader.broker_session import (
    CONNECT_TIMEOUT_S,
    REAUTH_STEP,
    SESSION_UNAVAILABLE_PREFIX,
    BrokerSessionUnavailableError,
    BrokerSettingsError,
    GatewayAddress,
    OrderRefusedError,
    ReadOnlyBroker,
    connect,
    load_sdk,
)

ADDRESS = GatewayAddress("127.0.0.1", 4002, 7)


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
