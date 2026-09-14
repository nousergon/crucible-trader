"""The IB Gateway PAPER session: connect, classify why a connect failed, and a
read-only view that cannot place an order.

`alpha-engine-config-I10649` (the trader smoke) and `-I10650` (the kill switch
and its drill) both open a broker session. Three properties live here so
neither re-derives them:

1. **"The broker session has expired" is a different failure from "the trader is
   broken"** (plan §10.7 row 9: IB Gateway paper re-authentication is a standing
   human touch). A refused or timed-out connect, or a connected session that
   manages no account, raises :class:`BrokerSessionUnavailableError`, whose text
   starts with :data:`SESSION_UNAVAILABLE_PREFIX` and names the human step. Every
   other exception propagates as itself -- a defect. Collapsing the two produces
   a page nobody can act on.
2. **Paper only.** A session whose single managed account is not `DU…` is
   refused. The paper -> live crossing is a reserved ruling (plan §9.5, phase 6).
3. **Read-only by construction, not by convention.** :class:`ReadOnlyBroker`
   exposes the three reads a statement needs and nothing else; any other
   attribute -- `placeOrder`, `cancelOrder`, `reqGlobalCancel` -- raises
   :class:`OrderRefusedError`. The connect itself also asks the gateway for a
   `readonly` API session, so a bypass would be refused twice.

The SDK is `ib_async` (the maintained successor of `ib_insync`, which the v1
executor used), declared as this package's `ib` extra and imported lazily so the
library and its tests carry no broker SDK. Host, port and client id come from the
environment with no defaults: a default port is how a paper tool connects to a
live gateway.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from collections.abc import Callable, Iterable
from typing import Any

from crucible_trader.broker_statement import PAPER_ACCOUNT_PREFIX

HOST_VAR = "CRUCIBLE_TRADER_IB_HOST"
PORT_VAR = "CRUCIBLE_TRADER_IB_PORT"
CLIENT_ID_VAR = "CRUCIBLE_TRADER_IB_CLIENT_ID"

#: Seconds a connect may take before it is read as "no gateway session".
CONNECT_TIMEOUT_S = 15.0

#: The prefix every session-unavailable reason carries, so a manifest `reason`
#: (and the page built from it) is classifiable by a reader without a regex over
#: prose.
SESSION_UNAVAILABLE_PREFIX = "broker_session_unavailable"

#: The human step, stated once. The gateway's API port closes when its daily
#: session lapses and it returns to the login screen, which is why a refused
#: connect is read as this and not as a defect.
REAUTH_STEP = (
    "HUMAN ACTION: re-authenticate IB Gateway PAPER on the trader box -- open the "
    "gateway, log in to the paper account (complete 2FA), confirm the API port is "
    "listening, then re-run this job. Nothing in the trader needs changing."
)


class BrokerSessionUnavailableError(RuntimeError):
    """No authenticated paper gateway session. A human action, not a defect."""


class OrderRefusedError(RuntimeError):
    """Something tried to reach an order-capable call through a read-only session."""


class BrokerSettingsError(RuntimeError):
    """The gateway address is not configured. Page and exit; never default."""


@dataclasses.dataclass(frozen=True)
class GatewayAddress:
    host: str
    port: int
    client_id: int

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> GatewayAddress:
        source = os.environ if env is None else env
        values = {
            var: (source.get(var) or "").strip() for var in (HOST_VAR, PORT_VAR, CLIENT_ID_VAR)
        }
        missing = sorted(var for var, value in values.items() if not value)
        if missing:
            raise BrokerSettingsError(
                f"{', '.join(missing)} unset. There is no default gateway address: a default "
                "port is how a paper tool connects to a live gateway."
            )
        try:
            port, client_id = int(values[PORT_VAR]), int(values[CLIENT_ID_VAR])
        except ValueError as exc:
            raise BrokerSettingsError(
                f"{PORT_VAR}/{CLIENT_ID_VAR} must be integers: {exc}"
            ) from exc
        return cls(values[HOST_VAR], port, client_id)


def load_sdk() -> Any:
    """`ib_async`, imported only when a real session is opened."""
    return importlib.import_module("ib_async")


def connect(
    address: GatewayAddress,
    *,
    readonly: bool,
    sdk_loader: Callable[[], Any] = load_sdk,
) -> Any:
    """A connected, authenticated, PAPER `IB` client. See the module docstring.

    Only a refused or timed-out connect and an account-less session are read as
    the expired-session condition; any other exception is a defect and
    propagates unchanged.
    """
    sdk = sdk_loader()
    client = sdk.IB()
    try:
        client.connect(
            address.host,
            address.port,
            clientId=address.client_id,
            timeout=CONNECT_TIMEOUT_S,
            readonly=readonly,
        )
    except (ConnectionRefusedError, TimeoutError) as exc:
        raise BrokerSessionUnavailableError(
            f"{SESSION_UNAVAILABLE_PREFIX}: the gateway at {address.host}:{address.port} "
            f"refused or timed out the API connect ({type(exc).__name__}: {exc}). {REAUTH_STEP}"
        ) from exc
    accounts = list(client.managedAccounts())
    if not accounts:
        client.disconnect()
        raise BrokerSessionUnavailableError(
            f"{SESSION_UNAVAILABLE_PREFIX}: the gateway accepted the API connect but manages "
            f"no account, which is a session that is not logged in. {REAUTH_STEP}"
        )
    live = [a for a in accounts if not a.startswith(PAPER_ACCOUNT_PREFIX)]
    if live:
        client.disconnect()
        raise OrderRefusedError(
            f"the gateway manages non-paper account(s) {live}. This tool acts on IB paper "
            f"({PAPER_ACCOUNT_PREFIX}…) accounts only; the paper -> live crossing is a "
            "reserved ruling (plan §9.5)."
        )
    return client


class ReadOnlyBroker:
    """The statement reads of an `IB` client, and nothing that can trade."""

    _READS = frozenset({"managedAccounts", "positions", "accountSummary"})

    def __init__(self, client: Any) -> None:
        object.__setattr__(self, "_client", client)

    def managedAccounts(self) -> list[str]:  # noqa: N802 - SDK spelling
        return list(self._client.managedAccounts())

    def positions(self) -> Iterable[Any]:
        return self._client.positions()

    def accountSummary(self) -> Iterable[Any]:  # noqa: N802 - SDK spelling
        return self._client.accountSummary()

    def __getattr__(self, name: str) -> Any:
        raise OrderRefusedError(
            f"{name!r} is not available on a read-only broker session (reads: "
            f"{sorted(self._READS)}). The smoke places no order, and that is enforced here "
            "rather than promised."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise OrderRefusedError(f"a read-only broker session cannot be modified ({name!r})")
