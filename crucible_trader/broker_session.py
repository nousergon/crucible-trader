"""The IB Gateway PAPER session: connect, classify why a connect failed, and a
read-only view that cannot place an order.

`alpha-engine-config-I10649` (the trader smoke) and `-I10650` (the kill switch
and its drill) both open a broker session. Four properties live here so
neither re-derives them:

1. **"The broker session has expired" is a different failure from "the trader is
   broken"** (plan §10.7 row 9: IB Gateway paper re-authentication is a standing
   human touch). A refused or timed-out connect, or a connected session that
   manages no account, raises :class:`BrokerSessionUnavailableError`, whose text
   starts with :data:`SESSION_UNAVAILABLE_PREFIX` and names the human step. Every
   other exception propagates as itself -- a defect. Collapsing the two produces
   a page nobody can act on.
2. **"The gateway is still starting" is a third condition, not the first.** A
   box started for a job runs its broker connect about a minute after boot,
   while the dockerised gateway is still completing its login and its API port
   is not yet listening. So :func:`connect` first waits, bounded, for the port
   to listen (:func:`wait_for_port`), and a connect that still fails is
   classified by box uptime: inside :attr:`GatewayAddress.boot_grace_s` of boot
   it raises :class:`GatewayNotReadyError` (reason prefix
   :data:`GATEWAY_NOT_READY_PREFIX`, no re-authentication step, because none is
   indicated); only past it does the human re-login text apply. Reading a boot
   race as an expired session sends a human to fix a login that was fine.
3. **Paper only.** A session whose single managed account is not `DU…` is
   refused. The paper -> live crossing is a reserved ruling (plan §9.5, phase 6).
4. **Read-only by construction, not by convention.** :class:`ReadOnlyBroker`
   exposes the three reads a statement needs and nothing else; any other
   attribute -- `placeOrder`, `cancelOrder`, `reqGlobalCancel` -- raises
   :class:`OrderRefusedError`. The connect itself also asks the gateway for a
   `readonly` API session, so a bypass would be refused twice.

The SDK is `ib_async` (the maintained successor of `ib_insync`, which the v1
executor used), declared as this package's `ib` extra and imported lazily so the
library and its tests carry no broker SDK. Host, port and client id come from the
environment with no defaults: a default port is how a paper tool connects to a
live gateway. The readiness wait and the boot grace do have defaults
(:data:`READY_DEADLINE_S`, :data:`BOOT_GRACE_S`), overridable through
:data:`READY_DEADLINE_VAR` and :data:`BOOT_GRACE_VAR`: a wait length cannot
reach the wrong gateway.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import socket
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from crucible_trader.broker_statement import PAPER_ACCOUNT_PREFIX

HOST_VAR = "CRUCIBLE_TRADER_IB_HOST"
PORT_VAR = "CRUCIBLE_TRADER_IB_PORT"
CLIENT_ID_VAR = "CRUCIBLE_TRADER_IB_CLIENT_ID"
READY_DEADLINE_VAR = "CRUCIBLE_TRADER_IB_READY_DEADLINE_S"
BOOT_GRACE_VAR = "CRUCIBLE_TRADER_IB_BOOT_GRACE_S"

#: Seconds :func:`connect` waits for the gateway's API port to listen before it
#: stops waiting. Sized for a cold boot: the gateway container starts with the
#: box and its login (including the second factor) takes minutes, not seconds.
READY_DEADLINE_S = 300.0

#: Seconds between two probes of the gateway port while waiting.
READY_POLL_S = 5.0

#: Seconds after boot inside which a gateway that is still not accepting the API
#: is read as STARTING, not as an expired session. Larger than
#: :data:`READY_DEADLINE_S` plus the boot-to-connect delay, so a job started on
#: a cold box that exhausts the whole wait is still classified as a boot race.
BOOT_GRACE_S = 900.0

#: Where the kernel publishes seconds since boot (first field).
UPTIME_PATH = "/proc/uptime"

#: Seconds a connect may take before it is read as "no gateway session".
CONNECT_TIMEOUT_S = 15.0

#: The prefix every session-unavailable reason carries, so a manifest `reason`
#: (and the page built from it) is classifiable by a reader without a regex over
#: prose.
SESSION_UNAVAILABLE_PREFIX = "broker_session_unavailable"

#: The prefix of a session-unavailable reason that is a boot race: the box came
#: up recently and the gateway had not finished starting. Distinct from
#: :data:`SESSION_UNAVAILABLE_PREFIX` so a reader never pages a human to
#: re-authenticate a gateway that was simply not up yet.
GATEWAY_NOT_READY_PREFIX = "gateway_not_ready"

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


class GatewayNotReadyError(BrokerSessionUnavailableError):
    """The gateway had not finished starting on a recently booted box.

    A subclass, so every caller that handles :class:`BrokerSessionUnavailableError`
    still handles this; its reason carries :data:`GATEWAY_NOT_READY_PREFIX` and
    no human re-authentication step.
    """


class OrderRefusedError(RuntimeError):
    """Something tried to reach an order-capable call through a read-only session."""


class BrokerSettingsError(RuntimeError):
    """The gateway address is not configured. Page and exit; never default."""


@dataclasses.dataclass(frozen=True)
class GatewayAddress:
    host: str
    port: int
    client_id: int
    #: How long :func:`connect` waits for the port to listen; 0 probes once.
    ready_deadline_s: float = READY_DEADLINE_S
    #: How long after boot a failed connect is read as a boot race.
    boot_grace_s: float = BOOT_GRACE_S

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
        timing = {
            var: _non_negative_seconds(source, var, default)
            for var, default in (
                (READY_DEADLINE_VAR, READY_DEADLINE_S),
                (BOOT_GRACE_VAR, BOOT_GRACE_S),
            )
        }
        return cls(
            values[HOST_VAR],
            port,
            client_id,
            ready_deadline_s=timing[READY_DEADLINE_VAR],
            boot_grace_s=timing[BOOT_GRACE_VAR],
        )


def _non_negative_seconds(source: Any, var: str, default: float) -> float:
    """An optional duration override. Unlike the address these have safe
    defaults (a wait length cannot reach the wrong gateway), but a value that is
    set and malformed is refused, never silently replaced by the default."""
    raw = (source.get(var) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise BrokerSettingsError(f"{var} must be a number of seconds: {exc}") from exc
    if not value >= 0:
        raise BrokerSettingsError(f"{var} must be a non-negative number of seconds, got {raw!r}")
    return value


def port_listening(host: str, port: int, *, timeout_s: float = READY_POLL_S) -> bool:
    """True when a TCP connect to ``host:port`` is accepted.

    Only a refusal or a timeout reads as "not listening yet". Any other socket
    error (an unresolvable host, an unreachable network) is a defect in the
    configured address and propagates as itself.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (ConnectionRefusedError, TimeoutError):
        return False


def read_uptime_s(path: str = UPTIME_PATH) -> float | None:
    """Seconds since boot, or None where the platform does not publish it.

    None (a laptop, a non-Linux host) classifies a failed connect as the
    pre-existing expired-session condition: the boot-race reading needs evidence
    of a recent boot and never assumes one.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    return float(text.split()[0])


def wait_for_port(
    address: GatewayAddress,
    *,
    probe: Callable[[str, int], bool],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    poll_s: float = READY_POLL_S,
) -> bool:
    """Probe the gateway port every ``poll_s`` until it listens or
    ``address.ready_deadline_s`` has elapsed. True when it listened."""
    start = monotonic()
    while True:
        if probe(address.host, address.port):
            return True
        remaining = address.ready_deadline_s - (monotonic() - start)
        if remaining <= 0:
            return False
        sleep(min(poll_s, remaining))


def _unavailable(
    address: GatewayAddress, what: str, uptime_s: float | None
) -> BrokerSessionUnavailableError:
    """Classify a gateway that is not serving an authenticated session."""
    if uptime_s is not None and uptime_s < address.boot_grace_s:
        return GatewayNotReadyError(
            f"{GATEWAY_NOT_READY_PREFIX}: {what}. The box has been up {uptime_s:.0f}s, inside "
            f"the {address.boot_grace_s:.0f}s boot grace, so the gateway is still starting "
            "(its login had not completed). This is a boot race, not an expired session: "
            "re-run the job once the gateway is up."
        )
    up = "unknown" if uptime_s is None else f"{uptime_s:.0f}s"
    return BrokerSessionUnavailableError(
        f"{SESSION_UNAVAILABLE_PREFIX}: {what} (box uptime {up}, boot grace "
        f"{address.boot_grace_s:.0f}s). {REAUTH_STEP}"
    )


def load_sdk() -> Any:
    """`ib_async`, imported only when a real session is opened."""
    return importlib.import_module("ib_async")


def connect(
    address: GatewayAddress,
    *,
    readonly: bool,
    sdk_loader: Callable[[], Any] = load_sdk,
    probe: Callable[[str, int], bool] | None = None,
    uptime: Callable[[], float | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Any:
    """A connected, authenticated, PAPER `IB` client. See the module docstring.

    First waits, bounded by ``address.ready_deadline_s``, for the gateway port to
    listen. Then only a port that never listened, a refused or timed-out API
    connect, and an account-less session are read as no session -- classified
    by box uptime into :class:`GatewayNotReadyError` or the expired-session
    :class:`BrokerSessionUnavailableError`. Any other exception is a defect and
    propagates unchanged.

    ``probe`` and ``uptime`` default to :func:`port_listening` and
    :func:`read_uptime_s`, resolved at call time.
    """
    probe = port_listening if probe is None else probe
    uptime = read_uptime_s if uptime is None else uptime
    where = f"the gateway at {address.host}:{address.port}"
    if not wait_for_port(address, probe=probe, sleep=sleep, monotonic=monotonic):
        raise _unavailable(
            address,
            f"{where} did not open its API port within {address.ready_deadline_s:.0f}s",
            uptime(),
        )
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
        raise _unavailable(
            address,
            f"{where} refused or timed out the API connect ({type(exc).__name__}: {exc})",
            uptime(),
        ) from exc
    accounts = list(client.managedAccounts())
    if not accounts:
        client.disconnect()
        raise _unavailable(
            address,
            f"{where} accepted the API connect but manages no account, which is a session "
            "that is not logged in",
            uptime(),
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
