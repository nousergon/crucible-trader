"""The kill switch: one command flattens or freezes the paper book.

`alpha-engine-config-I10650` deliverable 1; plan §9.5 ("one command flattens or
freezes; the degenerate-prediction gate holds the book").

**Three modes, one halt document.** Every mode writes :data:`KILL_SWITCH_KEY`
FIRST, before any broker call, so the trader session -- which calls
:func:`assert_trading_permitted` before it sends any order -- stops sending
orders even if the broker calls below then fail:

    flatten   halt + cancel every working order + close every position at market.
              Persists until a human runs `release`.
    freeze    halt + cancel every working order; positions are kept.
              Persists until a human runs `release`.
    hold      halt for THIS trading day only; working orders are left alone.
              What the degenerate-prediction gate engages (`crucible_trader.hold_book`),
              lifted from the v1 executor's `_should_hold_book`, which suppressed the
              rebalance and retained the book (and its stops) for the session.

**The outcome is measured, not assumed.** :func:`fire` reads the book before,
polls until the book reaches its mode's target (flat and no working order;
no working order; halted) or the declared bound elapses, reads the book after,
and lists every order the broker ACCEPTED after the fire instant that the switch
did not place itself. That outcome is the fire drill's artifact
(`crucible_trader.fire_drill`), and a switch that did not settle inside its bound
or let a foreign order through fails its run.

**Paper only.** Every broker read goes through
:class:`~crucible_trader.broker_statement.IbPaperStatementSource`, which refuses
a non-`DU` account.

**Recorded contract dependency:** `trader.kill_switch` is not yet in
`crucible.models.JOB_VALUES`. The halt document is written directly (never only
at manifest time) precisely so the protection does not wait on that
registration; the run's manifest is refused until it lands, and a test pins it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

from crucible.runner import RunContext
from crucible.store import Store

from crucible_trader.broker_statement import BrokerStatement, IbPaperStatementSource

KILL_SWITCH_JOB = "trader.kill_switch"
KILL_SWITCH_KEY = "trader/kill_switch.json"
KILL_SWITCH_EVENTS_PREFIX = "trader/kill_switch/events/"
KILL_SWITCH_SCHEMA_VERSION = "kill_switch.v1"
KILL_SWITCH_OUTCOME_SCHEMA_VERSION = "kill_switch_outcome.v1"
METRIC_MODULE = "crucible_trader.kill_switch"

#: Every order the switch places carries this `orderRef`, so its own closing
#: orders are told apart from an order that got through after the fire.
KILL_SWITCH_ORDER_REF = "crucible-kill-switch"

Mode = Literal["flatten", "freeze", "hold"]
MODES: tuple[str, ...] = ("flatten", "freeze", "hold")
PERSISTENT_MODES: frozenset[str] = frozenset({"flatten", "freeze"})

#: The declared bound, per mode, inside which the book must reach its target.
#: Flatten waits on paper market-order fills during regular hours; freeze and
#: hold wait only on cancels and on the halt document itself.
SETTLE_BOUND_S: Mapping[str, float] = {"flatten": 300.0, "freeze": 60.0, "hold": 60.0}
POLL_INTERVAL_S = 2.0

#: IB order statuses that mean the broker accepted the order.
ACCEPTED_STATUSES: frozenset[str] = frozenset({"PreSubmitted", "Submitted", "Filled"})


def kill_switch_event_key(trading_day: str, run_id: str) -> str:
    """Declared here only until `crucible.keys` owns it (as `reconciliation_key`)."""
    return f"{KILL_SWITCH_EVENTS_PREFIX}{trading_day}/{run_id}.json"


class TradingHaltedError(RuntimeError):
    """The kill switch (or the hold) is engaged. The session sends no order."""


class KillSwitchStateError(RuntimeError):
    """The halt document exists and cannot be read. Read as halted, loudly."""


class KillSwitchNotSettledError(RuntimeError):
    """The book did not reach its target inside the bound, or an order got through."""


@dataclasses.dataclass(frozen=True)
class AcceptedOrder:
    order_id: int
    symbol: str
    action: str
    quantity: float
    order_ref: str
    accepted_at: str

    def to_document(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class BrokerControl(Protocol):
    """What the switch needs from a broker. The live one is :class:`IbBrokerControl`."""

    def statement(self, trading_day: str) -> BrokerStatement: ...
    def open_order_count(self) -> int: ...
    def cancel_all(self) -> None: ...
    def submit_close(self, symbol: str, shares: int) -> None: ...
    def orders_accepted_since(self, instant: dt.datetime) -> list[AcceptedOrder]: ...
    def wait(self, seconds: float) -> None: ...


class IbBrokerControl:
    """:class:`BrokerControl` over a connected `ib_async.IB` client (paper)."""

    def __init__(self, client: Any, sdk: Any) -> None:
        self._client = client
        self._sdk = sdk

    def statement(self, trading_day: str) -> BrokerStatement:
        return IbPaperStatementSource(self._client).read_statement(trading_day)

    def open_order_count(self) -> int:
        return len(list(self._client.openTrades()))

    def cancel_all(self) -> None:
        self._client.reqGlobalCancel()

    def submit_close(self, symbol: str, shares: int) -> None:
        contract = self._sdk.Stock(symbol, "SMART", "USD")
        order = self._sdk.MarketOrder("SELL" if shares > 0 else "BUY", abs(shares))
        order.orderRef = KILL_SWITCH_ORDER_REF
        order.tif = "DAY"
        self._client.placeOrder(contract, order)

    def orders_accepted_since(self, instant: dt.datetime) -> list[AcceptedOrder]:
        accepted: list[AcceptedOrder] = []
        for trade in self._client.trades():
            times = [entry.time for entry in trade.log if entry.status in ACCEPTED_STATUSES]
            if not times:
                continue
            first = min(times)
            if first.tzinfo is None:
                raise KillSwitchStateError(
                    f"order {trade.order.orderId} carries a naive acceptance time {first!r}; "
                    "it cannot be placed before or after the fire instant"
                )
            if first >= instant:
                accepted.append(
                    AcceptedOrder(
                        order_id=int(trade.order.orderId),
                        symbol=trade.contract.symbol,
                        action=trade.order.action,
                        quantity=float(trade.order.totalQuantity),
                        order_ref=trade.order.orderRef or "",
                        accepted_at=_iso(first),
                    )
                )
        return accepted

    def wait(self, seconds: float) -> None:
        self._client.sleep(seconds)


@dataclasses.dataclass(frozen=True)
class KillSwitchOutcome:
    mode: str
    cause: str
    account: str
    trading_day: str
    run_id: str
    fired_at: str
    settled_at: str | None
    settle_seconds: float | None
    bound_seconds: float
    before: BrokerStatement
    after: BrokerStatement
    foreign_orders_after_fire: tuple[AcceptedOrder, ...]

    @property
    def reached(self) -> bool:
        return self.settled_at is not None

    @property
    def passed(self) -> bool:
        return self.reached and not self.foreign_orders_after_fire

    def reason(self) -> str:
        parts = []
        if not self.reached:
            parts.append(
                f"{self.mode} did not reach its target within {self.bound_seconds:.0f}s "
                f"(after: {len(self.after.positions)} positions)"
            )
        if self.foreign_orders_after_fire:
            parts.append(
                f"{len(self.foreign_orders_after_fire)} order(s) accepted after the fire "
                "instant: "
                + ", ".join(
                    f"{o.order_id} {o.action} {o.quantity:g} {o.symbol} at {o.accepted_at}"
                    for o in self.foreign_orders_after_fire
                )
            )
        return "; ".join(parts)

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": KILL_SWITCH_OUTCOME_SCHEMA_VERSION,
            "mode": self.mode,
            "cause": self.cause,
            "account": self.account,
            "trading_day": self.trading_day,
            "run_id": self.run_id,
            "fired_at": self.fired_at,
            "settled_at": self.settled_at,
            "settle_seconds": self.settle_seconds,
            "bound_seconds": self.bound_seconds,
            "reached": self.reached,
            "passed": self.passed,
            "state_before": self.before.to_document(),
            "state_after": self.after.to_document(),
            "orders_accepted_after_fire": [o.to_document() for o in self.foreign_orders_after_fire],
        }


def _iso(instant: dt.datetime) -> str:
    return instant.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def halt_document(
    *, engaged: bool, mode: str | None, cause: str, trading_day: str, since: str, run_id: str
) -> dict[str, Any]:
    return {
        "schema_version": KILL_SWITCH_SCHEMA_VERSION,
        "engaged": engaged,
        "mode": mode,
        "cause": cause,
        "trading_day": trading_day,
        "since": since,
        "run_id": run_id,
    }


def read_state(store: Store) -> dict[str, Any] | None:
    """The halt document, or ``None`` when the switch has never been written.

    Anything present that is not exactly a halt document raises
    :class:`KillSwitchStateError`: an unreadable switch is never read as "off".
    """
    try:
        payload = store.get_bytes(KILL_SWITCH_KEY)
    except KeyError:
        return None
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KillSwitchStateError(f"{KILL_SWITCH_KEY} is not readable JSON: {exc}") from exc
    expected = set(
        halt_document(engaged=False, mode=None, cause="", trading_day="", since="", run_id="")
    )
    if (
        not isinstance(document, dict)
        or set(document) != expected
        or document["schema_version"] != KILL_SWITCH_SCHEMA_VERSION
        or not isinstance(document["engaged"], bool)
        or (document["engaged"] and document["mode"] not in MODES)
    ):
        raise KillSwitchStateError(
            f"{KILL_SWITCH_KEY} is not a {KILL_SWITCH_SCHEMA_VERSION} halt document: {document!r}"
        )
    return document


def assert_trading_permitted(store: Store, trading_day: str) -> None:
    """The one check the trader session makes before it sends any order.

    Raises :class:`TradingHaltedError` when flatten/freeze is engaged (any day,
    until released) or a hold is engaged for ``trading_day``. A hold filed for an
    earlier session has expired: v1's hold was a per-session decision.
    """
    state = read_state(store)
    if state is None or not state["engaged"]:
        return
    if state["mode"] in PERSISTENT_MODES or state["trading_day"] == trading_day:
        raise TradingHaltedError(
            f"trading halted: {state['mode']} engaged since {state['since']} "
            f"(cause: {state['cause']}, run {state['run_id']}). "
            + (
                "A human releases it with the kill-switch `release` command."
                if state["mode"] in PERSISTENT_MODES
                else f"The hold lapses after session {state['trading_day']}."
            )
        )


def fire(
    ctx: RunContext,
    broker: BrokerControl,
    *,
    mode: str,
    cause: str,
    clock: Callable[[], dt.datetime] = utcnow,
    bound_seconds: float | None = None,
) -> KillSwitchOutcome:
    """Engage ``mode`` and measure what the book did. See the module docstring."""
    if mode not in MODES:
        raise ValueError(f"kill-switch mode {mode!r} not in {MODES}")
    if not cause.strip():
        raise ValueError(
            "a kill-switch fire needs a cause; an unexplained halt is not reconstructible"
        )
    bound = SETTLE_BOUND_S[mode] if bound_seconds is None else bound_seconds
    day = ctx.trading_day.isoformat()
    before = broker.statement(day)
    fired = clock()
    ctx.record_output(
        KILL_SWITCH_KEY,
        json.dumps(
            halt_document(
                engaged=True,
                mode=mode,
                cause=cause,
                trading_day=day,
                since=_iso(fired),
                run_id=ctx.run_id,
            ),
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
        KILL_SWITCH_SCHEMA_VERSION,
    )
    if mode in PERSISTENT_MODES:
        broker.cancel_all()
    if mode == "flatten":
        for symbol, shares in before.positions.items():
            broker.submit_close(symbol, shares)

    settled: dt.datetime | None = None
    while True:
        after = broker.statement(day)
        if mode == "hold" or (
            broker.open_order_count() == 0 and (mode == "freeze" or not after.positions)
        ):
            settled = clock()
            break
        if (clock() - fired).total_seconds() >= bound:
            break
        broker.wait(POLL_INTERVAL_S)

    foreign = tuple(
        order
        for order in broker.orders_accepted_since(fired)
        if order.order_ref != KILL_SWITCH_ORDER_REF
    )
    return KillSwitchOutcome(
        mode=mode,
        cause=cause,
        account=before.account,
        trading_day=day,
        run_id=ctx.run_id,
        fired_at=_iso(fired),
        settled_at=None if settled is None else _iso(settled),
        settle_seconds=None if settled is None else round((settled - fired).total_seconds(), 3),
        bound_seconds=bound,
        before=before,
        after=after,
        foreign_orders_after_fire=foreign,
    )


def record_outcome(ctx: RunContext, outcome: KillSwitchOutcome) -> None:
    """File the outcome and its metric; raise when it did not pass (the page)."""
    key = kill_switch_event_key(outcome.trading_day, outcome.run_id)
    ctx.record_output(
        key,
        json.dumps(outcome.to_document(), indent=2, sort_keys=True).encode("utf-8"),
        KILL_SWITCH_OUTCOME_SCHEMA_VERSION,
    )
    ctx.record_metric(
        {
            "name": "kill_switch_settle_seconds",
            "module": METRIC_MODULE,
            "metric_type": "operational",
            "value": outcome.settle_seconds,
            "unit": "seconds" if outcome.settle_seconds is not None else None,
            "n_floor": 1,
            "status": "OK" if outcome.passed else "FAIL",
            "status_reason": (
                f"{outcome.mode} reached its target in {outcome.settle_seconds}s "
                f"(bound {outcome.bound_seconds:.0f}s), zero orders accepted after the fire"
                if outcome.passed
                else outcome.reason()
            ),
            "source_path": key,
            "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if not outcome.passed:
        raise KillSwitchNotSettledError(outcome.reason())


def release(ctx: RunContext, *, cause: str, clock: Callable[[], dt.datetime] = utcnow) -> dict:
    """Disengage. A human command: nothing in this package calls it automatically."""
    if not cause.strip():
        raise ValueError("a release needs a cause")
    previous = read_state(ctx.store)
    if previous is None or not previous["engaged"]:
        raise KillSwitchStateError("the kill switch is not engaged; there is nothing to release")
    document = halt_document(
        engaged=False,
        mode=None,
        cause=f"released ({cause}); was {previous['mode']} since {previous['since']}",
        trading_day=ctx.trading_day.isoformat(),
        since=_iso(clock()),
        run_id=ctx.run_id,
    )
    ctx.record_output(
        KILL_SWITCH_KEY,
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        KILL_SWITCH_SCHEMA_VERSION,
    )
    return document
