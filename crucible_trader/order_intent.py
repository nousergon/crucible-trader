"""The durable order intent: written, immutably, before every order is submitted.

`alpha-engine-config-I12558` (CR-07, P1) and `nous-ergon-ops`
`policies/portfolio-accounting-policy.md`: a broker effect must never depend on
an illusion of distributed exactly-once. Before this module the router read the
broker's own order refs and then called ``placeOrder`` with nothing durable on
our side. A submit whose response was lost (a crash, a dropped socket, a
gateway restart before the order showed in the broker's lists) left no record
that it was ever attempted, so a retried session could send it again.

**The boundary.** Immediately before ``placeOrder`` the router writes ONE intent
document per (trading day, symbol) with a conditional create
(:meth:`crucible.store.Store.compare_and_swap` from
:data:`crucible.store.ETAG_ABSENT`):

* the write fails → the order is not sent;
* the create loses (the key already exists) → the order is not sent. Either a
  concurrent session claimed it, or an earlier attempt today got as far as the
  submit and the broker does not know the order. Both are **unknown**, and
  unknown is never "none": it is refused, never re-sent, and left for
  reconciliation to resolve against the broker's executions.

The document is never updated or deleted; what became of the order is the
broker's record (adopted refs, executions) and the session's shortfall
artifact, both keyed by the same ``orderRef``.

The key lives here, not in `crucible.keys`, because adding it there moves the
trader pin. It joins `crucible.keys` with the next pin.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from crucible.execution import execution_shortfall_key, validate_execution_shortfall
from crucible.store import ETAG_ABSENT, PointerConflictError, Store

__all__ = [
    "ORDER_INTENTS_PREFIX",
    "ORDER_INTENT_SCHEMA_VERSION",
    "OrderIntentConflictError",
    "OrderIntentError",
    "bought_by_this_trader",
    "claim_order_intent",
    "order_intent_document",
    "order_intent_key",
    "read_order_intent",
]

ORDER_INTENT_SCHEMA_VERSION = "order_intent.v1"
ORDER_INTENTS_PREFIX = "trader/order_intents/"

_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]*$")
_STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"

_REQUIRED: tuple[str, ...] = (
    "schema_version",
    "order_key",
    "order_ref",
    "trading_day",
    "symbol",
    "side",
    "quantity",
    "decision_price",
    "account",
    "intended_at",
)


class OrderIntentError(RuntimeError):
    """An intent key or document that cannot be trusted. Never guessed around."""


class OrderIntentConflictError(OrderIntentError):
    """The intent for this (day, symbol) already exists: the order is unknown."""

    def __init__(self, key: str, existing: Mapping[str, Any] | None) -> None:  # pyright: ignore[reportExplicitAny]  # raw JSON document
        self.key = key
        self.existing = dict(existing) if existing is not None else None
        prior = (existing or {}).get("order_ref", "<unreadable>")
        super().__init__(
            f"an order intent already exists at {key!r} (order_ref {prior!r}). An earlier or "
            "concurrent submit for this name today may have reached the broker; it is not "
            "re-sent. Reconcile it against the broker's executions."
        )


def order_intent_key(trading_day: str, symbol: str) -> str:
    """``trader/order_intents/<day>/<symbol>.json``; refuses a malformed part."""
    if not _ISO_DAY.match(trading_day):
        raise OrderIntentError(f"trading day {trading_day!r} is not an ISO date")
    dt.date.fromisoformat(trading_day)
    if not _SYMBOL.match(symbol):
        raise OrderIntentError(f"symbol {symbol!r} is not a key-safe ticker")
    return f"{ORDER_INTENTS_PREFIX}{trading_day}/{symbol}.json"


def order_intent_document(
    *,
    order_key: str,
    order_ref: str,
    trading_day: str,
    symbol: str,
    side: str,
    quantity: int,
    decision_price: float,
    account: str,
    intended_at: dt.datetime,
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw JSON document
    """The immutable record of one order about to be submitted."""
    if side not in ("buy", "sell"):
        raise OrderIntentError(f"side {side!r} is not buy or sell")
    if quantity <= 0:
        raise OrderIntentError(f"quantity {quantity!r} is not a positive share count")
    return {
        "schema_version": ORDER_INTENT_SCHEMA_VERSION,
        "order_key": order_key,
        "order_ref": order_ref,
        "trading_day": trading_day,
        "symbol": symbol,
        "side": side,
        "quantity": int(quantity),
        "decision_price": float(decision_price),
        "account": account,
        "intended_at": intended_at.astimezone(dt.UTC).strftime(_STAMP),
    }


def _parse(key: str, payload: bytes) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw JSON document
    try:
        document = json.loads(payload)
    except ValueError as exc:
        raise OrderIntentError(f"{key!r} is not readable JSON: {exc}") from exc
    if not isinstance(document, dict) or any(f not in document for f in _REQUIRED):
        raise OrderIntentError(f"{key!r} is not an {ORDER_INTENT_SCHEMA_VERSION} document")
    if document["schema_version"] != ORDER_INTENT_SCHEMA_VERSION:
        raise OrderIntentError(
            f"{key!r} is schema {document['schema_version']!r}, not {ORDER_INTENT_SCHEMA_VERSION}"
        )
    return document  # pyright: ignore[reportUnknownVariableType]  # json.loads output, checked against _REQUIRED above


def read_order_intent(store: Store, trading_day: str, symbol: str) -> dict[str, Any] | None:  # pyright: ignore[reportExplicitAny]  # raw JSON document
    """The intent for (day, symbol), or ``None`` when none was ever written."""
    key = order_intent_key(trading_day, symbol)
    if not store.exists(key):
        return None
    return _parse(key, store.get_bytes(key))


def claim_order_intent(store: Store, document: Mapping[str, Any]) -> str:  # pyright: ignore[reportExplicitAny]  # raw JSON document
    """Create the intent, or raise. Returns the store's version token.

    The conditional create IS the guard; there is no read-then-write window.
    A lost create raises :class:`OrderIntentConflictError` carrying whatever
    the existing document says (``None`` if it cannot be read).
    """
    key = order_intent_key(str(document.get("trading_day")), str(document.get("symbol")))
    payload = json.dumps(dict(document), indent=2, sort_keys=True).encode("utf-8")
    _parse(key, payload)
    try:
        return store.compare_and_swap(key, ETAG_ABSENT, payload)
    except PointerConflictError:
        try:
            existing: dict[str, Any] | None = _parse(key, store.get_bytes(key))  # pyright: ignore[reportExplicitAny]  # raw JSON document
        except (KeyError, OrderIntentError):
            existing = None
        raise OrderIntentConflictError(key, existing) from None


def _filled_by_order(store: Store, day: str) -> dict[str, float]:
    """Shares filled per order key in book ``day``'s session record (`execution_shortfall`).

    Empty when the session filed no record of fills: none written, ``no_orders``,
    or ``not_computed``. A record that cannot be read raises.
    """
    key = execution_shortfall_key(day)
    if not store.exists(key):
        return {}
    try:
        document = validate_execution_shortfall(json.loads(store.get_bytes(key)), origin=key)
    except ValueError as exc:  # bad JSON, or ExecutionArtifactError
        raise OrderIntentError(f"{key!r} is not a readable session record: {exc}") from exc
    if document.get("outcome") != "computed":
        return {}
    orders: list[dict[str, Any]] = document.get("orders") or []  # pyright: ignore[reportExplicitAny]  # validated JSON document
    return {str(o["order_id"]): float(o["filled_quantity"]) for o in orders}


def bought_by_this_trader(
    store: Store, symbols: Iterable[str], *, account: str, before: str
) -> frozenset[str]:
    """Of ``symbols``, the names this trader itself still holds shares of in ``account``.

    A name counts when the shares its own orders FILLED on book days before
    ``before`` -- bought minus sold, from each session's ``execution_shortfall``
    record, matched by order key -- are still positive. The router's own
    position is not foreign to it: on 2026-11-04 the trader bought SNDK (book
    2026-11-03), and SNDK is neither one of v1's declared inherited names nor
    certain to be in the next book, so
    :func:`~crucible_trader.order_router.plan_orders` would refuse the whole
    next session over the trader's own holding (``ForeignPositionError``,
    alpha-engine-config-I13164) the first time the champion rotated out of it.

    An intent alone is not a holding: it is written BEFORE the submit, so a buy
    that never filled owns nothing, and a sell that trimmed or partly filled
    leaves the trader's remaining shares its own (alpha-engine-config-I13237).
    A buy with no fill on record counts as nothing filled -- at worst the name
    stays refused. A sell with no fill on record raises
    :class:`OrderIntentError`: counting it as unfilled would call shares owned
    that the trader may already have sold. So does an intent or session record
    that cannot be read: a name is never called owned on a guess, and a name
    never called owned stays refused.

    Only ``symbols`` are read (one list of the prefix, the intents per wanted
    name, and one session record per day they span), so the cost tracks the
    names the book dropped, not the whole intent history.
    """
    wanted = set(symbols)
    if not wanted:
        return frozenset()
    order_intent_key(before, "A")  # validates the day before any read
    days_by_symbol: dict[str, list[str]] = {}
    for key in store.list_keys(ORDER_INTENTS_PREFIX):
        rest = key[len(ORDER_INTENTS_PREFIX) :]
        day, sep, name = rest.partition("/")
        if not sep or "/" in name or not name.endswith(".json"):
            raise OrderIntentError(f"{key!r} is not an intent key under {ORDER_INTENTS_PREFIX!r}")
        symbol = name[: -len(".json")]
        if symbol in wanted and _ISO_DAY.match(day) and day < before:
            days_by_symbol.setdefault(symbol, []).append(day)
    sessions: dict[str, dict[str, float]] = {}
    owned: set[str] = set()
    for symbol, days in days_by_symbol.items():
        held = 0.0
        for day in sorted(days):
            intent = read_order_intent(store, day, symbol)
            if intent is None:  # listed, then gone: never read as absent
                raise OrderIntentError(f"intent for {day}/{symbol} vanished while read")
            if intent["account"] != account or intent["symbol"] != symbol:
                continue
            if day not in sessions:
                sessions[day] = _filled_by_order(store, day)
            filled = sessions[day].get(str(intent["order_key"]))
            if intent["side"] == "buy":
                held += filled or 0.0
            elif filled is None:
                raise OrderIntentError(
                    f"sell intent {day}/{symbol} has no fill on record in "
                    f"{execution_shortfall_key(day)!r}; what it sold is unknown, so the name "
                    "is not called owned. Reconcile it against the broker's executions."
                )
            else:
                held -= filled
        if held > 0:
            owned.add(symbol)
    return frozenset(owned)
