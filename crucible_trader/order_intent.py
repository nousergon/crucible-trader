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
from collections.abc import Mapping
from typing import Any

from crucible.store import ETAG_ABSENT, PointerConflictError, Store

__all__ = [
    "ORDER_INTENTS_PREFIX",
    "ORDER_INTENT_SCHEMA_VERSION",
    "OrderIntentConflictError",
    "OrderIntentError",
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

    def __init__(self, key: str, existing: Mapping[str, Any] | None) -> None:
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
) -> dict[str, Any]:
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


def _parse(key: str, payload: bytes) -> dict[str, Any]:
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
    return document


def read_order_intent(store: Store, trading_day: str, symbol: str) -> dict[str, Any] | None:
    """The intent for (day, symbol), or ``None`` when none was ever written."""
    key = order_intent_key(trading_day, symbol)
    if not store.exists(key):
        return None
    return _parse(key, store.get_bytes(key))


def claim_order_intent(store: Store, document: Mapping[str, Any]) -> str:
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
            existing: dict[str, Any] | None = _parse(key, store.get_bytes(key))
        except (KeyError, OrderIntentError):
            existing = None
        raise OrderIntentConflictError(key, existing) from None
