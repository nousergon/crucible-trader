"""Reconcile a day's durable order intents against the broker. Read-only.

`alpha-engine-config-I12558` deliverable 3: "persist intent, reconcile unknown
broker responses". :mod:`crucible_trader.order_intent` is the first half -- one
immutable intent per (day, symbol), written before every submit. This module is
the second: it reads every intent for a trading day and the broker's own
records (open and completed orders, and every execution on the account), and
classifies each intent. It places, cancels and writes nothing; the caller
decides what to do with the document it returns.

Classes, per intent:

* ``filled`` -- executions under its ``orderRef`` add up to EXACTLY the
  intended quantity;
* ``overfilled`` -- they add up to MORE than it. A wrong effect, never a fill;
* ``effect_mismatch`` -- an execution under the intent's ``orderRef`` was on
  another account, instrument or side than the intent. Some other effect
  carries this decision's key;
* ``corrupt_execution`` -- an execution under the intent's key cannot be
  admitted: no ``execId``, a quantity that is not finite and positive, or one
  ``execId`` reported twice with different content. It is never summed;
* ``working`` -- the order is open at the broker (filled so far is carried);
* ``closed_partial`` -- the broker closed it with some, but not all, shares;
* ``closed_unfilled`` -- the broker closed it (cancelled, inactive) with none;
* ``unknown`` -- the broker has NO record under the intent's key. The submit
  may never have left, or its record has not surfaced yet. Unknown is never
  "none": it is a break, carried with its age;
* ``ref_mismatch`` -- the broker holds an order under this key whose ``orderRef``
  is not the intended one. Some other decision reached the broker.

And per broker record: ``orphan`` -- an order under this router's prefix for
the day with no intent at all (sent around the boundary, or before it existed).

**Executions are counted once, by the broker's own ``execId``**
(`alpha-engine-config-I12558`, audit FIN-04). The same execution delivered
twice (a replayed ``reqExecutions`` answer) is one effect; two executions with
distinct ids are two partial fills. An identical replay is dropped and counted
on the row as ``duplicate_executions``; a replay that disagrees with itself is
``corrupt_execution``. Before this, every record was summed and ``filled`` was
``>=`` the intent, so an overfill, a NaN quantity and a doubled delivery all
read clean.

``clean`` is true only when no row is in :data:`BREAK_CLASSES`. A day with no
intents and no broker orders is clean and says so with zero counts, never a
vacuous pass over an unread broker: the broker records are an input the caller
must supply.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from crucible.store import Store  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs; same import is baselined in every other module

from crucible_trader.order_intent import (
    ORDER_INTENTS_PREFIX,
    OrderIntentError,
    order_intent_key,
    read_order_intent,
)

__all__ = [
    "BREAK_CLASSES",
    "ORDER_INTENT_RECONCILIATION_PREFIX",
    "RECONCILIATION_SCHEMA_VERSION",
    "BrokerExecution",
    "BrokerOrder",
    "broker_records_from_client",
    "intents_for_day",
    "order_intent_reconciliation_key",
    "reconcile_order_intents",
]

RECONCILIATION_SCHEMA_VERSION = "order_intent_reconciliation.v2"

#: Where the daily job files the document. Beside the intents, and for the same
#: reason (see :mod:`crucible_trader.order_intent`): a key in `crucible.keys`
#: moves the trader pin. It joins `crucible.keys` with the next pin.
ORDER_INTENT_RECONCILIATION_PREFIX = "trader/order_intent_reconciliation/"

#: The broker statuses that mean the order is still live at the broker.
OPEN_STATUSES: frozenset[str] = frozenset(
    {"PendingSubmit", "ApiPending", "PreSubmitted", "Submitted", "PendingCancel"}
)

#: The classes that make a day not clean.
BREAK_CLASSES: frozenset[str] = frozenset(
    {
        "unknown",
        "ref_mismatch",
        "effect_mismatch",
        "corrupt_execution",
        "overfilled",
        "closed_partial",
        "orphan",
    }
)

#: The intent's side, as the broker spells it on an execution.
_EXEC_SIDE = {"buy": "BOT", "sell": "SLD"}

_REF_SEP = "|"
_STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclasses.dataclass(frozen=True)
class BrokerOrder:
    """One order the broker knows (open or completed)."""

    order_ref: str
    status: str


@dataclasses.dataclass(frozen=True)
class BrokerExecution:
    """One execution on the account, with the identity the broker gave it.

    ``exec_id`` is the broker's own execution id; ``account``, ``symbol`` and
    ``side`` (``BOT``/``SLD``) are what the effect actually was. All four are
    checked against the intent; none is optional (an empty one is a
    ``corrupt_execution``, never a pass).
    """

    order_ref: str
    shares: float
    exec_id: str = ""
    account: str = ""
    symbol: str = ""
    side: str = ""


def _status(trade: Any) -> str:  # pyright: ignore[reportExplicitAny]  # untyped IB SDK object
    return str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")


def broker_records_from_client(
    client: Any,  # pyright: ignore[reportExplicitAny]  # untyped IB SDK object
) -> tuple[list[BrokerOrder], list[BrokerExecution]]:
    """The broker's orders and executions, read from a connected session.

    Reads only: ``reqAllOpenOrders``, ``reqCompletedOrders(True)`` and
    ``reqExecutions`` (every execution on the account when the session is on
    the gateway's Master API client ID, which the reconcile job connects on).
    """
    orders = [
        BrokerOrder(order_ref=str(t.order.orderRef or ""), status=_status(t))
        for source in (client.reqAllOpenOrders(), client.reqCompletedOrders(True))
        for t in source
    ]
    executions = [
        BrokerExecution(
            order_ref=str(r.execution.orderRef or ""),
            shares=float(r.execution.shares),
            exec_id=str(getattr(r.execution, "execId", "") or ""),
            account=str(getattr(r.execution, "acctNumber", "") or ""),
            symbol=str(getattr(getattr(r, "contract", None), "symbol", "") or ""),
            side=str(getattr(r.execution, "side", "") or ""),
        )
        for r in client.reqExecutions()
    ]
    return orders, executions


def order_intent_reconciliation_key(trading_day: str) -> str:
    """The store key of ``trading_day``'s reconciliation document."""
    order_intent_key(trading_day, "A")  # validates the day
    return f"{ORDER_INTENT_RECONCILIATION_PREFIX}{trading_day}.json"


def intents_for_day(store: Store, trading_day: str) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]  # raw JSON document
    """Every intent written for ``trading_day``, sorted by symbol.

    An unreadable intent raises :class:`OrderIntentError`: a reconciliation
    that skipped it would report clean over the one record it could not read.
    """
    prefix = f"{ORDER_INTENTS_PREFIX}{trading_day}/"
    order_intent_key(trading_day, "A")  # validates the day before any read
    symbols: list[str] = []
    for key in store.list_keys(prefix):
        name = key[len(prefix) :]
        if "/" in name or not name.endswith(".json"):
            raise OrderIntentError(f"{key!r} is not an intent key under {prefix!r}")
        symbols.append(name[: -len(".json")])
    intents: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]  # raw JSON document
    for symbol in sorted(symbols):
        intent = read_order_intent(store, trading_day, symbol)
        if intent is None:  # listed, then gone: never read as absent
            raise OrderIntentError(f"intent for {trading_day}/{symbol} vanished while read")
        intents.append(intent)
    return intents


def _key_of(ref: str) -> str:
    return _REF_SEP.join(ref.split(_REF_SEP)[:3])


def _admissible(e: BrokerExecution) -> bool:
    return bool(e.exec_id) and math.isfinite(e.shares) and e.shares > 0


def _admit(
    executions: Sequence[BrokerExecution],
) -> tuple[list[BrokerExecution], int, bool]:
    """``executions`` counted once each by ``execId``.

    Returns the distinct executions, how many identical replays were dropped,
    and whether any record is corrupt (no id, a non-finite or non-positive
    quantity, or one id carrying two different records). A corrupt record is
    never summed: the caller classifies the whole key ``corrupt_execution``.
    """
    distinct: dict[str, BrokerExecution] = {}
    replays = 0
    corrupt = False
    for e in executions:
        if not _admissible(e):
            corrupt = True
            continue
        prior = distinct.get(e.exec_id)
        if prior is None:
            distinct[e.exec_id] = e
        elif prior == e:
            replays += 1
        else:
            corrupt = True
    return list(distinct.values()), replays, corrupt


def _effect_matches(
    intent: Mapping[str, Any],  # pyright: ignore[reportExplicitAny]  # raw JSON document
    e: BrokerExecution,
) -> bool:
    return (
        e.account == str(intent["account"])
        and e.symbol == str(intent["symbol"])
        and e.side == _EXEC_SIDE.get(str(intent["side"]), "")
    )


def _classify(
    intent: Mapping[str, Any],  # pyright: ignore[reportExplicitAny]  # raw JSON document
    orders: Sequence[BrokerOrder],
    executions: Sequence[BrokerExecution],
    filled: float,
    corrupt: bool,
) -> str:
    if corrupt:
        return "corrupt_execution"
    if not orders:
        return "unknown"
    refs = {o.order_ref for o in orders}
    if refs - {intent["order_ref"]}:
        return "ref_mismatch"
    if not all(_effect_matches(intent, e) for e in executions):
        return "effect_mismatch"
    intended = float(intent["quantity"])
    if filled > intended:
        return "overfilled"
    if filled == intended:
        return "filled"
    if any(o.status in OPEN_STATUSES for o in orders):
        return "working"
    return "closed_partial" if filled > 0 else "closed_unfilled"


def reconcile_order_intents(
    *,
    trading_day: str,
    intents: Iterable[Mapping[str, Any]],  # pyright: ignore[reportExplicitAny]  # raw JSON document
    orders: Iterable[BrokerOrder],
    executions: Iterable[BrokerExecution],
    key_prefix: str,
    now: dt.datetime,
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw JSON document
    """The ``order_intent_reconciliation.v1`` document for one trading day.

    ``key_prefix`` is the router's own ``orderRef`` prefix (``crucible-trader``);
    broker records under any other prefix are not this router's and are ignored.
    """
    day_prefix = _REF_SEP.join((key_prefix, trading_day)) + _REF_SEP
    by_key_orders: dict[str, list[BrokerOrder]] = {}
    for o in orders:
        if o.order_ref.startswith(day_prefix):
            by_key_orders.setdefault(_key_of(o.order_ref), []).append(o)
    raw_by_key: dict[str, list[BrokerExecution]] = {}
    for e in executions:
        if e.order_ref.startswith(day_prefix):
            raw_by_key.setdefault(_key_of(e.order_ref), []).append(e)
    admitted: dict[str, tuple[list[BrokerExecution], int, bool]] = {
        key: _admit(raw) for key, raw in raw_by_key.items()
    }
    filled_by_key = {key: sum(e.shares for e in a[0]) for key, a in admitted.items()}
    exec_refs = {key: {e.order_ref for e in raw} for key, raw in raw_by_key.items()}

    rows: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]  # raw JSON document
    seen: set[str] = set()
    for intent in intents:
        key = str(intent["order_key"])
        seen.add(key)
        orders_here = by_key_orders.get(key, [])
        # An execution under an unexpected ref is a mismatch even with no order record.
        orders_here = orders_here + [
            BrokerOrder(order_ref=ref, status="Executed")
            for ref in sorted(exec_refs.get(key, set()) - {o.order_ref for o in orders_here})
        ]
        filled = filled_by_key.get(key, 0.0)
        distinct, replays, corrupt = admitted.get(key, ([], 0, False))
        cls = _classify(intent, orders_here, distinct, filled, corrupt)
        intended_at = dt.datetime.strptime(str(intent["intended_at"]), _STAMP).replace(
            tzinfo=dt.UTC
        )
        rows.append(
            {
                "order_key": key,
                "symbol": intent["symbol"],
                "side": intent["side"],
                "intended_quantity": int(intent["quantity"]),
                "intended_order_ref": intent["order_ref"],
                "class": cls,
                "filled_quantity": filled,
                "execution_ids": sorted(e.exec_id for e in distinct),
                "duplicate_executions": replays,
                "broker_refs": sorted({o.order_ref for o in orders_here}),
                "broker_statuses": sorted({o.status for o in orders_here}),
                "age_hours": round((now - intended_at).total_seconds() / 3600.0, 2),
            }
        )
    for key in sorted((set(by_key_orders) | set(exec_refs)) - seen):
        refs = {o.order_ref for o in by_key_orders.get(key, [])} | exec_refs.get(key, set())
        rows.append(
            {
                "order_key": key,
                "symbol": key.split(_REF_SEP)[2],
                "class": "orphan",
                "filled_quantity": filled_by_key.get(key, 0.0),
                "broker_refs": sorted(refs),
                "broker_statuses": sorted({o.status for o in by_key_orders.get(key, [])}),
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    breaks = [r["order_key"] for r in rows if r["class"] in BREAK_CLASSES]
    return {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "trading_day": trading_day,
        "generated_at": now.astimezone(dt.UTC).strftime(_STAMP),
        "intents": sum(1 for r in rows if r["class"] != "orphan"),
        "counts": dict(sorted(counts.items())),
        "breaks": breaks,
        "clean": not breaks,
        "rows": rows,
    }
