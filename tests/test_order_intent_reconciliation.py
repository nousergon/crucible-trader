"""Reconciling a day's durable order intents against the broker (alpha-engine-config-I12558).

Read-only: every input is a record the broker or the store already holds.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest
from crucible.store import LocalStore

from crucible_trader.order_intent import (
    OrderIntentError,
    claim_order_intent,
    order_intent_document,
    order_intent_key,
)
from crucible_trader.order_intent_reconciliation import (
    BREAK_CLASSES,
    RECONCILIATION_SCHEMA_VERSION,
    BrokerExecution,
    BrokerOrder,
    broker_records_from_client,
    intents_for_day,
    reconcile_order_intents,
)

DAY = "2026-09-08"
PREFIX = "crucible-trader"
WHEN = dt.datetime(2026, 9, 8, 13, 30, tzinfo=dt.UTC)
NOW = WHEN + dt.timedelta(hours=7)


def _ref(symbol: str, qty: int = 100, side: str = "buy", price: float = 100.0) -> str:
    return f"{PREFIX}|{DAY}|{symbol}|{side}|{qty}|{price!r}|2026-09-08T13:30:00.000000Z"


def _intent(symbol: str, qty: int = 100) -> dict:
    return order_intent_document(
        order_key=f"{PREFIX}|{DAY}|{symbol}",
        order_ref=_ref(symbol, qty),
        trading_day=DAY,
        symbol=symbol,
        side="buy",
        quantity=qty,
        decision_price=100.0,
        account="DU1",
        intended_at=WHEN,
    )


def _run(intents, orders=(), executions=()):
    return reconcile_order_intents(
        trading_day=DAY,
        intents=intents,
        orders=orders,
        executions=executions,
        key_prefix=PREFIX,
        now=NOW,
    )


def _row(doc, symbol):
    return next(r for r in doc["rows"] if r["symbol"] == symbol)


def test_every_class_is_named_and_breaks_are_not_clean() -> None:
    intents = [_intent(s) for s in ("FIL", "WRK", "PAR", "CAN", "UNK", "MIS")]
    orders = [
        BrokerOrder(_ref("FIL"), "Filled"),
        BrokerOrder(_ref("WRK"), "Submitted"),
        BrokerOrder(_ref("PAR"), "Cancelled"),
        BrokerOrder(_ref("CAN"), "Cancelled"),
        BrokerOrder(_ref("MIS", qty=7), "Filled"),
        BrokerOrder(_ref("ORF"), "Filled"),
        BrokerOrder("someone-else|x|FIL", "Filled"),
        BrokerOrder(f"{PREFIX}|2026-09-04|FIL|buy|1|1.0|2026-09-04T13:30:00.000000Z", "Filled"),
    ]
    executions = [
        BrokerExecution(_ref("FIL"), 60.0),
        BrokerExecution(_ref("FIL"), 40.0),
        BrokerExecution(_ref("WRK"), 10.0),
        BrokerExecution(_ref("PAR"), 30.0),
        BrokerExecution(_ref("MIS", qty=7), 7.0),
        BrokerExecution("someone-else|x|FIL", 5.0),
    ]
    doc = _run(intents, orders, executions)
    classes = {r["symbol"]: r["class"] for r in doc["rows"]}
    assert classes == {
        "FIL": "filled",
        "WRK": "working",
        "PAR": "closed_partial",
        "CAN": "closed_unfilled",
        "UNK": "unknown",
        "MIS": "ref_mismatch",
        "ORF": "orphan",
    }
    assert doc["schema_version"] == RECONCILIATION_SCHEMA_VERSION
    assert doc["clean"] is False and doc["intents"] == 6
    assert sorted(doc["breaks"]) == sorted(
        f"{PREFIX}|{DAY}|{s}" for s in ("PAR", "UNK", "MIS", "ORF")
    )
    assert set(doc["breaks"]) == {
        r["order_key"] for r in doc["rows"] if r["class"] in BREAK_CLASSES
    }
    assert _row(doc, "FIL")["filled_quantity"] == 100.0
    assert _row(doc, "WRK")["filled_quantity"] == 10.0
    assert _row(doc, "UNK")["age_hours"] == 7.0 and _row(doc, "UNK")["broker_refs"] == []
    assert doc["counts"]["orphan"] == 1


def test_an_execution_under_another_ref_with_no_order_record_is_a_mismatch() -> None:
    doc = _run([_intent("AAA")], executions=[BrokerExecution(_ref("AAA", qty=5), 5.0)])
    assert _row(doc, "AAA")["class"] == "ref_mismatch"
    assert _row(doc, "AAA")["broker_statuses"] == ["Executed"]


def test_an_execution_with_no_order_record_still_counts_as_the_fill() -> None:
    doc = _run([_intent("AAA")], executions=[BrokerExecution(_ref("AAA"), 100.0)])
    assert _row(doc, "AAA")["class"] == "filled" and doc["clean"] is True


def test_an_orphan_known_only_by_its_execution_is_reported() -> None:
    doc = _run([], executions=[BrokerExecution(_ref("ZZZ"), 3.0)])
    (row,) = doc["rows"]
    assert row["class"] == "orphan" and row["filled_quantity"] == 3.0
    assert row["broker_statuses"] == [] and doc["intents"] == 0 and doc["clean"] is False


def test_an_empty_day_is_clean_with_zero_counts() -> None:
    doc = _run([])
    assert doc["clean"] is True and doc["counts"] == {} and doc["rows"] == []


def test_records_are_read_from_a_connected_session_without_writing() -> None:
    def trade(ref, status):
        return SimpleNamespace(
            order=SimpleNamespace(orderRef=ref), orderStatus=SimpleNamespace(status=status)
        )

    client = SimpleNamespace(
        reqAllOpenOrders=lambda: [trade(_ref("A"), "Submitted")],
        reqCompletedOrders=lambda api_only: [
            trade(_ref("B"), "Filled"),
            SimpleNamespace(order=SimpleNamespace(orderRef=None)),
        ],
        reqExecutions=lambda: [
            SimpleNamespace(execution=SimpleNamespace(orderRef=_ref("B"), shares="100"))
        ],
    )
    orders, executions = broker_records_from_client(client)
    assert orders == [
        BrokerOrder(_ref("A"), "Submitted"),
        BrokerOrder(_ref("B"), "Filled"),
        BrokerOrder("", ""),
    ]
    assert executions == [BrokerExecution(_ref("B"), 100.0)]


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def test_intents_are_read_for_one_day_in_symbol_order(store) -> None:
    for symbol in ("BBB", "AAA"):
        claim_order_intent(store, _intent(symbol))
    other = {**_intent("CCC"), "trading_day": "2026-09-09"}
    claim_order_intent(store, other)
    assert [i["symbol"] for i in intents_for_day(store, DAY)] == ["AAA", "BBB"]
    assert intents_for_day(store, "2026-09-10") == []


def test_a_malformed_day_is_refused_before_any_read(store) -> None:
    with pytest.raises(OrderIntentError):
        intents_for_day(store, "2026-9-8")


@pytest.mark.parametrize("name", ["AAA.txt", "nested/AAA.json"])
def test_a_foreign_key_under_the_day_prefix_is_refused(store, name) -> None:
    store.put_bytes(f"trader/order_intents/{DAY}/{name}", b"{}")
    with pytest.raises(OrderIntentError, match="not an intent key"):
        intents_for_day(store, DAY)


def test_an_unreadable_intent_is_refused_never_skipped(store) -> None:
    store.put_bytes(order_intent_key(DAY, "AAA"), json.dumps({"x": 1}).encode())
    with pytest.raises(OrderIntentError):
        intents_for_day(store, DAY)


class _VanishingStore(LocalStore):
    def exists(self, key):
        return False


def test_an_intent_that_vanishes_while_read_is_refused(tmp_path) -> None:
    store = _VanishingStore(tmp_path)
    claim_order_intent(store, _intent("AAA"))
    with pytest.raises(OrderIntentError, match="vanished"):
        intents_for_day(store, DAY)
