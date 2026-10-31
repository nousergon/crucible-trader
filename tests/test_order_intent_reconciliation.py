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


def _exe(
    symbol: str,
    shares: float,
    exec_id: str,
    *,
    qty: int = 100,
    account: str = "DU1",
    side: str = "BOT",
    on: str | None = None,
) -> BrokerExecution:
    """An execution under ``symbol``'s intended ref (or ``qty``'s), with identity."""
    return BrokerExecution(
        _ref(symbol, qty),
        shares,
        exec_id=exec_id,
        account=account,
        symbol=on or symbol,
        side=side,
    )


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
        _exe("FIL", 60.0, "e1"),
        _exe("FIL", 40.0, "e2"),
        _exe("WRK", 10.0, "e3"),
        _exe("PAR", 30.0, "e4"),
        _exe("MIS", 7.0, "e5", qty=7),
        BrokerExecution("someone-else|x|FIL", 5.0, exec_id="e6"),
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
    doc = _run([_intent("AAA")], executions=[_exe("AAA", 5.0, "e1", qty=5)])
    assert _row(doc, "AAA")["class"] == "ref_mismatch"
    assert _row(doc, "AAA")["broker_statuses"] == ["Executed"]


def test_an_execution_with_no_order_record_still_counts_as_the_fill() -> None:
    doc = _run([_intent("AAA")], executions=[_exe("AAA", 100.0, "e1")])
    assert _row(doc, "AAA")["class"] == "filled" and doc["clean"] is True


def test_an_orphan_known_only_by_its_execution_is_reported() -> None:
    doc = _run([], executions=[_exe("ZZZ", 3.0, "e1")])
    (row,) = doc["rows"]
    assert row["class"] == "orphan" and row["filled_quantity"] == 3.0
    assert row["broker_statuses"] == [] and doc["intents"] == 0 and doc["clean"] is False


# --- FIN-04 (alpha-engine-config-I12558): effects are admitted, identified, counted once.


def test_an_overfill_is_a_break_never_a_fill() -> None:
    doc = _run(
        [_intent("AAA")],
        [BrokerOrder(_ref("AAA"), "Filled")],
        [_exe("AAA", 200.0, "e1")],
    )
    assert _row(doc, "AAA")["class"] == "overfilled" and doc["clean"] is False
    assert doc["breaks"] == [f"{PREFIX}|{DAY}|AAA"]


@pytest.mark.parametrize("shares", [float("nan"), float("inf"), -float("inf"), -100.0, 0.0])
def test_a_quantity_that_is_not_finite_and_positive_is_corrupt(shares: float) -> None:
    doc = _run(
        [_intent("AAA")],
        [BrokerOrder(_ref("AAA"), "Filled")],
        [_exe("AAA", shares, "e1")],
    )
    row = _row(doc, "AAA")
    assert row["class"] == "corrupt_execution" and doc["clean"] is False
    assert row["filled_quantity"] == 0.0  # never summed


def test_an_execution_with_no_broker_id_is_corrupt() -> None:
    doc = _run([_intent("AAA")], [BrokerOrder(_ref("AAA"), "Filled")], [_exe("AAA", 100.0, "")])
    assert _row(doc, "AAA")["class"] == "corrupt_execution" and doc["clean"] is False


def test_an_identical_replay_counts_once() -> None:
    once = _exe("AAA", 100.0, "e1")
    doc = _run([_intent("AAA")], [BrokerOrder(_ref("AAA"), "Filled")], [once, once])
    row = _row(doc, "AAA")
    assert row["class"] == "filled" and doc["clean"] is True
    assert row["filled_quantity"] == 100.0 and row["duplicate_executions"] == 1
    assert row["execution_ids"] == ["e1"]


def test_one_id_reported_with_two_different_records_is_corrupt() -> None:
    doc = _run(
        [_intent("AAA")],
        [BrokerOrder(_ref("AAA"), "Filled")],
        [_exe("AAA", 100.0, "e1"), _exe("AAA", 50.0, "e1")],
    )
    assert _row(doc, "AAA")["class"] == "corrupt_execution" and doc["clean"] is False


def test_distinct_partial_executions_reconcile_exactly_once() -> None:
    doc = _run(
        [_intent("AAA")],
        [BrokerOrder(_ref("AAA"), "Filled")],
        [_exe("AAA", 30.0, "e1"), _exe("AAA", 70.0, "e2"), _exe("AAA", 30.0, "e1")],
    )
    row = _row(doc, "AAA")
    assert row["class"] == "filled" and row["filled_quantity"] == 100.0
    assert row["execution_ids"] == ["e1", "e2"] and row["duplicate_executions"] == 1


@pytest.mark.parametrize(
    "wrong",
    [{"account": "DU2"}, {"on": "BBB"}, {"side": "SLD"}],
    ids=["account", "instrument", "side"],
)
def test_the_same_ref_on_another_account_instrument_or_side_is_a_mismatch(wrong) -> None:
    doc = _run(
        [_intent("AAA")],
        [BrokerOrder(_ref("AAA"), "Filled")],
        [_exe("AAA", 100.0, "e1", **wrong)],
    )
    assert _row(doc, "AAA")["class"] == "effect_mismatch" and doc["clean"] is False


def test_an_execution_missing_its_identity_is_never_a_match() -> None:
    bare = BrokerExecution(_ref("AAA"), 100.0, exec_id="e1")
    doc = _run([_intent("AAA")], [BrokerOrder(_ref("AAA"), "Filled")], [bare])
    assert _row(doc, "AAA")["class"] == "effect_mismatch" and doc["clean"] is False


def test_a_partial_unmatched_close_stays_a_break() -> None:
    doc = _run(
        [_intent("AAA")],
        [BrokerOrder(_ref("AAA"), "Cancelled")],
        [_exe("AAA", 40.0, "e1")],
    )
    assert _row(doc, "AAA")["class"] == "closed_partial" and doc["clean"] is False


def test_every_new_class_is_a_break() -> None:
    assert {"overfilled", "corrupt_execution", "effect_mismatch"} <= BREAK_CLASSES


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
            SimpleNamespace(
                contract=SimpleNamespace(symbol="B"),
                execution=SimpleNamespace(
                    orderRef=_ref("B"), shares="100", execId="0001.01", acctNumber="DU1", side="BOT"
                ),
            )
        ],
    )
    orders, executions = broker_records_from_client(client)
    assert orders == [
        BrokerOrder(_ref("A"), "Submitted"),
        BrokerOrder(_ref("B"), "Filled"),
        BrokerOrder("", ""),
    ]
    assert executions == [_exe("B", 100.0, "0001.01")]


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
