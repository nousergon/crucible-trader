"""The durable order intent (alpha-engine-config-I12558): one immutable record per
(day, symbol), created conditionally before the submit, and never guessed around."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from crucible.execution import execution_shortfall_key
from crucible.store import LocalStore

from crucible_trader.execution_shortfall import (
    Fill,
    build_no_orders_document,
    build_shortfall_document,
    decide,
    write_shortfall_document,
)
from crucible_trader.order_intent import (
    ORDER_INTENT_SCHEMA_VERSION,
    OrderIntentConflictError,
    OrderIntentError,
    bought_by_this_trader,
    claim_order_intent,
    order_intent_document,
    order_intent_key,
    read_order_intent,
)
from crucible_trader.order_intent_reconciliation import (
    BrokerExecution,
    BrokerOrder,
    intents_for_day,
    order_intent_reconciliation_key,
    reconcile_order_intents,
)

DAY = "2026-09-08"
WHEN = dt.datetime(2026, 9, 8, 13, 30, tzinfo=dt.UTC)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _doc(**overrides):
    kwargs = {
        "order_key": f"crucible-trader|{DAY}|AAA",
        "order_ref": f"crucible-trader|{DAY}|AAA|buy|100|100.0|2026-09-08T13:30:00.000000Z",
        "trading_day": DAY,
        "symbol": "AAA",
        "side": "buy",
        "quantity": 100,
        "decision_price": 100.0,
        "account": "DU1",
        "intended_at": WHEN,
    }
    kwargs.update(overrides)
    return order_intent_document(**kwargs)


def test_the_key_binds_day_and_symbol() -> None:
    assert order_intent_key(DAY, "BRK.B") == f"trader/order_intents/{DAY}/BRK.B.json"


@pytest.mark.parametrize(
    ("day", "symbol"),
    [("2026-9-8", "AAA"), ("20260908", "AAA"), (DAY, "../AAA"), (DAY, "aaa"), (DAY, "")],
)
def test_a_malformed_key_part_is_refused(day, symbol) -> None:
    with pytest.raises(OrderIntentError):
        order_intent_key(day, symbol)


def test_an_impossible_date_is_refused() -> None:
    with pytest.raises(ValueError):
        order_intent_key("2026-13-01", "AAA")


def test_the_document_records_the_whole_order_in_utc() -> None:
    doc = _doc(intended_at=WHEN.astimezone(dt.timezone(dt.timedelta(hours=-4))))
    assert doc["schema_version"] == ORDER_INTENT_SCHEMA_VERSION
    assert doc["intended_at"] == "2026-09-08T13:30:00.000000Z"
    assert doc["quantity"] == 100 and doc["side"] == "buy" and doc["account"] == "DU1"


@pytest.mark.parametrize(("field", "value"), [("side", "hold"), ("quantity", 0)])
def test_a_document_that_cannot_be_an_order_is_refused(field, value) -> None:
    with pytest.raises(OrderIntentError):
        _doc(**{field: value})


def test_absent_is_none_and_a_claim_reads_back(store) -> None:
    assert read_order_intent(store, DAY, "AAA") is None
    claim_order_intent(store, _doc())
    assert read_order_intent(store, DAY, "AAA") == _doc()


def test_a_second_claim_loses_and_names_the_first(store) -> None:
    claim_order_intent(store, _doc())
    with pytest.raises(OrderIntentConflictError, match="already exists") as exc:
        claim_order_intent(store, _doc(quantity=7))
    assert exc.value.existing["quantity"] == 100  # the first claim stands, unchanged
    assert exc.value.key == order_intent_key(DAY, "AAA")
    assert read_order_intent(store, DAY, "AAA")["quantity"] == 100


def test_a_lost_claim_over_an_unreadable_record_is_still_a_conflict(store) -> None:
    store.put_bytes(order_intent_key(DAY, "AAA"), b"not json")
    with pytest.raises(OrderIntentConflictError, match="<unreadable>") as exc:
        claim_order_intent(store, _doc())
    assert exc.value.existing is None


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        json.dumps({"schema_version": ORDER_INTENT_SCHEMA_VERSION}).encode(),
    ],
)
def test_an_unreadable_record_is_refused_not_read_as_absent(store, payload) -> None:
    store.put_bytes(order_intent_key(DAY, "AAA"), payload)
    with pytest.raises(OrderIntentError):
        read_order_intent(store, DAY, "AAA")


def test_another_schema_version_is_refused(store) -> None:
    store.put_bytes(
        order_intent_key(DAY, "AAA"),
        json.dumps({**_doc(), "schema_version": "order_intent.v0"}).encode(),
    )
    with pytest.raises(OrderIntentError, match="order_intent.v0"):
        read_order_intent(store, DAY, "AAA")


def test_a_document_missing_a_field_is_never_written(store) -> None:
    doc = _doc()
    del doc["order_ref"]
    with pytest.raises(OrderIntentError):
        claim_order_intent(store, doc)
    assert read_order_intent(store, DAY, "AAA") is None


# -- which held names this trader bought itself (alpha-engine-config-I13164) --------


def _claim(
    store, day: str, symbol: str, side: str = "buy", account: str = "DU1", quantity: int = 10
) -> None:
    claim_order_intent(
        store,
        _doc(
            order_key=f"crucible-trader|{day}|{symbol}",
            order_ref=f"crucible-trader|{day}|{symbol}|{side}|10|10.0|2026-09-08T13:30:00.000000Z",
            trading_day=day,
            symbol=symbol,
            side=side,
            account=account,
            quantity=quantity,
        ),
    )


def _session(store, day: str, *orders: tuple[str, str, int, int]) -> None:
    """The session record for book ``day``: (symbol, side, intended, filled) per order."""
    fills = [
        Fill(
            order_id=f"crucible-trader|{day}|{symbol}",
            decision=decide(symbol, side, intended, price=10.0, now=WHEN),  # type: ignore[arg-type]
            submitted_at=WHEN,
            filled_quantity=filled,
            fill_price=10.0 if filled else None,
        )
        for symbol, side, intended, filled in orders
    ]
    write_shortfall_document(
        store, build_shortfall_document(trading_day=day, champion="c", fills=fills, now=WHEN)
    )


def _reconcile(store, book_day: str, executed: str, *orders: tuple[str, str, int, str]) -> None:
    """The reconcile job's document for ``book_day``, run on execution day ``executed``:
    (symbol, side, shares the broker executed, final broker status) per order."""
    intents = intents_for_day(store, book_day)
    ref = {i["symbol"]: i["order_ref"] for i in intents}
    broker_orders = [BrokerOrder(order_ref=ref[sym], status=status) for sym, _, _, status in orders]
    executions = [
        BrokerExecution(
            order_ref=ref[sym],
            shares=float(shares),
            exec_id=f"e-{book_day}-{sym}",
            account="DU1",
            symbol=sym,
            side={"buy": "BOT", "sell": "SLD"}[side],
        )
        for sym, side, shares, _ in orders
        if shares
    ]
    document = reconcile_order_intents(
        trading_day=executed,
        book_day=book_day,
        intents=intents,
        orders=broker_orders,
        executions=executions,
        key_prefix="crucible-trader",
        now=WHEN,
    )
    store.put_bytes(order_intent_reconciliation_key(executed), json.dumps(document).encode())


def test_a_sell_that_left_shares_held_does_not_revoke_ownership(store) -> None:
    """Codex bf44c6c0: a trim, or a partly filled sell, leaves the trader's own shares."""
    _claim(store, "2026-09-03", "AAA")
    _claim(store, "2026-09-03", "BBB")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 10), ("BBB", "buy", 10, 10))
    _claim(store, "2026-09-04", "AAA", side="sell", quantity=4)
    _claim(store, "2026-09-04", "BBB", side="sell")
    _session(store, "2026-09-04", ("AAA", "sell", 4, 4), ("BBB", "sell", 10, 3))
    # The broker closed BBB's sell with 3 of 10 sold: settled, so 7 are still ours.
    _reconcile(
        store,
        "2026-09-04",
        "2026-09-08",
        ("AAA", "sell", 4, "Filled"),
        ("BBB", "sell", 3, "Cancelled"),
    )
    owned = bought_by_this_trader(store, {"AAA", "BBB"}, account="DU1", before=DAY)
    assert owned == frozenset({"AAA", "BBB"})


def test_a_buy_that_never_filled_is_not_ownership(store) -> None:
    """Codex af59f570: an intent is written BEFORE the submit; it is not a holding."""
    _claim(store, "2026-09-03", "AAA")  # the broker closed it with nothing filled
    _claim(store, "2026-09-03", "BBB")  # written, then the submit never reached the broker
    _session(store, "2026-09-03", ("AAA", "buy", 10, 0))
    assert bought_by_this_trader(store, {"AAA", "BBB"}, account="DU1", before=DAY) == frozenset()


def test_a_name_whose_buy_filled_is_owned(store) -> None:
    _claim(store, "2026-09-03", "SNDK")
    _claim(store, "2026-09-03", "PLTR")
    _session(store, "2026-09-03", ("SNDK", "buy", 10, 10), ("PLTR", "buy", 10, 10))
    owned = bought_by_this_trader(store, {"SNDK", "ZZZ"}, account="DU1", before=DAY)
    assert owned == frozenset({"SNDK"})  # PLTR was not asked about; ZZZ was never bought


def test_filled_shares_net_across_days_before_the_book_day(store) -> None:
    _claim(store, "2026-09-03", "AAA", side="buy")
    _claim(store, "2026-09-03", "BBB", side="buy", quantity=5)
    _session(store, "2026-09-03", ("AAA", "buy", 10, 10), ("BBB", "buy", 5, 5))
    _claim(store, "2026-09-04", "AAA", side="sell")
    _claim(store, "2026-09-04", "BBB", side="buy")
    _session(store, "2026-09-04", ("AAA", "sell", 10, 10), ("BBB", "buy", 10, 10))
    _claim(store, DAY, "CCC", side="buy")  # the book day itself is not "before" it
    _session(store, DAY, ("CCC", "buy", 10, 10))
    owned = bought_by_this_trader(store, {"AAA", "BBB", "CCC"}, account="DU1", before=DAY)
    assert owned == frozenset({"BBB"})  # AAA was sold whole


def test_a_session_record_without_fills_counts_a_buy_as_nothing_filled(store) -> None:
    _claim(store, "2026-09-03", "AAA")
    write_shortfall_document(
        store,
        build_no_orders_document(trading_day="2026-09-03", champion="c", reason="r", now=WHEN),
    )
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset()


def test_a_sell_with_no_fill_on_record_counts_as_sold_whole_not_unfilled(store) -> None:
    _claim(store, "2026-09-03", "AAA")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 10))
    _claim(store, "2026-09-04", "AAA", side="sell")  # the session filed no record
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset()


# -- codex 609afe10: the session snapshot is not the final fill ----------------------


def test_a_buy_that_filled_after_the_snapshot_is_owned_once_reconciled(store) -> None:
    """The router's fill wait ended with 0 filled; the DAY order filled later."""
    _claim(store, "2026-09-03", "AAA")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 0))
    _reconcile(store, "2026-09-03", "2026-09-04", ("AAA", "buy", 10, "Filled"))
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset({"AAA"})


def test_a_sell_that_finished_after_the_snapshot_leaves_no_phantom_shares(store) -> None:
    """The snapshot saw 3 of 10 sold; the rest sold later. Nothing of ours is left."""
    _claim(store, "2026-09-03", "AAA")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 10))
    _claim(store, "2026-09-04", "AAA", side="sell")
    _session(store, "2026-09-04", ("AAA", "sell", 10, 3))
    _reconcile(store, "2026-09-04", "2026-09-08", ("AAA", "sell", 10, "Filled"))
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset()


def test_an_unsettled_sell_never_reads_as_shares_still_owned(store) -> None:
    """No reconciliation yet: the sell may have finished, so nothing is called owned."""
    _claim(store, "2026-09-03", "AAA")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 10))
    _claim(store, "2026-09-04", "AAA", side="sell")
    _session(store, "2026-09-04", ("AAA", "sell", 10, 3))
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset()


def test_an_order_still_working_at_reconciliation_counts_as_unsettled(store) -> None:
    _claim(store, "2026-09-03", "AAA")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 0))
    _reconcile(store, "2026-09-03", "2026-09-04", ("AAA", "buy", 4, "Submitted"))
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset()


def test_a_settled_order_does_not_need_its_session_record(store) -> None:
    _claim(store, "2026-09-03", "AAA")
    store.put_bytes(execution_shortfall_key("2026-09-03"), b"not json")
    _reconcile(store, "2026-09-03", "2026-09-04", ("AAA", "buy", 10, "Filled"))
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset({"AAA"})


def test_a_reconciliation_of_another_book_is_not_read_as_this_one(store) -> None:
    _claim(store, "2026-09-03", "AAA")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 0))
    _reconcile(store, "2026-09-03", "2026-09-04", ("AAA", "buy", 10, "Filled"))
    key = order_intent_reconciliation_key("2026-09-04")
    document = json.loads(store.get_bytes(key))
    document["book_day"] = "2026-09-02"
    store.put_bytes(key, json.dumps(document).encode())
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset()


@pytest.mark.parametrize(
    "payload",
    [b"not json", b"[]", json.dumps({"schema_version": "order_intent_reconciliation.v1"}).encode()],
)
def test_an_unreadable_reconciliation_is_refused_not_skipped(store, payload) -> None:
    _claim(store, "2026-09-03", "AAA")
    _session(store, "2026-09-03", ("AAA", "buy", 10, 10))
    store.put_bytes(order_intent_reconciliation_key("2026-09-04"), payload)
    with pytest.raises(OrderIntentError, match="not a readable reconciliation"):
        bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY)


def test_an_unreadable_session_record_is_refused_not_skipped(store) -> None:
    _claim(store, "2026-09-03", "AAA")
    store.put_bytes(execution_shortfall_key("2026-09-03"), b"not json")
    with pytest.raises(OrderIntentError, match="not a readable session record"):
        bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY)


def test_a_buy_for_another_account_is_not_ownership(store) -> None:
    _claim(store, "2026-09-03", "AAA", account="DU2")
    assert bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY) == frozenset()


def test_nothing_asked_reads_nothing(store) -> None:
    store.put_bytes("trader/order_intents/garbage", b"x")  # never listed
    assert bought_by_this_trader(store, set(), account="DU1", before=DAY) == frozenset()


def test_an_unreadable_intent_for_a_wanted_name_is_refused_not_skipped(store) -> None:
    store.put_bytes(order_intent_key("2026-09-03", "AAA"), b"not json")
    with pytest.raises(OrderIntentError):
        bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY)


def test_a_stray_key_under_the_prefix_is_refused(store) -> None:
    store.put_bytes("trader/order_intents/2026-09-03/AAA/x.json", b"{}")
    with pytest.raises(OrderIntentError, match="not an intent key"):
        bought_by_this_trader(store, {"AAA"}, account="DU1", before=DAY)


def test_an_intent_listed_then_gone_is_refused_not_read_as_absent(tmp_path) -> None:
    class ListsAGhost(LocalStore):
        def list_keys(self, prefix: str = ""):
            yield order_intent_key("2026-09-03", "AAA")

    with pytest.raises(OrderIntentError, match="vanished"):
        bought_by_this_trader(ListsAGhost(tmp_path), {"AAA"}, account="DU1", before=DAY)
