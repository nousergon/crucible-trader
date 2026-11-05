"""The durable order intent (alpha-engine-config-I12558): one immutable record per
(day, symbol), created conditionally before the submit, and never guessed around."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from crucible.store import LocalStore

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


def _claim(store, day: str, symbol: str, side: str = "buy", account: str = "DU1") -> None:
    claim_order_intent(
        store,
        _doc(
            order_key=f"crucible-trader|{day}|{symbol}",
            order_ref=f"crucible-trader|{day}|{symbol}|{side}|10|10.0|2026-09-08T13:30:00.000000Z",
            trading_day=day,
            symbol=symbol,
            side=side,
            account=account,
        ),
    )


def test_a_name_whose_latest_intent_is_a_buy_is_owned(store) -> None:
    _claim(store, "2026-09-03", "SNDK")
    _claim(store, "2026-09-03", "PLTR")
    owned = bought_by_this_trader(store, {"SNDK", "ZZZ"}, account="DU1", before=DAY)
    assert owned == frozenset({"SNDK"})  # PLTR was not asked about; ZZZ was never bought


def test_only_the_latest_intent_before_the_book_day_counts(store) -> None:
    _claim(store, "2026-09-03", "AAA", side="buy")
    _claim(store, "2026-09-04", "AAA", side="sell")
    _claim(store, "2026-09-03", "BBB", side="sell")
    _claim(store, "2026-09-04", "BBB", side="buy")
    _claim(store, DAY, "CCC", side="buy")  # the book day itself is not "before" it
    owned = bought_by_this_trader(store, {"AAA", "BBB", "CCC"}, account="DU1", before=DAY)
    assert owned == frozenset({"BBB"})


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
