"""`exception.v1` (alpha-engine-config-I12892 phase 0): the record, its closed
field set, the converters from both detectors' documents, and the day file.
Synthetic documents only."""

from __future__ import annotations

import copy
import math

import pytest

from crucible_trader.exception_record import (
    CLASSES_BY_SOURCE,
    CONTROL_ARM_VOID,
    DETECTING_JOB,
    SOURCE_BROKER_RECONCILIATION,
    SOURCE_ORDER_INTENT_RECONCILIATION,
    ExceptionRecordError,
    exception_day_document,
    exception_id,
    exceptions_key,
    records_from_broker_reconciliation,
    records_from_order_intent_reconciliation,
    validate_exception_day_document,
    validate_exception_record,
)
from crucible_trader.order_intent_reconciliation import BREAK_CLASSES
from crucible_trader.reconciliation import DISCREPANCY_CLASSES
from crucible_trader.reconciliation_job import RECONCILE_JOB

DAY = "2026-10-30"
STAMP = "2026-10-30T20:45:00Z"


def broker_document(*, void=False, discrepancies=None, missed=()):
    return {
        "trading_day": DAY,
        "account": "DU1",
        "void": void,
        "control_arm": {"missed": list(missed)},
        "discrepancies": [
            {
                "klass": "share_delta",
                "instrument": "AAA",
                "expected": 12.0,
                "actual": 9.0,
                "delta": -3.0,
                "side": "broker_under",
                "detail": "broker share count differs",
            },
            {
                "klass": "cash_delta",
                "instrument": "USD",
                "expected": 800.0,
                "actual": 790.0,
                "delta": -10.0,
                "side": "broker_under",
                "detail": "cash differs",
            },
        ]
        if discrepancies is None
        else discrepancies,
    }


def intent_document(rows):
    return {"trading_day": DAY, "generated_at": "2026-10-30T20:45:01.123456Z", "rows": rows}


INTENT_ROW = {
    "order_key": f"crucible-trader|{DAY}|AAA",
    "symbol": "AAA",
    "side": "buy",
    "intended_quantity": 4,
    "class": "overfilled",
    "filled_quantity": 8.0,
    "broker_refs": ["ref-1"],
    "broker_statuses": ["Filled"],
}


def a_record():
    (record,) = records_from_broker_reconciliation(
        broker_document(discrepancies=broker_document()["discrepancies"][:1]), detected_at=STAMP
    )
    return record.to_document()


class TestVocabulary:
    def test_the_detecting_job_is_the_reconcile_job(self):
        assert DETECTING_JOB == RECONCILE_JOB

    def test_each_source_raises_exactly_its_detectors_classes(self):
        assert CLASSES_BY_SOURCE[SOURCE_BROKER_RECONCILIATION] == (
            frozenset(DISCREPANCY_CLASSES) | {CONTROL_ARM_VOID}
        )
        assert CLASSES_BY_SOURCE[SOURCE_ORDER_INTENT_RECONCILIATION] == BREAK_CLASSES

    def test_key_and_a_bad_day(self):
        assert exceptions_key(DAY) == f"trader/exceptions/{DAY}.json"
        with pytest.raises(ExceptionRecordError, match="YYYY-MM-DD"):
            exceptions_key("10/30/2026")

    def test_the_id_is_the_identity_not_the_time(self):
        one = exception_id("s", DAY, ("share_delta", "AAA"))
        assert one == exception_id("s", DAY, ("share_delta", "AAA"))
        assert one != exception_id("s", DAY, ("share_delta", "BBB"))
        assert len(one) == 32


class TestBrokerConverter:
    def test_each_finding_is_a_record_with_both_records_named(self):
        share, cash = records_from_broker_reconciliation(broker_document(), detected_at=STAMP)
        doc = share.to_document()
        assert doc["class"] == "share_delta" and doc["instrument"] == "AAA"
        assert doc["amount"] == -3.0 and doc["unit"] == "shares"
        assert doc["expected"] == {"value": 12.0, "record": f"trader/reconciliation/{DAY}.json"}
        assert doc["actual"] == {"value": 9.0, "record": f"trader/broker_statements/{DAY}.json"}
        assert doc["detected_by"] == {
            "job": "trader.reconcile",
            "manifest_key": f"runs/trader.reconcile/{DAY}/run.json",
            "document_key": f"trader/reconciliation/{DAY}.json",
            "void": False,
        }
        assert cash.to_document()["unit"] == "usd"

    def test_a_class_without_one_unit_carries_none(self):
        (record,) = records_from_broker_reconciliation(
            broker_document(
                discrepancies=[
                    {
                        "klass": "corporate_action",
                        "instrument": "AAA",
                        "expected": 5.0,
                        "actual": None,
                        "delta": -5.0,
                        "side": "broker",
                        "detail": "dividend not credited",
                    }
                ]
            ),
            detected_at=STAMP,
        )
        assert record.to_document()["unit"] is None

    def test_a_void_run_is_a_break_first_and_marks_every_finding(self):
        records = records_from_broker_reconciliation(
            broker_document(void=True, missed=("cash_delta",)), detected_at=STAMP
        )
        assert [r.klass for r in records] == [CONTROL_ARM_VOID, "share_delta", "cash_delta"]
        assert all(r.to_document()["detected_by"]["void"] for r in records)
        assert "cash_delta" in records[0].detail

    def test_a_void_with_nothing_named_still_records(self):
        (record,) = records_from_broker_reconciliation(
            broker_document(void=True, discrepancies=[]), detected_at=STAMP
        )
        assert record.klass == CONTROL_ARM_VOID and "nothing named" in record.detail

    def test_a_clean_run_has_no_records(self):
        assert (
            records_from_broker_reconciliation(broker_document(discrepancies=[]), detected_at=STAMP)
            == []
        )


class TestIntentConverter:
    def test_only_break_classes_become_records(self):
        rows = [INTENT_ROW, {**INTENT_ROW, "order_key": "k2", "class": "filled"}]
        (record,) = records_from_order_intent_reconciliation(intent_document(rows))
        doc = record.to_document()
        assert doc["source"] == SOURCE_ORDER_INTENT_RECONCILIATION
        assert doc["amount"] == 4.0 and doc["unit"] == "shares"
        assert doc["expected"] == {"value": 4.0, "record": f"trader/order_intents/{DAY}/AAA.json"}
        assert doc["actual"]["record"] == f"trader/order_intent_reconciliation/{DAY}.json"
        assert doc["detected_at"] == "2026-10-30T20:45:01.123456Z"
        assert "ref-1" in doc["detail"] and "Filled" in doc["detail"]

    def test_an_orphan_has_no_intent_side(self):
        orphan = {
            "order_key": f"crucible-trader|{DAY}|BBB",
            "symbol": "BBB",
            "class": "orphan",
            "filled_quantity": 2.0,
            "broker_refs": [],
            "broker_statuses": [],
        }
        (record,) = records_from_order_intent_reconciliation(intent_document([orphan]), void=True)
        doc = record.to_document()
        assert doc["expected"] == {"value": None, "record": None}
        assert doc["amount"] is None and doc["side"] is None
        assert doc["detected_by"]["void"] is True
        assert "refs none; statuses none" in doc["detail"]

    def test_an_unknown_intent_with_no_fill_quantity_has_no_amount(self):
        row = {**INTENT_ROW, "class": "unknown", "filled_quantity": None}
        (record,) = records_from_order_intent_reconciliation(intent_document([row]))
        assert record.amount is None and record.to_document()["unit"] is None


class TestDayDocument:
    def test_a_valid_day(self):
        records = records_from_broker_reconciliation(broker_document(), detected_at=STAMP)
        day = exception_day_document(DAY, records)
        validate_exception_day_document(day)
        assert [r["class"] for r in day["records"]] == ["share_delta", "cash_delta"]

    def test_a_record_from_another_day_is_refused(self):
        records = records_from_broker_reconciliation(broker_document(), detected_at=STAMP)
        with pytest.raises(ExceptionRecordError, match="not 2026-10-29"):
            exception_day_document("2026-10-29", records)

    def test_two_records_with_one_id_are_refused(self):
        records = records_from_broker_reconciliation(broker_document(), detected_at=STAMP)
        with pytest.raises(ExceptionRecordError, match="share id"):
            exception_day_document(DAY, [records[0], records[0]])

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda d: d.pop("records"), "missing"),
            (lambda d: d.update(schema_version="exception.v0"), "schema_version"),
            (lambda d: d.update(trading_day="x"), "YYYY-MM-DD"),
            (lambda d: d.update(trading_day="2026-10-29"), "is not for"),
        ],
    )
    def test_invalid_day_documents(self, mutate, match):
        day = exception_day_document(
            DAY, records_from_broker_reconciliation(broker_document(), detected_at=STAMP)
        )
        mutate(day)
        with pytest.raises(ExceptionRecordError, match=match):
            validate_exception_day_document(day)


def _set(path, value):
    def apply(doc):
        target = doc
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value

    return apply


class TestValidation:
    def test_a_built_record_is_valid(self):
        validate_exception_record(a_record())

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda d: d.pop("detail"), "missing"),
            (lambda d: d.update(extra=1), "unexpected"),
            (_set(("schema_version",), "exception.v2"), "schema_version"),
            (_set(("source",), "elsewhere"), "unknown source"),
            (_set(("class",), "overfilled"), "is not one"),
            (_set(("source",), 3), "not a string"),
            (_set(("account",), 3), "string or null"),
            (_set(("unit",), "usd"), "is not the unit"),
            (_set(("amount",), True), "not a number"),
            (_set(("amount",), "1.0"), "not a number"),
            (_set(("amount",), math.nan), "not finite"),
            (_set(("expected",), []), "not an object"),
            (_set(("expected",), {"value": 1.0}), "missing"),
            (_set(("expected",), {1: 1.0}), "non-string key"),
            (_set(("detected_by", "job"), "trader.fire"), "detected_by.job"),
            (_set(("detected_by", "void"), "no"), "not a boolean"),
            (_set(("detected_at",), "2026-10-30 20:45"), "not a UTC stamp"),
            (_set(("exception_id",), "XYZ"), "not 32 hex"),
        ],
    )
    def test_invalid_records(self, mutate, match):
        doc = copy.deepcopy(a_record())
        mutate(doc)
        with pytest.raises(ExceptionRecordError, match=match):
            validate_exception_record(doc)


class TestMalformedDetectorDocuments:
    @pytest.mark.parametrize(
        ("document", "match"),
        [
            ({**broker_document(), "discrepancies": {}}, "not a list"),
            ({**broker_document(), "discrepancies": [1]}, "non-object"),
            ({**broker_document(void=True), "control_arm": {"missed": [1]}}, "list of strings"),
            ({**broker_document(), "void": "no"}, "not a boolean"),
            ({k: v for k, v in broker_document().items() if k != "account"}, "is missing"),
        ],
    )
    def test_raise_rather_than_skip(self, document, match):
        with pytest.raises(ExceptionRecordError, match=match):
            records_from_broker_reconciliation(document, detected_at=STAMP)
