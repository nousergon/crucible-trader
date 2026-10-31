"""`exception.v1`: one reconciliation break, as a record a review queue can hold.

`alpha-engine-config-I12892` phase 0 (the exception desk). Until now a break was
a page -- the `trader.reconcile` manifest going `status: failed` with
:func:`~crucible_trader.reconciliation_job.page_text` as its reason -- and
nothing recorded the break as an item that a proposal, a decision or a later
score could point at. This module is that record, and nothing more: it carries
what was DETECTED. Proposals, decisions and scores are later phases and do not
appear in this schema.

One record names:

* ``source`` -- which detector raised it (:data:`SOURCES`), and ``class``, one
  of that detector's own closed set of finding classes;
* ``trading_day``, ``account`` and ``instrument`` -- what the break is about;
* ``amount`` and ``unit`` -- the detector's own delta, with a unit only where
  the class fixes one (:data:`_UNIT_BY_CLASS`); never inferred;
* ``expected`` and ``actual`` -- the two records that disagree: each a value
  and the store key of the document that carries it (``None`` where that side
  does not exist, e.g. an orphan broker order has no intent);
* ``detected_by`` -- the job, its manifest key, the detector's document key, and
  whether the control arm voided the run;
* ``detected_at`` -- when, in UTC;
* ``exception_id`` -- DERIVED from the break's identity (:func:`exception_id`),
  so a re-run or a historical export of the same break yields the same id and a
  queue can de-duplicate on it.

The day's records are filed together at :func:`exceptions_key` by the daily
reconcile job (:func:`crucible_trader.reconciliation_job.file_exception_records`)
and can be rebuilt from the detectors' stored documents by
:mod:`crucible_trader.exception_export`. Both go through the converters below,
so a live record and an exported one carry the same ``exception_id`` for the
same break (only ``detected_at`` differs: the job's clock, or the manifest's
``finished``).

Writing a record never changes what the reconcile job decides or pages. The
records are built FROM the detectors' documents after the detectors ran.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

from crucible.keys import (  # pyright: ignore[reportMissingTypeStubs]  # crucible ships no stubs; same import is baselined in every other module
    manifest_key,
    trader_broker_statement_key,
    trader_reconciliation_key,
)

from crucible_trader.order_intent import order_intent_key
from crucible_trader.order_intent_reconciliation import (
    BREAK_CLASSES as INTENT_BREAK_CLASSES,
)
from crucible_trader.order_intent_reconciliation import (
    order_intent_reconciliation_key,
)
from crucible_trader.reconciliation import DISCREPANCY_CLASSES

__all__ = [
    "CLASSES_BY_SOURCE",
    "CONTROL_ARM_VOID",
    "DETECTING_JOB",
    "EXCEPTIONS_PREFIX",
    "EXCEPTION_SCHEMA_VERSION",
    "SOURCES",
    "SOURCE_BROKER_RECONCILIATION",
    "SOURCE_ORDER_INTENT_RECONCILIATION",
    "ExceptionRecord",
    "ExceptionRecordError",
    "exception_day_document",
    "exception_id",
    "exceptions_key",
    "records_from_broker_reconciliation",
    "records_from_order_intent_reconciliation",
    "validate_exception_day_document",
    "validate_exception_record",
]

EXCEPTION_SCHEMA_VERSION = "exception.v1"

#: Where the day's records are filed. Beside the other `trader/` documents that
#: are not yet in `crucible.keys`, for the same reason as
#: :data:`crucible_trader.order_intent_reconciliation.ORDER_INTENT_RECONCILIATION_PREFIX`:
#: a key in `crucible.keys` moves the trader pin. It joins with the next pin.
EXCEPTIONS_PREFIX = "trader/exceptions/"

#: The job whose run detects every break this module records. Equal to
#: `crucible_trader.reconciliation_job.RECONCILE_JOB` (asserted by a test; not
#: imported, because that module imports this one).
DETECTING_JOB = "trader.reconcile"

SOURCE_BROKER_RECONCILIATION = "crucible-trader.broker_reconciliation"
SOURCE_ORDER_INTENT_RECONCILIATION = "crucible-trader.order_intent_reconciliation"

#: A run whose control arm missed a planted discrepancy is VOID and pages for
#: that reason alone (`ReconciliationVoidError`), so the void is a break in its
#: own right: the detector cannot be trusted that day.
CONTROL_ARM_VOID = "control_arm_void"

#: Every source, and the closed set of classes it can raise. The broker
#: reconciler's set is `reconciliation.DISCREPANCY_CLASSES` plus the void; the
#: intent reconciler's is its `BREAK_CLASSES` (its non-break classes, such as
#: ``filled`` and ``working``, are not exceptions).
CLASSES_BY_SOURCE: Mapping[str, frozenset[str]] = {
    SOURCE_BROKER_RECONCILIATION: frozenset(DISCREPANCY_CLASSES) | {CONTROL_ARM_VOID},
    SOURCE_ORDER_INTENT_RECONCILIATION: INTENT_BREAK_CLASSES,
}
SOURCES: tuple[str, ...] = tuple(CLASSES_BY_SOURCE)

#: The classes whose delta has one fixed unit. Every other class carries the
#: detector's delta with ``unit: None`` -- e.g. a ``corporate_action`` finding
#: is in shares for a split and in dollars for an unpaid dividend, and guessing
#: which from the instrument would put a wrong unit on a real break.
_UNIT_BY_CLASS: Mapping[tuple[str, str], str] = {
    (SOURCE_BROKER_RECONCILIATION, "cash_delta"): "usd",
    (SOURCE_BROKER_RECONCILIATION, "share_delta"): "shares",
    (SOURCE_BROKER_RECONCILIATION, "position_count"): "positions",
    (SOURCE_ORDER_INTENT_RECONCILIATION, "overfilled"): "shares",
    (SOURCE_ORDER_INTENT_RECONCILIATION, "closed_partial"): "shares",
}

_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EXCEPTION_ID = re.compile(r"^[0-9a-f]{32}$")
_UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")

_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "exception_id",
        "source",
        "class",
        "trading_day",
        "account",
        "instrument",
        "side",
        "amount",
        "unit",
        "expected",
        "actual",
        "detail",
        "detected_by",
        "detected_at",
    }
)
_SIDE_FIELDS = frozenset({"value", "record"})
_DETECTED_BY_FIELDS = frozenset({"job", "manifest_key", "document_key", "void"})
_DAY_FIELDS = frozenset({"schema_version", "trading_day", "records"})


class ExceptionRecordError(ValueError):
    """A document that is not a valid `exception.v1` record, or a detector
    document a record cannot be built from. Raised, never defaulted around."""


def exceptions_key(trading_day: str) -> str:
    """The store key of ``trading_day``'s exception records."""
    _require_day(trading_day)
    return f"{EXCEPTIONS_PREFIX}{trading_day}.json"


def exception_id(source: str, trading_day: str, identity: Sequence[str | None]) -> str:
    """The record's id: a digest of what the break IS, never of when it was
    seen, so the same break always has the same id."""
    payload = json.dumps([source, trading_day, *identity], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclasses.dataclass(frozen=True)
class ExceptionRecord:
    """One break. See the module docstring for every field."""

    source: str
    klass: str
    trading_day: str
    account: str | None
    instrument: str | None
    side: str | None
    amount: float | None
    expected_value: float | None
    expected_record: str | None
    actual_value: float | None
    actual_record: str | None
    detail: str
    document_key: str
    void: bool
    detected_at: str
    #: What, beyond source, day and class, makes this break this break.
    identity: tuple[str | None, ...]

    @property
    def exception_id(self) -> str:
        return exception_id(self.source, self.trading_day, (self.klass, *self.identity))

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": EXCEPTION_SCHEMA_VERSION,
            "exception_id": self.exception_id,
            "source": self.source,
            "class": self.klass,
            "trading_day": self.trading_day,
            "account": self.account,
            "instrument": self.instrument,
            "side": self.side,
            "amount": self.amount,
            "unit": _UNIT_BY_CLASS.get((self.source, self.klass)),
            "expected": {"value": self.expected_value, "record": self.expected_record},
            "actual": {"value": self.actual_value, "record": self.actual_record},
            "detail": self.detail,
            "detected_by": {
                "job": DETECTING_JOB,
                "manifest_key": manifest_key(DETECTING_JOB, self.trading_day),
                "document_key": self.document_key,
                "void": self.void,
            },
            "detected_at": self.detected_at,
        }
        validate_exception_record(document)
        return document


def records_from_broker_reconciliation(
    document: Mapping[str, object], *, detected_at: str
) -> list[ExceptionRecord]:
    """The breaks in one stored ``broker_reconciliation.v1`` document (as the
    job files it: the result plus ``control_arm`` and ``void``).

    A void run yields a :data:`CONTROL_ARM_VOID` record FIRST, because the void
    is what that run pages on; its findings follow, each marked ``void``.
    """
    day = _str(document, "trading_day")
    account = _str(document, "account")
    void = _bool(document, "void")
    doc_key = trader_reconciliation_key(day)
    statement_key = trader_broker_statement_key(day)
    records: list[ExceptionRecord] = []
    if void:
        control = _mapping(document, "control_arm")
        missed = ", ".join(_str_list(control, "missed")) or "nothing named"
        records.append(
            ExceptionRecord(
                source=SOURCE_BROKER_RECONCILIATION,
                klass=CONTROL_ARM_VOID,
                trading_day=day,
                account=account,
                instrument=None,
                side=None,
                amount=None,
                expected_value=None,
                expected_record=None,
                actual_value=None,
                actual_record=None,
                detail=f"the control arm missed planted {missed}; the run is VOID",
                document_key=doc_key,
                void=True,
                detected_at=detected_at,
                identity=(account,),
            )
        )
    for finding in _mapping_list(document, "discrepancies"):
        klass = _str(finding, "klass")
        instrument = _str(finding, "instrument")
        side = _str(finding, "side")
        records.append(
            ExceptionRecord(
                source=SOURCE_BROKER_RECONCILIATION,
                klass=klass,
                trading_day=day,
                account=account,
                instrument=instrument,
                side=side,
                amount=_opt_float(finding, "delta"),
                expected_value=_opt_float(finding, "expected"),
                expected_record=doc_key,
                actual_value=_opt_float(finding, "actual"),
                actual_record=statement_key,
                detail=_str(finding, "detail"),
                document_key=doc_key,
                void=void,
                detected_at=detected_at,
                identity=(account, instrument, side),
            )
        )
    return records


def records_from_order_intent_reconciliation(
    document: Mapping[str, object], *, void: bool = False
) -> list[ExceptionRecord]:
    """The breaks in one ``order_intent_reconciliation`` document: every row
    whose class is in its ``BREAK_CLASSES``. ``detected_at`` is the document's
    own ``generated_at``. ``void`` is the same run's broker control-arm verdict.

    An intent row's expected side is the intent (its key and quantity); its
    actual side is the broker's, as the document records it. An orphan has no
    intent, so its expected side is empty.
    """
    day = _str(document, "trading_day")
    detected_at = _str(document, "generated_at")
    doc_key = order_intent_reconciliation_key(day)
    records: list[ExceptionRecord] = []
    for row in _mapping_list(document, "rows"):
        klass = _str(row, "class")
        if klass not in INTENT_BREAK_CLASSES:
            continue
        order_key = _str(row, "order_key")
        symbol = _str(row, "symbol")
        filled = _opt_float(row, "filled_quantity")
        refs = ", ".join(_str_list(row, "broker_refs")) or "none"
        statuses = ", ".join(_str_list(row, "broker_statuses")) or "none"
        if klass == "orphan":
            intended = None
            intent_record = None
            side = None
        else:
            intended = _opt_float(row, "intended_quantity")
            intent_record = order_intent_key(day, symbol)
            side = _str(row, "side")
        records.append(
            ExceptionRecord(
                source=SOURCE_ORDER_INTENT_RECONCILIATION,
                klass=klass,
                trading_day=day,
                account=None,
                instrument=symbol,
                side=side,
                amount=None if intended is None or filled is None else filled - intended,
                expected_value=intended,
                expected_record=intent_record,
                actual_value=filled,
                actual_record=doc_key,
                detail=f"order {order_key}: {klass}; broker refs {refs}; statuses {statuses}",
                document_key=doc_key,
                void=void,
                detected_at=detected_at,
                identity=(order_key,),
            )
        )
    return records


def exception_day_document(
    trading_day: str, records: Sequence[ExceptionRecord]
) -> dict[str, object]:
    """The day's file: every record for ``trading_day``, in detection order.
    Records from another day, or two records with one id, are refused."""
    _require_day(trading_day)
    documents = [r.to_document() for r in records]
    seen: set[object] = set()
    for doc in documents:
        if doc["trading_day"] != trading_day:
            raise ExceptionRecordError(
                f"record {doc['exception_id']} is for {doc['trading_day']}, not {trading_day}"
            )
        if doc["exception_id"] in seen:
            raise ExceptionRecordError(f"two records share id {doc['exception_id']}")
        seen.add(doc["exception_id"])
    document: dict[str, object] = {
        "schema_version": EXCEPTION_SCHEMA_VERSION,
        "trading_day": trading_day,
        "records": documents,
    }
    return document


def validate_exception_day_document(document: Mapping[str, object]) -> None:
    """Raise :class:`ExceptionRecordError` unless ``document`` is a valid day
    file: the three fields, and every record valid and for that day."""
    _exact_fields(document, _DAY_FIELDS, "exception day document")
    if document["schema_version"] != EXCEPTION_SCHEMA_VERSION:
        raise ExceptionRecordError(f"schema_version {document['schema_version']!r}")
    day = _str(document, "trading_day")
    _require_day(day)
    for record in _mapping_list(document, "records"):
        validate_exception_record(record)
        if record["trading_day"] != day:
            raise ExceptionRecordError(f"record {record['exception_id']} is not for {day}")


def validate_exception_record(document: Mapping[str, object]) -> None:
    """Raise :class:`ExceptionRecordError` unless ``document`` is a valid
    `exception.v1` record. The field set is closed, the class must be one its
    source can raise, and the unit is the one its class fixes (or null)."""
    _exact_fields(document, _RECORD_FIELDS, "exception record")
    if document["schema_version"] != EXCEPTION_SCHEMA_VERSION:
        raise ExceptionRecordError(f"schema_version {document['schema_version']!r}")
    source = _str(document, "source")
    if source not in CLASSES_BY_SOURCE:
        raise ExceptionRecordError(f"unknown source {source!r}")
    klass = _str(document, "class")
    if klass not in CLASSES_BY_SOURCE[source]:
        raise ExceptionRecordError(f"class {klass!r} is not one {source} raises")
    _require_day(_str(document, "trading_day"))
    for name in ("account", "instrument", "side", "unit"):
        _opt_str(document, name)
    if document["unit"] != _UNIT_BY_CLASS.get((source, klass)):
        raise ExceptionRecordError(f"unit {document['unit']!r} is not the unit of {klass}")
    _opt_float(document, "amount")
    for name in ("expected", "actual"):
        side = _mapping(document, name)
        _exact_fields(side, _SIDE_FIELDS, name)
        _opt_float(side, "value")
        _opt_str(side, "record")
    _str(document, "detail")
    detected_by = _mapping(document, "detected_by")
    _exact_fields(detected_by, _DETECTED_BY_FIELDS, "detected_by")
    if detected_by["job"] != DETECTING_JOB:
        raise ExceptionRecordError(f"detected_by.job {detected_by['job']!r}")
    _str(detected_by, "manifest_key")
    _str(detected_by, "document_key")
    _bool(detected_by, "void")
    if not _UTC_STAMP.match(_str(document, "detected_at")):
        raise ExceptionRecordError(f"detected_at {document['detected_at']!r} is not a UTC stamp")
    if not _EXCEPTION_ID.match(_str(document, "exception_id")):
        raise ExceptionRecordError(f"exception_id {document['exception_id']!r} is not 32 hex")


def _require_day(day: str) -> None:
    if not _ISO_DAY.match(day):
        raise ExceptionRecordError(f"trading day {day!r} is not YYYY-MM-DD")


def _exact_fields(document: Mapping[str, object], fields: frozenset[str], what: str) -> None:
    keys = frozenset(document)
    if keys != fields:
        raise ExceptionRecordError(
            f"{what}: missing {sorted(fields - keys)}, unexpected {sorted(keys - fields)}"
        )


def _field(document: Mapping[str, object], name: str) -> object:
    if name not in document:
        raise ExceptionRecordError(f"field {name!r} is missing")
    return document[name]


def _str(document: Mapping[str, object], name: str) -> str:
    value = _field(document, name)
    if not isinstance(value, str):
        raise ExceptionRecordError(f"field {name!r} is not a string: {value!r}")
    return value


def _opt_str(document: Mapping[str, object], name: str) -> str | None:
    value = _field(document, name)
    if value is not None and not isinstance(value, str):
        raise ExceptionRecordError(f"field {name!r} is not a string or null: {value!r}")
    return value


def _bool(document: Mapping[str, object], name: str) -> bool:
    value = _field(document, name)
    if not isinstance(value, bool):
        raise ExceptionRecordError(f"field {name!r} is not a boolean: {value!r}")
    return value


def _opt_float(document: Mapping[str, object], name: str) -> float | None:
    """A finite number or null. A bool is not a number here, and NaN is
    refused: a break whose amount is NaN cannot be scored."""
    value = _field(document, name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExceptionRecordError(f"field {name!r} is not a number or null: {value!r}")
    if not math.isfinite(value):
        raise ExceptionRecordError(f"field {name!r} is not finite: {value!r}")
    return float(value)


def _mapping(document: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = _field(document, name)
    if not isinstance(value, Mapping):
        raise ExceptionRecordError(f"field {name!r} is not an object: {value!r}")
    return _as_object_mapping(value)  # pyright: ignore[reportUnknownArgumentType]  # isinstance(Mapping) narrows to Mapping[Unknown, Unknown]


def _mapping_list(document: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    value = _field(document, name)
    if not isinstance(value, list):
        raise ExceptionRecordError(f"field {name!r} is not a list: {value!r}")
    out: list[Mapping[str, object]] = []
    for item in value:  # pyright: ignore[reportUnknownVariableType]  # raw JSON list
        if not isinstance(item, Mapping):
            raise ExceptionRecordError(f"field {name!r} holds a non-object: {item!r}")
        out.append(_as_object_mapping(item))  # pyright: ignore[reportUnknownArgumentType]  # raw JSON object
    return out


def _str_list(document: Mapping[str, object], name: str) -> list[str]:
    value = _field(document, name)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):  # pyright: ignore[reportUnknownVariableType]  # raw JSON list
        raise ExceptionRecordError(f"field {name!r} is not a list of strings: {value!r}")
    return [str(v) for v in value]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # checked above


def _as_object_mapping(value: Mapping[object, object]) -> Mapping[str, object]:
    if not all(isinstance(k, str) for k in value):
        raise ExceptionRecordError(f"an object has a non-string key: {value!r}")
    return {str(k): v for k, v in value.items()}
