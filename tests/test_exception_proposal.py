"""Shadow proposals over `exception.v1` records (alpha-engine-config-I12892
phase 1b): every class has a rule, defects and detector findings are never
promotable, a void run holds everything, and the CLI only reads. Synthetic
documents only."""

from __future__ import annotations

import json

import pytest
from crucible.store import LocalStore

from crucible_trader.exception_proposal import (
    ACTIONS,
    NATURES,
    PROPOSAL_SCHEMA_VERSION,
    ROUNDING_ADJUSTMENT_MAX_USD,
    main,
    proposals_from_store,
    propose,
    propose_day,
)
from crucible_trader.exception_record import (
    CLASSES_BY_SOURCE,
    CONTROL_ARM_VOID,
    SOURCE_BROKER_RECONCILIATION,
    SOURCE_ORDER_INTENT_RECONCILIATION,
    ExceptionRecordError,
    exception_day_document,
    exceptions_key,
    records_from_broker_reconciliation,
    records_from_order_intent_reconciliation,
)

DAY = "2026-10-30"
STAMP = "2026-10-30T20:45:00Z"
LATER = "2026-10-31T01:00:00Z"


def broker_doc(discrepancies, *, void=False):
    return {
        "trading_day": DAY,
        "account": "DU1",
        "void": void,
        "control_arm": {"missed": ["cash_delta"] if void else []},
        "discrepancies": discrepancies,
    }


def finding(klass, *, instrument="AAA", delta=-3.0):
    return {
        "klass": klass,
        "instrument": instrument,
        "expected": 12.0,
        "actual": 9.0,
        "delta": delta,
        "side": "broker_under",
        "detail": klass,
    }


def broker_records(discrepancies, *, void=False):
    return [
        r.to_document()
        for r in records_from_broker_reconciliation(
            broker_doc(discrepancies, void=void), detected_at=STAMP
        )
    ]


def intent_record(klass):
    doc = {
        "trading_day": DAY,
        "generated_at": STAMP,
        "rows": [
            {
                "class": klass,
                "order_key": "k1",
                "symbol": "AAA",
                "side": "buy",
                "intended_quantity": 10.0,
                "filled_quantity": 4.0,
                "broker_refs": ["r1"],
                "broker_statuses": ["Cancelled"],
            }
        ],
    }
    (record,) = records_from_order_intent_reconciliation(doc)
    return record.to_document()


def test_every_class_of_every_source_gets_a_proposal_from_the_closed_sets():
    for source, classes in CLASSES_BY_SOURCE.items():
        for klass in classes:
            if klass == CONTROL_ARM_VOID:
                continue
            if source == SOURCE_BROKER_RECONCILIATION:
                (record,) = broker_records([finding(klass)])
            else:
                assert source == SOURCE_ORDER_INTENT_RECONCILIATION
                record = intent_record(klass)
            doc = propose(record, proposed_at=LATER).to_document()
            assert doc["schema_version"] == PROPOSAL_SCHEMA_VERSION
            assert doc["nature"] in NATURES
            assert doc["action"] in ACTIONS
            assert doc["exception_id"] == record["exception_id"]


def test_only_events_are_ever_promotable():
    records = broker_records(
        [
            finding(k)
            for k in CLASSES_BY_SOURCE[SOURCE_BROKER_RECONCILIATION]
            if k != CONTROL_ARM_VOID
        ]
    )
    records += [intent_record(k) for k in CLASSES_BY_SOURCE[SOURCE_ORDER_INTENT_RECONCILIATION]]
    for record in records:
        proposal = propose(record, proposed_at=LATER)
        assert proposal.promotable == (proposal.nature == "event"), proposal


def test_small_cash_gap_is_a_rounding_event_and_a_large_one_a_defect():
    small, large = broker_records(
        [
            finding("cash_delta", instrument="USD", delta=-ROUNDING_ADJUSTMENT_MAX_USD),
            finding("cash_delta", instrument="USD2", delta=-10.0),
        ]
    )
    assert propose(small, proposed_at=LATER).action == "post_rounding_adjustment"
    assert propose(small, proposed_at=LATER).nature == "event"
    assert propose(large, proposed_at=LATER).action == "investigate_cash"
    assert propose(large, proposed_at=LATER).nature == "defect"


def test_a_void_run_holds_every_proposal_and_promotes_nothing():
    records = broker_records([finding("corporate_action")], void=True)
    void, held = (propose(r, proposed_at=LATER) for r in records)
    assert void.klass == CONTROL_ARM_VOID and void.nature == "detector" and not void.held
    assert held.nature == "event" and held.held and not held.promotable
    assert "HELD" in held.rationale


def test_partial_fill_is_an_event_and_an_orphan_a_defect():
    assert (
        propose(intent_record("closed_partial"), proposed_at=LATER).action == "accept_partial_fill"
    )
    orphan = propose(intent_record("orphan"), proposed_at=LATER)
    assert orphan.nature == "defect" and not orphan.promotable


def test_proposal_id_is_stable_across_runs_and_distinct_per_break():
    a, b = broker_records([finding("share_delta"), finding("share_delta", instrument="BBB")])
    assert propose(a, proposed_at=LATER).proposal_id == propose(a, proposed_at=STAMP).proposal_id
    assert propose(a, proposed_at=LATER).proposal_id != propose(b, proposed_at=LATER).proposal_id


def _day_file(records):
    raw = records_from_broker_reconciliation(broker_doc(records), detected_at=STAMP)
    return exception_day_document(DAY, raw)


def test_propose_day_validates_the_file_first():
    document = _day_file([finding("share_delta")])
    assert len(propose_day(document, proposed_at=LATER)) == 1
    document["schema_version"] = "exception.v0"
    with pytest.raises(ExceptionRecordError):
        propose_day(document, proposed_at=LATER)


def test_store_read_filters_by_day_and_names_a_bad_file(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.put_bytes(exceptions_key(DAY), json.dumps(_day_file([finding("share_delta")])).encode())
    assert len(proposals_from_store(store, since=DAY, until=DAY, proposed_at=LATER)) == 1
    assert proposals_from_store(store, since="2026-10-31", until=None, proposed_at=LATER) == []
    store.put_bytes(exceptions_key("2026-10-31"), b'{"schema_version": "x"}')
    with pytest.raises(ExceptionRecordError, match="2026-10-31"):
        proposals_from_store(store, since=DAY, until=None, proposed_at=LATER)


def test_cli_prints_json_lines_and_writes_nothing_to_the_store(tmp_path):
    root = tmp_path / "store"
    store = LocalStore(root)
    store.put_bytes(
        exceptions_key(DAY), json.dumps(_day_file([finding("corporate_action")])).encode()
    )
    before = sorted(p.name for p in root.rglob("*"))
    out = tmp_path / "proposals.jsonl"
    assert main(["--store", str(root), "--since", DAY, "--out", str(out)]) == 0
    (line,) = out.read_text().splitlines()
    assert json.loads(line)["action"] == "await_broker_corporate_action"
    assert sorted(p.name for p in root.rglob("*")) == before


def test_the_rule_table_must_cover_every_class_exactly():
    from crucible_trader import exception_proposal

    exception_proposal._check_exhaustive(exception_proposal._RULES)  # pyright: ignore[reportPrivateUsage]
    short = dict(exception_proposal._RULES)  # pyright: ignore[reportPrivateUsage]
    short.pop((SOURCE_BROKER_RECONCILIATION, "cash_delta"))
    with pytest.raises(RuntimeError, match="cash_delta"):
        exception_proposal._check_exhaustive(short)  # pyright: ignore[reportPrivateUsage]


def test_store_read_skips_non_json_keys_and_refuses_a_non_object(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.put_bytes(exceptions_key(DAY).replace(".json", ".txt"), b"notes")
    assert proposals_from_store(store, since=DAY, until=None, proposed_at=LATER) == []
    store.put_bytes(exceptions_key(DAY), b"[]")
    with pytest.raises(ExceptionRecordError, match="not a JSON object"):
        proposals_from_store(store, since=DAY, until=None, proposed_at=LATER)


def test_module_runs_as_a_script_and_prints_to_stdout(tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    root = tmp_path / "store"
    store = LocalStore(root)
    store.put_bytes(exceptions_key(DAY), json.dumps(_day_file([finding("share_delta")])).encode())
    monkeypatch.setattr(sys, "argv", ["exception_proposal", "--store", str(root), "--since", DAY])
    monkeypatch.delitem(sys.modules, "crucible_trader.exception_proposal")
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("crucible_trader.exception_proposal", run_name="__main__")
    assert exited.value.code == 0
    (line,) = capsys.readouterr().out.splitlines()
    assert json.loads(line)["action"] == "investigate_shares"
